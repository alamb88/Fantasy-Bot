import os
import re
import json
from dataclasses import dataclass, asdict
from typing import List, Optional

import faiss
import numpy as np
import tiktoken
import requests
import yt_dlp

from openai import OpenAI
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    NoTranscriptFound,
    TranscriptsDisabled,
)

# ----- Paths -----
DATA_DIR = "data"
META_PATH = os.path.join(DATA_DIR, "meta.json")
FAISS_PATH = os.path.join(DATA_DIR, "index.faiss")
YAHOO_CACHE = os.path.join(DATA_DIR, "yahoo_cache.json")

client = OpenAI()  # Requires OPENAI_API_KEY env var
enc = tiktoken.get_encoding("cl100k_base")


@dataclass
class Chunk:
    video_id: str
    url: str
    text: str


# ----------------------- helpers -----------------------

def _extract_video_id(url: str) -> str:
    q = parse_qs(urlparse(url).query).get("v")
    if q and len(q) > 0:
        return q[0]
    # fallback for youtu.be/<id> or other forms
    path = urlparse(url).path.strip("/")
    return path.split("/")[-1] if path else url

_VTT_TS = re.compile(r"^\d{1,2}:\d{2}:\d{2}\.\d{3}\s-->\s\d{1,2}:\d{2}:\d{2}\.\d{3}")
_SRT_TS = re.compile(r"^\d{2}:\d{2}:\d{2},\d{3}\s-->\s\d{2}:\d{2}:\d{2},\d{3}")

def _vtt_srt_to_text(raw: str) -> str:
    lines = []
    for line in raw.splitlines():
        l = line.strip()
        if not l:
            continue
        if l.isdigit():
            # srt cue index
            continue
        if _VTT_TS.match(l) or _SRT_TS.match(l):
            continue
        if l.upper().startswith("WEBVTT"):
            continue
        # remove simple tags (italics, font)
        l = re.sub(r"</?[^>]+>", "", l)
        lines.append(l)
    # collapse duplicate captions
    text = " ".join(lines)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def _fetch_transcript_via_transcript_api(video_id: str) -> Optional[str]:
    # Try manual English
    try:
        srt = YouTubeTranscriptApi.get_transcript(video_id, languages=["en", "en-US", "en-GB"])
        return " ".join([x.get("text", "") for x in srt if x.get("text")]).strip()
    except (NoTranscriptFound, TranscriptsDisabled, Exception):
        pass
    # Fallback: auto-generated English
    try:
        transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
        gen = transcripts.find_generated_transcript(["en", "en-US", "en-GB"])
        srt = gen.fetch()
        return " ".join([x.get("text", "") for x in srt if x.get("text")]).strip()
    except Exception:
        return None

def _fetch_transcript_via_ytdlp(video_url: str) -> Optional[str]:
    """
    Use yt-dlp to get subtitle URLs (manual or automatic) and download the file.
    """
    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        "extract_flat": False,  # need full info
        "writesubtitles": False,
        "writeautomaticsub": False,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
    except Exception:
        return None

    # Try manual subtitles first, then auto
    sub_map = info.get("subtitles") or {}
    auto_map = info.get("automatic_captions") or {}

    def pick_url(track_map: dict) -> Optional[str]:
        # Prefer English keys; fall back to any
        for key in ["en", "en-US", "en-GB", "a.en", "en-uk", "en-us"]:
            if key in track_map and track_map[key]:
                return track_map[key][0].get("url")
        for _, lst in track_map.items():
            if lst and lst[0].get("url"):
                return lst[0]["url"]
        return None

    url = pick_url(sub_map) or pick_url(auto_map)
    if not url:
        return None

    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        text = _vtt_srt_to_text(r.text)
        return text if text else None
    except Exception:
        return None


# ----------------------- main index -----------------------

class VideoIndex:
    def __init__(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        self.index: faiss.Index | None = None
        self.meta: List[Chunk] = []

    # ---------- Embedding ----------
    def _embed(self, texts: List[str]) -> np.ndarray:
        res = client.embeddings.create(model="text-embedding-3-large", input=texts)
        X = np.array([d.embedding for d in res.data], dtype="float32")
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        return X

    # ---------- Ingest one video ----------
    def add_video(self, url: str) -> int:
        """
        Fetch transcript (Transcript API first, yt-dlp fallback), chunk, embed, and add to index.
        Returns number of chunks added for this video.
        """
        vid = _extract_video_id(url)

        # 1) Normal API path
        full_text = _fetch_transcript_via_transcript_api(vid)

        # 2) Fallback via yt-dlp subtitle URL (auto captions)
        if not full_text:
            full_text = _fetch_transcript_via_ytdlp(url)

        if not full_text:
            print(f"[skip] No transcript available (API/yt-dlp) for {url}")
            return 0

        # Simple chunking (character-based)
        chunk_size = 1500
        chunks = [full_text[i:i + chunk_size] for i in range(0, len(full_text), chunk_size)]
        if not chunks:
            print(f"[skip] No chunks produced for {url}")
            return 0

        vecs = self._embed(chunks)
        if self.index is None:
            self.index = faiss.IndexFlatIP(vecs.shape[1])

        self.index.add(vecs)
        for c in chunks:
            self.meta.append(Chunk(video_id=vid, url=url, text=c))

        print(f"[ok] {len(chunks)} chunks added from {url}")
        return len(chunks)

    # ---------- Persistence ----------
    def save(self):
        if self.index is None:
            print("Nothing to save (index empty).")
            return
        faiss.write_index(self.index, FAISS_PATH)
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump([asdict(m) for m in self.meta], f, ensure_ascii=False, indent=2)
        print("[ok] Index saved.")

    def load(self) -> bool:
        if not (os.path.exists(FAISS_PATH) and os.path.exists(META_PATH)):
            self.index = None
            self.meta = []
            print("No existing index yet (will build after ingest).")
            return False
        self.index = faiss.read_index(FAISS_PATH)
        with open(META_PATH, "r", encoding="utf-8") as f:
            meta_list = json.load(f)
        self.meta = [Chunk(**m) for m in meta_list]
        print(f"[ok] Loaded {len(self.meta)} chunks from disk.")
        return True

    # ---------- Query ----------
    def search(self, query: str, k: int = 5) -> List[Chunk]:
        if self.index is None:
            raise RuntimeError("Index empty. Ingest videos first.")
        qv = self._embed([query])
        D, I = self.index.search(qv, min(k, len(self.meta)))
        return [self.meta[i] for i in I[0] if i != -1]


SYSTEM_PROMPT = """You are an NBA fantasy assistant for 9-category leagues.
Base your answer on the provided snippets; be concise and practical.
Cite numbered sources like [1], [2] with the video URLs.
"""

def _read_yahoo_context() -> str:
    try:
        if os.path.exists(YAHOO_CACHE):
            snap = json.load(open(YAHOO_CACHE, "r", encoding="utf-8"))
            settings = snap.get("settings", {})
            cats = [c.get("display_name") for c in settings.get("stat_categories", {}).get("stats", []) if c.get("display_name")]
            roster_positions = [p.get("position") for p in settings.get("roster_positions", []) if p.get("position")]
            return (
                f"Yahoo league context: categories={', '.join(cats[:9]) or '(unknown)'}; "
                f"roster positions={', '.join(roster_positions) or '(unknown)'}."
            )
    except Exception:
        pass
    return ""

def answer(query: str, hits: List[Chunk]) -> str:
    context = ""
    for i, h in enumerate(hits, start=1):
        short = (h.text[:300] + "…") if len(h.text) > 300 else h.text
        context += f"[{i}] {short}\n(Source: {h.url})\n\n"

    yahoo_ctx = _read_yahoo_context()

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{query}\n\n{yahoo_ctx}\n\nContext:\n{context}"},
        ],
    )
    return resp.choices[0].message.content
