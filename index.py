import os
import json
from dataclasses import dataclass, asdict
from typing import List

import faiss
import numpy as np
import tiktoken
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

client = OpenAI()
enc = tiktoken.get_encoding("cl100k_base")


@dataclass
class Chunk:
    video_id: str
    url: str
    text: str


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
        Fetch transcript (manual or auto), chunk, embed, and add to index.
        Returns number of chunks added for this video.
        """
        vid = parse_qs(urlparse(url).query).get("v", [url.split("/")[-1]])[0]

        # Try manual English transcript first
        srt = None
        try:
            srt = YouTubeTranscriptApi.get_transcript(vid, languages=["en", "en-US", "en-GB"])
        except (NoTranscriptFound, TranscriptsDisabled):
            srt = None

        # Fallback: auto-generated English transcript
        if srt is None:
            try:
                transcripts = YouTubeTranscriptApi.list_transcripts(vid)
                gen = transcripts.find_generated_transcript(["en", "en-US", "en-GB"])
                srt = gen.fetch()
            except Exception:
                srt = None

        if not srt:
            print(f"[skip] No transcript available for {url}")
            return 0

        full_text = " ".join([x.get("text", "") for x in srt if x.get("text")]).strip()
        if not full_text:
            print(f"[skip] Empty transcript for {url}")
            return 0

        # Simple chunking by characters (safe & fast)
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
Base your answer on the provided snippets; keep it concise and practical.
Cite numbered sources like [1], [2] including the video URLs.
"""

def answer(query: str, hits: List[Chunk]) -> str:
    context = ""
    for i, h in enumerate(hits, start=1):
        short = (h.text[:300] + "…") if len(h.text) > 300 else h.text
        context += f"[{i}] {short}\n(Source: {h.url})\n\n"

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{query}\n\nContext:\n{context}"},
        ],
    )
    return resp.choices[0].message.content
