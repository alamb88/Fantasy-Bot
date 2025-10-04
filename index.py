import os, json, faiss, numpy as np, tiktoken
from openai import OpenAI
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
from dataclasses import dataclass, asdict
from config import CREATOR_WHITELIST, LEAGUE_SETTINGS

client = OpenAI()
enc = tiktoken.get_encoding("cl100k_base")

DATA_DIR = "data"
META_PATH = f"{DATA_DIR}/meta.json"
FAISS_PATH = f"{DATA_DIR}/index.faiss"

@dataclass
class Chunk:
    video_id: str
    url: str
    text: str

class VideoIndex:
    def __init__(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        self.vecs, self.meta = None, []
        self.index = None

    def _embed(self, texts):
        res = client.embeddings.create(model="text-embedding-3-large", input=texts)
        X = np.array([r.embedding for r in res.data], dtype="float32")
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        return X

    def add_video(self, url):
        vid = parse_qs(urlparse(url).query).get("v", [url.split("/")[-1]])[0]
        srt = YouTubeTranscriptApi.get_transcript(vid)
        text = " ".join([x["text"] for x in srt])
        chunks = [text[i:i+1500] for i in range(0, len(text), 1500)]
        vecs = self._embed(chunks)
        self.index = faiss.IndexFlatIP(vecs.shape[1]) if self.index is None else self.index
        self.index.add(vecs)
        for c in chunks:
            self.meta.append(Chunk(vid, url, c))
        print(f"Added {len(chunks)} chunks from {url}")

    def save(self):
        faiss.write_index(self.index, FAISS_PATH)
        json.dump([asdict(m) for m in self.meta], open(META_PATH,"w"), indent=2)
        print("Index saved.")

    def load(self):
        self.index = faiss.read_index(FAISS_PATH)
        self.meta = [Chunk(**m) for m in json.load(open(META_PATH))]
        print(f"Loaded {len(self.meta)} chunks.")

    def search(self, query, k=5):
        qv = self._embed([query])
        D, I = self.index.search(qv, k)
        return [self.meta[i] for i in I[0]]

SYSTEM_PROMPT = """You are an NBA fantasy assistant for 9-cat leagues.
Use only Josh Lloyd sources; cite [1], [2] with URLs. Be concise."""

def answer(query, hits):
    ctx = ""
    for i, h in enumerate(hits, 1):
        ctx += f"[{i}] {h.text[:300]}...(Source: {h.url})\n"
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"system","content":SYSTEM_PROMPT},
                  {"role":"user","content":f"{query}\n\nContext:\n{ctx}"}]
    )
    return resp.choices[0].message.content

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--add", nargs="+")
    p.add_argument("--ask", type=str)
    args = p.parse_args()
    vi = VideoIndex()
    if args.add:
        for u in args.add: vi.add_video(u)
        vi.save()
    if args.ask:
        vi.load()
        print(answer(args.ask, vi.search(args.ask)))
