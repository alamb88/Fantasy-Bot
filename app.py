from fastapi import FastAPI, Query
from fastapi.responses import PlainTextResponse
from index import VideoIndex, answer

app = FastAPI(title="NBA Fantasy Bot (9-cat)")

vi = VideoIndex()
INDEX_LOADED = False
try:
    INDEX_LOADED = vi.load()
except Exception as e:
    # Keep running; you will build the index via Shell after first boot.
    print("Index load skipped:", repr(e))
    INDEX_LOADED = False


@app.get("/health", response_class=PlainTextResponse)
def health():
    return f"ok (index_loaded={INDEX_LOADED}, chunks={len(vi.meta)})"


@app.get("/ask", response_class=PlainTextResponse)
def ask(q: str = Query(..., description="Your question")):
    if vi.index is None or len(vi.meta) == 0:
        return (
            "Index not built yet.\n"
            "Open the Railway Shell and run:\n"
            "mkdir -p /app/data && python index.py --add <youtube_url_1> <youtube_url_2> --save"
        )
    hits = vi.search(q, k=5)
    return answer(q, hits)
