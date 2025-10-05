import os
from fastapi import FastAPI, Query, Header, HTTPException
from fastapi.responses import PlainTextResponse
import yt_dlp

from index import VideoIndex, answer

app = FastAPI(title="NBA Fantasy Bot (9-cat)")

# ----- Init index (tolerate empty) -----
vi = VideoIndex()
INDEX_LOADED = False
try:
    INDEX_LOADED = vi.load()
except Exception as e:
    print("Index load skipped:", repr(e))
    INDEX_LOADED = False

# ============== Public routes ==============

@app.get("/health", response_class=PlainTextResponse)
def health():
    return f"ok (index_loaded={INDEX_LOADED}, chunks={len(vi.meta)})"

@app.get("/ask", response_class=PlainTextResponse)
def ask(q: str = Query(..., description="Your question")):
    if vi.index is None or len(vi.meta) == 0:
        return "Index not built yet. POST to /admin/ingest_latest with your admin token to build it."
    hits = vi.search(q, k=5)
    return answer(q, hits)

# ============== Admin helpers (no Shell needed) ==============

ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")
JOSH_URLS = os.getenv("JOSH_URLS", "")

def fetch_latest_youtube_urls(channel: str, limit: int = 12) -> list[str]:
    # channel can be '@JoshLloyd48' or a full channel URL
    channel_url = f"https://www.youtube.com/{channel}/videos" if channel.startswith("@") else channel
    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        "extract_flat": True,
        "playlistend": limit,
    }
    urls: list[str] = []
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(channel_url, download=False)
        entries = info.get("entries") or []
        for e in entries[:limit]:
            vid = e.get("id")
            if vid:
                urls.append(f"https://www.youtube.com/watch?v={vid}")
    return urls

@app.post("/admin/ingest", response_class=PlainTextResponse)
def admin_ingest(x_admin_token: str = Header(default="")):
    if not ADMIN_TOKEN or x_admin_token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="unauthorized")
    urls = [u.strip() for u in JOSH_URLS.replace("\n", " ").split(" ") if u.strip()]
    if not urls:
        return "Set JOSH_URLS env var first."
    for u in urls:
        try:
            vi.add_video(u)
        except Exception as e:
            print("Ingest error:", u, repr(e))
    vi.save()
    global INDEX_LOADED
    INDEX_LOADED = True
    return f"Ingest complete. Added {len(urls)} videos."

@app.post("/admin/ingest_latest", response_class=PlainTextResponse)
def admin_ingest_latest(
    x_admin_token: str = Header(default=""),
    channel: str = "@JoshLloyd48",
    limit: int = 12,
):
    if not ADMIN_TOKEN or x_admin_token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="unauthorized")
    urls = fetch_latest_youtube_urls(channel, limit=limit)
    if not urls:
        return "No videos found for channel."
    for u in urls:
        try:
            vi.add_video(u)
        except Exception as e:
            print("Ingest error:", u, repr(e))
    vi.save()
    global INDEX_LOADED
    INDEX_LOADED = True
    return f"Ingest complete. Added {len(urls)} videos from {channel}."

