import os
from fastapi import FastAPI, Query, Header, HTTPException
from fastapi.responses import PlainTextResponse
import yt_dlp

from index import VideoIndex, answer

app = FastAPI(title="NBA Fantasy Bot (9-cat)")

# ---- Init index (tolerate empty) ----
vi = VideoIndex()
INDEX_LOADED = False
try:
    INDEX_LOADED = vi.load()
except Exception as e:
    print("Index load skipped:", repr(e))
    INDEX_LOADED = False

# ================= Public =================

@app.get("/health", response_class=PlainTextResponse)
def health():
    return f"ok (index_loaded={INDEX_LOADED}, chunks={len(vi.meta)})"

@app.get("/ask", response_class=PlainTextResponse)
def ask(q: str = Query(..., description="Your question")):
    if vi.index is None or len(vi.meta) == 0:
        return "Index not built yet. Use /admin/ingest_latest to build it."
    hits = vi.search(q, k=5)
    return answer(q, hits)

# ================= Admin (no Shell needed) =================

ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")
JOSH_URLS = os.getenv("JOSH_URLS", "")

def _require_admin(token_from_header: str):
    if not ADMIN_TOKEN or token_from_header != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="unauthorized")

def fetch_latest_youtube_urls(channel: str, limit: int = 12) -> list[str]:
    """Return latest video URLs from a channel/handle or channel URL."""
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
    _require_admin(x_admin_token)
    urls = [u.strip() for u in JOSH_URLS.replace("\n", " ").split(" ") if u.strip()]
    if not urls:
        return "Set JOSH_URLS env var first."

    added_chunks = 0
    added_videos = 0
    for u in urls:
        try:
            n = vi.add_video(u)
            if n > 0:
                added_videos += 1
                added_chunks += n
        except Exception as e:
            print("Ingest error:", u, repr(e))

    if added_chunks > 0:
        vi.save()
        global INDEX_LOADED
        INDEX_LOADED = True

    return f"Ingest complete. videos_added={added_videos}, chunks_added={added_chunks}, total_chunks={len(vi.meta)}"

@app.post("/admin/ingest_latest", response_class=PlainTextResponse)
def admin_ingest_latest(
    x_admin_token: str = Header(default=""),
    channel: str = "@LockedOnFantasyBasketball",
    limit: int = 12,
):
    _require_admin(x_admin_token)
    urls = fetch_latest_youtube_urls(channel, limit=limit)
    if not urls:
        return "No videos found for channel."

    added_chunks = 0
    added_videos = 0
    for u in urls:
        try:
            n = vi.add_video(u)
            if n > 0:
                added_videos += 1
                added_chunks += n
        except Exception as e:
            print("Ingest error:", u, repr(e))

    if added_chunks > 0:
        vi.save()
        global INDEX_LOADED
        INDEX_LOADED = True

    return (
        f"Ingest complete. videos_added={added_videos}, "
        f"chunks_added={added_chunks}, total_chunks={len(vi.meta)}"
    )

# ---------- Diagnostics ----------

from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled

@app.get("/admin/diagnose_latest", response_class=PlainTextResponse)
def admin_diagnose_latest(
    x_admin_token: str = Header(default=""),
    channel: str = "@LockedOnFantasyBasketball",
    limit: int = 8,
):
    _require_admin(x_admin_token)
    urls = fetch_latest_youtube_urls(channel, limit=limit)
    if not urls:
        return "No videos found."

    lines = []
    ok = 0
    for u in urls:
        vid = u.split("v=")[-1]
        has = False
        try:
            srt = YouTubeTranscriptApi.get_transcript(vid, languages=["en", "en-US", "en-GB"])
            has = bool(srt)
        except (NoTranscriptFound, TranscriptsDisabled, Exception):
            has = False
        if not has:
            try:
                trs = YouTubeTranscriptApi.list_transcripts(vid)
                gen = trs.find_generated_transcript(["en","en-US","en-GB"])
                srt = gen.fetch()
                has = bool(srt)
            except Exception:
                has = False
        lines.append(("✅" if has else "❌") + " " + u)
        ok += 1 if has else 0

    return f"Transcripts found: {ok}/{len(urls)}\n" + "\n".join(lines)

@app.post("/admin/ingest_one", response_class=PlainTextResponse)
def admin_ingest_one(
    x_admin_token: str = Header(default=""),
    url: str = Query(..., description="Full YouTube watch URL"),
):
    _require_admin(x_admin_token)
    try:
        n = vi.add_video(url)
        if n > 0:
            vi.save()
            global INDEX_LOADED
            INDEX_LOADED = True
        return f"URL ingested. chunks_added={n}, total_chunks={len(vi.meta)}"
    except Exception as e:
        return f"Error: {repr(e)}"

    global INDEX_LOADED
    INDEX_LOADED = True
    return f"Ingest complete. Added {len(urls)} videos from {channel}."

