import os
from fastapi import FastAPI, Query, Header, HTTPException
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled

from index import VideoIndex, answer

# 1) Create the app first
app = FastAPI(title="NBA Fantasy Bot (9-cat)")

# 2) Then add CORS (for Lovable UI)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://*.lovable.dev",
        "https://*.lovable.app",
        # add your own UI domain(s) here if needed:
        # "https://your-domain.com",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ================= Config / Admin =================

ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")
JOSH_URLS = os.getenv("JOSH_URLS", "")

def _require_admin(token_from_header: str):
    if not ADMIN_TOKEN or token_from_header != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="unauthorized")

# ---------- YouTube helpers ----------

def fetch_latest_youtube_items(channel: str, limit: int = 50) -> list[dict]:
    """
    Return items: id,url,title,duration from a channel handle (@handle) or /videos URL.
    We use yt-dlp metadata extraction (no downloads).
    """
    channel_url = f"https://www.youtube.com/{channel}/videos" if channel.startswith("@") else channel
    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        "extract_flat": True,  # metadata only
        "playlistend": limit,
    }
    items: list[dict] = []
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(channel_url, download=False)
        for e in (info.get("entries") or []):
            vid = e.get("id")
            title = e.get("title") or ""
            duration = e.get("duration")
            if vid:
                items.append({
                    "id": vid,
                    "url": f"https://www.youtube.com/watch?v={vid}",
                    "title": title,
                    "duration": duration if isinstance(duration, (int, float)) else None
                })
    return items

def fetch_latest_youtube_urls(channel: str, limit: int = 12) -> list[str]:
    return [it["url"] for it in fetch_latest_youtube_items(channel, limit)]

def _passes_filters(item: dict, min_duration: int, include_kw: list[str], exclude_kw: list[str]) -> bool:
    title = (item.get("title") or "").lower()
    dur = item.get("duration") or 0
    if dur and dur < min_duration:
        return False
    if include_kw and not any(kw in title for kw in include_kw):
        return False
    if exclude_kw and any(kw in title for kw in exclude_kw):
        return False
    return True

# ================= Index init =================

vi = VideoIndex()
try:
    vi.load()
except Exception as e:
    print("Index load skipped:", repr(e))

# ---- Auto-ingest on startup if index is empty (for setups without Volumes) ----
AUTO_INGEST = os.getenv("AUTO_INGEST", "0") == "1"
AUTO_CHANNEL = os.getenv("AUTO_CHANNEL", "https://www.youtube.com/@LockedOnFantasyBasketball/videos")
AUTO_LIMIT = int(os.getenv("AUTO_LIMIT", "20"))
AUTO_FILTERED = os.getenv("AUTO_FILTERED", "1") == "1"
AUTO_MIN_DURATION = int(os.getenv("AUTO_MIN_DURATION", "600"))  # seconds
AUTO_INCLUDE = os.getenv("AUTO_INCLUDE", "tier,sleeper,bust,punt,rank,draft")
AUTO_EXCLUDE = os.getenv("AUTO_EXCLUDE", "short,stream,live,clip")

try:
    needs_build = (vi.index is None or len(vi.meta) == 0)
    if needs_build and AUTO_INGEST:
        if AUTO_FILTERED:
            items = fetch_latest_youtube_items(AUTO_CHANNEL, limit=AUTO_LIMIT)
            include_kw = [s.strip().lower() for s in AUTO_INCLUDE.split(",") if s.strip()]
            exclude_kw = [s.strip().lower() for s in AUTO_EXCLUDE.split(",") if s.strip()]
            for it in items:
                if _passes_filters(it, AUTO_MIN_DURATION, include_kw, exclude_kw):
                    vi.add_video(it["url"])
        else:
            for url in fetch_latest_youtube_urls(AUTO_CHANNEL, limit=AUTO_LIMIT):
                vi.add_video(url)
        if vi.index is not None and len(vi.meta) > 0:
            vi.save()
            print(f"[auto-ingest] Built index with {len(vi.meta)} chunks.")
        else:
            print("[auto-ingest] No items ingested on startup.")
except Exception as e:
    print("[auto-ingest] skipped due to error:", repr(e))

# ================= Public routes =================

@app.get("/health", response_class=PlainTextResponse)
def health():
    index_loaded = (vi.index is not None and len(vi.meta) > 0)
    return f"ok (index_loaded={index_loaded}, chunks={len(vi.meta)})"

@app.get("/ask", response_class=PlainTextResponse)
def ask(q: str = Query(..., description="Your question")):
    if vi.index is None or len(vi.meta) == 0:
        return "Index not built yet. Use /admin/ingest_latest or /admin/ingest_filtered to build it."
    hits = vi.search(q, k=5)
    return answer(q, hits)

# Optional POST variant if you prefer JSON body from Lovable
# from pydantic import BaseModel
# class AskBody(BaseModel): q: str
# @app.post("/ask", response_class=PlainTextResponse)
# def ask_post(body: AskBody):
#     if vi.index is None or len(vi.meta) == 0:
#         return "Index not built yet. Use /admin/ingest_filtered to build it."
#     hits = vi.search(body.q, k=5)
#     return answer(body.q, hits)

# ================= Admin routes (no Shell needed) =================

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

    return (
        f"Ingest complete. videos_added={added_videos}, "
        f"chunks_added={added_chunks}, total_chunks={len(vi.meta)}"
    )

@app.post("/admin/ingest_filtered", response_class=PlainTextResponse)
def admin_ingest_filtered(
    x_admin_token: str = Header(default=""),
    channel: str = "@LockedOnFantasyBasketball",
    limit: int = 50,
    min_duration: int = 600,  # 10+ minutes
    include: str = "tier,sleeper,bust,punt,rank,draft",
    exclude: str = "short,stream,live,clip",
):
    _require_admin(x_admin_token)

    include_kw = [s.strip().lower() for s in include.split(",") if s.strip()]
    exclude_kw = [s.strip().lower() for s in exclude.split(",") if s.strip()]

    items = fetch_latest_youtube_items(channel, limit=limit)
    if not items:
        return "No videos found."

    filtered = [it for it in items if _passes_filters(it, min_duration, include_kw, exclude_kw)]
    if not filtered:
        return "No videos matched filters. Try lowering min_duration or changing include/exclude."

    added_videos = 0
    added_chunks = 0
    for it in filtered:
        try:
            n = vi.add_video(it["url"])
            if n > 0:
                added_videos += 1
                added_chunks += n
        except Exception as e:
            print("Ingest error:", it["url"], repr(e))

    if added_chunks > 0:
        vi.save()

    return (
        f"Filtered {len(filtered)}/{len(items)} items "
        f"(min_duration={min_duration}, include={include}, exclude={exclude})\n"
        f"Ingest complete. videos_added={added_videos}, chunks_added={added_chunks}, total_chunks={len(vi.meta)}"
    )

# ---------- Diagnostics ----------

@app.get("/admin/diagnose_latest", response_class=PlainTextResponse)
def admin_diagnose_latest(
    x_admin_token: str = Header(default=""),
    channel: str = "@LockedOnFantasyBasketball",
    limit: int = 20,
):
    _require_admin(x_admin_token)
    urls = fetch_latest_youtube_urls(channel, limit=limit)
    if not urls:
        return "No videos found."

    # Quick probe using Transcript API; yt-dlp fallback presence check via metadata
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
                ydl_opts = {"quiet": True, "skip_download": True, "extract_flat": False}
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(u, download=False)
                has = bool((info.get("subtitles") or {}) or (info.get("automatic_captions") or {}))
            except Exce
