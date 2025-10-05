import os
from fastapi import FastAPI, Query, Header, HTTPException
from fastapi.responses import PlainTextResponse
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled

from index import VideoIndex, answer

app = FastAPI(title="NBA Fantasy Bot (9-cat)")

# ---- Init index (tolerate empty) ----
vi = VideoIndex()
try:
    vi.load()
except Exception as e:
    print("Index load skipped:", repr(e))

# ================= Public =================

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

# ================= Admin (no Shell needed) =================

ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")
JOSH_URLS = os.getenv("JOSH_URLS", "")

def _require_admin(token_from_header: str):
    if not ADMIN_TOKEN or token_from_header != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="unauthorized")

def fetch_latest_youtube_items(channel: str, limit: int = 50) -> list[dict]:
    """Return items: id,url,title,duration from a channel handle or /videos URL."""
    channel_url = f"https://www.youtube.com/{channel}/videos" if channel.startswith("@") else channel
    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        "extract_flat": True,   # metadata only
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
    # Quick probe using Transcript API; yt-dlp fallback requires full extraction per URL (slower)
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
            # Try detecting presence via yt-dlp metadata quickly
            try:
                ydl_opts = {"quiet": True, "skip_download": True, "extract_flat": False}
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(u, download=False)
                has = bool((info.get("subtitles") or {}) or (info.get("automatic_captions") or {}))
            except Exception:
                has = False
        lines.append(("✅" if has else "❌") + " " + u)
        ok += 1 if has else 0

    return f"Transcripts found (API or yt-dlp): {ok}/{len(urls)}\n" + "\n".join(lines)

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
        return f"URL ingested. chunks_added={n}, total_chunks={len(vi.meta)}"
    except Exception as e:
        return f"Error: {repr(e)}"


    global INDEX_LOADED
    INDEX_LOADED = True
    return f"Ingest complete. Added {len(urls)} videos from {channel}."

