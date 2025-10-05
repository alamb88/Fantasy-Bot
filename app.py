import os
import json
import time
import threading
from typing import Set, Any, Dict, Optional, List

from fastapi import FastAPI, Query, Header, HTTPException, BackgroundTasks
from fastapi.responses import (
    PlainTextResponse,
    RedirectResponse,
    JSONResponse,
    StreamingResponse,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled

# ---- core index + helpers (your existing index.py / yahoo_integration.py) ----
from index import (
    VideoIndex,
    answer,
    SYSTEM_PROMPT,
    _read_yahoo_context,
    client as openai_client,  # reuse OpenAI client for streaming
)
from yahoo_integration import _load_oauth, save_token, get_game_id, snapshot_league


# ================= FastAPI App =================
# ---- CORS Configuration ----
# For development: allow all origins (simplest fix)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=False,  # Must be False when using "*"
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],  # Allow all headers including Content-Type, Accept, x-admin-token
    expose_headers=["*"],
)

# Global OPTIONS responder (helps strict preflights)
@app.options("/{path:path}")
def options_handler():
    return PlainTextResponse("", status_code=204, headers={
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "*",
    })


# Global OPTIONS responder (helps strict preflights)
@app.options("/{path:path}")
def options_handler():
    return PlainTextResponse("", status_code=204)


# ================= Config / Admin =================

ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")
JOSH_URLS = os.getenv("JOSH_URLS", "")

def _require_admin(token_from_header: str):
    if not ADMIN_TOKEN or token_from_header != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="unauthorized")


# ---------- YouTube helpers ----------

def fetch_latest_youtube_items(channel: str, limit: int = 50) -> List[dict]:
    """
    Return items: id,url,title,duration from a channel handle (@handle),
    playlist URL, or /videos URL.
    """
    channel_url = (
        f"https://www.youtube.com/{channel}/videos" if channel.startswith("@") else channel
    )
    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        "extract_flat": True,  # metadata only
        "playlistend": limit,
        "retries": 3,
        "file_access_retries": 3,
        "nocheckcertificate": True,
        "geo_bypass": True,
        "http_headers": {"User-Agent": "Mozilla/5.0"},
    }
    items: List[dict] = []
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(channel_url, download=False)
        for e in (info.get("entries") or []):
            vid = e.get("id")
            title = e.get("title") or ""
            duration = e.get("duration")
            if vid:
                items.append(
                    {
                        "id": vid,
                        "url": f"https://www.youtube.com/watch?v={vid}",
                        "title": title,
                        "duration": duration if isinstance(duration, (int, float)) else None,
                    }
                )
    return items

def fetch_latest_youtube_urls(channel: str, limit: int = 12) -> List[str]:
    return [it["url"] for it in fetch_latest_youtube_items(channel, limit)]

def _passes_filters(item: dict, min_duration: int, include_kw: List[str], exclude_kw: List[str]) -> bool:
    title = (item.get("title") or "").lower()
    dur = item.get("duration") or 0
    if dur and dur < min_duration:
        return False
    if include_kw and not any(kw in title for kw in include_kw):
        return False
    if exclude_kw and any(kw in title for kw in exclude_kw):
        return False
    return True

def fetch_channel_items_all(channel: str, max_items: int = 5000) -> List[dict]:
    """
    Traverse a channel/playlist and return up to max_items entries with id,title,duration,url.
    Using extract_flat lets yt-dlp fetch long playlists without downloading media.
    """
    channel_url = (
        f"https://www.youtube.com/{channel}/videos" if channel.startswith("@") else channel
    )
    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        "extract_flat": True,
        "playlistend": max_items,
        "noplaylist": False,
        "retries": 3,
        "file_access_retries": 3,
        "nocheckcertificate": True,
        "geo_bypass": True,
        "http_headers": {"User-Agent": "Mozilla/5.0"},
    }
    items: List[dict] = []
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(channel_url, download=False)
        for e in (info.get("entries") or []):
            vid = e.get("id")
            title = e.get("title") or ""
            duration = e.get("duration")
            if vid:
                items.append(
                    {
                        "id": vid,
                        "url": f"https://www.youtube.com/watch?v={vid}",
                        "title": title,
                        "duration": duration if isinstance(duration, (int, float)) else None,
                    }
                )
    return items


# ================= Index init =================

vi = VideoIndex()
try:
    vi.load()
except Exception as e:
    print("Index load skipped:", repr(e))

def _seen_ids_from_meta() -> Set[str]:
    return set(m.video_id for m in vi.meta)

# ---- Auto-ingest on startup (optional) ----
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

# ---- Optional: auto-cache Yahoo on boot ----
AUTO_YAHOO_LEAGUE_KEY = os.getenv("AUTO_YAHOO_LEAGUE_KEY", "")
if AUTO_YAHOO_LEAGUE_KEY:
    try:
        snapshot_league(AUTO_YAHOO_LEAGUE_KEY)
        print("[auto] Yahoo league cached.")
    except Exception as e:
        print("[auto] Yahoo cache failed:", repr(e))


# ================= Public routes =================

@app.get("/health", response_class=PlainTextResponse)
def health():
    index_loaded = (vi.index is not None and len(vi.meta) > 0)
    return f"ok (index_loaded={index_loaded}, chunks={len(vi.meta)})"


# --------- GET /ask (legacy) ---------
@app.get("/ask", response_class=PlainTextResponse)
def ask_get(q: str = Query(..., description="Your question")):
    if vi.index is None or len(vi.meta) == 0:
        return "Index not built yet. Use /admin/ingest_latest or /admin/ingest_filtered to build it."
    hits = vi.search(q, k=5)
    return answer(q, hits)


# --------- POST /ask (for Lovable chat) ---------

class AskRequest(BaseModel):
    message: str
    draftContext: Optional[Dict[str, Any]] = None
    k: int = 5  # number of context snippets

@app.post("/ask", response_class=PlainTextResponse)
def ask_post(payload: AskRequest):
    if vi.index is None or len(vi.meta) == 0:
        return "Index not built yet. Use /admin/ingest_latest or /admin/ingest_filtered to build it."

    # Flatten draft context into a readable preface
    ctx_lines = []
    if payload.draftContext:
        for key, val in payload.draftContext.items():
            ctx_lines.append(f"- {key}: {val}")
    ctx_block = "\n".join(ctx_lines)

    full_query = f"{payload.message}\n\nDraft context:\n{ctx_block}" if ctx_block else payload.message
    hits = vi.search(full_query, k=max(1, payload.k))
    return answer(full_query, hits)


# --------- SSE streaming: POST + GET fallbacks ---------

def _build_context_snippets(hits):
    blocks = []
    for i, h in enumerate(hits, start=1):
        short = (h.text[:300] + "…") if len(h.text) > 300 else h.text
        blocks.append(f"[{i}] {short}\n(Source: {h.url})\n")
    return "\n".join(blocks)

def _sse_wrap(text: str) -> str:
    # SSE: each event line begins with "data:" and ends with a blank line
    return f"data: {text}\n\n"

@app.post("/ask_stream")
def ask_stream_post(payload: AskRequest):
    if vi.index is None or len(vi.meta) == 0:
        return PlainTextResponse("Index not built yet.", status_code=400)

    # Flatten draft context
    ctx_lines = []
    if payload.draftContext:
        for key, val in payload.draftContext.items():
            ctx_lines.append(f"- {key}: {val}")
    ctx_block = "\n".join(ctx_lines)
    full_query = f"{payload.message}\n\nDraft context:\n{ctx_block}" if ctx_block else payload.message

    hits = vi.search(full_query, k=max(1, payload.k))
    context_snippets = _build_context_snippets(hits)
    yahoo_ctx = _read_yahoo_context()
    user_content = f"{full_query}\n\n{yahoo_ctx}\n\nContext:\n{context_snippets}"

    def sse_gen():
        stream = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            stream=True,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        )
        for chunk in stream:
            delta = getattr(chunk.choices[0].delta, "content", None) or ""
            if delta:
                yield _sse_wrap(delta)

    return StreamingResponse(
        sse_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )

@app.get("/ask_stream")
def ask_stream_get(q: str = Query(..., description="Full query including any context")):
    if vi.index is None or len(vi.meta) == 0:
        return PlainTextResponse("Index not built yet.", status_code=400)

    hits = vi.search(q, k=5)
    context_snippets = _build_context_snippets(hits)
    yahoo_ctx = _read_yahoo_context()
    user_content = f"{q}\n\n{yahoo_ctx}\n\nContext:\n{context_snippets}"

    def sse_gen():
        stream = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            stream=True,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        )
        for chunk in stream:
            delta = getattr(chunk.choices[0].delta, "content", None) or ""
            if delta:
                yield _sse_wrap(delta)

    return StreamingResponse(
        sse_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


# ================= Admin routes =================

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


# ---------------- Full-Channel Ingest (Background) + Failure tracking ----------------

PROGRESS_PATH = os.path.join("data", "ingest_progress.json")
FAIL_PATH = os.path.join("data", "ingest_failures.json")
_ingest_lock = threading.Lock()
_ingest_running = False
_progress = {
    "running": False,
    "started_at": 0,
    "finished_at": 0,
    "added_videos": 0,
    "added_chunks": 0,
    "skipped": 0,
    "errors": 0,
    "last": "",
    "total_candidates": 0,
    "done": False,
}

def _progress_save():
    try:
        os.makedirs("data", exist_ok=True)
        with open(PROGRESS_PATH, "w") as f:
            json.dump(_progress, f, indent=2)
    except Exception:
        pass

def _fail_list_load() -> list:
    try:
        if os.path.exists(FAIL_PATH):
            return json.load(open(FAIL_PATH))
    except Exception:
        pass
    return []

def _fail_list_save(lst: list):
    try:
        os.makedirs("data", exist_ok=True)
        json.dump(lst, open(FAIL_PATH, "w"), indent=2)
    except Exception:
        pass

def _progress_reset():
    global _progress
    _progress = {
        "running": True,
        "started_at": int(time.time()),
        "finished_at": 0,
        "added_videos": 0,
        "added_chunks": 0,
        "skipped": 0,
        "errors": 0,
        "last": "",
        "total_candidates": 0,
        "done": False,
    }
    _progress_save()

def _ingest_channel_worker(channel: str, max_items: int, min_duration: int,
                           include: List[str], exclude: List[str]):
    global _ingest_running
    try:
        items = fetch_channel_items_all(channel, max_items=max_items)
        _progress["total_candidates"] = len(items)
        _progress_save()

        seen = _seen_ids_from_meta()
        failures = _fail_list_load()

        for it in items:
            try:
                _progress["last"] = it.get("url", "")
                title = (it.get("title") or "").lower()
                dur = it.get("duration") or 0
                vid = it.get("id")
                url = it.get("url")

                # Filters
                if dur and dur < min_duration:
                    _progress["skipped"] += 1; _progress_save(); continue
                if include and not any(kw in title for kw in include):
                    _progress["skipped"] += 1; _progress_save(); continue
                if exclude and any(kw in title for kw in exclude):
                    _progress["skipped"] += 1; _progress_save(); continue
                if vid in seen:
                    _progress["skipped"] += 1; _progress_save(); continue

                # Ingest (index.py handles all transcript fallbacks)
                added = vi.add_video(url)
                if added > 0:
                    _progress["added_videos"] += 1
                    _progress["added_chunks"] += added
                    seen.add(vid)
                    if (_progress["added_videos"] % 5) == 0:
                        vi.save()
                else:
                    _progress["skipped"] += 1
                    if url not in failures:
                        failures.append(url)
                        _fail_list_save(failures)
                _progress_save()

            except Exception:
                _progress["errors"] += 1
                if it.get("url") and it["url"] not in failures:
                    failures.append(it["url"])
                    _fail_list_save(failures)
                _progress_save()

        if vi.index is not None and len(vi.meta) > 0:
            vi.save()

    finally:
        _progress["done"] = True
        _progress["running"] = False
        _progress["finished_at"] = int(time.time())
        _progress_save()
        with _ingest_lock:
            _ingest_running = False

@app.post("/admin/ingest_channel_full", response_class=PlainTextResponse)
def admin_ingest_channel_full(
    background_tasks: BackgroundTasks,
    x_admin_token: str = Header(default=""),
    channel: str = Query("@LockedOnFantasyBasketball"),
    max_items: int = Query(5000, ge=1, le=20000),
    min_duration: int = Query(600, ge=0),  # 10+ minutes by default
    include: str = Query("tier,sleeper,bust,punt,rank,draft"),
    exclude: str = Query("short,stream,live,clip"),
):
    _require_admin(x_admin_token)

    include_kw = [s.strip().lower() for s in include.split(",") if s.strip()]
    exclude_kw = [s.strip().lower() for s in exclude.split(",") if s.strip()]

    global _ingest_running
    with _ingest_lock:
        if _ingest_running:
            return "Ingest already running. Check /admin/ingest_progress."
        _ingest_running = True
        _progress_reset()
        background_tasks.add_task(_ingest_channel_worker, channel, max_items, min_duration, include_kw, exclude_kw)

    return "Full-channel ingest started. Check /admin/ingest_progress."

@app.get("/admin/ingest_progress")
def admin_ingest_progress(x_admin_token: str = Header(default="")):
    _require_admin(x_admin_token)
    try:
        if os.path.exists(PROGRESS_PATH):
            return JSONResponse(json.load(open(PROGRESS_PATH)))
    except Exception:
        pass
    return JSONResponse(_progress)

# ---- Failures: see & retry ----

@app.get("/admin/ingest_failures")
def admin_ingest_failures(x_admin_token: str = Header(default="")):
    _require_admin(x_admin_token)
    return JSONResponse({"failures": _fail_list_load()})

@app.post("/admin/retry_failures", response_class=PlainTextResponse)
def admin_retry_failures(
    x_admin_token: str = Header(default=""),
    limit: int = Query(50, ge=1, le=1000)
):
    _require_admin(x_admin_token)
    failures = _fail_list_load()
    if not failures:
        return "No failures recorded."

    to_retry = failures[:limit]
    still_failed = []
    added_videos = 0
    added_chunks = 0

    for url in to_retry:
        try:
            n = vi.add_video(url)
            if n > 0:
                added_videos += 1
                added_chunks += n
            else:
                still_failed.append(url)
        except Exception:
            still_failed.append(url)

    remaining = still_failed + failures[len(to_retry):]
    _fail_list_save(remaining)

    if added_videos > 0:
        vi.save()

    return f"Retried {len(to_retry)}. Success: {added_videos} (chunks {added_chunks}). Remaining failures: {len(remaining)}"


# ================= Yahoo Auth & Data =================

@app.get("/auth/yahoo/login")
def yahoo_login():
    try:
        oauth = _load_oauth()
        if oauth.token and oauth.token_is_valid():
            return PlainTextResponse("Already authorized with Yahoo.")
        auth_url = oauth.generate_authorize_url()
        return RedirectResponse(url=auth_url)
    except Exception as e:
        return PlainTextResponse(f"Error generating Yahoo login: {e}", status_code=500)

@app.get("/auth/yahoo/callback")
def yahoo_callback(code: str = "", state: str = ""):
    try:
        oauth = _load_oauth()
        if not code:
            return PlainTextResponse("Missing 'code' in callback.", status_code=400)
        if not oauth.request_token(code):
            return PlainTextResponse("Failed to exchange code for token.", status_code=401)
        save_token(oauth)
        return PlainTextResponse("Yahoo authorized! You can close this tab.")
    except Exception as e:
        return PlainTextResponse(f"Callback error: {e}", status_code=500)

@app.post("/admin/yahoo/cache_league")
def yahoo_cache_league(
    x_admin_token: str = Header(default=""),
    league_key: str = Query(..., description="Format: {game_id}.l.{league_id} (e.g., 423.l.12345)"),
):
    _require_admin(x_admin_token)
    try:
        snap = snapshot_league(league_key)
        return JSONResponse(snap)
    except Exception as e:
        return PlainTextResponse(f"Yahoo snapshot error: {e}", status_code=500)

@app.get("/yahoo/game_id")
def yahoo_game_id():
    try:
        oauth = _load_oauth()
        gid = get_game_id(oauth, "nba")
        return PlainTextResponse(str(gid))
    except Exception as e:
        return PlainTextResponse(f"Error: {e}", status_code=500)

@app.get("/yahoo/cache")
def yahoo_cache_read():
    p = os.path.join("data", "yahoo_cache.json")
    if not os.path.exists(p):
        return PlainTextResponse("No Yahoo cache yet. Run /auth/yahoo/login then /admin/yahoo/cache_league.", status_code=404)
    return JSONResponse(json.load(open(p)))

