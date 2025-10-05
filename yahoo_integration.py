# yahoo_integration.py
import os, json, time
from pathlib import Path
from typing import Any, Dict, Optional

from yahoo_oauth import OAuth2
from yahoo_fantasy_api import game as yf_game
from yahoo_fantasy_api import league as yf_league

DATA_DIR = Path("data")
TOKEN_PATH = DATA_DIR / "yahoo_token.json"

# --- OAuth session management ---

def _load_oauth() -> OAuth2:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    client_id = os.getenv("YAHOO_CLIENT_ID", "")
    client_secret = os.getenv("YAHOO_CLIENT_SECRET", "")
    redirect_uri = os.getenv("YAHOO_REDIRECT_URI", "")  # e.g. https://<railway-domain>/auth/yahoo/callback
    if not client_id or not client_secret or not redirect_uri:
        raise RuntimeError("Set YAHOO_CLIENT_ID, YAHOO_CLIENT_SECRET, YAHOO_REDIRECT_URI env vars.")

    # yahoo_oauth can read from a token file; we manage it ourselves
    token_dict = None
    if TOKEN_PATH.exists():
        token_dict = json.loads(TOKEN_PATH.read_text())

    oauth = OAuth2(None, None, from_file=False, consumer_key=client_id, consumer_secret=client_secret, redirect_uri=redirect_uri)
    if token_dict:
        oauth.token = token_dict
        # refresh if needed
        if oauth.token_is_valid() is False:
            if not oauth.refresh_token():
                # force re-login if refresh fails
                raise RuntimeError("Yahoo token expired and refresh failed. Re-auth required.")
    return oauth

def save_token(oauth: OAuth2):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TOKEN_PATH.write_text(json.dumps(oauth.token, indent=2))

# --- High-level helpers ---

def get_game_id(oauth: OAuth2, code: str = "nba") -> str:
    g = yf_game.Game(oauth, code)
    return g.game_id()  # current season’s game id (e.g., NBA current year)  # noqa

def get_league(oauth: OAuth2, league_key: str) -> yf_league.League:
    return yf_league.League(oauth, league_key)

def snapshot_league(league_key: str) -> Dict[str, Any]:
    """
    Pull a compact JSON snapshot of league settings, teams, rosters, and free agents.
    Cache-friendly: call this on demand or via an admin button.
    """
    oauth = _load_oauth()
    lg = get_league(oauth, league_key)

    # Basic meta
    settings = lg.settings()
    teams = lg.teams()
    # Team rosters (could be 10–12 calls; fine for admin refresh)
    rosters = {}
    for t in teams:
        tid = t["team_id"]
        try:
            rosters[tid] = lg.roster(tid)
        except Exception:
            rosters[tid] = []

    # Available players (free agents) – page through a few chunks by position if you like
    # Here we pull a small slice as a demo
    fa_pg = lg.free_agents("PG", 0) or []
    fa_sg = lg.free_agents("SG", 0) or []
    free_agents = fa_pg[:50] + fa_sg[:50]

    snap = {
        "fetched_at": int(time.time()),
        "league_key": league_key,
        "settings": settings,
        "teams": teams,
        "rosters": rosters,
        "free_agents_sample": free_agents,
    }
    # Persist to disk for quick access
    (DATA_DIR / "yahoo_cache.json").write_text(json.dumps(snap, indent=2))
    save_token(oauth)
    return snap
