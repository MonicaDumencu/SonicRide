import os
import argparse
import sys
from pathlib import Path
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyOAuth

try:
    from dotenv import load_dotenv, find_dotenv  # optional
    _dotenv_path = find_dotenv(usecwd=True)
    if not _dotenv_path:
        candidate = os.path.join(os.getcwd(), ".env")
        if os.path.isfile(candidate):
            _dotenv_path = candidate
    if _dotenv_path:
        load_dotenv(dotenv_path=_dotenv_path)
except Exception:
    _dotenv_path = None

# Fallback manual parser if vars still missing after attempted load
if not os.getenv("SPOTIFY_CLIENT_ID") or not os.getenv("SPOTIFY_CLIENT_SECRET"):
    candidate = os.path.join(os.getcwd(), ".env")
    if os.path.isfile(candidate):
        try:
            with open(candidate, 'r', encoding='utf-8') as fh:
                for line in fh:
                    line = line.strip()\
                        .replace('\ufeff', '')  # strip BOM if present
                    if not line or line.startswith('#'):
                        continue
                    if '=' in line:
                        k, v = line.split('=', 1)
                        k = k.strip()
                        v = v.strip().strip('"').strip("'")
                        if k and v and k.startswith('SPOTIFY_') and not os.getenv(k):
                            os.environ[k] = v
        except Exception:
            pass


def determine_mood(avg_speed: float, avg_lean: float):
    if (avg_speed or 0) > 80 or (avg_lean or 0) > 25:
        return "high_energy"
    if (avg_speed or 0) > 50:
        return "moderate"
    return "chill"


def mood_params(mood: str):
    if mood == "high_energy":
        return ["rock", "edm"], 0.85, 130
    if mood == "moderate":
        return ["pop", "indie"], 0.6, 110
    return ["jazz", "chill"], 0.4, 90


def validate_seed_genres(sp, desired, needed=2):
    """Return a list of valid seed genres filtered from desired. If insufficient,
    augment with other available seeds. Ensures at least 1 (preferably 'needed')."""
    try:
        available = set(sp.recommendation_genre_seeds())
    except Exception:
        # Fallback static minimal list likely to exist
        available = {"rock", "pop", "indie", "jazz", "edm", "hip-hop", "acoustic", "ambient"}
    valid = [g for g in desired if g in available]
    if len(valid) < 1:
        # pick a safe default
        valid = ["rock"]
    # If we want more variety and have room, add others from available
    if len(valid) < needed:
        for g in sorted(available):
            if g not in valid:
                valid.append(g)
            if len(valid) >= needed:
                break
    return valid


def main():
    parser = argparse.ArgumentParser(description="Create a Spotify playlist based on ride metrics.")
    parser.add_argument("--metrics", default="metrics.csv", help="Path to ride metrics CSV (default: metrics.csv)")
    parser.add_argument("--limit", type=int, default=15, help="Number of recommendation tracks")
    parser.add_argument("--name-prefix", default="RideTrack Playlist", help="Playlist name prefix")
    parser.add_argument("--public", action="store_true", help="Create playlist as public (default private)")
    parser.add_argument("--debug", action="store_true", help="Print debug info for env loading")
    parser.add_argument("--redirect-uri", dest="redirect_uri", help="Override redirect URI (must match Spotify app)")
    args = parser.parse_args()

    metrics_path = Path(args.metrics)
    if not metrics_path.exists():
        print(f"Error: metrics file not found: {metrics_path}", file=sys.stderr)
        sys.exit(1)

    try:
        df = pd.read_csv(metrics_path)
    except Exception as e:
        print(f"Failed to read metrics CSV: {e}", file=sys.stderr)
        sys.exit(1)

    speed_series = df.get('speed_kmh')
    lean_series = df.get('lean_angle_deg')
    if speed_series is None or lean_series is None:
        print("Error: metrics CSV missing required columns 'speed_kmh' and/or 'lean_angle_deg'", file=sys.stderr)
        sys.exit(1)

    avg_speed = speed_series.dropna().mean() if not speed_series.dropna().empty else 0.0
    avg_lean = lean_series.dropna().mean() if not lean_series.dropna().empty else 0.0
    print(f"Avg Speed: {avg_speed:.1f} km/h, Avg Lean: {avg_lean:.1f}°")

    mood = determine_mood(avg_speed, avg_lean)
    print(f"Ride mood: {mood}")
    seed_genres, target_energy, target_tempo = mood_params(mood)

    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
    redirect_uri = args.redirect_uri or os.getenv("SPOTIFY_REDIRECT_URI", "http://localhost:8888/callback")

    if args.debug:
        print(f"[DEBUG] dotenv path: {_dotenv_path}")
        print(f"[DEBUG] CWD: {os.getcwd()}")
        print(f"[DEBUG] SPOTIFY_CLIENT_ID present? {'yes' if client_id else 'no'}")
        print(f"[DEBUG] SPOTIFY_CLIENT_SECRET present? {'yes' if client_secret else 'no'}")
        # show first 4 chars only if present
        if client_id:
            print(f"[DEBUG] SPOTIFY_CLIENT_ID startswith: {client_id[:4]}")

    missing = [name for name, val in [("SPOTIFY_CLIENT_ID", client_id), ("SPOTIFY_CLIENT_SECRET", client_secret)] if not val]
    if missing:
        print(f"Error: Missing required environment variables: {', '.join(missing)}", file=sys.stderr)
        hint = "Ensure you created a .env file with SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET."\
               " Example lines:\nSPOTIFY_CLIENT_ID=...\nSPOTIFY_CLIENT_SECRET=..."
        if not _dotenv_path:
            hint += "\n(No .env file was found by python-dotenv.)"
        else:
            hint += f"\nLoaded .env path: {_dotenv_path}"
        print(hint, file=sys.stderr)
        sys.exit(1)

    scope = "playlist-modify-private user-read-private"
    if args.public:
        scope = scope.replace("playlist-modify-private", "playlist-modify-public")

    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri,
        scope=scope
    ))

    if args.debug:
        try:
            me = sp.me()
            print(f"[DEBUG] Authenticated user id: {me.get('id')}")
        except Exception as e:
            print(f"[DEBUG] Failed fetching current user: {e}", file=sys.stderr)


    seed_genres = validate_seed_genres(sp, seed_genres, needed=2)
    print(f"Requesting {args.limit} recommendations for genres={seed_genres}, energy={target_energy}, tempo={target_tempo}")
    try:
        rec = sp.recommendations(
            seed_genres=seed_genres,
            limit=args.limit,
            target_energy=target_energy,
            target_tempo=target_tempo
        )
    except Exception as e:
        if args.debug:
            # Raw request attempt
            import requests
            from spotipy.oauth2 import SpotifyOAuth as _So
            # get token directly
            # access underlying auth manager via sp.auth_manager
            token_info = getattr(sp.auth_manager, 'cache_handler', None)
            print(f"[DEBUG] Recommendation failure: {e}")
            try:
                auth_mgr = sp.auth_manager
                access_token = auth_mgr.get_access_token(as_dict=False)
                print(f"[DEBUG] Access token length: {len(access_token)}")
                hdrs = {"Authorization": f"Bearer {access_token}"}
                test_url = "https://api.spotify.com/v1/recommendations/available-genre-seeds"
                r = requests.get(test_url, headers=hdrs, timeout=10)
                print(f"[DEBUG] Direct genre seeds status: {r.status_code}")
                print(f"[DEBUG] Direct genre seeds body (truncated 300): {r.text[:300]}")
            except Exception as ie:
                print(f"[DEBUG] Raw token/genre fetch failed: {ie}")
        print(f"Error fetching recommendations: {e}", file=sys.stderr)
        sys.exit(1)

    tracks = rec.get('tracks', [])
    if not tracks:
        print("No tracks returned from Spotify recommendations.", file=sys.stderr)
        sys.exit(1)

    track_uris = [t['uri'] for t in tracks]
    user_id = sp.me()['id']
    playlist_name = f"{args.name_prefix} ({mood.capitalize()})"
    playlist = sp.user_playlist_create(
        user=user_id,
        name=playlist_name,
        public=args.public,
        description="Auto-generated by RideTrack based on ride metrics"
    )

    sp.playlist_add_items(playlist_id=playlist['id'], items=track_uris)

    print(f"✅ Playlist created: {playlist['external_urls']['spotify']}")
    print("Tracks added:")
    for t in tracks:
        artist_names = ", ".join(a['name'] for a in t['artists'])
        print(f"- {t['name']} by {artist_names}")


if __name__ == "__main__":
    main()
    
