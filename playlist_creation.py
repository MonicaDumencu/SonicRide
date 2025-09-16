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
    return ["jazz", "acoustic"], 0.4, 90


def validate_seed_genres(sp, desired, needed=2):
    """Return a list of valid seed genres filtered from desired. If insufficient,
    augment with other available seeds. Ensures at least 1 (preferably 'needed')."""
    # Use a curated list of definitely valid genres instead of API call
    # These are based on the official Spotify genre seeds documentation
    available = {
        "acoustic", "afrobeat", "alt-rock", "alternative", "ambient", "blues", "bossanova", "brazil",
        "breakbeat", "british", "chill", "classical", "club", "country", "dance", "deep-house", 
        "disco", "drum-and-bass", "dub", "dubstep", "edm", "electronic", "folk", "funk", "garage",
        "gospel", "groove", "grunge", "hip-hop", "house", "idm", "indie", "indie-pop", "jazz",
        "latin", "metal", "new-age", "pop", "punk", "r-n-b", "reggae", "rock", "soul", "techno",
        "trance", "world-music"
    }
    
    valid = [g for g in desired if g in available]
    print(f"[DEBUG] Desired genres: {desired}, Valid from desired: {valid}")
    
    if len(valid) < 1:
        # pick safe defaults based on mood
        fallback_map = {
            ("jazz", "acoustic"): ["jazz", "acoustic"],
            ("pop", "indie"): ["pop", "indie"],  
            ("rock", "edm"): ["rock", "electronic"]
        }
        desired_tuple = tuple(sorted(desired))
        valid = fallback_map.get(desired_tuple, ["pop", "rock"])
        print(f"[DEBUG] No valid genres from desired, using fallback: {valid}")
    
    # If we want more variety and have room, add others from available
    if len(valid) < needed:
        additional = []
        fallback_genres = ["pop", "rock", "electronic", "indie", "jazz", "acoustic"]
        for g in fallback_genres:
            if g not in valid and g in available:
                additional.append(g)
                if len(valid) + len(additional) >= needed:
                    break
        valid.extend(additional)
        if additional:
            print(f"[DEBUG] Added additional genres for variety: {additional}")
    
    return valid[:needed]  # Ensure we don't exceed the limit


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
    redirect_uri = args.redirect_uri or os.getenv("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8888/callback")

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

    # Try using Client Credentials first for recommendations, then OAuth for playlist creation
    client_creds_sp = spotipy.Spotify(client_credentials_manager=spotipy.SpotifyClientCredentials(
        client_id=client_id,
        client_secret=client_secret
    ))
    
    # OAuth for playlist creation (requires user permissions)
    scope = "playlist-modify-private user-read-private"
    if args.public:
        scope = scope.replace("playlist-modify-private", "playlist-modify-public")

    oauth_sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri,
        scope=scope
    ))
    
    # Use client credentials for recommendations, OAuth for user operations
    sp = oauth_sp  # For compatibility with existing code

    if args.debug:
        # Test Client Credentials
        try:
            search_test = client_creds_sp.search("test", type="track", limit=1)
            print(f"[DEBUG] Client Credentials working - search returned {len(search_test['tracks']['items'])} items")
        except Exception as e:
            print(f"[DEBUG] Client Credentials failed: {e}", file=sys.stderr)
            
        # Test OAuth
        try:
            me = oauth_sp.me()
            print(f"[DEBUG] OAuth working - user id: {me.get('id')}")
        except Exception as e:
            print(f"[DEBUG] OAuth failed: {e}", file=sys.stderr)


    seed_genres = validate_seed_genres(client_creds_sp, seed_genres, needed=2)
    print(f"Requesting {args.limit} recommendations for genres={seed_genres}, energy={target_energy}, tempo={target_tempo}")
    
    # Note: Recommendations API requires 250k+ MAUs as of May 2025 - skip directly to search
    tracks = []
    print("[DEBUG] Note: Spotify Recommendations API requires 250k+ MAUs as of May 2025")
    print("📻 Using search-based track selection...")
    
    # Use search to find tracks by genre
    search_queries = [
            f"genre:{genre}" for genre in seed_genres
        ] + [
            f"{genre} music" for genre in seed_genres
        ] + [
            "instrumental", "jazz music", "acoustic guitar", "ambient music", "chill music"
        ]
    
    tracks = []
    tracks_found = set()  # Avoid duplicates
    
    for query in search_queries:
            if len(tracks) >= args.limit:
                break
                
        try:
            print(f"[DEBUG] Searching for: {query}")
            search_result = client_creds_sp.search(
                q=query, 
                type='track', 
                limit=min(10, args.limit - len(tracks))
            )
            
            for track in search_result['tracks']['items']:
                if len(tracks) >= args.limit:
                    break
                if track['id'] not in tracks_found:
                    tracks.append(track)
                    tracks_found.add(track['id'])
                    
        except Exception as search_error:
            print(f"[DEBUG] Search query '{query}' failed: {search_error}")
            continue
    
    if not tracks:
        print("❌ Search-based track selection failed", file=sys.stderr)
        sys.exit(1)
    
    print(f"✅ Found {len(tracks)} tracks via search")

    if not tracks:
        print("No tracks found.", file=sys.stderr)
        sys.exit(1)

    track_uris = [t['uri'] for t in tracks]
    user_id = oauth_sp.me()['id']
    playlist_name = f"{args.name_prefix} ({mood.capitalize()})"
    playlist = oauth_sp.user_playlist_create(
        user=user_id,
        name=playlist_name,
        public=args.public,
        description="Auto-generated by RideTrack based on ride metrics"
    )

    oauth_sp.playlist_add_items(playlist_id=playlist['id'], items=track_uris)

    print(f"✅ Playlist created: {playlist['external_urls']['spotify']}")
    print("Tracks added:")
    for t in tracks:
        artist_names = ", ".join(a['name'] for a in t['artists'])
        print(f"- {t['name']} by {artist_names}")


if __name__ == "__main__":
    main()
    
