#!/usr/bin/env python3
"""
SonicRide: Smart Spotify Playlist Generator
Creates personalized playlists based on motorcycle ride metrics using intelligent search.
"""
import os
import argparse
import sys
from pathlib import Path
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyOAuth, SpotifyClientCredentials
import random
import time
import math
import io
import base64

from PIL import Image, ImageDraw, ImageFont

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def determine_mood(avg_speed: float, avg_lean: float) -> str:
    """Return a more granular mood label for a ride.

    The function uses simple thresholds on average speed (km/h) and
    average lean (degrees) to choose a descriptive mood. Labels are:
    - 'adrenaline' : extreme riding (very fast or very aggressive lean)
    - 'energetic'  : fast / sporty riding
    - 'upbeat'     : lively but controlled (higher-than-average speed)
    - 'steady'     : regular, confident cruising
    - 'relaxed'    : gentle pace
    - 'serene'     : slow, calm, very relaxed

    These are intentionally coarse and meant to drive different
    music parameter presets in `get_mood_parameters`.
    """
    s = (avg_speed or 0)
    lean = (avg_lean or 0)

    # Adrenaline: very high speed or very large lean angle
    if s >= 110 or lean >= 35:
        return "adrenaline"

    # Energetic: fast or aggressive
    if s >= 85 or lean >= 25:
        return "energetic"

    # Upbeat: above-average speed
    if s >= 60:
        return "upbeat"

    # Steady: normal cruising pace
    if s >= 40:
        return "steady"

    # Relaxed: gentle pace
    if s >= 20:
        return "relaxed"

    # Serene: very slow / calm
    return "serene"


def get_mood_parameters(mood: str) -> tuple:
    """Map granular mood labels to (genres, energy, tempo).

    Returns a tuple (list[str], energy:float, tempo:int).
    """
    # Tuned mapping: genres chosen for playlist search effectiveness and
    # target_energy/tempo chosen to reflect typical musical ranges.
    mood_config = {
        # Very intense riding: fast BPM, high energy, aggressive genres
        "adrenaline": (["hard-rock", "metal", "punk"], 0.95, 150),

        # Sporty / thrilling: driving EDM and rock with high energy
        "energetic": (["electronic", "dance", "alt-rock"], 0.85, 135),

        # Lively but melodic: modern pop/indie with upbeat tempo
        "upbeat": (["pop", "indie-pop", "dance-pop"], 0.72, 122),

        # Confident cruising: rhythmic indie/folk with moderate tempo
        "steady": (["indie", "folk", "folk-pop"], 0.58, 100),

        # Gentle, intimate: acoustic and jazzy selections at mellow tempo
        "relaxed": (["acoustic", "singer-songwriter", "jazz"], 0.45, 82),

        # Very calm: ambient, minimal classical textures — low energy & tempo
        "serene": (["ambient", "neo-classical", "minimal"], 0.22, 60),

        # legacy labels for compatibility
        "high_energy": (["rock", "electronic"], 0.85, 130),
        "moderate": (["pop", "indie"], 0.6, 110),
        "chill": (["jazz", "acoustic"], 0.4, 90),
    }
    return mood_config.get(mood, mood_config["chill"])


def score_track_match(audio_features: dict, target_energy: float, target_tempo: int, target_valence: float = None) -> float:
    """Score how well a track matches target audio features (0-100)."""
    if not audio_features:
        return 0
    
    score = 0
    
    # Energy matching (40% weight) - Most important for ride matching
    energy_diff = abs(audio_features.get('energy', 0.5) - target_energy)
    score += (1 - energy_diff) * 40
    
    # Tempo matching (30% weight) - BPM matching for ride rhythm
    tempo_diff = abs(audio_features.get('tempo', 120) - target_tempo) / 50
    score += max(0, (1 - tempo_diff)) * 30
    
    # Valence/mood matching (20% weight) - Happiness/energy level
    if target_valence:
        valence_diff = abs(audio_features.get('valence', 0.5) - target_valence)
        score += (1 - valence_diff) * 20
    else:
        score += 20  # Neutral bonus
    
    # Danceability bonus for high energy rides (10% weight)
    if target_energy > 0.7:
        score += audio_features.get('danceability', 0.5) * 10
    
    return min(100, max(0, score))


def build_search_queries(genres: list, mood: str) -> list:
    """Build intelligent search queries based on genres and mood."""
    queries = []
    
    # Genre-based searches
    for genre in genres:
        queries.extend([
            genre,
            f"{genre} music",
            f"best {genre} songs"
        ])
    
    # Mood-specific searches with riding context
    mood_queries = {
        "high_energy": [
            "high energy rock", "electronic dance", "workout music",
            "energetic", "powerful", "intense music", "driving rock"
        ],
        "moderate": [
            "upbeat pop", "indie hits", "feel good music",
            "moderate tempo", "driving music", "road trip songs"
        ],
        "chill": [
            "chill jazz", "acoustic chill", "relaxing music",
            "mellow", "calm", "peaceful", "cruising music"
        ]
    }
    
    queries.extend(mood_queries.get(mood, mood_queries["chill"]))
    return queries


def get_target_valence(mood: str) -> float:
    """Get target valence (happiness) for mood."""
    return {"high_energy": 0.7, "moderate": 0.6, "chill": 0.4}.get(mood, 0.5)


def search_and_score_tracks(sp, genres: list, mood: str, limit: int, target_energy: float, target_tempo: int, debug: bool = False) -> list:
    """Search for tracks and score them based on audio features."""
    
    search_queries = build_search_queries(genres, mood)
    target_valence = get_target_valence(mood)
    
    # Collect candidate tracks
    candidate_tracks = []
    track_ids_seen = set()
    
    if debug:
        print(f"🔍 Searching with {len(search_queries)} queries...")

    # Reduce query volume: use a smaller subset of queries and request more
    # results per-query (Spotify supports up to 50). This reduces total API
    # calls and helps avoid rate limits in constrained environments.
    max_queries = min(6, len(search_queries))
    compressed_queries = search_queries[:max_queries]
    per_query_limit = 50
    if debug:
        print(f"[DEBUG] Using {len(compressed_queries)} queries, up to {per_query_limit} results each")
    
    # Search phase
    for query in compressed_queries:
        if len(candidate_tracks) >= limit * 3:  # Get 3x candidates for better selection
            break

        # Perform search with retries and exponential backoff to handle 429/502 rate errors
        search_attempts = 0
        max_search_attempts = 5
        while search_attempts < max_search_attempts:
            try:
                # Request more results per query to reduce total requests.
                per_call = max(10, min(per_query_limit, limit * 3 - len(candidate_tracks)))
                if per_call <= 0:
                    break
                results = sp.search(q=query, type='track', limit=per_call)
                tracks_found = len(results.get('tracks', {}).get('items', []))

                if debug and tracks_found > 0:
                    print(f"   '{query}': found {tracks_found} tracks")

                for track in results['tracks']['items']:
                    if track['id'] not in track_ids_seen:
                        candidate_tracks.append(track)
                        track_ids_seen.add(track['id'])
                        if len(candidate_tracks) >= limit * 3:
                            break

                # success -> break retry loop
                break

            except Exception as e:
                search_attempts += 1
                msg = str(e)
                # Detect rate-limit-like failures (429 or repeated server errors)
                is_rate = '429' in msg or 'rate' in msg.lower() or 'Retry' in msg
                backoff = 1.0 * (2 ** (search_attempts - 1))
                if debug:
                    print(f"[DEBUG] Search attempt {search_attempts} for '{query}' failed: {e}")
                    print(f"[DEBUG] Backing off {backoff}s before retrying")
                time.sleep(backoff)
                if search_attempts >= max_search_attempts:
                    if debug:
                        print(f"[DEBUG] Giving up search for '{query}' after {search_attempts} attempts")
                    break
                continue
    
    if not candidate_tracks:
        return []
    
    if debug:
        print(f"📊 Found {len(candidate_tracks)} candidate tracks")
    
    # NOTE: audio-features disabled — use search-only results and preserve
    # debug info. This avoids 403s or permission issues when fetching
    # audio features and still returns candidate tracks for playlist creation.
    if debug:
        print("[DEBUG] Audio-features disabled; returning top search results")
    return candidate_tracks[:min(limit, len(candidate_tracks))]


def create_playlist(oauth_sp, tracks: list, mood: str, target_energy: float, target_tempo: int, playlist_prefix: str, public: bool = False, debug: bool = False) -> str:
    """Create Spotify playlist with selected tracks."""
    user_id = oauth_sp.me()['id']
    playlist_name = f"{playlist_prefix} ({mood.capitalize()})"
    
    description = f"Auto-generated by SonicRide | Mood: {mood} | Target: {target_energy:.1f} energy, {target_tempo} BPM"
    
    playlist = oauth_sp.user_playlist_create(
        user=user_id,
        name=playlist_name,
        public=public,
        description=description
    )
    
    track_uris = [t['uri'] for t in tracks]
    oauth_sp.playlist_add_items(playlist_id=playlist['id'], items=track_uris)
    # Try to generate and upload a cover image (optional)
    try:
        img_b64 = generate_cover_image_bytes(mood, playlist_prefix)
        if img_b64:
            # Decode to bytes to measure exact size for debug
            try:
                img_bytes = base64.b64decode(img_b64)
            except Exception:
                img_bytes = None

            if debug:
                if img_bytes is None:
                    print("[DEBUG] Generated image invalid base64; skipping upload")
                else:
                    print(f"[DEBUG] Generated image bytes: {len(img_bytes)}")

            # Spotify expects base64-encoded JPEG bytes without data URI prefix
            try:
                oauth_sp.playlist_upload_cover_image(playlist_id=playlist['id'], image_b64=img_b64)
                if debug:
                    print("[DEBUG] Cover image uploaded successfully")
            except Exception as e:
                if debug:
                    print(f"[DEBUG] Cover upload failed: {e}")
                # Non-fatal: upload can fail due to permissions or size; continue
                pass
    except Exception as e:
        if debug:
            print(f"[DEBUG] Cover generation failed: {e}")
        # Non-fatal: if generating the image fails, we still return the playlist URL
        pass

    return playlist['external_urls']['spotify']


def generate_cover_image_bytes(mood: str, prefix: str, size: int = 640) -> str:
    """Generate a simple playlist cover image and return base64 JPEG bytes.

    The image is a solid color chosen by mood with overlaid text for prefix
    and mood. Returns a base64-encoded string suitable for
    `playlist_upload_cover_image`.
    """
    # Mood color palette (pleasant, contrasted)
    colors = {
        'adrenaline': (30, 30, 30),
        'energetic': (220, 50, 47),
        'upbeat': (255, 165, 0),
        'steady': (34, 139, 34),
        'relaxed': (70, 130, 180),
        'serene': (100, 149, 237),
    }
    bg = colors.get(mood, (50, 50, 50))

    img = Image.new('RGB', (size, size), color=bg)
    draw = ImageDraw.Draw(img)

    # Load a default font; fallback to basic if not available
    try:
        font_title = ImageFont.truetype('arial.ttf', size // 12)
        font_mood = ImageFont.truetype('arial.ttf', size // 10)
    except Exception:
        font_title = ImageFont.load_default()
        font_mood = ImageFont.load_default()

    # Draw prefix (top) and mood (center)
    padding = size // 20
    title_text = prefix
    mood_text = mood.capitalize()

    # Title at top-left
    draw.text((padding, padding), title_text, fill=(255, 255, 255), font=font_title)

    # Mood centered — use textbbox for compatibility across Pillow versions
    try:
        bbox = draw.textbbox((0, 0), mood_text, font=font_mood)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
    except Exception:
        # Fallback: estimate using font.getsize if available
        try:
            w, h = font_mood.getsize(mood_text)
        except Exception:
            w, h = (len(mood_text) * (size // 20), size // 10)

    draw.text(((size - w) / 2, (size - h) / 2), mood_text, fill=(255, 255, 255), font=font_mood)

    # Encode to JPEG and base64, ensuring size <= 256 KB (Spotify limit)
    max_bytes = 256 * 1024
    quality = 85
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=quality)
    img_bytes = buf.getvalue()

    # If image is too large, iteratively reduce quality and/or size
    while len(img_bytes) > max_bytes and quality > 20:
        quality -= 5
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=quality)
        img_bytes = buf.getvalue()

    # If still too large, downscale the image progressively
    current_size = size
    while len(img_bytes) > max_bytes and current_size > 128:
        current_size = int(current_size * 0.8)
        img_small = img.resize((current_size, current_size), Image.LANCZOS)
        buf = io.BytesIO()
        img_small.save(buf, format='JPEG', quality=max(30, quality))
        img_bytes = buf.getvalue()

    if len(img_bytes) > max_bytes:
        # As a last resort, return None so upload is skipped
        return None

    img_b64 = base64.b64encode(img_bytes).decode('ascii')
    return img_b64


def generate_playlist_prefix(df: pd.DataFrame, mood: str) -> str:
    """Generate a friendly playlist prefix based on ride metrics.

    Uses the date (if present), mood, average speed and an approximate
    duration to create a memorable prefix like "Morning Ride — Steady 50kmh".
    """
    # Try to extract a date/time from the `time` column and produce a
    # friendly label (e.g. 'Mon 2025-09-16' and time-of-day 'Morning').
    date_label = None
    time_of_day = None
    if 'time' in df.columns:
        try:
            times = pd.to_datetime(df['time'], errors='coerce')
            if times.notnull().any():
                dt = times.min()
                date_label = dt.strftime('%b %d')
                hour = dt.hour
                if 5 <= hour < 12:
                    time_of_day = 'Morning'
                elif 12 <= hour < 17:
                    time_of_day = 'Afternoon'
                elif 17 <= hour < 22:
                    time_of_day = 'Evening'
                else:
                    time_of_day = 'Night'
        except Exception:
            date_label = None
            time_of_day = None

    # Average speed
    try:
        avg_speed = df.get('speed_kmh')
        avg_speed_val = float(avg_speed.dropna().mean()) if avg_speed is not None and not avg_speed.dropna().empty else None
    except Exception:
        avg_speed_val = None

    # Duration in minutes
    duration_minutes = None
    if 'dt_s' in df.columns and df['dt_s'].dropna().sum() > 0:
        duration_minutes = int(df['dt_s'].dropna().sum() / 60)
    else:
        try:
            times = pd.to_datetime(df['time'], errors='coerce')
            if times.notnull().any():
                duration_minutes = int((times.max() - times.min()).total_seconds() / 60)
        except Exception:
            duration_minutes = None

    # Build friendly pieces
    pieces = []
    if time_of_day:
        pieces.append(time_of_day)
    if date_label:
        pieces.append(date_label)

    main = f"{mood.capitalize()}"
    extras = []
    if avg_speed_val:
        extras.append(f"{int(round(avg_speed_val))} km/h")
    if duration_minutes and duration_minutes > 0:
        extras.append(f"{duration_minutes} min")

    # Compose: "Morning Sep 16 — Steady · 50 km/h · 30 min"
    left = " ".join(pieces) if pieces else None
    right = " · ".join(extras) if extras else None

    if left:
        prefix = f"{left} — {main}"
    else:
        prefix = main

    if right:
        prefix = f"{prefix} · {right}"

    return prefix or "SonicRide Playlist"


def main():
    parser = argparse.ArgumentParser(
        description="SonicRide: Create Spotify playlists based on motorcycle ride metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python sonicride.py --metrics ride.csv --limit 12
  python sonicride.py --metrics fast_ride.csv --limit 20 --public --debug
        """
    )
    parser.add_argument("--metrics", default="metrics.csv", 
                       help="Path to ride metrics CSV file (default: metrics.csv)")
    parser.add_argument("--limit", type=int, default=15, 
                       help="Number of tracks in playlist (default: 15)")
    parser.add_argument("--name-prefix", default="SonicRide Playlist", 
                       help="Playlist name prefix (default: 'SonicRide Playlist')")
    parser.add_argument("--public", action="store_true", 
                       help="Create public playlist (default: private)")
    parser.add_argument("--debug", action="store_true", 
                       help="Show detailed debug information")
    
    args = parser.parse_args()

    # Load and validate metrics
    metrics_path = Path(args.metrics)
    if not metrics_path.exists():
        print(f"❌ Error: Metrics file not found: {metrics_path}", file=sys.stderr)
        sys.exit(1)

    try:
        df = pd.read_csv(metrics_path)
        speed_series = df.get('speed_kmh')
        lean_series = df.get('lean_angle_deg')
        
        if speed_series is None or lean_series is None:
            print("❌ Error: CSV missing required columns 'speed_kmh' and/or 'lean_angle_deg'", file=sys.stderr)
            sys.exit(1)
            
        avg_speed = speed_series.dropna().mean() if not speed_series.dropna().empty else 0.0
        avg_lean = lean_series.dropna().mean() if not lean_series.dropna().empty else 0.0
        
    except Exception as e:
        print(f"❌ Error reading metrics: {e}", file=sys.stderr)
        sys.exit(1)

    # Compute ride duration (seconds). Prefer `dt_s` if present, else fall back to timestamp diff.
    duration_seconds = None
    if 'dt_s' in df.columns and df['dt_s'].dropna().sum() > 0:
        duration_seconds = int(df['dt_s'].dropna().sum())
    else:
        try:
            times = pd.to_datetime(df['time'], errors='coerce')
            if times.notnull().any():
                duration_seconds = int((times.max() - times.min()).total_seconds())
        except Exception:
            duration_seconds = None

    if not duration_seconds or duration_seconds <= 0:
        # Fallback to approximate duration from number of rows assuming dt_s if missing
        if 'dt_s' in df.columns:
            duration_seconds = int(df['dt_s'].fillna(0).sum()) or max(60, len(df) * 30)
        else:
            duration_seconds = max(60, len(df) * 30)

    if args.debug:
        print(f"[DEBUG] Ride duration (seconds): {duration_seconds}")

    # Analyze ride
    print(f"🏍️  Ride Analysis: {avg_speed:.1f} km/h avg speed, {avg_lean:.1f}° avg lean")
    
    mood = determine_mood(avg_speed, avg_lean)
    genres, target_energy, target_tempo = get_mood_parameters(mood)
    
    print(f"🎵 Ride mood: {mood}")
    if args.debug:
        print(f"[DEBUG] Target: {genres} genres, {target_energy} energy, {target_tempo} BPM")

    # Provide a generated playlist prefix when the user didn't override it
    if args.name_prefix == "SonicRide Playlist":
        args.name_prefix = generate_playlist_prefix(df, mood)

    # Set up Spotify
    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
    redirect_uri = os.getenv("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8888/callback")

    missing = [name for name, val in [("SPOTIFY_CLIENT_ID", client_id), ("SPOTIFY_CLIENT_SECRET", client_secret)] if not val]
    if missing:
        print(f"❌ Missing environment variables: {', '.join(missing)}", file=sys.stderr)
        print("💡 Create a .env file with SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET", file=sys.stderr)
        sys.exit(1)

    # Initialize Spotify clients
    try:
        client_creds_sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(
            client_id=client_id, client_secret=client_secret
        ))
        
        # Request full user scopes required for playlist creation and optional playback control
        scope_parts = [
            "playlist-modify-private",
            "playlist-modify-public",
            "user-read-private",
            "user-modify-playback-state",
        ]
        scope = " ".join(scope_parts)

        oauth_sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
            scope=scope
        ))
        
    except Exception as e:
        print(f"❌ Spotify authentication failed: {e}", file=sys.stderr)
        sys.exit(1)

    # Find tracks using smart search
    print("🎯 Searching for perfect tracks...")
    if args.debug:
        print("💡 Note: Using smart search for robust results")
    
    # Estimate how many tracks we'll need to cover the ride duration.
    # Use average track length ~180s and add a safety multiplier.
    est_per_track = 180
    est_needed = max(3, math.ceil((duration_seconds / est_per_track) * 1.6))
    # Allow user override via --limit; otherwise use estimated number
    search_limit = args.limit if args.limit and args.limit > est_needed else est_needed

    tracks = search_and_score_tracks(
        client_creds_sp, genres, mood, search_limit,
        target_energy, target_tempo, args.debug
    )

    # Select tracks until we reach ride duration (use track['duration_ms']).
    selected = []
    total_sec = 0
    seen = set()
    for t in tracks:
        tid = t.get('id')
        if not tid or tid in seen:
            continue
        seen.add(tid)
        dur = t.get('duration_ms') or 180000
        dur_s = int(dur / 1000)
        selected.append(t)
        total_sec += dur_s
        if total_sec >= duration_seconds:
            break

    # If not enough duration, try to fetch more candidates (larger search) once.
    if total_sec < duration_seconds:
        if args.debug:
            print(f"[DEBUG] Collected {total_sec}s, need {duration_seconds}s — expanding search")
        more_limit = max(search_limit * 2, 30)
        more_tracks = search_and_score_tracks(
            client_creds_sp, genres, mood, more_limit,
            target_energy, target_tempo, args.debug
        )
        for t in more_tracks:
            tid = t.get('id')
            if not tid or tid in seen:
                continue
            seen.add(tid)
            dur = t.get('duration_ms') or 180000
            dur_s = int(dur / 1000)
            selected.append(t)
            total_sec += dur_s
            if total_sec >= duration_seconds:
                break

    tracks = selected
    
    if not tracks:
        print("❌ No suitable tracks found", file=sys.stderr)
        sys.exit(1)

    print(f"✅ Found {len(tracks)} perfect tracks")

    # Create playlist
    try:
        playlist_url = create_playlist(
            oauth_sp, tracks, mood, target_energy, target_tempo,
            args.name_prefix, args.public, debug=args.debug
        )
        
        print(f"🎶 Playlist created: {playlist_url}")
        
        if args.debug:
            print("\n📝 Tracks added:")
            for i, track in enumerate(tracks, 1):
                artist_names = ", ".join(a['name'] for a in track['artists'])
                print(f"   {i:2d}. {track['name']} by {artist_names}")
        else:
            print(f"📝 Added {len(tracks)} tracks to your {mood} riding playlist")
            
    except Exception as e:
        print(f"❌ Failed to create playlist: {e}", file=sys.stderr)
        sys.exit(1)

    print("🏍️ Ready to ride! Enjoy your personalized soundtrack! 🎵")


if __name__ == "__main__":
    main()