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

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def determine_mood(avg_speed: float, avg_lean: float) -> str:
    """Determine riding mood based on speed and lean angle."""
    if (avg_speed or 0) > 80 or (avg_lean or 0) > 25:
        return "high_energy"
    if (avg_speed or 0) > 50:
        return "moderate"
    return "chill"


def get_mood_parameters(mood: str) -> tuple:
    """Get music parameters for the given mood."""
    mood_config = {
        "high_energy": (["rock", "electronic"], 0.85, 130),
        "moderate": (["pop", "indie"], 0.6, 110),
        "chill": (["jazz", "acoustic"], 0.4, 90)
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
    
    # Search phase
    for query in search_queries:
        if len(candidate_tracks) >= limit * 3:  # Get 3x candidates for better selection
            break
            
        try:
            results = sp.search(q=query, type='track', limit=10)
            tracks_found = len(results['tracks']['items'])
            
            if debug and tracks_found > 0:
                print(f"   '{query}': found {tracks_found} tracks")
            
            for track in results['tracks']['items']:
                if track['id'] not in track_ids_seen:
                    candidate_tracks.append(track)
                    track_ids_seen.add(track['id'])
                    if len(candidate_tracks) >= limit * 3:
                        break
                        
        except Exception as e:
            if debug:
                print(f"   Search failed for '{query}': {e}")
            continue
    
    if not candidate_tracks:
        return []
    
    if debug:
        print(f"📊 Found {len(candidate_tracks)} candidate tracks")
    
    # Audio features scoring phase
    track_ids = [t['id'] for t in candidate_tracks]
    scored_tracks = []
    
    try:
        # Process in batches (API limit is 100, using 50 for safety)
        for i in range(0, len(track_ids), 50):
            batch_ids = track_ids[i:i+50]
            batch_features = sp.audio_features(batch_ids)
            
            for j, features in enumerate(batch_features):
                if features:  # Some tracks might not have audio features
                    track_idx = i + j
                    score = score_track_match(features, target_energy, target_tempo, target_valence)
                    scored_tracks.append({
                        'track': candidate_tracks[track_idx],
                        'features': features,
                        'score': score
                    })
    except Exception as e:
        if debug:
            print(f"⚠️  Audio features unavailable: {e}")
            print("   Using genre-based selection instead")
        # Fallback to genre-based selection without scoring
        return random.sample(candidate_tracks, min(limit, len(candidate_tracks)))
    
    if not scored_tracks:
        return random.sample(candidate_tracks, min(limit, len(candidate_tracks)))
    
    # Sort by score and return best matches
    scored_tracks.sort(key=lambda x: x['score'], reverse=True)
    
    if debug:
        print(f"🎯 Top matches:")
        for i, item in enumerate(scored_tracks[:min(5, len(scored_tracks))]):
            track = item['track']
            score = item['score']
            features = item['features']
            artist_names = ", ".join(a['name'] for a in track['artists'])
            print(f"   {i+1}. {track['name']} by {artist_names}")
            print(f"      Score: {score:.1f} | Energy: {features['energy']:.2f} | Tempo: {features['tempo']:.0f} BPM")
    
    return [item['track'] for item in scored_tracks[:limit]]


def create_playlist(oauth_sp, tracks: list, mood: str, target_energy: float, target_tempo: int, playlist_prefix: str, public: bool = False) -> str:
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
    
    return playlist['external_urls']['spotify']


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

    # Analyze ride
    print(f"🏍️  Ride Analysis: {avg_speed:.1f} km/h avg speed, {avg_lean:.1f}° avg lean")
    
    mood = determine_mood(avg_speed, avg_lean)
    genres, target_energy, target_tempo = get_mood_parameters(mood)
    
    print(f"🎵 Ride mood: {mood}")
    if args.debug:
        print(f"[DEBUG] Target: {genres} genres, {target_energy} energy, {target_tempo} BPM")

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
        
        scope = "playlist-modify-private user-read-private"
        if args.public:
            scope = scope.replace("playlist-modify-private", "playlist-modify-public")

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
        print("💡 Note: Using smart search (Spotify Recommendations API requires 250k+ MAUs)")
    
    tracks = search_and_score_tracks(
        client_creds_sp, genres, mood, args.limit, 
        target_energy, target_tempo, args.debug
    )
    
    if not tracks:
        print("❌ No suitable tracks found", file=sys.stderr)
        sys.exit(1)

    print(f"✅ Found {len(tracks)} perfect tracks")

    # Create playlist
    try:
        playlist_url = create_playlist(
            oauth_sp, tracks, mood, target_energy, target_tempo, 
            args.name_prefix, args.public
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