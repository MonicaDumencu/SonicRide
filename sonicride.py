#!/usr/bin/env python3
"""
SonicRide: Smart Spotify Playlist Generator

Creates personalized playlists based on motorcycle ride metrics using
intelligent search.
"""
import argparse
import base64
import math
import os
import random
import sys
import time
from pathlib import Path

import pandas as pd
import requests
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


def generate_cover_image_bytes(
    mood: str | None = None,
    prefix: str | None = None,
    size: int = 296,
    genres: list | None = None,
    tracks: list | None = None,
    show_text: bool = False,
    mode: str = "hf",
) -> str | None:
    """Generate a cover image and return a base64-encoded image string.

    Uses the Hugging Face Inference API. It reads `HF_API_TOKEN` and optional
    `HF_MODEL` from environment variables. Defaults to
    `stabilityai/stable-diffusion-xl-base-1.0`.

    Returns base64 image data (no data URL prefix) or ``None`` on failure.
    """
    # Only HF mode is supported in this simplified function
    if mode != "hf":
        return None

    hf_token = os.environ.get("HF_API_TOKEN")
    if not hf_token:
        if os.environ.get("DEBUG"):
            print("[DEBUG] HF_API_TOKEN not set")
        return None

    model = os.environ.get("HF_MODEL", "stabilityai/stable-diffusion-xl-base-1.0")

    # Build a basic prompt from the playlist inputs
    prompt_parts = []
    if mood:
        prompt_parts.append(mood)
    if genres:
        prompt_parts.append(", ".join(genres[:3]))
    if prefix:
        prompt_parts.append(prefix)

    # Compose a detailed prompt that encourages realistic, photorealistic
    # album covers while asking the model to optimize for small web-sized
    # JPEG output (helps keep Spotify upload under the size limit).
    prompt = (
        " | ".join([p for p in prompt_parts if p])
        or "realistic album cover, cinematic photography"
    )
    prompt += (
        ", photorealistic, studio lighting, high detail, no text, "
        "natural colors, cinematic composition, film grain, "
        "shot on 35mm, shallow depth of field. "
        "Optimize for web: small file size, high JPEG compression, 8-bit RGB."
    )

    # Prefer binary JPEG response from HF if the model supports it
    # and hint at web-optimized/formats in parameters. Keep timeout moderate.
    headers = {"Authorization": f"Bearer {hf_token}", "Accept": "image/jpeg"}

    # Inference API for image generation accepts content-type and prompt payload
    url = f"https://api-inference.huggingface.co/models/{model}"

    # Ensure dimensions are multiples of 8 (many SD endpoints require this)
    size_aligned = max(8, (size // 8) * 8)

    # Parameters: ask for small dimensions, jpeg format, and safety/compression.
    # Include a soft hint for maximum bytes; not all models honor it.
    payload = {
        "inputs": prompt,
        "options": {"wait_for_model": True},
        "parameters": {
            "width": size_aligned,
            "height": size_aligned,
            "format": "jpeg",
            "jpeg_quality": 65,
            "optimize": True,
            "max_bytes": 200000,
            "return_full_text": False,
        },
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        if resp.status_code != 200:
            if os.environ.get("DEBUG"):
                print(f"[DEBUG] HF API error {resp.status_code}: {resp.text}")
            return None

        # Response content should be image bytes (if model returns binary) or JSON with base64
        content_type = resp.headers.get("content-type", "")
        if "application/json" in content_type:
            data = resp.json()
            # Some HF responses return a raw base64 string
            if isinstance(data, str):
                return data
            # Some HF responses embed base64 under 'image' or are an array
            if isinstance(data, dict) and "image" in data:
                return data["image"]
            # Some responses are a list of objects containing generated image
            if isinstance(data, list) and data:
                # list of strings
                if all(isinstance(x, str) for x in data):
                    return data[0]
                # list of dicts
                if isinstance(data[0], dict):
                    if "generated_image" in data[0]:
                        return data[0]["generated_image"]
                    if "image" in data[0]:
                        return data[0]["image"]
            # Fallback: no base64 found
            if os.environ.get("DEBUG"):
                try:
                    print(
                        "[DEBUG] HF JSON response did not contain an image field"
                    )
                except Exception:
                    print(
                        "[DEBUG] HF JSON response did not contain an image field"
                    )
            return None

    # If the server honored `Accept: image/jpeg`, return the binary as
    # base64.
        content_type = resp.headers.get("content-type", "")
        if content_type.startswith("image"):
            img_bytes = resp.content
            b64 = base64.b64encode(img_bytes).decode("utf-8")
            return b64

    # Otherwise fall back to JSON handling above (handled earlier) — try to
    # parse JSON
        if "application/json" in content_type:
            data = resp.json()
            if isinstance(data, str):
                return data
            if isinstance(data, dict) and "image" in data:
                return data["image"]
            if isinstance(data, list) and data:
                if all(isinstance(x, str) for x in data):
                    return data[0]
                if isinstance(data[0], dict):
                    if "generated_image" in data[0]:
                        return data[0]["generated_image"]
                    if "image" in data[0]:
                        return data[0]["image"]

        # As a last resort, return raw base64 of response content
        img_bytes = resp.content
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        return b64

    except Exception as e:
        if os.environ.get("DEBUG"):
            print(f"[DEBUG] Error generating cover via HF: {e}")
        return None


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
    s = avg_speed or 0
    lean = avg_lean or 0

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


def score_track_match(
    audio_features: dict,
    target_energy: float,
    target_tempo: int,
    target_valence: float = None,
) -> float:
    """Score how well a track matches target audio features (0-100)."""
    if not audio_features:
        return 0

    score = 0

    # Energy matching (40% weight) - Most important for ride matching
    energy_diff = abs(audio_features.get("energy", 0.5) - target_energy)
    score += (1 - energy_diff) * 40

    # Tempo matching (30% weight) - BPM matching for ride rhythm
    tempo_diff = abs(audio_features.get("tempo", 120) - target_tempo) / 50
    score += max(0, (1 - tempo_diff)) * 30

    # Valence/mood matching (20% weight) - Happiness/energy level
    if target_valence:
        valence_diff = abs(audio_features.get("valence", 0.5) - target_valence)
        score += (1 - valence_diff) * 20
    else:
        score += 20  # Neutral bonus

    # Danceability bonus for high energy rides (10% weight)
    if target_energy > 0.7:
        score += audio_features.get("danceability", 0.5) * 10

    return min(100, max(0, score))


def build_search_queries(
    genres: list, mood: str, user_prefs: list | None = None
) -> list:
    """Build intelligent search queries based on genres and mood."""
    queries = []

    # Genre-based searches
    for genre in genres:
        queries.extend([genre, f"{genre} music", f"best {genre} songs"])

    # Mood-specific searches with riding context
    mood_queries = {
        "high_energy": [
            "high energy rock",
            "electronic dance",
            "workout music",
            "energetic",
            "powerful",
            "intense music",
            "driving rock",
        ],
        "moderate": [
            "upbeat pop",
            "indie hits",
            "feel good music",
            "moderate tempo",
            "driving music",
            "road trip songs",
        ],
        "chill": [
            "chill jazz",
            "acoustic chill",
            "relaxing music",
            "mellow",
            "calm",
            "peaceful",
            "cruising music",
        ],
    }

    queries.extend(mood_queries.get(mood, mood_queries["chill"]))
    # Incorporate user preferences as higher-priority queries
    if user_prefs:
        # place prefs at front to bias results toward user's tastes
        pref_queries = []
        for p in user_prefs:
            p = p.strip()
            if not p:
                continue
            pref_queries.extend(
                [
                    p,
                    f"{p} music",
                    f"best {p} songs",
                ]
            )
        # de-duplicate while preserving order: prefs first
        seen = set()
        combined = []
        for q in pref_queries + queries:
            if q not in seen:
                seen.add(q)
                combined.append(q)
        queries = combined
    return queries


def get_target_valence(mood: str) -> float:
    """Get target valence (happiness) for mood."""
    return {"high_energy": 0.7, "moderate": 0.6, "chill": 0.4}.get(mood, 0.5)


def search_and_score_tracks(
    sp,
    genres: list,
    mood: str,
    limit: int,
    target_energy: float,
    target_tempo: int,
    debug: bool = False,
    user_prefs: list | None = None,
) -> list:
    """Search for tracks and score them based on audio features."""

    search_queries = build_search_queries(genres, mood, user_prefs)
    # target_valence intentionally unused when audio-features are disabled

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
        print(
            "[DEBUG] Using %d queries, up to %d results each"
            % (len(compressed_queries), per_query_limit)
        )

    # Search phase
    for query in compressed_queries:
        if len(candidate_tracks) >= limit * 3:
            # Get 3x candidates for better selection
            break

        # Perform search with retries and exponential backoff to handle 429/502 rate errors
        search_attempts = 0
        max_search_attempts = 5
        while search_attempts < max_search_attempts:
            try:
                # Request more results per query to reduce total requests.
                per_call = max(
                    10, min(per_query_limit, limit * 3 - len(candidate_tracks))
                )
                if per_call <= 0:
                    break
                results = sp.search(q=query, type="track", limit=per_call)
                tracks_found = len(results.get("tracks", {}).get("items", []))

                if debug and tracks_found > 0:
                    print(f"   '{query}': found {tracks_found} tracks")

                for track in results["tracks"]["items"]:
                    if track["id"] not in track_ids_seen:
                        candidate_tracks.append(track)
                        track_ids_seen.add(track["id"])
                        if len(candidate_tracks) >= limit * 3:
                            break

                # success -> break retry loop
                break

            except Exception as e:
                search_attempts += 1
                msg = str(e)
                # Detect rate-limit-like failures (429 or repeated server errors)
                # (we don't explicitly use the boolean below but keep the check
                # available for debugging/extension)
                _ = "429" in msg or "rate" in msg.lower() or "Retry" in msg
                backoff = 1.0 * (2 ** (search_attempts - 1))
                if debug:
                    print(
                        "[DEBUG] Search attempt %d for '%s' failed: %s"
                        % (search_attempts, query, e)
                    )
                    print("[DEBUG] Backing off %ss before retrying" % (backoff,))
                time.sleep(backoff)
                if search_attempts >= max_search_attempts:
                    if debug:
                        print(
                            "[DEBUG] Giving up search for '%s' after %d attempts"
                            % (query, search_attempts)
                        )
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

    # Build a simple weight for each candidate: preference matches receive
    # higher weight so they're more likely to be sampled, but sampling is
    # still random. This preserves variety while respecting user tastes.
    try:
        prefs_lower = [p.lower() for p in user_prefs] if user_prefs else []
        weights = []
        for t in candidate_tracks:
            name = (t.get("name") or "").lower()
            artists = ", ".join(a.get("name", "") for a in t.get("artists", [])).lower()
            w = 1.0
            # Increase weight if any preference appears in title or artist
            for p in prefs_lower:
                if p and (p in name or p in artists):
                    w += 3.0
            weights.append(w)

        # Normalize weights to probabilities
        total = sum(weights) if weights else 0.0
        if total <= 0:
            probs = None
        else:
            probs = [w / total for w in weights]

        # Sample without replacement up to `limit*3` candidates using weights
        max_candidates = min(len(candidate_tracks), max(10, limit * 3))
        chosen = []
        indices = list(range(len(candidate_tracks)))
        random.seed(None)
        if probs is None:
            random.shuffle(indices)
            chosen_idx = indices[:max_candidates]
        else:
            # Weighted sampling without replacement via cumulative method
            cum = list(probs)
            for _ in range(max_candidates):
                r = random.random()
                s = 0.0
                for i, idx in enumerate(indices):
                    s += cum[i]
                    if r <= s:
                        chosen.append(candidate_tracks[idx])
                        # remove chosen index and its weight
                        indices.pop(i)
                        cum.pop(i)
                        # re-normalize cumulative weights
                        total_remain = sum(cum) if cum else 0.0
                        if total_remain > 0:
                            cum = [c / total_remain for c in cum]
                        break
                if not indices:
                    break
            chosen_idx = None

        if chosen:
            final_candidates = chosen
        else:
            final_candidates = [candidate_tracks[i] for i in chosen_idx]

        return final_candidates[: min(limit, len(final_candidates))]
    except Exception:
        # On any failure, fall back to simple shuffle-based randomness
        try:
            random.seed(None)
            random.shuffle(candidate_tracks)
        except Exception:
            pass
        return candidate_tracks[: min(limit, len(candidate_tracks))]


def create_playlist(
    oauth_sp,
    tracks: list,
    mood: str,
    target_energy: float,
    target_tempo: int,
    playlist_prefix: str,
    genres: list = None,
    public: bool = False,
    debug: bool = False,
) -> str:
    """Create Spotify playlist with selected tracks and realistic cover art."""
    user_id = oauth_sp.me()["id"]
    playlist_name = f"{playlist_prefix} ({mood.capitalize()})"

    description = (
        f"Auto-generated by SonicRide | Mood: {mood} | "
        f"Target: {target_energy:.1f} energy, {target_tempo} BPM"
    )

    playlist = oauth_sp.user_playlist_create(
        user=user_id, name=playlist_name, public=public, description=description
    )

    track_uris = [t["uri"] for t in tracks]
    oauth_sp.playlist_add_items(playlist_id=playlist["id"], items=track_uris)

    # Generate and upload realistic cover art
    try:
        if debug:
            print("[DEBUG] Generating realistic album cover...")
        # Generate cover using the module-level HF generator
        cover_image_bytes = generate_cover_image_bytes(
            mood=mood,
            prefix=playlist_prefix,
            size=640,
            genres=genres or [],
            tracks=tracks,
            show_text=False,
            mode="hf",
        )

        if cover_image_bytes:
            oauth_sp.playlist_upload_cover_image(
                playlist_id=playlist["id"], image_b64=cover_image_bytes
            )
            if debug:
                print("[DEBUG] Cover art uploaded successfully")
        else:
            if debug:
                print("[DEBUG] Cover generation failed, " "proceeding without image")

    except Exception as e:
        if debug:
            print(f"[DEBUG] Cover upload failed: {e}")
        # Continue without cover - not a critical failure
        pass

    # Get updated playlist info with cover image
    try:
        updated_playlist = oauth_sp.playlist(playlist["id"])
        return {
            'url': playlist["external_urls"]["spotify"],
            'name': updated_playlist["name"],
            'id': updated_playlist["id"],
            'image_url': updated_playlist["images"][0]["url"] if updated_playlist["images"] else None,
            'description': updated_playlist["description"],
            'tracks_total': updated_playlist["tracks"]["total"]
        }
    except Exception as e:
        if debug:
            print(f"[DEBUG] Failed to get updated playlist info: {e}")
        # Fallback to basic info
        return {
            'url': playlist["external_urls"]["spotify"],
            'name': playlist_name,
            'id': playlist["id"],
            'image_url': None,
            'description': description,
            'tracks_total': len(tracks)
        }


def analyze_playlist_themes(tracks: list, genres: list, mood: str) -> dict:
    """Analyze playlist tracks to determine visual themes for cover generation.

    Returns a dictionary with visual theme information based on:
    - Track titles and artist names for keyword analysis
    - Genre information for overall aesthetic
    - Mood for color palette and composition style
    """
    if not tracks:
        return {
            "primary_theme": "road",
            "color_scheme": "neutral",
            "elements": ["motorcycle"],
        }

    # Keywords that suggest different visual themes
    theme_keywords = {
        "road": [
            "highway",
            "road",
            "drive",
            "cruise",
            "journey",
            "travel",
            "mile",
            "route",
        ],
        "urban": [
            "city",
            "street",
            "neon",
            "lights",
            "downtown",
            "metro",
            "urban",
            "night",
        ],
        "nature": [
            "mountain",
            "forest",
            "river",
            "sky",
            "wind",
            "rain",
            "sun",
            "valley",
            "hill",
        ],
        "dark": [
            "dark",
            "black",
            "shadow",
            "midnight",
            "storm",
            "thunder",
            "steel",
            "iron",
        ],
        "energy": [
            "fire",
            "electric",
            "power",
            "energy",
            "lightning",
            "burn",
            "rush",
            "speed",
        ],
        "vintage": [
            "classic",
            "old",
            "vintage",
            "retro",
            "golden",
            "rusty",
            "worn",
            "aged",
        ],
    }

    # Color schemes based on genre and mood
    color_schemes = {
        "rock": "dark_red",
        "metal": "black_steel",
        "electronic": "electric_blue",
        "pop": "vibrant",
        "indie": "warm_vintage",
        "folk": "earth_tones",
        "jazz": "sepia_gold",
        "acoustic": "natural_wood",
        "ambient": "cool_blue",
    }

    # Analyze track and artist names for theme keywords
    all_text = ""
    for track in tracks:
        all_text += (track.get("name", "") + " ").lower()
        for artist in track.get("artists", []):
            all_text += (artist.get("name", "") + " ").lower()

    # Count theme keyword occurrences
    theme_scores = {}
    for theme, keywords in theme_keywords.items():
        score = sum(all_text.count(keyword) for keyword in keywords)
        theme_scores[theme] = score

    # Determine primary theme
    if any(theme_scores.values()):
        primary_theme = max(theme_scores, key=theme_scores.get)
    else:
        primary_theme = "road"

    # Determine color scheme from genres
    primary_genre = genres[0] if genres else "rock"
    color_scheme = color_schemes.get(primary_genre, "neutral")

    # Mood-based adjustments
    if mood in ["adrenaline", "energetic"]:
        if primary_theme == "road":
            primary_theme = "energy"
        color_scheme = color_scheme.replace("cool_", "warm_").replace(
            "sepia_", "bright_"
        )
    elif mood in ["serene", "relaxed"]:
        if color_scheme.startswith("dark_"):
            color_scheme = "cool_blue"

    # Define visual elements based on theme
    elements = ["motorcycle"]  # Always include motorcycle for SonicRide
    if primary_theme == "road":
        elements.extend(["highway", "horizon", "asphalt"])
    elif primary_theme == "urban":
        elements.extend(["cityscape", "streetlights", "reflections"])
    elif primary_theme == "nature":
        elements.extend(["landscape", "sky", "mountains"])
    elif primary_theme == "dark":
        elements.extend(["shadows", "steel", "concrete"])
    elif primary_theme == "energy":
        elements.extend(["motion_blur", "sparks", "trails"])
    elif primary_theme == "vintage":
        elements.extend(["weathered", "classic_bike", "sepia"])

    return {
        "primary_theme": primary_theme,
        "color_scheme": color_scheme,
        "elements": elements,
        "mood": mood,
        "genre": primary_genre,
    }


# Legacy PIL-based image generators removed. Use `generate_cover_image_bytes` (HF) above.


def generate_playlist_prefix(df: pd.DataFrame, mood: str) -> str:
    """Return a single creative word as playlist name influenced by mood.

    The goal is to produce a memorable, one-word title (no extra punctuation)
    that reflects the ride mood, average speed and duration. We sample from
    curated word lists and optionally combine short modifiers for variety.
    """
    # Lightweight, defensive extraction of metrics
    try:
        avg_speed = (
            float(df.get("speed_kmh").dropna().mean())
            if "speed_kmh" in df.columns
            else 0.0
        )
    except Exception:
        avg_speed = 0.0
    try:
        duration_minutes = (
            int(df.get("dt_s").dropna().sum() / 60)
            if "dt_s" in df.columns and df["dt_s"].dropna().sum() > 0
            else None
        )
    except Exception:
        duration_minutes = None

    # Word banks tuned to mood and ride energy
    word_banks = {
        "adrenaline": ["Sprint", "Razor", "Blaze", "Rogue", "Nitro"],
        "energetic": ["Surge", "Pulse", "Throttle", "Charge", "Vigor"],
        "upbeat": ["Groove", "Bounce", "Spark", "Glide", "Radiant"],
        "steady": ["Cruise", "Drift", "Steady", "Horizon", "Compass"],
        "relaxed": ["Driftwood", "Mellow", "Ease", "Sway", "Lull"],
        "serene": ["Calm", "Still", "Haven", "Quiet", "Aether"],
    }

    # Fallback pool
    fallback = ["Vibe", "Pulse", "Journey", "Echo", "Orbit", "Halo"]

    candidates = word_banks.get(mood, fallback)[:]

    # Influence selection by speed/duration: faster -> prefer shorter punchy words
    if avg_speed and avg_speed >= 80:
        candidates = [w for w in candidates if len(w) <= 6] or candidates
    # longer rides -> pick more evocative names
    if duration_minutes and duration_minutes >= 90:
        candidates = candidates + ["Odyssey", "Marathon"]

    # Randomized per-run selection using OS entropy so names vary per run
    try:
        import secrets

        name = secrets.choice(candidates)
    except Exception:
        # Fallback to deterministic selection if secrets not available
        key = f"{mood}:{int(avg_speed)}:{duration_minutes or 0}"
        idx = abs(hash(key)) % len(candidates)
        name = candidates[idx]

    # Ensure single word, capitalize for display
    name = "".join(ch for ch in str(name).split()[0] if ch.isalnum())
    return name or "SonicRide"


def main():
    parser = argparse.ArgumentParser(
        description="SonicRide: Create Spotify playlists based on motorcycle ride metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python sonicride.py --metrics ride.csv --limit 12
  python sonicride.py --metrics fast_ride.csv --limit 20 --public --debug
        """,
    )
    parser.add_argument(
        "--metrics",
        default="metrics.csv",
        help="Path to ride metrics CSV file (default: metrics.csv)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=15,
        help="Number of tracks in playlist (default: 15)",
    )
    parser.add_argument(
        "--name-prefix",
        default="SonicRide Playlist",
        help="Playlist name prefix (default: 'SonicRide Playlist')",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Create public playlist (default: private)",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Show detailed debug information"
    )
    # `--prefs` removed: selection is randomized and genre/mood-driven only

    args = parser.parse_args()

    # Load and validate metrics
    metrics_path = Path(args.metrics)
    if not metrics_path.exists():
        print(f"❌ Error: Metrics file not found: {metrics_path}", file=sys.stderr)
        sys.exit(1)

    try:
        df = pd.read_csv(metrics_path)
        speed_series = df.get("speed_kmh")
        lean_series = df.get("lean_angle_deg")

        if speed_series is None or lean_series is None:
            print(
                "❌ Error: CSV missing required columns 'speed_kmh' and/or 'lean_angle_deg'",
                file=sys.stderr,
            )
            sys.exit(1)

        avg_speed = (
            speed_series.dropna().mean() if not speed_series.dropna().empty else 0.0
        )
        avg_lean = (
            lean_series.dropna().mean() if not lean_series.dropna().empty else 0.0
        )

    except Exception as e:
        print(f"❌ Error reading metrics: {e}", file=sys.stderr)
        sys.exit(1)

    # Compute ride duration (seconds). Prefer `dt_s` if present, else fall back to timestamp diff.
    duration_seconds = None
    if "dt_s" in df.columns and df["dt_s"].dropna().sum() > 0:
        duration_seconds = int(df["dt_s"].dropna().sum())
    else:
        try:
            times = pd.to_datetime(df["time"], errors="coerce")
            if times.notnull().any():
                duration_seconds = int((times.max() - times.min()).total_seconds())
        except Exception:
            duration_seconds = None

    if not duration_seconds or duration_seconds <= 0:
        # Fallback to approximate duration from number of rows assuming dt_s if missing
        if "dt_s" in df.columns:
            duration_seconds = int(df["dt_s"].fillna(0).sum()) or max(60, len(df) * 30)
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
        print(
            f"[DEBUG] Target: {genres} genres, {target_energy} energy, {target_tempo} BPM"
        )

    # Provide a generated playlist prefix when the user didn't override it
    if args.name_prefix == "SonicRide Playlist":
        args.name_prefix = generate_playlist_prefix(df, mood)

    # Set up Spotify
    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
    redirect_uri = os.getenv("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8888/callback")

    missing = [
        name
        for name, val in [
            ("SPOTIFY_CLIENT_ID", client_id),
            ("SPOTIFY_CLIENT_SECRET", client_secret),
        ]
        if not val
    ]
    if missing:
        print(
            f"❌ Missing environment variables: {', '.join(missing)}", file=sys.stderr
        )
        print(
            "💡 Create a .env file with SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET",
            file=sys.stderr,
        )
        sys.exit(1)

    # Initialize Spotify clients
    try:
        client_creds_sp = spotipy.Spotify(
            client_credentials_manager=SpotifyClientCredentials(
                client_id=client_id, client_secret=client_secret
            )
        )

        # Request full user scopes required for playlist creation and optional playback control
        scope_parts = [
            "playlist-modify-private",
            "playlist-modify-public",
            "user-read-private",
            "user-top-read",
            "user-modify-playback-state",
            # Required for uploading user-provided images to playlists
            "ugc-image-upload",
        ]
        scope = " ".join(scope_parts)

        oauth_sp = spotipy.Spotify(
            auth_manager=SpotifyOAuth(
                client_id=client_id,
                client_secret=client_secret,
                redirect_uri=redirect_uri,
                scope=scope,
            )
        )
        # If debug mode, show cached token info (no raw tokens printed)
        try:
            if args.debug:
                get_cached = getattr(oauth_sp.auth_manager, "get_cached_token", None)
                token_info = get_cached() if callable(get_cached) else None
                if token_info:
                    scopes = token_info.get("scope") or token_info.get("scopes") or ""
                    expiry = token_info.get("expires_at")
                    now = int(time.time())
                    expired = bool(expiry and expiry <= now)
                    print(
                        f"[DEBUG] Spotify token: present, expires_at={expiry}, expired={expired}, scopes={scopes}"
                    )
                else:
                    print(
                        "[DEBUG] No cached Spotify token found (interactive auth may be required)."
                    )
        except Exception as e:
            if args.debug:
                print(f"[DEBUG] Unable to read cached token info: {e}")
        # Ensure the cached token includes the image upload scope; if not,
        # remove the cache and force re-auth so the user can grant it.
        try:
            if args.debug:
                # Re-fetch token_info safely
                get_cached = getattr(oauth_sp.auth_manager, "get_cached_token", None)
                token_info = get_cached() if callable(get_cached) else None
            else:
                token_info = None

            has_scope = False
            if token_info:
                token_scopes = token_info.get("scope") or token_info.get("scopes") or ""
                has_scope = "ugc-image-upload" in token_scopes

            if token_info and not has_scope:
                # Attempt to remove cache file
                cache_path = getattr(oauth_sp.auth_manager, "cache_path", None)
                try:
                    if cache_path and os.path.exists(cache_path):
                        os.remove(cache_path)
                        if args.debug:
                            print(
                                "[DEBUG] Removed cached token missing 'ugc-image-upload'; forcing re-auth"
                            )
                except Exception as e:
                    if args.debug:
                        print(f"[DEBUG] Failed to remove cache file: {e}")

                # Recreate auth manager to prompt for fresh auth with required scopes
                try:
                    oauth_sp = spotipy.Spotify(
                        auth_manager=SpotifyOAuth(
                            client_id=client_id,
                            client_secret=client_secret,
                            redirect_uri=redirect_uri,
                            scope=scope,
                        )
                    )
                    if args.debug:
                        print(
                            "[DEBUG] Re-created Spotify auth manager to acquire missing scopes"
                        )
                except Exception as e:
                    if args.debug:
                        print(f"[DEBUG] Re-auth failed: {e}")
        except Exception:
            # Non-fatal; continue with whatever auth state is present
            pass

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

    # Preferences removed — search driven only by mood/genres and randomized selection
    tracks = search_and_score_tracks(
        client_creds_sp,
        genres,
        mood,
        search_limit,
        target_energy,
        target_tempo,
        args.debug,
        user_prefs=None,
    )

    # No preference-based prepending; keep tracks from search results

    # Select tracks until we reach ride duration (use track['duration_ms']).
    selected = []
    total_sec = 0
    seen = set()
    for t in tracks:
        tid = t.get("id")
        if not tid or tid in seen:
            continue
        seen.add(tid)
        dur = t.get("duration_ms") or 180000
        dur_s = int(dur / 1000)
        selected.append(t)
        total_sec += dur_s
        if total_sec >= duration_seconds:
            break

    # If not enough duration, try to fetch more candidates (larger search) once.
    if total_sec < duration_seconds:
        if args.debug:
            print(
                f"[DEBUG] Collected {total_sec}s, need {duration_seconds}s — expanding search"
            )
        more_limit = max(search_limit * 2, 30)
        more_tracks = search_and_score_tracks(
            client_creds_sp,
            genres,
            mood,
            more_limit,
            target_energy,
            target_tempo,
            args.debug,
            user_prefs=None,
        )
        for t in more_tracks:
            tid = t.get("id")
            if not tid or tid in seen:
                continue
            seen.add(tid)
            dur = t.get("duration_ms") or 180000
            dur_s = int(dur / 1000)
            selected.append(t)
            total_sec += dur_s
            if total_sec >= duration_seconds:
                break

    tracks = selected
    # Shuffle final track order so repeated runs with identical input
    # still produce different playlists. Use OS entropy for randomness.
    try:
        random.seed(None)
        random.shuffle(tracks)
    except Exception:
        print("[DEBUG] Final shuffle failed; proceeding with original order")
    if not tracks:
        print("❌ No suitable tracks found", file=sys.stderr)
        sys.exit(1)

    print(f"✅ Found {len(tracks)} perfect tracks")

    # Create playlist
    try:
        playlist_info = create_playlist(
            oauth_sp,
            tracks,
            mood,
            target_energy,
            target_tempo,
            args.name_prefix,
            genres=genres,
            public=args.public,
            debug=args.debug,
        )

        # Handle both old string format and new dict format for backward compatibility
        if isinstance(playlist_info, dict):
            playlist_url = playlist_info['url']
            playlist_name = playlist_info['name']
            playlist_image = playlist_info.get('image_url')
        else:
            # Fallback for old format
            playlist_url = playlist_info
            playlist_name = f"{args.name_prefix} ({mood.capitalize()})"
            playlist_image = None

        print(f"🎶 Playlist created: {playlist_url}")
        print(f"📝 Playlist name: {playlist_name}")
        if playlist_image:
            print(f"🖼️ Playlist image: {playlist_image}")
        
        # Calculate total playlist duration
        total_duration_ms = sum(track.get('duration_ms', 0) for track in tracks)
        total_minutes = total_duration_ms // 60000
        total_seconds = (total_duration_ms % 60000) // 1000
        
        # Output playlist statistics for web app parsing
        print(f"📊 {len(tracks)} tracks added")
        print(f"⏱️ Total duration: {total_minutes}m {total_seconds}s")

        if args.debug:
            print("\n📝 Tracks added:")
            for i, track in enumerate(tracks, 1):
                artist_names = ", ".join(a["name"] for a in track["artists"])
                duration_ms = track.get('duration_ms', 0)
                duration_min = duration_ms // 60000
                duration_sec = (duration_ms % 60000) // 1000
                print(f"   {i:2d}. {track['name']} by {artist_names} ({duration_min:2d}:{duration_sec:02d})")
        else:
            print(f"📝 Added {len(tracks)} tracks to your {mood} riding playlist")

    except Exception as e:
        print(f"❌ Failed to create playlist: {e}", file=sys.stderr)
        sys.exit(1)

    print("🏍️ Ready to ride! Enjoy your personalized soundtrack! 🎵")


if __name__ == "__main__":
    main()
