#!/usr/bin/env python3
"""
GPX Timestamp Generator
Parses GPX files, uses map-matching with OSRM to get realistic travel times,
and generates new GPX files with accurate timestamps for every track point.

This tool:
1. Parses input GPX file and extracts the track polyline
2. Sends the polyline to OSRM for map-matching
3. Gets realistic distances and durations for each segment
4. Assigns cumulative timestamps to each track point
5. Writes a new GPX file with accurate <time> elements

Author: SonicRide GPX Tools
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Tuple, Optional
import xml.etree.ElementTree as ET
import requests
import json

try:
    import gpxpy
    import gpxpy.gpx
except ImportError:
    print("❌ Error: gpxpy not installed. Run: pip install gpxpy")
    sys.exit(1)

try:
    import polyline
except ImportError:
    try:
        # Try alternative installation
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "polyline"])
        import polyline
    except Exception:
        print("❌ Error: polyline not installed. Run: pip install polyline")
        sys.exit(1)

# Elevation API configuration
ELEVATION_API_URL = "https://api.open-elevation.com/api/v1/lookup"
MAX_ELEVATION_BATCH = 100  # Max points per elevation API request


# Default GPX folder relative to script location
GPX_FOLDER = Path(__file__).parent / "gpx"


def resolve_gpx_path(filename: str, create_folder: bool = False) -> Path:
    """Resolve GPX file path, looking in gpx folder first."""
    if create_folder:
        GPX_FOLDER.mkdir(exist_ok=True)
    
    # If it's already an absolute path, use it as-is
    path = Path(filename)
    if path.is_absolute():
        return path
    
    # Check if file exists in gpx folder
    gpx_path = GPX_FOLDER / filename
    if gpx_path.exists() or create_folder:
        return gpx_path
    
    # Check if file exists in current directory
    current_path = Path(filename)
    if current_path.exists():
        return current_path
    
    # Default to gpx folder for new files
    return gpx_path


class GPXTimestampGenerator:
    """GPX timestamp generator using OSRM map-matching."""
    
    def __init__(self, osrm_server: str = "http://router.project-osrm.org", 
                 profile: str = "car", debug: bool = False):
        """
        Initialize the GPX timestamp generator.
        
        Args:
            osrm_server: OSRM server URL (default: public OSRM server)
            profile: OSRM routing profile (car, bike, foot)
            debug: Enable debug output
        """
        self.osrm_server = osrm_server.rstrip('/')
        self.profile = profile
        self.debug = debug
        
    def parse_gpx(self, gpx_file: Path) -> List[Tuple[float, float, Optional[float]]]:
        """
        Parse GPX file and extract track points or route points.
        
        Args:
            gpx_file: Path to GPX file
            
        Returns:
            List of (lat, lon, elevation) tuples
        """
        if self.debug:
            print(f"📖 Parsing GPX file: {gpx_file}")
            
        try:
            with open(gpx_file, 'r', encoding='utf-8') as f:
                gpx = gpxpy.parse(f)
        except Exception as e:
            raise Exception(f"Failed to parse GPX file: {e}")
            
        points = []
        total_points = 0
        
        # First try to get track points
        for track in gpx.tracks:
            for segment in track.segments:
                for point in segment.points:
                    elevation = point.elevation if point.elevation is not None else None
                    points.append((point.latitude, point.longitude, elevation))
                    total_points += 1
        
        # If no tracks found, try routes
        if not points:
            for route in gpx.routes:
                for point in route.points:
                    elevation = point.elevation if point.elevation is not None else None
                    points.append((point.latitude, point.longitude, elevation))
                    total_points += 1
                    
        if not points:
            raise Exception("No track points or route points found in GPX file")
            
        if self.debug:
            if gpx.tracks:
                print(f"   Found {total_points} track points in {len(gpx.tracks)} tracks")
            elif gpx.routes:
                print(f"   Found {total_points} route points in {len(gpx.routes)} routes")
            
        return points
        
    def create_polyline(self, points: List[Tuple[float, float, Optional[float]]]) -> str:
        """
        Create a polyline string from track points.
        
        Args:
            points: List of (lat, lon, elevation) tuples
            
        Returns:
            Polyline encoded string
        """
        # Convert to (lat, lon) tuples for polyline encoding
        coords = [(lat, lon) for lat, lon, _ in points]
        return polyline.encode(coords, precision=5)
        
    def map_match_osrm(self, points: List[Tuple[float, float, Optional[float]]]) -> dict:
        """
        Send track to OSRM for map-matching, handling large routes by chunking.
        
        Args:
            points: List of (lat, lon, elevation) tuples
            
        Returns:
            OSRM map-matching response (combined if chunked)
        """
        if self.debug:
            print(f"🗺️  Map-matching {len(points)} points with OSRM...")
            
        # For routes with many points, we need to chunk them to avoid URL length limits
        max_points_per_request = 100  # Conservative limit to avoid 414 errors
        
        if len(points) <= max_points_per_request:
            return self._single_osrm_request(points)
        else:
            return self._chunked_osrm_request(points, max_points_per_request)
    
    def _single_osrm_request(self, points: List[Tuple[float, float, Optional[float]]]) -> dict:
        """Send a single OSRM request."""
        # Create coordinate string for OSRM (lon,lat format)
        coordinates = ";".join(f"{lon},{lat}" for lat, lon, _ in points)
        
        # OSRM match service URL
        url = f"{self.osrm_server}/match/v1/{self.profile}/{coordinates}"
        
        params = {
            'overview': 'full',
            'geometries': 'geojson',
            'annotations': 'true',
            'steps': 'false'
        }
        
        try:
            if self.debug:
                print(f"   Sending single request with {len(points)} points...")
                
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('code') != 'Ok':
                raise Exception(f"OSRM error: {data.get('code')} - {data.get('message', 'Unknown error')}")
                
            if self.debug:
                matchings = data.get('matchings', [])
                if matchings:
                    confidence = matchings[0].get('confidence', 0)
                    print(f"   Map-matching successful (confidence: {confidence:.2f})")
                else:
                    print("   Warning: No matchings returned")
                    
            return data
            
        except requests.RequestException as e:
            raise Exception(f"OSRM request failed: {e}")
        except json.JSONDecodeError as e:
            raise Exception(f"Invalid OSRM response: {e}")
    
    def _chunked_osrm_request(self, points: List[Tuple[float, float, Optional[float]]], chunk_size: int) -> dict:
        """Send multiple OSRM requests for large routes and combine results."""
        if self.debug:
            num_chunks = (len(points) + chunk_size - 1) // chunk_size
            print(f"   Route too large, splitting into {num_chunks} chunks of ~{chunk_size} points each...")
        
        all_durations = []
        total_duration = 0
        successful_chunks = 0
        
        # Process in overlapping chunks to maintain continuity
        for i in range(0, len(points), chunk_size - 1):  # Overlap by 1 point
            end_idx = min(i + chunk_size, len(points))
            chunk_points = points[i:end_idx]
            
            if len(chunk_points) < 2:  # Need at least 2 points for routing
                continue
                
            try:
                chunk_data = self._single_osrm_request(chunk_points)
                
                # Extract durations from this chunk
                matchings = chunk_data.get('matchings', [])
                if matchings:
                    for leg in matchings[0].get('legs', []):
                        if 'annotation' in leg and 'duration' in leg['annotation']:
                            leg_durations = leg['annotation']['duration']
                            # Skip first duration if this isn't the first chunk (avoid double-counting overlap)
                            start_idx = 1 if i > 0 and len(leg_durations) > 1 else 0
                            all_durations.extend(leg_durations[start_idx:])
                            total_duration += sum(leg_durations[start_idx:])
                
                successful_chunks += 1
                
            except Exception as e:
                if self.debug:
                    print(f"   Chunk {i//chunk_size + 1} failed: {e}")
                continue
        
        if self.debug:
            print(f"   Successfully processed {successful_chunks} chunks")
            print(f"   Total estimated duration: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
        
        # Create a combined response structure
        if all_durations:
            return {
                'code': 'Ok',
                'matchings': [{
                    'legs': [{
                        'annotation': {
                            'duration': all_durations
                        }
                    }]
                }]
            }
        else:
            return {'code': 'NoMatch'}
            
    def calculate_timestamps(self, points: List[Tuple[float, float, Optional[float]]], 
                           osrm_data: dict, start_time: datetime) -> List[datetime]:
        """
        Calculate timestamps for each track point based on OSRM durations.
        
        Args:
            points: Original track points
            osrm_data: OSRM map-matching response
            start_time: Starting timestamp for the track
            
        Returns:
            List of timestamps for each point
        """
        if self.debug:
            print(f"⏱️  Calculating timestamps from {start_time}...")
            
        matchings = osrm_data.get('matchings', [])
        if not matchings:
            # Fallback: distribute evenly over 1 hour
            if self.debug:
                print("   Warning: No matchings found, using fallback timing")
            return self._fallback_timestamps(points, start_time)
            
        matching = matchings[0]  # Use first (and usually only) matching
        legs = matching.get('legs', [])
        
        if not legs:
            return self._fallback_timestamps(points, start_time)
            
        # Extract segment durations from OSRM annotations
        segment_durations = []
        total_duration = 0
        
        for leg in legs:
            if 'annotation' in leg and 'duration' in leg['annotation']:
                leg_durations = leg['annotation']['duration']
                segment_durations.extend(leg_durations)
                total_duration += sum(leg_durations)
                
        if self.debug:
            print(f"   Total estimated duration: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
            print(f"   OSRM provided {len(segment_durations)} segment durations")
            print(f"   Distributing over {len(points)} route points")
            
        if total_duration <= 0 or not segment_durations:
            return self._fallback_timestamps(points, start_time)
            
        # Map OSRM segment durations to our route points
        timestamps = [start_time]  # First point gets start time
        current_time = start_time
        
        # Always use proportional distribution based on distance, but with OSRM's total realistic timing
        if len(segment_durations) > 0:
            if self.debug:
                print(f"   Using OSRM total duration ({total_duration:.1f}s) with distance-based distribution + speed variation")
                
            # Calculate total distance between points for proportional distribution
            distances = []
            total_distance = 0
            
            for i in range(1, len(points)):
                lat1, lon1, _ = points[i-1]
                lat2, lon2, _ = points[i]
                # Use simple distance calculation
                import math
                R = 6371000  # Earth radius in meters
                phi1, phi2 = math.radians(lat1), math.radians(lat2)
                dphi = phi2 - phi1
                dlambda = math.radians(lon2 - lon1)
                a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2
                distance = 2 * R * math.asin(math.sqrt(a))
                distances.append(distance)
                total_distance += distance
            
            # Base speed from OSRM total timing
            base_speed = total_distance / total_duration  # m/s
            base_speed_kmh = base_speed * 3.6
            
            if self.debug:
                print(f"   Base speed from OSRM: {base_speed_kmh:.1f} km/h")
            
            # Apply realistic speed variations for each route segment
            for i, distance in enumerate(distances):
                # Start with base speed
                speed = base_speed
                
                # Add realistic variations based on segment characteristics
                if distance < 10:  # Very short segment - likely tight turn or intersection
                    speed *= 0.5  # Much slower for very tight areas
                elif distance < 30:  # Short segment - tight turn
                    speed *= 0.7  # Slower for tight areas  
                elif distance > 150:  # Long segment - likely straight road or highway
                    speed *= 1.3  # Faster for straight sections
                elif distance > 80:  # Medium-long segment - country road
                    speed *= 1.1  # Slightly faster
                
                # Add some randomness to avoid perfect uniformity (±10%)
                import random
                variation = 0.9 + (random.random() * 0.2)  # 0.9 to 1.1 multiplier
                speed *= variation
                
                # Calculate time for this segment
                segment_duration = distance / speed if speed > 0 else distance / base_speed
                current_time += timedelta(seconds=segment_duration)
                timestamps.append(current_time)
        else:
            # No OSRM segments available - fallback
            return self._fallback_timestamps(points, start_time)
            
        if self.debug:
            actual_duration = (timestamps[-1] - timestamps[0]).total_seconds()
            print(f"   Applied duration: {actual_duration:.1f} seconds ({actual_duration/60:.1f} minutes)")
            
            # Show speed variation in first few segments for verification
            if len(timestamps) > 10:
                print("   Speed verification (first 10 segments):")
                for i in range(1, min(11, len(timestamps))):
                    lat1, lon1, _ = points[i-1]
                    lat2, lon2, _ = points[i]
                    import math
                    R = 6371000
                    phi1, phi2 = math.radians(lat1), math.radians(lat2)
                    dphi = phi2 - phi1
                    dlambda = math.radians(lon2 - lon1)
                    a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2
                    distance = 2 * R * math.asin(math.sqrt(a))
                    
                    time_diff = (timestamps[i] - timestamps[i-1]).total_seconds()
                    speed_kmh = (distance / time_diff) * 3.6 if time_diff > 0 else 0
                    print(f"     Segment {i}: {speed_kmh:.1f} km/h")
            
        return timestamps
        
    def _fallback_timestamps(self, points: List[Tuple[float, float, Optional[float]]], 
                           start_time: datetime) -> List[datetime]:
        """
        Fallback timestamp calculation - distribute points evenly over time.
        
        Args:
            points: Track points
            start_time: Starting timestamp
            
        Returns:
            List of evenly distributed timestamps
        """
        if self.debug:
            print("   Using fallback: evenly distributed timestamps over 1 hour")
            
        total_seconds = 3600  # 1 hour default
        interval = total_seconds / max(1, len(points) - 1)
        
        timestamps = []
        for i in range(len(points)):
            timestamp = start_time + timedelta(seconds=i * interval)
            timestamps.append(timestamp)
            
        return timestamps
        
    def fetch_elevations(self, points: List[Tuple[float, float, Optional[float]]]) -> List[Tuple[float, float, Optional[float]]]:
        """
        Fetch elevation data for points that don't have it.
        
        Args:
            points: List of (lat, lon, elevation) tuples
            
        Returns:
            List of (lat, lon, elevation) tuples with elevations filled in
        """
        if self.debug:
            points_without_elevation = sum(1 for _, _, ele in points if ele is None)
            print(f"🏔️  Fetching elevation data for {points_without_elevation}/{len(points)} points...")
            
        # If all points already have elevation, return as-is
        if all(ele is not None for _, _, ele in points):
            if self.debug:
                print("   All points already have elevation data")
            return points
            
        # Prepare points that need elevation
        updated_points = []
        points_to_fetch = []
        indices_to_update = []
        
        for i, (lat, lon, ele) in enumerate(points):
            if ele is None:
                points_to_fetch.append({"latitude": lat, "longitude": lon})
                indices_to_update.append(i)
            updated_points.append([lat, lon, ele])
            
        if not points_to_fetch:
            return points
            
        # Fetch elevations in batches
        try:
            fetched_elevations = []
            for i in range(0, len(points_to_fetch), MAX_ELEVATION_BATCH):
                batch = points_to_fetch[i:i + MAX_ELEVATION_BATCH]
                
                if self.debug and len(points_to_fetch) > MAX_ELEVATION_BATCH:
                    batch_num = i // MAX_ELEVATION_BATCH + 1
                    total_batches = (len(points_to_fetch) + MAX_ELEVATION_BATCH - 1) // MAX_ELEVATION_BATCH
                    print(f"   Fetching elevation batch {batch_num}/{total_batches} ({len(batch)} points)...")
                    
                response = requests.post(
                    ELEVATION_API_URL,
                    json={"locations": batch},
                    timeout=30
                )
                response.raise_for_status()
                
                data = response.json()
                if "results" in data:
                    for result in data["results"]:
                        elevation = result.get("elevation")
                        fetched_elevations.append(elevation)
                else:
                    # Fallback: use None for failed requests
                    fetched_elevations.extend([None] * len(batch))
                    
            # Update points with fetched elevations
            for idx, elevation in zip(indices_to_update, fetched_elevations):
                updated_points[idx][2] = elevation
                
            # Convert back to tuples
            result = [(lat, lon, ele) for lat, lon, ele in updated_points]
            
            if self.debug:
                successful_fetches = sum(1 for _, _, ele in result if ele is not None)
                print(f"   Successfully fetched {successful_fetches}/{len(points)} elevations")
                
            return result
            
        except Exception as e:
            if self.debug:
                print(f"   Elevation fetch failed: {e}")
                print("   Continuing without elevation data...")
            return points
        
    def create_output_gpx(self, points: List[Tuple[float, float, Optional[float]]], 
                         timestamps: List[datetime], original_gpx_file: Path, 
                         output_file: Path) -> None:
        """
        Create new GPX file with timestamps.
        
        Args:
            points: Track points
            timestamps: Corresponding timestamps
            original_gpx_file: Original GPX file for metadata
            output_file: Output GPX file path
        """
        if self.debug:
            print(f"📝 Creating output GPX: {output_file}")
            
        # Parse original GPX to preserve metadata
        try:
            with open(original_gpx_file, 'r', encoding='utf-8') as f:
                original_gpx = gpxpy.parse(f)
        except Exception:
            # Create minimal GPX if original can't be parsed
            original_gpx = gpxpy.gpx.GPX()
            
        # Create new GPX
        new_gpx = gpxpy.gpx.GPX()
        
        # Copy metadata
        if original_gpx.name:
            new_gpx.name = f"{original_gpx.name} (Timestamped)"
        else:
            new_gpx.name = "GPX Track with Timestamps"
            
        if original_gpx.description:
            new_gpx.description = f"{original_gpx.description} | Generated by GPX Timestamp Generator"
        else:
            new_gpx.description = "Generated by GPX Timestamp Generator using OSRM map-matching"
            
        # Create track (convert routes to tracks for consistency)
        track = gpxpy.gpx.GPXTrack()
        track.name = new_gpx.name
        new_gpx.tracks.append(track)
        
        # Create track segment
        segment = gpxpy.gpx.GPXTrackSegment()
        track.segments.append(segment)
        
        # Add timestamped points
        for i, ((lat, lon, elevation), timestamp) in enumerate(zip(points, timestamps)):
            point = gpxpy.gpx.GPXTrackPoint(
                latitude=lat,
                longitude=lon,
                elevation=elevation,
                time=timestamp
            )
            segment.points.append(point)
            
        # Write GPX file
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(new_gpx.to_xml())
                
            if self.debug:
                print(f"   Successfully wrote {len(points)} timestamped points")
                
        except Exception as e:
            raise Exception(f"Failed to write output GPX: {e}")
            
    def process_gpx(self, input_file: Path, output_file: Path, 
                   start_time: Optional[datetime] = None) -> None:
        """
        Process a GPX file and add timestamps.
        
        Args:
            input_file: Input GPX file path
            output_file: Output GPX file path  
            start_time: Starting timestamp (default: now)
        """
        if start_time is None:
            start_time = datetime.now()
            
        if self.debug:
            print(f"🚀 Processing GPX file: {input_file}")
            print(f"   Start time: {start_time}")
            print(f"   Output file: {output_file}")
            
        # Step 1: Parse GPX
        points = self.parse_gpx(input_file)
        
        # Step 2: Fetch elevation data if missing
        points = self.fetch_elevations(points)
        
        # Step 3: Map-match with OSRM
        try:
            osrm_data = self.map_match_osrm(points)
        except Exception as e:
            print(f"⚠️  Map-matching failed: {e}")
            print("   Using fallback timing calculation...")
            osrm_data = {}
            
        # Step 4: Calculate timestamps
        timestamps = self.calculate_timestamps(points, osrm_data, start_time)
        
        # Step 5: Create output GPX
        self.create_output_gpx(points, timestamps, input_file, output_file)
        
        print(f"✅ Successfully generated timestamped GPX: {output_file}")
        print(f"   Processed {len(points)} points from {timestamps[0]} to {timestamps[-1]}")


def parse_datetime(date_string: str) -> datetime:
    """Parse datetime string in various formats."""
    formats = [
        "%Y-%m-%dT%H:%M:%SZ",  # ISO format with Z
        "%Y-%m-%dT%H:%M:%S",   # ISO format without Z
        "%Y-%m-%d %H:%M:%S",   # Space separated
        "%Y-%m-%d %H:%M",      # Without seconds
        "%Y-%m-%d",            # Date only
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_string, fmt)
        except ValueError:
            continue
            
    raise ValueError(f"Unable to parse datetime: {date_string}")


def main():
    parser = argparse.ArgumentParser(
        description="GPX Timestamp Generator - Add accurate timestamps using OSRM map-matching",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python gpx_timestamp_generator.py input.gpx -o output.gpx
  python gpx_timestamp_generator.py track.gpx -o timed_track.gpx --start "2025-09-16 08:00:00"
  python gpx_timestamp_generator.py ride.gpx -o ride_timed.gpx --profile bike --debug
  python gpx_timestamp_generator.py track.gpx --osrm "http://localhost:5000" --debug

The tool automatically looks for input files in the gpx/ folder and saves output files there.
You can still use absolute paths if needed.

Map-matching uses OSRM (Open Source Routing Machine) to get realistic travel times
between track points. The tool supports car, bike, and foot profiles.

For best results with motorcycle rides, use 'car' profile (default).
        """
    )
    
    parser.add_argument("input_gpx", type=str,
                       help="Input GPX file to process (automatically looks in gpx/ folder)")
    parser.add_argument("-o", "--output", type=str,
                       help="Output GPX file (default: input_timestamped.gpx in gpx/ folder)")
    parser.add_argument("--start", type=str,
                       help="Start timestamp (default: now). Format: 'YYYY-MM-DD HH:MM:SS' or ISO")
    parser.add_argument("--osrm", default="http://router.project-osrm.org",
                       help="OSRM server URL (default: public server)")
    parser.add_argument("--profile", choices=["car", "bike", "foot"], default="car",
                       help="OSRM routing profile (default: car - good for motorcycles)")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug output")
    parser.add_argument("--skip-elevation", action="store_true",
                       help="Skip fetching elevation data (faster processing)")
    
    args = parser.parse_args()
    
    # Resolve input file path (check gpx folder first)
    input_file = resolve_gpx_path(args.input_gpx)
    if not input_file.exists():
        print(f"❌ Error: Input GPX file not found: {input_file}", file=sys.stderr)
        print(f"   Looked in: {GPX_FOLDER} and current directory", file=sys.stderr)
        sys.exit(1)
        
    # Set output file (default to gpx folder)
    if args.output is None:
        stem = input_file.stem
        suffix = input_file.suffix
        output_file = resolve_gpx_path(f"{stem}_timestamped{suffix}", create_folder=True)
    else:
        output_file = resolve_gpx_path(args.output, create_folder=True)
        
    # Parse start time
    start_time = None
    if args.start:
        try:
            start_time = parse_datetime(args.start)
        except ValueError as e:
            print(f"❌ Error: {e}", file=sys.stderr)
            sys.exit(1)
    
    # Initialize generator
    generator = GPXTimestampGenerator(
        osrm_server=args.osrm,
        profile=args.profile,
        debug=args.debug
    )
    
    # Skip elevation fetching if requested
    if args.skip_elevation:
        generator.fetch_elevations = lambda points: points
    
    # Process GPX file
    try:
        generator.process_gpx(input_file, output_file, start_time)
        
        if args.debug:
            file_size = output_file.stat().st_size
            print(f"📊 Output file size: {file_size:,} bytes")
            
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
        
    print("🏍️ GPX timestamp generation complete! Ready for analysis.")


if __name__ == "__main__":
    main()