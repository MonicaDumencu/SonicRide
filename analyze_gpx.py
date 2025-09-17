#!/usr/bin/env python3
"""
GPX analysis script - check the raw GPX data for timing and distance patterns.
"""

import gpxpy
import math
from datetime import datetime

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in meters."""
    R = 6371000  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def analyze_gpx_raw(gpx_file):
    """Analyze raw GPX data to check timing and distances."""
    print(f"🔍 Analyzing raw GPX data: {gpx_file}")
    print("=" * 60)
    
    with open(gpx_file, 'r', encoding='utf-8') as f:
        gpx = gpxpy.parse(f)
    
    points = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                points.append({
                    'lat': point.latitude,
                    'lon': point.longitude,
                    'ele': point.elevation,
                    'time': point.time
                })
    
    print(f"📊 GPX FILE ANALYSIS")
    print(f"   Total points: {len(points)}")
    
    if len(points) < 2:
        print("   Not enough points for analysis")
        return
    
    # Check timestamps
    first_time = points[0]['time']
    last_time = points[-1]['time']
    total_duration = (last_time - first_time).total_seconds()
    
    print(f"   Start time: {first_time}")
    print(f"   End time: {last_time}")
    print(f"   Total duration: {total_duration/60:.1f} minutes")
    
    # Calculate raw distances and time differences
    distances = []
    time_diffs = []
    speeds = []
    
    for i in range(1, min(11, len(points))):  # Check first 10 segments
        prev_point = points[i-1]
        curr_point = points[i]
        
        distance = haversine_distance(
            prev_point['lat'], prev_point['lon'],
            curr_point['lat'], curr_point['lon']
        )
        
        time_diff = (curr_point['time'] - prev_point['time']).total_seconds()
        speed_ms = distance / time_diff if time_diff > 0 else 0
        speed_kmh = speed_ms * 3.6
        
        distances.append(distance)
        time_diffs.append(time_diff)
        speeds.append(speed_kmh)
        
        print(f"   Segment {i}: {distance:.2f}m in {time_diff:.2f}s = {speed_kmh:.1f} km/h")
    
    print()
    print("🕐 TIMING ANALYSIS")
    print(f"   Average time difference: {sum(time_diffs)/len(time_diffs):.2f} seconds")
    print(f"   Min time difference: {min(time_diffs):.2f} seconds")
    print(f"   Max time difference: {max(time_diffs):.2f} seconds")
    print(f"   Time differences vary: {'Yes' if max(time_diffs) - min(time_diffs) > 1 else 'No'}")
    
    print()
    print("📏 DISTANCE ANALYSIS")
    print(f"   Average segment distance: {sum(distances)/len(distances):.2f} meters")
    print(f"   Min segment distance: {min(distances):.2f} meters")
    print(f"   Max segment distance: {max(distances):.2f} meters")
    print(f"   Distance varies: {'Yes' if max(distances) - min(distances) > 10 else 'No'}")
    
    print()
    print("🚗 SPEED ANALYSIS (First 10 segments)")
    print(f"   Average speed: {sum(speeds)/len(speeds):.1f} km/h")
    print(f"   Min speed: {min(speeds):.1f} km/h")
    print(f"   Max speed: {max(speeds):.1f} km/h")
    print(f"   Speed variation: {max(speeds) - min(speeds):.1f} km/h")
    
    # Check if speeds are suspiciously constant
    speed_std = (sum([(s - sum(speeds)/len(speeds))**2 for s in speeds]) / len(speeds))**0.5
    print(f"   Speed standard deviation: {speed_std:.2f} km/h")
    
    if speed_std < 1.0:
        print("   ⚠️  ISSUE: Speeds are suspiciously constant!")
        print("   This suggests timing distribution might be too uniform.")
    else:
        print("   ✅ Speed variation looks normal")

if __name__ == "__main__":
    analyze_gpx_raw("gpx/trip1_complete.gpx")