#!/usr/bin/env python3
"""
ride_metrics.py — Parse a Calimoto GPX ride and compute speed, altitude, and realistic motorcycle lean angles.

Outputs a CSV with:
timestamp, lat, lon, altitude_m, speed_kmh, lean_angle_deg, turn_radius_m, dt_s, distance_m

Lean angle calculation uses physics (centripetal force) but applies realistic street riding caps:
- City/tight corners: ~25° (cautious street riding)
- Normal corners: ~32° (typical spirited street riding)
- Country roads: ~38° (aggressive street riding)
- Highway sweepers: up to 40° (maximum realistic street limit)

Usage:
  python ride_metrics.py --input ride.gpx --output metrics.csv --smooth-speed 5 --smooth-lean 5

Dependencies:
  pip install gpxpy
"""

import argparse
import csv
import math
from typing import List, Optional

import gpxpy

G = 9.81  # gravitational acceleration (m/s^2)
R_EARTH = 6371000.0  # meters

# -------------------------
# Geodesy helpers
# -------------------------
def haversine_distance(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance (meters) between two lat/lon points."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2
    return 2 * R_EARTH * math.asin(math.sqrt(a))

def initial_bearing_deg(lat1, lon1, lat2, lon2) -> float:
    """Initial bearing from point 1 to point 2 in degrees [0, 360)."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)
    y = math.sin(dlambda) * math.cos(phi2)
    x = math.cos(phi1)*math.sin(phi2) - math.sin(phi1)*math.cos(phi2)*math.cos(dlambda)
    theta = math.degrees(math.atan2(y, x))
    return (theta + 360.0) % 360.0

def local_xy(lat_ref, lon_ref, lat, lon):
    """
    Convert lat/lon (deg) to local tangent plane x,y (meters) using equirectangular projection
    referenced at (lat_ref, lon_ref). Assumes small distances between points.
    """
    x = math.radians(lon - lon_ref) * R_EARTH * math.cos(math.radians(lat_ref))
    y = math.radians(lat - lat_ref) * R_EARTH
    return x, y

# -------------------------
# Curvature / turn radius
# -------------------------
def turn_radius_from_three_points(p1, p2, p3) -> Optional[float]:
    """
    Approximate turn radius (meters) from three consecutive points using circumcircle method in local XY.
    Returns None if points are nearly collinear or degenerate.
    p1, p2, p3: dicts with 'lat', 'lon'
    """
    lat0, lon0 = p2['lat'], p2['lon']
    x1, y1 = local_xy(lat0, lon0, p1['lat'], p1['lon'])
    x2, y2 = local_xy(lat0, lon0, p2['lat'], p2['lon'])
    x3, y3 = local_xy(lat0, lon0, p3['lat'], p3['lon'])

    d = 2 * (x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))
    if abs(d) < 1e-6:
        return None  # nearly collinear

    ux = ((x1**2 + y1**2)*(y2 - y3) + (x2**2 + y2**2)*(y3 - y1) + (x3**2 + y3**2)*(y1 - y2)) / d
    uy = ((x1**2 + y1**2)*(x3 - x2) + (x2**2 + y2**2)*(x1 - x3) + (x3**2 + y3**2)*(x2 - x1)) / d

    r = math.hypot(ux - x1, uy - y1)
    return r if r > 0 else None

# -------------------------
# Smoothing helpers
# -------------------------
def moving_average(values: List[Optional[float]], window: int) -> List[Optional[float]]:
    """Simple centered moving average; None values are skipped; returns None where insufficient data."""
    if window <= 1:
        return values[:]
    half = window // 2
    out: List[Optional[float]] = [None]*len(values)
    for i in range(len(values)):
        lo, hi = max(0, i - half), min(len(values), i + half + 1)
        window_vals = [v for v in values[lo:hi] if v is not None]
        out[i] = (sum(window_vals)/len(window_vals)) if window_vals else None
    return out

# -------------------------
# GPX processing
# -------------------------
def extract_points_from_gpx(path: str) -> List[dict]:
    """
    Flatten points from all tracks/segments into a single time-ordered list of dicts:
    {'time': datetime|None, 'lat': float, 'lon': float, 'ele': float|None}
    Preserves original order; assumes GPX track points are already time ordered.
    """
    with open(path, "r", encoding="utf-8") as f:
        gpx = gpxpy.parse(f)

    points: List[dict] = []
    for trk in gpx.tracks:
        for seg in trk.segments:
            for pt in seg.points:
                points.append({
                    'time': pt.time,          # datetime or None
                    'lat': pt.latitude,
                    'lon': pt.longitude,
                    'ele': pt.elevation       # may be None
                })
    return points

def compute_metrics(points: List[dict],
                    min_dt_s: float = 1.0,
                    min_move_speed_kmh: float = 3.0,
                    smooth_speed_window: int = 5,
                    smooth_lean_window: int = 5,
                    max_lean_deg_cap: Optional[float] = 40.0) -> List[dict]:
    """
    For each point, compute:
      - distance to previous point (m)
      - dt (s)
      - speed_kmh (None if cannot compute)
      - altitude_m
      - turn_radius_m (from triplets)
      - lean_angle_deg (approx; None if cannot compute)

    Guards:
      - If dt < min_dt_s, skip speed/lean for that sample to avoid division noise spikes.
      - Speeds < min_move_speed_kmh are treated as stationary for lean calc.
      - Optionally cap lean angles to a realistic maximum (e.g., 60°).

    Smoothing:
      - Apply centered moving average to speed and lean series if windows > 1.
    """
    n = len(points)
    out = []
    # First pass: raw per-point metrics
    for i in range(n):
        cur = points[i]
        prev = points[i-1] if i > 0 else None
        nxt  = points[i+1] if i < n-1 else None

        altitude_m = cur['ele'] if cur['ele'] is not None else None

        # distance and dt
        distance_m = None
        dt_s = None
        speed_kmh = None
        if prev:
            distance_m = haversine_distance(prev['lat'], prev['lon'], cur['lat'], cur['lon'])
            if cur['time'] and prev['time']:
                dt_s = (cur['time'] - prev['time']).total_seconds()
                # Only calculate speed if we have sufficient time difference and reasonable distance
                if dt_s and dt_s >= min_dt_s and distance_m is not None:
                    calculated_speed = (distance_m / dt_s) * 3.6
                    # Apply realistic speed cap for European road conditions (max ~200 km/h)
                    speed_kmh = min(calculated_speed, 200.0) if calculated_speed > 0 else None

        # turn radius and lean
        turn_radius_m = None
        lean_angle_deg = None
        if prev and nxt and speed_kmh is not None and speed_kmh >= min_move_speed_kmh:
            turn_radius_m = turn_radius_from_three_points(prev, cur, nxt)
            if turn_radius_m and turn_radius_m > 0:
                v = speed_kmh / 3.6  # m/s
                # Only calculate lean if speed is reasonable (between 10-180 km/h for meaningful lean)
                if 10.0 <= speed_kmh <= 180.0:
                    # Calculate theoretical lean angle from physics
                    theoretical_lean_rad = math.atan((v*v) / (turn_radius_m * G))
                    theoretical_lean_deg = math.degrees(theoretical_lean_rad)
                    
                    # Apply realistic caps for street riding:
                    # - Tight city corners: 25-30° (cautious street riding)
                    # - Normal corners: 30-35° (typical spirited street riding)
                    # - Highway sweepers: up to 40° (aggressive but realistic street limit)
                    if turn_radius_m < 30:  # Very tight city/parking lot turns
                        max_realistic = 25.0
                    elif turn_radius_m < 100:  # Normal street corners
                        max_realistic = 32.0
                    elif turn_radius_m < 300:  # Spirited country road corners
                        max_realistic = 38.0
                    else:  # Highway on-ramps and sweeping curves
                        max_realistic = 40.0
                    
                    # Use the smaller of theoretical physics or realistic riding limit
                    lean_angle_deg = min(theoretical_lean_deg, max_realistic)
                    
                    # Final safety cap
                    if max_lean_deg_cap is not None:
                        lean_angle_deg = min(lean_angle_deg, max_lean_deg_cap)

        out.append({
            'time': cur['time'].isoformat() if cur['time'] else None,
            'lat': cur['lat'],
            'lon': cur['lon'],
            'altitude_m': altitude_m,
            'distance_m': distance_m,
            'dt_s': dt_s,
            'speed_kmh_raw': speed_kmh,
            'turn_radius_m_raw': turn_radius_m,
            'lean_angle_deg_raw': lean_angle_deg,
        })

    # Second pass: outlier removal and smoothing
    speeds = [row['speed_kmh_raw'] for row in out]
    leans  = [row['lean_angle_deg_raw'] for row in out]
    
    # Remove extreme outliers in speed data
    valid_speeds = [s for s in speeds if s is not None and 0 <= s <= 200]
    if valid_speeds:
        # Calculate reasonable bounds (mean ± 3 standard deviations)
        import statistics
        mean_speed = statistics.mean(valid_speeds)
        try:
            std_speed = statistics.stdev(valid_speeds) if len(valid_speeds) > 1 else 0
            max_reasonable = min(200.0, mean_speed + 3 * std_speed)
            min_reasonable = max(0.0, mean_speed - 3 * std_speed)
        except:
            max_reasonable, min_reasonable = 200.0, 0.0
            
        # Filter out unrealistic speeds
        speeds_filtered = []
        for s in speeds:
            if s is not None and min_reasonable <= s <= max_reasonable:
                speeds_filtered.append(s)
            else:
                speeds_filtered.append(None)
        speeds = speeds_filtered
    
    speeds_sm = moving_average(speeds, smooth_speed_window) if smooth_speed_window > 1 else speeds
    leans_sm  = moving_average(leans, smooth_lean_window)   if smooth_lean_window  > 1 else leans

    for i, row in enumerate(out):
        row['speed_kmh'] = speeds_sm[i]
        row['lean_angle_deg'] = leans_sm[i]
        # keep raw if you want to inspect:
        # del row['speed_kmh_raw']; del row['lean_angle_deg_raw']

    return out

def write_csv(rows: List[dict], path: str):
    fieldnames = [
        'time', 'lat', 'lon',
        'altitude_m',
        'speed_kmh', 'lean_angle_deg',
        'turn_radius_m_raw',  # helpful for debugging
        'dt_s', 'distance_m'
    ]
    with open(path, "w", newline='', encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, None) for k in fieldnames})

# -------------------------
# CLI
# -------------------------
def main():
    ap = argparse.ArgumentParser(description="Compute speed, altitude, and approximate lean angle from a GPX ride.")
    ap.add_argument("--input", "-i", required=True, help="Input GPX file (Calimoto export).")
    ap.add_argument("--output", "-o", required=True, help="Output CSV path.")
    ap.add_argument("--smooth-speed", type=int, default=5, help="Moving average window for speed (odd number recommended).")
    ap.add_argument("--smooth-lean", type=int, default=5, help="Moving average window for lean angle (odd number recommended).")
    ap.add_argument("--min-dt", type=float, default=1.0, help="Minimum dt (s) to compute speed (avoid noise).")
    ap.add_argument("--min-speed", type=float, default=3.0, help="Minimum speed (km/h) to compute lean.")
    ap.add_argument("--max-lean", type=float, default=40.0, help="Cap lean angle to this maximum (deg) for street riding; set -1 to disable.")
    args = ap.parse_args()

    points = extract_points_from_gpx(args.input)
    if not points:
        raise SystemExit("No track points found in GPX.")

    max_lean_cap = None if args.max_lean is not None and args.max_lean < 0 else args.max_lean
    rows = compute_metrics(
        points,
        min_dt_s=args.min_dt,
        min_move_speed_kmh=args.min_speed,
        smooth_speed_window=max(1, int(args.smooth_speed)),
        smooth_lean_window=max(1, int(args.smooth_lean)),
        max_lean_deg_cap=max_lean_cap
    )
    write_csv(rows, args.output)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
    