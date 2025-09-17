#!/usr/bin/env python3
"""
Quick ride analysis script - extract key statistics from ride metrics CSV.
"""

import pandas as pd
import numpy as np
from datetime import datetime

def analyze_ride(csv_file):
    """Analyze ride metrics and print key statistics."""
    print(f"🏍️ Analyzing ride data from: {csv_file}")
    print("=" * 60)
    
    # Load data
    df = pd.read_csv(csv_file)
    
    # Basic info
    print(f"📊 RIDE OVERVIEW")
    print(f"   Total data points: {len(df):,}")
    print(f"   Start time: {df['time'].iloc[0]}")
    print(f"   End time: {df['time'].iloc[-1]}")
    
    # Parse timestamps for duration
    start_time = pd.to_datetime(df['time'].iloc[0])
    end_time = pd.to_datetime(df['time'].iloc[-1])
    duration = end_time - start_time
    print(f"   Total duration: {duration}")
    
    # Distance analysis
    total_distance = df['distance_m'].sum() / 1000  # Convert to km
    print(f"   Total distance: {total_distance:.2f} km")
    
    # Elevation analysis
    if df['altitude_m'].notna().any():
        min_alt = df['altitude_m'].min()
        max_alt = df['altitude_m'].max()
        elevation_gain = max_alt - min_alt
        print(f"   Elevation range: {min_alt:.0f}m - {max_alt:.0f}m (gain: {elevation_gain:.0f}m)")
    
    print()
    
    # Speed analysis
    speed_data = df['speed_kmh'].dropna()
    if len(speed_data) > 0:
        print(f"🚗 SPEED ANALYSIS")
        print(f"   Average speed: {speed_data.mean():.1f} km/h")
        print(f"   Maximum speed: {speed_data.max():.1f} km/h")
        print(f"   Minimum speed: {speed_data.min():.1f} km/h")
        print(f"   Speed std dev: {speed_data.std():.1f} km/h")
        
        # Speed distribution
        city_speed = (speed_data <= 50).sum()
        highway_speed = (speed_data > 80).sum()
        country_speed = len(speed_data) - city_speed - highway_speed
        
        print(f"   Speed distribution:")
        print(f"     City speeds (≤50 km/h): {city_speed:,} points ({city_speed/len(speed_data)*100:.1f}%)")
        print(f"     Country speeds (50-80 km/h): {country_speed:,} points ({country_speed/len(speed_data)*100:.1f}%)")
        print(f"     Highway speeds (>80 km/h): {highway_speed:,} points ({highway_speed/len(speed_data)*100:.1f}%)")
    
    print()
    
    # Lean angle analysis
    lean_data = df['lean_angle_deg'].dropna()
    if len(lean_data) > 0:
        print(f"🏍️ LEAN ANGLE ANALYSIS")
        print(f"   Average lean angle: {lean_data.mean():.1f}°")
        print(f"   Maximum lean angle: {lean_data.max():.1f}°")
        print(f"   Lean angles > 20°: {(lean_data > 20).sum():,} points ({(lean_data > 20).sum()/len(lean_data)*100:.1f}%)")
        print(f"   Lean angles > 30°: {(lean_data > 30).sum():,} points ({(lean_data > 30).sum()/len(lean_data)*100:.1f}%)")
        
        # Cornering style analysis
        cautious = (lean_data <= 15).sum()
        moderate = ((lean_data > 15) & (lean_data <= 25)).sum()
        spirited = ((lean_data > 25) & (lean_data <= 35)).sum()
        aggressive = (lean_data > 35).sum()
        
        print(f"   Cornering style:")
        print(f"     Cautious (≤15°): {cautious:,} points ({cautious/len(lean_data)*100:.1f}%)")
        print(f"     Moderate (15-25°): {moderate:,} points ({moderate/len(lean_data)*100:.1f}%)")
        print(f"     Spirited (25-35°): {spirited:,} points ({spirited/len(lean_data)*100:.1f}%)")
        print(f"     Aggressive (>35°): {aggressive:,} points ({aggressive/len(lean_data)*100:.1f}%)")
    
    print()
    
    # Turn radius analysis
    radius_data = df['turn_radius_m_raw'].dropna()
    if len(radius_data) > 0:
        print(f"🔄 CORNERING ANALYSIS")
        tight_corners = (radius_data < 50).sum()
        normal_corners = ((radius_data >= 50) & (radius_data < 200)).sum()
        sweeping_corners = (radius_data >= 200).sum()
        
        print(f"   Turn types:")
        print(f"     Tight corners (<50m): {tight_corners:,} points")
        print(f"     Normal corners (50-200m): {normal_corners:,} points")
        print(f"     Sweeping curves (>200m): {sweeping_corners:,} points")
        print(f"   Average turn radius: {radius_data.mean():.1f}m")
    
    print()
    print("🎯 RIDE SUMMARY")
    if len(speed_data) > 0 and len(lean_data) > 0:
        avg_speed = speed_data.mean()
        max_lean = lean_data.max()
        
        if avg_speed < 40 and max_lean < 20:
            style = "Urban/City riding - Cautious and traffic-aware"
        elif avg_speed < 60 and max_lean < 30:
            style = "Mixed riding - Balanced urban and country"
        elif avg_speed < 80 and max_lean < 35:
            style = "Spirited country riding - Confident cornering"
        else:
            style = "Aggressive/Highway riding - High speeds and lean angles"
            
        print(f"   Riding style: {style}")
        print(f"   Performance: Avg {avg_speed:.1f} km/h, Max lean {max_lean:.1f}°")

if __name__ == "__main__":
    analyze_ride("trip1_complete_metrics.csv")