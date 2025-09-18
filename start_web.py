#!/usr/bin/env python3
"""
SonicRide Startup Script
Easy way to start the SonicRide web application
"""

import os
import sys
import subprocess
from pathlib import Path

def check_requirements():
    """Check if required files exist"""
    required_files = [
        'gpx_timestamp_generator.py',
        'ride_metrics.py',
        'sonicride.py',
        'app.py'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {', '.join(missing_files)}")
        return False
    
    print("✅ All required files found")
    return True

def check_python_packages():
    """Check if required Python packages are installed"""
    required_packages = [
        'flask',
        'requests',
        'gpxpy',
        'pandas',
        'spotipy',
        'dotenv'  # correct import name for python-dotenv
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing Python packages: {', '.join(missing_packages)}")
        venv_path = Path('.venv')
        if venv_path.exists():
            print("📦 Run: python update_venv.py")
            print("🔧 Then activate venv and try again:")
            print("   .venv\\Scripts\\activate  (Windows)")
            print("   source .venv/bin/activate  (Linux/macOS)")
        else:
            print("📦 Install with: pip install -r web_requirements.txt")
        return False
    
    print("✅ All required packages installed")
    return True

def check_environment():
    """Check if .env file exists and has required variables"""
    env_file = Path('.env')
    
    if not env_file.exists():
        print("⚠️  No .env file found")
        print("🔧 Create .env file with your Spotify credentials:")
        print("   SPOTIFY_CLIENT_ID=your_client_id")
        print("   SPOTIFY_CLIENT_SECRET=your_client_secret")
        return False
    
    # Read .env file and check for required variables
    with open(env_file, 'r') as f:
        content = f.read()
    
    required_vars = ['SPOTIFY_CLIENT_ID', 'SPOTIFY_CLIENT_SECRET']
    missing_vars = []
    
    for var in required_vars:
        if var not in content or f"{var}=" not in content:
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing environment variables in .env: {', '.join(missing_vars)}")
        return False
    
    print("✅ Environment variables configured")
    return True

def create_directories():
    """Create necessary directories"""
    directories = ['uploads', 'gpx', 'static', 'templates']
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("✅ Directories created")

def main():
    """Main startup function"""
    print("🏍️  SonicRide - Starting Web Application")
    print("=" * 50)
    
    # Check if we're in a virtual environment
    venv_path = Path('.venv')
    if venv_path.exists() and not sys.prefix != sys.base_prefix:
        print("⚠️  Virtual environment detected but not activated!")
        print("🔧 Activate it first:")
        print("   .venv\\Scripts\\activate  (Windows)")
        print("   source .venv/bin/activate  (Linux/macOS)")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    if not check_python_packages():
        sys.exit(1)
    
    if not check_environment():
        print("⚠️  Warning: Spotify features may not work without proper .env configuration")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Create directories
    create_directories()
    
    print("\n🚀 Starting SonicRide Web Application...")
    print("📱 Open your browser and go to: http://localhost:5000")
    print("🛑 Press Ctrl+C to stop the server")
    print("=" * 50)
    
    try:
        # Start Flask application
        subprocess.run([sys.executable, 'app.py'], check=True)
    except KeyboardInterrupt:
        print("\n🛑 SonicRide stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Failed to start SonicRide: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()