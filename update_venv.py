#!/usr/bin/env python3
"""
Update existing venv with SonicRide web application requirements
"""

import subprocess
import sys
import os
from pathlib import Path

def check_venv():
    """Check if .venv directory exists"""
    venv_path = Path(".venv")
    if not venv_path.exists():
        print("❌ No .venv directory found")
        print("Create one with: python -m venv .venv")
        return False
    
    print("✅ Found existing .venv directory")
    return True

def get_python_exe():
    """Get the Python executable for the venv"""
    if sys.platform == "win32":
        return ".venv\\Scripts\\python.exe"
    else:
        return ".venv/bin/python"

def get_pip_exe():
    """Get the pip executable for the venv"""
    if sys.platform == "win32":
        return ".venv\\Scripts\\pip.exe"
    else:
        return ".venv/bin/pip"

def install_web_requirements():
    """Install additional web requirements"""
    pip_exe = get_pip_exe()
    
    # Web-specific packages that might be missing
    web_packages = [
        "Flask>=2.3.3",
        "Werkzeug>=2.3.7", 
        "gunicorn>=21.2.0"
    ]
    
    print("📦 Installing web application dependencies...")
    
    for package in web_packages:
        try:
            print(f"   Installing {package}...")
            subprocess.run([pip_exe, "install", package], check=True, capture_output=True)
            print(f"   ✅ {package}")
        except subprocess.CalledProcessError as e:
            print(f"   ⚠️  Failed to install {package} (might already be installed)")
    
    print("✅ Web dependencies installation complete")

    # Install full requirements.txt if present to ensure all packages are available
    req_path = Path("requirements.txt")
    if req_path.exists():
        print("📦 Installing packages from requirements.txt...")
        try:
            subprocess.run([pip_exe, "install", "-r", str(req_path)], check=True)
            print("✅ requirements.txt installed")
        except subprocess.CalledProcessError:
            print("⚠️  Failed to install some packages from requirements.txt")
    else:
        print("⚠️  No requirements.txt found; skipping full requirements install")

def verify_installation():
    """Verify that all required packages are available"""
    python_exe = get_python_exe()
    
    required_packages = {
        'flask': 'Flask web framework',
        'requests': 'HTTP requests',
        'gpxpy': 'GPX file parsing', 
        'pandas': 'Data analysis',
        'spotipy': 'Spotify API',
        'dotenv': 'Environment variables'
    }
    
    print("🔍 Verifying installation...")
    all_good = True
    
    for package, description in required_packages.items():
        try:
            subprocess.run([
                python_exe, "-c", f"import {package}"
            ], check=True, capture_output=True)
            print(f"   ✅ {package} - {description}")
        except subprocess.CalledProcessError:
            print(f"   ❌ {package} - {description} (MISSING)")
            all_good = False
    
    return all_good

def main():
    """Main function"""
    print("🏍️  SonicRide - Update Existing Virtual Environment")
    print("=" * 55)
    
    if not check_venv():
        sys.exit(1)
    
    install_web_requirements()
    
    if verify_installation():
        print("\n🎉 Virtual environment updated successfully!")
        print("=" * 55)
        print("📋 Ready to start SonicRide:")
        print("1. Activate venv:")
        if sys.platform == "win32":
            print("   .venv\\Scripts\\activate")
        else:
            print("   source .venv/bin/activate")
        print("2. Start web app:")
        print("   python start_web.py")
        print("3. Open browser:")
        print("   http://localhost:5000")
    else:
        print("\n❌ Some packages are missing. You may need to install them manually.")
        sys.exit(1)

if __name__ == '__main__':
    main()