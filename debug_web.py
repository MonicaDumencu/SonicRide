#!/usr/bin/env python3
"""
SonicRide Debug Startup Script
Starts the web application with enhanced debugging features
"""

import os
import sys
from pathlib import Path

def main():
    """Start SonicRide in debug mode"""
    print("🐛 SonicRide Debug Mode")
    print("=" * 30)
    
    # Set debug environment variables
    os.environ['FLASK_ENV'] = 'development'
    os.environ['FLASK_DEBUG'] = '1'
    
    # Check if we're in venv
    venv_path = Path('.venv')
    if venv_path.exists() and sys.prefix == sys.base_prefix:
        print("⚠️  Virtual environment not activated!")
        print("🔧 Run: .venv\\Scripts\\activate")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    print("🚀 Starting in debug mode...")
    print("📱 Web interface: http://localhost:5000")
    print("📝 Logs: sonicride_web.log")
    print("🛑 Press Ctrl+C to stop")
    print("=" * 30)
    
    # Import and run app
    from app import app
    app.run(
        debug=True,
        host='0.0.0.0', 
        port=5000,
        use_reloader=True,
        use_debugger=True,
        threaded=True
    )

if __name__ == '__main__':
    main()