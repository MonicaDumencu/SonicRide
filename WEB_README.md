# SonicRide Web Application

A beautiful web interface for the SonicRide motorcycle GPS track analysis and Spotify playlist generator.

## Features

🏍️ **Upload GPX Files** - Drag and drop or browse for your motorcycle ride GPX files  
⏱️ **Real-time Progress** - Watch as your ride is processed through the complete pipeline  
🎵 **Spotify Integration** - Automatically creates personalized playlists based on your riding style  
📊 **Download Results** - Get timestamped GPX files and detailed ride metrics  
📱 **Responsive Design** - Works perfectly on desktop, tablet, and mobile devices  

## Quick Start

1. **Update Virtual Environment**
Since you already have a `.venv` directory, update it with web dependencies:
```bash
python update_venv.py
```

2. **Activate Virtual Environment**
```bash
# Windows
.venv\Scripts\activate

# Linux/macOS  
source .venv/bin/activate
```

3. **Set up Spotify Credentials**
Create a `.env` file with your Spotify app credentials:
```env
SPOTIFY_CLIENT_ID=your_spotify_client_id
SPOTIFY_CLIENT_SECRET=your_spotify_client_secret
```

4. **Start the Application**
```bash
python start_web.py
```

5. **Open Your Browser**
Navigate to `http://localhost:5000`

## How It Works

The web application runs the complete SonicRide pipeline:

1. **Upload GPX File** - Select your motorcycle ride GPX file
2. **Timestamp Generation** - Uses OSRM map-matching for realistic travel times
3. **Ride Metrics Analysis** - Calculates lean angles, cornering analysis, and speed data
4. **Spotify Playlist Creation** - Generates mood-based music selection matching your ride

## Processing Pipeline

```
Raw GPX → Timestamped GPX → Ride Metrics CSV → Spotify Playlist
```

### Stage 1: Timestamp Generation
- Uses OSRM routing service for map-matching
- Generates realistic travel times for each GPS point
- Handles elevation data from open-elevation.com

### Stage 2: Ride Metrics Analysis
- Physics-based motorcycle lean angle calculation (25-40° street caps)
- Turn radius analysis using circumcircle method
- Speed and acceleration analysis

### Stage 3: Spotify Playlist Creation
- Mood mapping based on speed and lean angle data
- Smart track selection with duration matching
- Automatic playlist creation in your Spotify account

## File Structure

```
SonicRide/
├── app.py                 # Flask web application
├── start_web.py           # Startup script
├── web_requirements.txt   # Web app dependencies
├── templates/
│   └── index.html        # Main HTML template
├── static/
│   ├── style.css         # CSS styling
│   └── script.js         # JavaScript functionality
├── uploads/              # Temporary file uploads
└── gpx/                  # GPX file processing directory
```

## API Endpoints

- `GET /` - Main web interface
- `POST /upload` - Upload GPX file and start processing
- `GET /status/<job_id>` - Check processing status
- `GET /download/<filename>` - Download generated files
- `GET /health` - Health check endpoint

## Configuration

### Environment Variables
- `SPOTIFY_CLIENT_ID` - Your Spotify app client ID
- `SPOTIFY_CLIENT_SECRET` - Your Spotify app client secret

### Flask Settings
- Maximum file upload size: 50MB
- Supported file types: `.gpx`
- Processing timeout: No limit (long rides may take several minutes)

## Development

### Running in Development Mode
Make sure your virtual environment is activated first:
```bash
# Activate venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/macOS

# Run the app
python app.py
```

### Production Deployment
```bash
# With virtual environment activated
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## Troubleshooting

### Common Issues

**Upload fails with "File too large"**
- Maximum file size is 50MB
- For larger files, process directly with command line tools

**Processing stalls at timestamp generation**
- Check internet connection (OSRM API required)
- Large routes may take several minutes to process

**Spotify playlist creation fails**
- Verify `.env` file has correct Spotify credentials
- Ensure Spotify app has proper permissions
- Check Spotify API rate limits

**Missing dependencies error**
- Make sure virtual environment is activated: `.venv\Scripts\activate`
- Run `python update_venv.py` to install web dependencies
- Ensure all original SonicRide Python files are present

### Getting Help

1. Check the browser console for JavaScript errors
2. Check the terminal where you started the app for Python errors
3. Verify all required files from original SonicRide are present
4. Test individual pipeline stages with command line tools first

## Security Notes

- Files are temporarily stored in `uploads/` directory
- GPX files are copied to `gpx/` directory for processing
- Clean up temporary files periodically in production
- Use HTTPS in production deployment
- Consider authentication for production use

## Performance Tips

- Smaller GPX files (< 1000 points) process faster
- Close browser tabs during processing to avoid timeout
- Use wired internet connection for better OSRM API performance
- Consider local OSRM server for better performance with large files