#!/usr/bin/env python3
"""
SonicRide Web Application
A Flask web interface for processing GPX files through the complete SonicRide pipeline:
1. GPX timestamp generation
2. Ride metrics analysis
3. Spotify playlist creation
"""

import os
import json
import uuid
import threading
import logging
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import subprocess
import sys

app = Flask(__name__)
app.config['SECRET_KEY'] = 'sonicride-secret-key-change-in-production'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sonicride_web.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global storage for processing status
processing_status = {}

class ProcessingPipeline:
    """Handles the complete SonicRide processing pipeline"""
    
    def __init__(self, job_id, gpx_file_path, fast_mode=True):
        self.job_id = job_id
        self.gpx_file_path = gpx_file_path
        self.fast_mode = fast_mode
        mode_text = "fast" if fast_mode else "detailed"
        self.status = {
            'stage': 'initializing',
            'progress': 0,
            'message': f'Starting {mode_text} processing...',
            'error': None,
            'completed': False,
            'results': {}
        }
        processing_status[job_id] = self.status
    
    def update_status(self, stage, progress, message, error=None):
        """Update processing status"""
        self.status.update({
            'stage': stage,
            'progress': progress,
            'message': message,
            'error': error,
            'timestamp': datetime.now().isoformat()
        })
        processing_status[self.job_id] = self.status
        
        # Log status updates
        if error:
            logger.error(f"Job {self.job_id} - {stage}: {error}")
        else:
            logger.info(f"Job {self.job_id} - {stage} ({progress}%): {message}")
    
    def run_command(self, cmd, stage_name):
        """Run a command and capture output"""
        try:
            # Set environment to use UTF-8 encoding
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            env['PYTHONLEGACYWINDOWSSTDIO'] = '0'  # Force UTF-8 on Windows
            
            result = subprocess.run(
                cmd, 
                shell=True, 
                capture_output=True, 
                text=True, 
                cwd=os.getcwd(),
                env=env,
                encoding='utf-8',
                errors='replace'  # Replace problematic characters instead of failing
            )
            
            # Log the raw return code and check for success/failure
            logger.info(f"Command '{cmd}' returned code: {result.returncode}")
            
            if result.returncode != 0:
                # Clean the error message of problematic Unicode characters
                if result.stderr:
                    error_msg = result.stderr.replace('✅', '[OK]').replace('❌', '[ERROR]')
                    error_msg = error_msg.encode('ascii', errors='ignore').decode('ascii')
                else:
                    error_msg = "Command failed with no error output"
                
                logger.error(f"Command failed: {error_msg}")
                raise Exception(f"{stage_name} failed: {error_msg}")
            
            # Check if output files were actually created instead of relying on stdout
            logger.info(f"{stage_name} completed successfully (return code 0)")
            
            # Clean the output for return
            if result.stdout:
                clean_output = result.stdout.replace('✅', '[OK]').replace('❌', '[ERROR]')
                return clean_output
            else:
                return "Command completed successfully"
                
        except UnicodeDecodeError as e:
            logger.error(f"Unicode encoding error in {stage_name}: {e}")
            raise Exception(f"Error in {stage_name}: Unicode encoding issue - command may have succeeded, checking output files...")
        except Exception as e:
            logger.error(f"Exception in {stage_name}: {e}")
            raise Exception(f"Error in {stage_name}: {str(e)}")
    
    def run_timestamp_generation_with_progress(self, input_filename):
        """Run timestamp generation with real-time progress updates"""
        try:
            # Import the GPX timestamp generator directly
            import sys
            import importlib.util
            
            # Load the GPX timestamp generator module
            spec = importlib.util.spec_from_file_location("gpx_timestamp_generator", "gpx_timestamp_generator.py")
            gpx_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(gpx_module)
            
            # Create progress callback
            def progress_callback(percentage, message):
                # Map timestamp generation progress (0-100%) to overall progress (5-40%)
                overall_progress = round(5 + (percentage * 0.35), 2)  # 35% of total progress, rounded to 2 decimals
                self.update_status('timestamp_generation', overall_progress, message)
            
            # Resolve paths
            input_file = gpx_module.resolve_gpx_path(input_filename)
            base_name = Path(input_filename).stem
            output_file = gpx_module.resolve_gpx_path(f"{base_name}_timestamped.gpx", create_folder=True)
            
            # Create generator with progress callback
            generator = gpx_module.GPXTimestampGenerator(
                osrm_server="http://router.project-osrm.org",
                profile="car",
                debug=False,  # Disable debug to avoid console spam
                progress_callback=progress_callback
            )
            
            # Configure generator settings
            generator.fetch_elevations = lambda points: points  # Skip elevation fetching
            
            if self.fast_mode:
                generator._fast_mode = True
            
            # Process the GPX file
            generator.process_gpx(input_file, output_file)
            
            logger.info("Timestamp generation completed successfully")
            
        except Exception as e:
            logger.error(f"Timestamp generation failed: {e}")
            raise Exception(f"Timestamp Generation failed: {str(e)}")
    
    def process(self):
        """Run the complete processing pipeline"""
        try:
            # self.gpx_file_path is now just the filename (e.g., "trip.gpx")
            # gpx_timestamp_generator.py will resolve it to gpx/trip.gpx
            input_filename = self.gpx_file_path
            base_name = Path(input_filename).stem
            
            # Stage 1: Generate timestamps with detailed progress tracking
            self.update_status('timestamp_generation', 5, 'Initializing timestamp generation...')
            timestamped_file = f"gpx/{base_name}_timestamped.gpx"
            
            mode_desc = "fast mode (optimized for speed)" if self.fast_mode else "detailed mode (maximum precision)"
            logger.info(f"Processing in {mode_desc}")
            logger.info(f"Looking for input file: gpx/{input_filename}")
            
            # Run timestamp generation with progress monitoring
            self.run_timestamp_generation_with_progress(input_filename)
            
            if not os.path.exists(timestamped_file):
                raise Exception("Timestamped GPX file was not created")
            
            # Stage 2: Generate ride metrics
            self.update_status('metrics_generation', 45, 'Analyzing ride metrics and lean angles...')
            metrics_file = f"{base_name}_metrics.csv"
            
            cmd2 = f'python ride_metrics.py -i "{timestamped_file}" -o "{metrics_file}"'
            logger.info(f"Running command: {cmd2}")
            self.run_command(cmd2, "Ride Metrics Generation")
            
            if not os.path.exists(metrics_file):
                raise Exception("Metrics CSV file was not created")
            
            # Stage 3: Generate Spotify playlist
            self.update_status('playlist_generation', 70, 'Creating Spotify playlist...')
            
            cmd3 = f'python sonicride.py --metrics "{metrics_file}"'
            logger.info(f"Running command: {cmd3}")
            playlist_output = self.run_command(cmd3, "Spotify Playlist Generation")
            
            # Parse playlist information from output
            playlist_info = self.parse_playlist_output(playlist_output)
            
            # Final status update
            self.status.update({
                'stage': 'completed',
                'progress': 100,
                'message': 'Processing completed successfully!',
                'completed': True,
                'results': {
                    'timestamped_gpx': timestamped_file,
                    'metrics_csv': metrics_file,
                    'playlist_info': playlist_info
                }
            })
            processing_status[self.job_id] = self.status
            
        except Exception as e:
            self.update_status('error', 0, 'Processing failed', str(e))
    
    def parse_playlist_output(self, output):
        """Parse Spotify playlist information from command output"""
        playlist_info = {
            'name': 'Unknown',
            'url': None,
            'tracks_count': 0,
            'duration': 'Unknown'
        }
        
        lines = output.split('\n')
        for line in lines:
            if 'Playlist created:' in line:
                # Extract URL from the "🎶 Playlist created: URL" line
                url_part = line.split('Playlist created:')[-1].strip()
                playlist_info['url'] = url_part
            elif 'Playlist name:' in line:
                # Extract name from the "📝 Playlist name: Name" line
                name_part = line.split('Playlist name:')[-1].strip()
                playlist_info['name'] = name_part
            elif 'Playlist image:' in line:
                # Extract image URL from the "🖼️ Playlist image: URL" line
                image_part = line.split('Playlist image:')[-1].strip()
                playlist_info['image_url'] = image_part
            elif 'Playlist URL:' in line:
                playlist_info['url'] = line.split('Playlist URL:')[-1].strip()
            elif 'tracks added' in line and line.strip().split()[0].isdigit():
                # Parse "📊 5 tracks added" format
                try:
                    playlist_info['tracks_count'] = int(line.strip().split()[1])
                except:
                    pass
            elif 'Total duration:' in line:
                # Parse "⏱️ Total duration: 15m 30s" format
                duration_part = line.split('Total duration:')[-1].strip()
                playlist_info['duration'] = duration_part
        
        # Set default name if not found
        if playlist_info['name'] == 'Unknown' and 'open.spotify.com' in playlist_info.get('url', ''):
            playlist_info['name'] = 'SonicRide Playlist'
        
        logger.info(f"Parsed playlist info: {playlist_info}")
        return playlist_info

def allowed_file(filename):
    """Check if uploaded file is a valid GPX file"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'gpx'

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle GPX file upload and start processing"""
    if 'gpx_file' not in request.files:
        return jsonify({'error': 'No file selected'}), 400
    
    file = request.files['gpx_file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Only GPX files are allowed'}), 400
    
    try:
        # Get processing mode preference
        fast_mode = request.form.get('fast_mode', '').lower() == 'true'
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        job_id = str(uuid.uuid4())
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{job_id}_{filename}")
        file.save(upload_path)
        
        # Copy to gpx folder for processing
        gpx_filename = f"gpx/{filename}"
        os.makedirs('gpx', exist_ok=True)
        
        # Read and write to ensure proper encoding
        with open(upload_path, 'r', encoding='utf-8') as src:
            content = src.read()
        with open(gpx_filename, 'w', encoding='utf-8') as dst:
            dst.write(content)
        
        # Start processing in background thread
        # Pass filename and processing mode preference
        pipeline = ProcessingPipeline(job_id, filename, fast_mode)
        thread = threading.Thread(target=pipeline.process)
        thread.daemon = True
        thread.start()
        
        mode_text = "fast" if fast_mode else "detailed"
        return jsonify({'job_id': job_id, 'message': f'Processing started in {mode_text} mode'})
        
    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/status/<job_id>')
def get_status(job_id):
    """Get processing status for a job"""
    if job_id not in processing_status:
        return jsonify({'error': 'Job not found'}), 404
    
    return jsonify(processing_status[job_id])

@app.route('/download/<path:filename>')
def download_file(filename):
    """Download generated files"""
    try:
        if filename.endswith('.gpx'):
            return send_from_directory('gpx', filename, as_attachment=True)
        else:
            return send_from_directory('.', filename, as_attachment=True)
    except FileNotFoundError:
        return jsonify({'error': 'File not found'}), 404

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

if __name__ == '__main__':
    # Check if required modules are available
    required_files = [
        'gpx_timestamp_generator.py',
        'ride_metrics.py', 
        'sonicride.py'
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print(f"Warning: Missing required files: {missing_files}")
    
    print("SonicRide Web Application Starting...")
    print("Upload a GPX file to generate timestamps, analyze ride metrics, and create a Spotify playlist!")
    print("🐛 Debug mode enabled - automatic reloading and error pages active")
    
    # Enable debug mode with detailed error pages
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=True, use_debugger=True)