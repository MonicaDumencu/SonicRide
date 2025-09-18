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

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global storage for processing status
processing_status = {}

class ProcessingPipeline:
    """Handles the complete SonicRide processing pipeline"""
    
    def __init__(self, job_id, gpx_file_path):
        self.job_id = job_id
        self.gpx_file_path = gpx_file_path
        self.status = {
            'stage': 'initializing',
            'progress': 0,
            'message': 'Starting processing...',
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
    
    def run_command(self, cmd, stage_name):
        """Run a command and capture output"""
        try:
            result = subprocess.run(
                cmd, 
                shell=True, 
                capture_output=True, 
                text=True, 
                cwd=os.getcwd()
            )
            
            if result.returncode != 0:
                raise Exception(f"{stage_name} failed: {result.stderr}")
            
            return result.stdout
        except Exception as e:
            raise Exception(f"Error in {stage_name}: {str(e)}")
    
    def process(self):
        """Run the complete processing pipeline"""
        try:
            base_name = Path(self.gpx_file_path).stem
            
            # Stage 1: Generate timestamps
            self.update_status('timestamp_generation', 10, 'Generating timestamps with OSRM...')
            timestamped_file = f"gpx/{base_name}_timestamped.gpx"
            
            cmd1 = f'python gpx_timestamp_generator.py "{self.gpx_file_path}" --output "{timestamped_file}"'
            self.run_command(cmd1, "Timestamp Generation")
            
            if not os.path.exists(timestamped_file):
                raise Exception("Timestamped GPX file was not created")
            
            # Stage 2: Generate ride metrics
            self.update_status('metrics_generation', 40, 'Analyzing ride metrics and lean angles...')
            metrics_file = f"{base_name}_metrics.csv"
            
            cmd2 = f'python ride_metrics.py -i "{timestamped_file}" -o "{metrics_file}"'
            self.run_command(cmd2, "Ride Metrics Generation")
            
            if not os.path.exists(metrics_file):
                raise Exception("Metrics CSV file was not created")
            
            # Stage 3: Generate Spotify playlist
            self.update_status('playlist_generation', 70, 'Creating Spotify playlist...')
            
            cmd3 = f'python sonicride.py --metrics "{metrics_file}"'
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
                playlist_info['name'] = line.split('Playlist created:')[-1].strip()
            elif 'Playlist URL:' in line:
                playlist_info['url'] = line.split('Playlist URL:')[-1].strip()
            elif 'tracks added' in line:
                try:
                    playlist_info['tracks_count'] = int(line.split()[0])
                except:
                    pass
            elif 'Total duration:' in line:
                playlist_info['duration'] = line.split('Total duration:')[-1].strip()
        
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
        pipeline = ProcessingPipeline(job_id, gpx_filename)
        thread = threading.Thread(target=pipeline.process)
        thread.daemon = True
        thread.start()
        
        return jsonify({'job_id': job_id, 'message': 'Processing started'})
        
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
    
    app.run(debug=True, host='0.0.0.0', port=5000)