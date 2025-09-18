/**
 * SonicRide Web Application JavaScript
 * Handles file upload, progress tracking, and UI interactions
 */

class SonicRideApp {
    constructor() {
        this.currentJobId = null;
        this.statusInterval = null;
        this.init();
    }

    init() {
        this.bindEvents();
        this.showSection('upload-section');
    }

    bindEvents() {
        // File upload form
        const uploadForm = document.getElementById('upload-form');
        const fileInput = document.getElementById('gpx-file');
        const uploadBtn = document.getElementById('upload-btn');

        uploadForm.addEventListener('submit', (e) => this.handleUpload(e));
        
        // File input change
        fileInput.addEventListener('change', (e) => {
            const fileName = e.target.files[0]?.name;
            const label = document.querySelector('.file-input-label span');
            if (fileName) {
                label.textContent = fileName;
                label.parentElement.style.borderColor = 'var(--primary-color)';
                label.parentElement.style.background = 'linear-gradient(45deg, rgba(29, 185, 84, 0.1), rgba(74, 144, 226, 0.1))';
            }
        });

        // Mode card selection
        const modeCards = document.querySelectorAll('.mode-card');
        const hiddenInput = document.getElementById('fast-mode');
        
        modeCards.forEach(card => {
            card.addEventListener('click', () => {
                // Remove active state from all cards
                modeCards.forEach(c => c.classList.remove('active'));
                
                // Add active state to clicked card
                card.classList.add('active');
                
                // Update hidden input value
                const isFastMode = card.dataset.mode === 'fast';
                hiddenInput.value = isFastMode ? 'true' : 'false';
                
                // Update upload button
                this.updateUploadButton(isFastMode);
                
                // Update selection indicators
                this.updateSelectionIndicators(card.dataset.mode);
            });
        });

        // Initialize with fast mode selected
        this.updateUploadButton(true);
        this.updateSelectionIndicators('fast');

        // Retry button
        const retryBtn = document.getElementById('retry-btn');
        retryBtn?.addEventListener('click', () => this.resetToUpload());

        // New upload button
        const newUploadBtn = document.getElementById('new-upload-btn');
        newUploadBtn?.addEventListener('click', () => this.resetToUpload());
    }

    async handleUpload(e) {
        e.preventDefault();
        
        const fileInput = document.getElementById('gpx-file');
        const fastModeInput = document.getElementById('fast-mode');
        const file = fileInput.files[0];
        
        if (!file) {
            this.showError('Please select a GPX file');
            return;
        }

        if (!file.name.toLowerCase().endsWith('.gpx')) {
            this.showError('Please select a valid GPX file');
            return;
        }

        // Show loading overlay
        this.showLoadingOverlay(true);

        try {
            // Upload file with processing mode preference
            const formData = new FormData();
            formData.append('gpx_file', file);
            formData.append('fast_mode', fastModeInput.value);

            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (!response.ok) {
                throw new Error(result.error || 'Upload failed');
            }

            // Start monitoring progress
            this.currentJobId = result.job_id;
            this.showSection('processing-section');
            this.showLoadingOverlay(false);
            this.startStatusPolling();

        } catch (error) {
            this.showLoadingOverlay(false);
            this.showError(error.message);
        }
    }

    startStatusPolling() {
        if (this.statusInterval) {
            clearInterval(this.statusInterval);
        }

        this.statusInterval = setInterval(async () => {
            try {
                await this.checkStatus();
            } catch (error) {
                console.error('Status check error:', error);
                this.stopStatusPolling();
                this.showError('Failed to check processing status');
            }
        }, 2000); // Check every 2 seconds

        // Initial check
        this.checkStatus();
    }

    stopStatusPolling() {
        if (this.statusInterval) {
            clearInterval(this.statusInterval);
            this.statusInterval = null;
        }
    }

    async checkStatus() {
        if (!this.currentJobId) return;

        const response = await fetch(`/status/${this.currentJobId}`);
        const status = await response.json();

        if (!response.ok) {
            throw new Error(status.error || 'Status check failed');
        }

        this.updateProgress(status);

        if (status.completed) {
            this.stopStatusPolling();
            this.showResults(status.results);
        } else if (status.error) {
            this.stopStatusPolling();
            this.showError(status.error);
        }
    }

    updateProgress(status) {
        // Update progress bar
        const progressFill = document.getElementById('progress-fill');
        const progressText = document.getElementById('progress-text');
        
        if (progressFill && progressText) {
            const roundedProgress = Math.round(status.progress * 100) / 100; // Round to 2 decimal places
            progressFill.style.width = `${roundedProgress}%`;
            progressText.textContent = `${roundedProgress}%`;
        }

        // Update status message
        const statusMessage = document.getElementById('status-message');
        if (statusMessage) {
            statusMessage.textContent = status.message;
        }

        // Update step indicators
        this.updateSteps(status.stage);
    }

    updateSteps(currentStage) {
        const steps = {
            'timestamp_generation': 'step-timestamp',
            'metrics_generation': 'step-metrics', 
            'playlist_generation': 'step-playlist',
            'completed': 'step-playlist'
        };

        // Reset all steps
        Object.values(steps).forEach(stepId => {
            const step = document.getElementById(stepId);
            if (step) {
                step.classList.remove('active', 'completed');
                const icon = step.querySelector('.step-status i');
                if (icon) {
                    icon.className = 'fas fa-hourglass-half';
                }
            }
        });

        // Mark completed steps
        const stageOrder = ['timestamp_generation', 'metrics_generation', 'playlist_generation'];
        const currentIndex = stageOrder.indexOf(currentStage);
        
        stageOrder.forEach((stage, index) => {
            const stepId = steps[stage];
            const step = document.getElementById(stepId);
            if (!step) return;

            if (index < currentIndex || currentStage === 'completed') {
                // Completed step
                step.classList.add('completed');
                const icon = step.querySelector('.step-status i');
                if (icon) {
                    icon.className = 'fas fa-check-circle';
                }
            } else if (index === currentIndex && currentStage !== 'completed') {
                // Active step
                step.classList.add('active');
                const icon = step.querySelector('.step-status i');
                if (icon) {
                    icon.className = 'fas fa-spinner fa-spin';
                }
            }
        });

        // If completed, mark all as completed
        if (currentStage === 'completed') {
            Object.values(steps).forEach(stepId => {
                const step = document.getElementById(stepId);
                if (step) {
                    step.classList.add('completed');
                    const icon = step.querySelector('.step-status i');
                    if (icon) {
                        icon.className = 'fas fa-check-circle';
                    }
                }
            });
        }
    }

    showResults(results) {
        // Update playlist information
        const playlistName = document.getElementById('playlist-name');
        const playlistStats = document.getElementById('playlist-stats');
        const playlistUrl = document.getElementById('playlist-url');

        if (results.playlist_info) {
            const info = results.playlist_info;
            
            if (playlistName) {
                playlistName.textContent = info.name || 'SonicRide Playlist';
            }
            
            if (playlistStats) {
                const stats = [];
                if (info.tracks_count) stats.push(`${info.tracks_count} tracks`);
                if (info.duration) stats.push(`Duration: ${info.duration}`);
                playlistStats.textContent = stats.join(' • ') || 'Playlist created successfully';
            }
            
            if (playlistUrl && info.url) {
                playlistUrl.href = info.url;
                playlistUrl.style.display = 'inline-flex';
            }
            
            // Update playlist icon with actual image if available
            const playlistIcon = document.querySelector('.playlist-icon');
            if (playlistIcon && info.image_url) {
                playlistIcon.innerHTML = `<img src="${info.image_url}" alt="Playlist Cover" class="playlist-cover-image">`;
            } else if (playlistIcon) {
                // Fallback to Spotify icon
                playlistIcon.innerHTML = '<i class="fab fa-spotify"></i>';
            }
        }

        // Update download links
        const downloadTimestamped = document.getElementById('download-timestamped');
        const downloadMetrics = document.getElementById('download-metrics');

        if (downloadTimestamped && results.timestamped_gpx) {
            downloadTimestamped.href = `/download/${encodeURIComponent(results.timestamped_gpx)}`;
        }

        if (downloadMetrics && results.metrics_csv) {
            downloadMetrics.href = `/download/${encodeURIComponent(results.metrics_csv)}`;
        }

        this.showSection('results-section');
    }

    showError(message) {
        const errorMessage = document.getElementById('error-message');
        if (errorMessage) {
            errorMessage.textContent = message;
        }
        this.showSection('error-section');
    }

    showSection(sectionId) {
        // Hide all sections
        const sections = [
            'upload-section',
            'processing-section', 
            'results-section',
            'error-section'
        ];

        sections.forEach(id => {
            const section = document.getElementById(id);
            if (section) {
                section.style.display = 'none';
            }
        });

        // Show target section
        const targetSection = document.getElementById(sectionId);
        if (targetSection) {
            targetSection.style.display = 'block';
            // Add fade-in animation
            targetSection.style.animation = 'fadeIn 0.5s ease-in-out';
        }
    }

    showLoadingOverlay(show) {
        const overlay = document.getElementById('loading-overlay');
        if (overlay) {
            overlay.style.display = show ? 'flex' : 'none';
        }
    }

    updateUploadButton(isFastMode) {
        const uploadBtn = document.getElementById('upload-btn');
        if (!uploadBtn) return;
        
        if (isFastMode) {
            uploadBtn.innerHTML = '<i class="fas fa-rocket"></i>Start Fast Processing';
        } else {
            uploadBtn.innerHTML = '<i class="fas fa-microscope"></i>Start Detailed Processing';
        }
    }
    
    updateSelectionIndicators(selectedMode) {
        const fastCard = document.querySelector('.fast-mode');
        const detailedCard = document.querySelector('.detailed-mode');
        
        if (fastCard && detailedCard) {
            const fastIndicator = fastCard.querySelector('.selection-indicator i');
            const detailedIndicator = detailedCard.querySelector('.selection-indicator i');
            
            if (selectedMode === 'fast') {
                if (fastIndicator) fastIndicator.className = 'fas fa-check-circle';
                if (detailedIndicator) detailedIndicator.className = 'far fa-circle';
            } else {
                if (fastIndicator) fastIndicator.className = 'far fa-circle';
                if (detailedIndicator) detailedIndicator.className = 'fas fa-check-circle';
            }
        }
    }

    resetToUpload() {
        // Reset form
        const form = document.getElementById('upload-form');
        if (form) {
            form.reset();
        }

        // Reset file input label
        const label = document.querySelector('.file-input-label span');
        if (label) {
            label.textContent = 'Choose GPX File';
            label.parentElement.style.borderColor = 'var(--text-muted)';
            label.parentElement.style.background = 'linear-gradient(45deg, var(--border-color), #333)';
        }

        // Reset mode selection to fast mode
        const modeCards = document.querySelectorAll('.mode-card');
        modeCards.forEach(c => c.classList.remove('active'));
        document.querySelector('.fast-mode')?.classList.add('active');
        document.getElementById('fast-mode').value = 'true';
        this.updateUploadButton(true);
        this.updateSelectionIndicators('fast');

        // Clear current job
        this.currentJobId = null;
        this.stopStatusPolling();

        // Reset progress
        const progressFill = document.getElementById('progress-fill');
        const progressText = document.getElementById('progress-text');
        if (progressFill) progressFill.style.width = '0%';
        if (progressText) progressText.textContent = '0%';

        // Reset steps
        this.updateSteps('');

        // Show upload section
        this.showSection('upload-section');
    }
}

// File validation helper
function validateGPXFile(file) {
    const errors = [];
    
    if (!file) {
        errors.push('No file selected');
        return errors;
    }
    
    // Check file extension
    if (!file.name.toLowerCase().endsWith('.gpx')) {
        errors.push('File must have .gpx extension');
    }
    
    // Check file size (50MB limit)
    const maxSize = 50 * 1024 * 1024; // 50MB
    if (file.size > maxSize) {
        errors.push('File size must be less than 50MB');
    }
    
    // Check if file is empty
    if (file.size === 0) {
        errors.push('File cannot be empty');
    }
    
    return errors;
}

// Utility functions
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function formatDuration(seconds) {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    
    if (hours > 0) {
        return `${hours}h ${minutes}m ${secs}s`;
    } else if (minutes > 0) {
        return `${minutes}m ${secs}s`;
    } else {
        return `${secs}s`;
    }
}

// Enhanced error handling
window.addEventListener('error', (event) => {
    console.error('Global error:', event.error);
    
    // Show user-friendly error message
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-toast';
    errorDiv.innerHTML = `
        <i class="fas fa-exclamation-triangle"></i>
        <span>An unexpected error occurred. Please try again.</span>
    `;
    errorDiv.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: var(--error-color);
        color: white;
        padding: 16px 20px;
        border-radius: var(--border-radius);
        display: flex;
        align-items: center;
        gap: 12px;
        z-index: 1001;
        box-shadow: var(--shadow);
        animation: slideIn 0.3s ease-out;
    `;
    
    document.body.appendChild(errorDiv);
    
    // Remove after 5 seconds
    setTimeout(() => {
        errorDiv.remove();
    }, 5000);
});

// Add slide-in animation for error toast
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
`;
document.head.appendChild(style);

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.sonicRideApp = new SonicRideApp();
});

// Prevent drag and drop on the page (except file input)
document.addEventListener('dragover', (e) => {
    e.preventDefault();
});

document.addEventListener('drop', (e) => {
    e.preventDefault();
});

// Add drag and drop support to file input
const uploadCard = document.querySelector('.upload-card');
if (uploadCard) {
    uploadCard.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadCard.classList.add('drag-over');
    });
    
    uploadCard.addEventListener('dragleave', (e) => {
        e.preventDefault();
        uploadCard.classList.remove('drag-over');
    });
    
    uploadCard.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadCard.classList.remove('drag-over');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            const fileInput = document.getElementById('gpx-file');
            if (fileInput) {
                fileInput.files = files;
                fileInput.dispatchEvent(new Event('change'));
            }
        }
    });
}

// Add CSS for drag over state
const dragStyles = document.createElement('style');
dragStyles.textContent = `
    .upload-card.drag-over {
        border-color: var(--primary-color);
        background: rgba(29, 185, 84, 0.1);
        transform: scale(1.02);
    }
`;
document.head.appendChild(dragStyles);