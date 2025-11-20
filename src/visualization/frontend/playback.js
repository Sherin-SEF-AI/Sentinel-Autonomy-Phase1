/**
 * SENTINEL Scenario Playback
 * 
 * Playback interface for recorded scenarios with frame-by-frame controls.
 */

class ScenarioPlayback {
    constructor() {
        this.scenarios = [];
        this.currentScenario = null;
        this.currentFrame = 0;
        this.isPlaying = false;
        this.playbackSpeed = 1.0;
        this.showAnnotations = true;
        this.playbackInterval = null;
        
        this.canvas = document.getElementById('playback-canvas');
        this.ctx = this.canvas.getContext('2d');
        
        this.init();
    }
    
    init() {
        console.log('Initializing Scenario Playback...');
        
        // Resize canvas
        this.resizeCanvas();
        window.addEventListener('resize', () => this.resizeCanvas());
        
        // Set up timeline dragging
        this.setupTimelineDragging();
        
        // Load scenarios
        this.loadScenarios();
    }
    
    resizeCanvas() {
        const content = document.querySelector('.playback-content');
        const rect = content.getBoundingClientRect();
        this.canvas.width = rect.width;
        this.canvas.height = rect.height;
    }
    
    setupTimelineDragging() {
        const handle = document.getElementById('timeline-handle');
        const timeline = document.getElementById('timeline');
        let isDragging = false;
        
        handle.addEventListener('mousedown', (e) => {
            isDragging = true;
            e.preventDefault();
        });
        
        document.addEventListener('mousemove', (e) => {
            if (!isDragging || !this.currentScenario) return;
            
            const rect = timeline.getBoundingClientRect();
            const x = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
            const progress = x / rect.width;
            
            this.seekToProgress(progress);
        });
        
        document.addEventListener('mouseup', () => {
            isDragging = false;
        });
    }
    
    async loadScenarios() {
        console.log('Loading scenarios...');
        
        try {
            const response = await fetch('/api/scenarios');
            const data = await response.json();
            
            this.scenarios = data.scenarios || [];
            this.renderScenarioList();
            
            console.log(`Loaded ${this.scenarios.length} scenarios`);
        } catch (error) {
            console.error('Error loading scenarios:', error);
            document.getElementById('scenarios-container').innerHTML = 
                '<div class="no-scenario">Error loading scenarios</div>';
        }
    }
    
    renderScenarioList() {
        const container = document.getElementById('scenarios-container');
        
        if (this.scenarios.length === 0) {
            container.innerHTML = '<div class="no-scenario">No scenarios recorded</div>';
            return;
        }
        
        container.innerHTML = this.scenarios.map(scenario => {
            const date = new Date(scenario.metadata.timestamp);
            const duration = scenario.metadata.duration.toFixed(1);
            const trigger = scenario.metadata.trigger;
            
            return `
                <div class="scenario-item" onclick="playback.loadScenario('${scenario.id}')">
                    <div class="scenario-time">${date.toLocaleString()}</div>
                    <div class="scenario-details">
                        Duration: ${duration}s | Trigger: ${trigger}
                    </div>
                </div>
            `;
        }).join('');
    }
    
    async loadScenario(scenarioId) {
        console.log(`Loading scenario: ${scenarioId}`);
        
        try {
            const response = await fetch(`/api/scenarios/${scenarioId}`);
            const data = await response.json();
            
            this.currentScenario = data;
            this.currentFrame = 0;
            this.isPlaying = false;
            
            // Update UI
            document.querySelectorAll('.scenario-item').forEach(item => {
                item.classList.remove('active');
            });
            event.target.closest('.scenario-item').classList.add('active');
            
            // Show canvas and controls
            document.getElementById('no-scenario-message').style.display = 'none';
            document.getElementById('playback-canvas').style.display = 'block';
            document.getElementById('info-panel').style.display = 'block';
            
            // Enable controls
            document.getElementById('play-btn').disabled = false;
            document.getElementById('prev-btn').disabled = false;
            document.getElementById('next-btn').disabled = false;
            
            // Update info panel
            this.updateInfoPanel();
            
            // Render first frame
            this.renderFrame();
            
            console.log(`Loaded scenario with ${this.currentScenario.annotations.frames.length} frames`);
        } catch (error) {
            console.error('Error loading scenario:', error);
            alert('Failed to load scenario');
        }
    }
    
    updateInfoPanel() {
        if (!this.currentScenario) return;
        
        const metadata = this.currentScenario.metadata;
        const annotations = this.currentScenario.annotations;
        
        document.getElementById('info-trigger').textContent = metadata.trigger;
        document.getElementById('info-duration').textContent = metadata.duration.toFixed(1) + 's';
        
        // Count total detections
        let totalDetections = 0;
        let maxRisk = 0;
        
        annotations.frames.forEach(frame => {
            if (frame.detections_3d) {
                totalDetections += frame.detections_3d.length;
            }
            if (frame.risk_assessment && frame.risk_assessment.top_risks) {
                frame.risk_assessment.top_risks.forEach(risk => {
                    maxRisk = Math.max(maxRisk, risk.contextual_score);
                });
            }
        });
        
        document.getElementById('info-detections').textContent = totalDetections;
        document.getElementById('info-risk').textContent = (maxRisk * 100).toFixed(0) + '%';
    }
    
    renderFrame() {
        if (!this.currentScenario) return;
        
        const frames = this.currentScenario.annotations.frames;
        if (this.currentFrame >= frames.length) {
            this.currentFrame = frames.length - 1;
        }
        
        const frame = frames[this.currentFrame];
        
        // Clear canvas
        this.ctx.fillStyle = '#1a1a1a';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Draw frame (placeholder - would load actual video frame)
        this.ctx.fillStyle = '#2a2a2a';
        this.ctx.fillRect(50, 50, this.canvas.width - 100, this.canvas.height - 100);
        
        // Draw frame info
        this.ctx.fillStyle = '#ffffff';
        this.ctx.font = '16px monospace';
        this.ctx.fillText(
            `Frame ${this.currentFrame + 1} / ${frames.length}`,
            20, 30
        );
        this.ctx.fillText(
            `Time: ${frame.timestamp.toFixed(2)}s`,
            20, 50
        );
        
        // Draw annotations if enabled
        if (this.showAnnotations) {
            this.drawAnnotations(frame);
        }
        
        // Update timeline
        this.updateTimeline();
    }
    
    drawAnnotations(frame) {
        // Draw detections
        if (frame.detections_3d) {
            frame.detections_3d.forEach(det => {
                const bbox = det.bbox_3d;
                
                // Convert 3D to 2D (simplified projection)
                const x = this.canvas.width / 2 + bbox.x * 10;
                const y = this.canvas.height / 2 - bbox.y * 10;
                const size = Math.max(bbox.w, bbox.l) * 10;
                
                // Draw bounding box
                this.ctx.strokeStyle = this.getClassColor(det.class_name);
                this.ctx.lineWidth = 2;
                this.ctx.strokeRect(x - size/2, y - size/2, size, size);
                
                // Draw label
                this.ctx.fillStyle = this.getClassColor(det.class_name);
                this.ctx.font = '12px sans-serif';
                this.ctx.fillText(
                    `${det.class_name} (${det.track_id})`,
                    x - size/2,
                    y - size/2 - 5
                );
            });
        }
        
        // Draw risk indicators
        if (frame.risk_assessment && frame.risk_assessment.top_risks) {
            let y = 80;
            frame.risk_assessment.top_risks.forEach(risk => {
                const score = (risk.contextual_score * 100).toFixed(0);
                const color = this.getUrgencyColor(risk.urgency);
                
                this.ctx.fillStyle = color;
                this.ctx.font = '14px sans-serif';
                this.ctx.fillText(
                    `⚠️ ${risk.hazard.type}: ${score}% (${risk.urgency})`,
                    20, y
                );
                y += 20;
            });
        }
    }
    
    getClassColor(className) {
        const colors = {
            'vehicle': '#00aaff',
            'pedestrian': '#ff4444',
            'cyclist': '#ffaa00',
            'traffic_sign': '#00ff88',
            'traffic_light': '#ff00ff'
        };
        return colors[className] || '#ffffff';
    }
    
    getUrgencyColor(urgency) {
        const colors = {
            'critical': '#ff4444',
            'high': '#ff8800',
            'medium': '#ffaa00',
            'low': '#00ff88'
        };
        return colors[urgency] || '#ffffff';
    }
    
    updateTimeline() {
        if (!this.currentScenario) return;
        
        const frames = this.currentScenario.annotations.frames;
        const progress = this.currentFrame / (frames.length - 1);
        
        document.getElementById('timeline-progress').style.width = (progress * 100) + '%';
        document.getElementById('timeline-handle').style.left = (progress * 100) + '%';
        
        const currentTime = frames[this.currentFrame].timestamp;
        const totalTime = frames[frames.length - 1].timestamp;
        
        document.getElementById('time-display').textContent = 
            `${currentTime.toFixed(2)} / ${totalTime.toFixed(2)}`;
    }
    
    togglePlayback() {
        if (!this.currentScenario) return;
        
        this.isPlaying = !this.isPlaying;
        
        const playBtn = document.getElementById('play-btn');
        playBtn.textContent = this.isPlaying ? '⏸' : '▶';
        
        if (this.isPlaying) {
            this.startPlayback();
        } else {
            this.stopPlayback();
        }
    }
    
    startPlayback() {
        const frames = this.currentScenario.annotations.frames;
        const frameInterval = 1000 / 30 / this.playbackSpeed; // 30 FPS adjusted by speed
        
        this.playbackInterval = setInterval(() => {
            this.currentFrame++;
            
            if (this.currentFrame >= frames.length) {
                this.currentFrame = 0; // Loop
            }
            
            this.renderFrame();
        }, frameInterval);
    }
    
    stopPlayback() {
        if (this.playbackInterval) {
            clearInterval(this.playbackInterval);
            this.playbackInterval = null;
        }
    }
    
    previousFrame() {
        if (!this.currentScenario) return;
        
        this.stopPlayback();
        this.isPlaying = false;
        document.getElementById('play-btn').textContent = '▶';
        
        this.currentFrame = Math.max(0, this.currentFrame - 1);
        this.renderFrame();
    }
    
    nextFrame() {
        if (!this.currentScenario) return;
        
        this.stopPlayback();
        this.isPlaying = false;
        document.getElementById('play-btn').textContent = '▶';
        
        const frames = this.currentScenario.annotations.frames;
        this.currentFrame = Math.min(frames.length - 1, this.currentFrame + 1);
        this.renderFrame();
    }
    
    seekToPosition(event) {
        if (!this.currentScenario) return;
        
        const timeline = document.getElementById('timeline');
        const rect = timeline.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const progress = x / rect.width;
        
        this.seekToProgress(progress);
    }
    
    seekToProgress(progress) {
        if (!this.currentScenario) return;
        
        const frames = this.currentScenario.annotations.frames;
        this.currentFrame = Math.floor(progress * (frames.length - 1));
        this.renderFrame();
    }
    
    setPlaybackSpeed(speed) {
        this.playbackSpeed = speed;
        
        // Update button states
        document.querySelectorAll('.speed-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        event.target.classList.add('active');
        
        // Restart playback if playing
        if (this.isPlaying) {
            this.stopPlayback();
            this.startPlayback();
        }
    }
    
    toggleAnnotations() {
        this.showAnnotations = !this.showAnnotations;
        
        const toggle = document.getElementById('annotation-toggle');
        if (this.showAnnotations) {
            toggle.classList.add('active');
        } else {
            toggle.classList.remove('active');
        }
        
        this.renderFrame();
    }
}

// Global functions for HTML onclick handlers
let playback;

function loadScenarios() {
    if (playback) {
        playback.loadScenarios();
    }
}

function togglePlayback() {
    if (playback) {
        playback.togglePlayback();
    }
}

function previousFrame() {
    if (playback) {
        playback.previousFrame();
    }
}

function nextFrame() {
    if (playback) {
        playback.nextFrame();
    }
}

function seekToPosition(event) {
    if (playback) {
        playback.seekToPosition(event);
    }
}

function setPlaybackSpeed(speed) {
    if (playback) {
        playback.setPlaybackSpeed(speed);
    }
}

function toggleAnnotations() {
    if (playback) {
        playback.toggleAnnotations();
    }
}

// Initialize playback when page loads
window.addEventListener('DOMContentLoaded', () => {
    console.log('Starting SENTINEL Playback...');
    playback = new ScenarioPlayback();
});
