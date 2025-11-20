/**
 * SENTINEL Visualization Frontend
 * 
 * Real-time visualization dashboard for SENTINEL safety system.
 * Connects to backend WebSocket for live data streaming.
 */

class SentinelDashboard {
    constructor() {
        this.ws = null;
        this.reconnectInterval = 3000;
        this.reconnectTimer = null;
        
        // Canvas contexts
        this.bevCanvas = document.getElementById('bev-canvas');
        this.bevCtx = this.bevCanvas.getContext('2d');
        this.cameraCanvas = document.getElementById('camera-canvas');
        this.cameraCtx = this.cameraCanvas.getContext('2d');
        
        // State
        this.currentView = 'bev';
        this.latestData = null;
        this.frameCount = 0;
        
        this.init();
    }
    
    init() {
        console.log('Initializing SENTINEL Dashboard...');
        
        // Set up canvas sizes
        this.resizeCanvases();
        window.addEventListener('resize', () => this.resizeCanvases());
        
        // Set up view tabs
        document.querySelectorAll('.tab').forEach(tab => {
            tab.addEventListener('click', (e) => {
                this.switchView(e.target.dataset.view);
            });
        });
        
        // Connect to WebSocket
        this.connect();
    }
    
    resizeCanvases() {
        const viewContent = document.querySelector('.view-content');
        const rect = viewContent.getBoundingClientRect();
        
        this.bevCanvas.width = rect.width;
        this.bevCanvas.height = rect.height;
        this.cameraCanvas.width = rect.width;
        this.cameraCanvas.height = rect.height;
    }
    
    connect() {
        console.log('Connecting to SENTINEL backend...');
        
        // Determine WebSocket URL
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const host = window.location.hostname || 'localhost';
        const port = window.location.port || '8080';
        const wsUrl = `${protocol}//${host}:${port}/ws/stream`;
        
        try {
            this.ws = new WebSocket(wsUrl);
            
            this.ws.onopen = () => {
                console.log('Connected to SENTINEL backend');
                this.updateConnectionStatus(true);
                
                // Clear reconnect timer
                if (this.reconnectTimer) {
                    clearTimeout(this.reconnectTimer);
                    this.reconnectTimer = null;
                }
            };
            
            this.ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.handleData(data);
                } catch (error) {
                    console.error('Error parsing message:', error);
                }
            };
            
            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
            };
            
            this.ws.onclose = () => {
                console.log('Disconnected from SENTINEL backend');
                this.updateConnectionStatus(false);
                
                // Attempt to reconnect
                this.reconnectTimer = setTimeout(() => {
                    console.log('Attempting to reconnect...');
                    this.connect();
                }, this.reconnectInterval);
            };
            
        } catch (error) {
            console.error('Error creating WebSocket:', error);
            this.updateConnectionStatus(false);
        }
    }
    
    updateConnectionStatus(connected) {
        const statusDot = document.getElementById('connection-status');
        const statusText = document.getElementById('connection-text');
        
        if (connected) {
            statusDot.classList.add('connected');
            statusText.textContent = 'Connected';
        } else {
            statusDot.classList.remove('connected');
            statusText.textContent = 'Disconnected';
        }
    }
    
    handleData(data) {
        this.latestData = data;
        this.frameCount++;
        
        // Update timestamp
        document.getElementById('timestamp').textContent = 
            `Timestamp: ${data.timestamp.toFixed(2)}s`;
        
        // Update FPS
        if (data.performance && data.performance.fps) {
            document.getElementById('fps-display').textContent = 
                `${data.performance.fps.toFixed(1)} FPS`;
        }
        
        // Update visualizations
        if (data.bev) {
            this.updateBEV(data.bev, data.segmentation);
        }
        
        if (data.detections) {
            this.updateDetections(data.detections);
        }
        
        if (data.driver_state) {
            this.updateDriverState(data.driver_state);
        }
        
        if (data.risk_assessment) {
            this.updateRisks(data.risk_assessment);
        }
        
        if (data.alerts && data.alerts.length > 0) {
            this.showAlerts(data.alerts);
        }
        
        if (data.performance) {
            this.updatePerformance(data.performance);
        }
    }
    
    updateBEV(bev, segmentation) {
        if (this.currentView !== 'bev') return;
        
        // Decode base64 image
        const img = new Image();
        img.onload = () => {
            // Clear canvas
            this.bevCtx.fillStyle = '#1a1a1a';
            this.bevCtx.fillRect(0, 0, this.bevCanvas.width, this.bevCanvas.height);
            
            // Calculate scaling to fit canvas
            const scale = Math.min(
                this.bevCanvas.width / img.width,
                this.bevCanvas.height / img.height
            );
            const x = (this.bevCanvas.width - img.width * scale) / 2;
            const y = (this.bevCanvas.height - img.height * scale) / 2;
            
            // Draw BEV image
            this.bevCtx.drawImage(img, x, y, img.width * scale, img.height * scale);
            
            // Draw segmentation overlay if available
            if (segmentation && segmentation.overlay) {
                const overlayImg = new Image();
                overlayImg.onload = () => {
                    this.bevCtx.globalAlpha = 0.5;
                    this.bevCtx.drawImage(overlayImg, x, y, img.width * scale, img.height * scale);
                    this.bevCtx.globalAlpha = 1.0;
                };
                overlayImg.src = 'data:image/jpeg;base64,' + segmentation.overlay;
            }
            
            // Draw detections on BEV
            if (this.latestData && this.latestData.detections) {
                this.drawDetectionsOnBEV(this.latestData.detections, x, y, scale);
            }
        };
        img.src = 'data:image/jpeg;base64,' + bev.image;
    }
    
    drawDetectionsOnBEV(detections, offsetX, offsetY, scale) {
        detections.forEach(det => {
            const bbox = det.bbox_3d;
            
            // Convert vehicle coordinates to BEV pixel coordinates
            // BEV: 640x640, 0.1m per pixel, vehicle at (320, 480)
            const bevX = 320 + bbox.x / 0.1;
            const bevY = 480 - bbox.y / 0.1;
            
            // Scale to canvas
            const canvasX = offsetX + bevX * scale;
            const canvasY = offsetY + bevY * scale;
            const size = Math.max(bbox.w, bbox.l) / 0.1 * scale;
            
            // Draw bounding box
            this.bevCtx.strokeStyle = this.getClassColor(det.class_name);
            this.bevCtx.lineWidth = 2;
            this.bevCtx.strokeRect(
                canvasX - size/2,
                canvasY - size/2,
                size,
                size
            );
            
            // Draw label
            this.bevCtx.fillStyle = this.getClassColor(det.class_name);
            this.bevCtx.font = '12px sans-serif';
            this.bevCtx.fillText(
                `${det.class_name} (${det.track_id})`,
                canvasX - size/2,
                canvasY - size/2 - 5
            );
        });
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
    
    updateDetections(detections) {
        // Detections are drawn on BEV view
        // Could also update a separate 3D view here
    }
    
    updateDriverState(driverState) {
        // Update readiness score
        const readiness = driverState.readiness_score || 0;
        document.getElementById('readiness-score').textContent = readiness.toFixed(0);
        document.getElementById('readiness-bar').style.width = `${readiness}%`;
        
        // Update attention zone
        const attentionZone = driverState.gaze?.attention_zone || '--';
        document.getElementById('attention-zone').textContent = attentionZone;
        
        // Update drowsiness
        const drowsiness = driverState.drowsiness?.score || 0;
        document.getElementById('drowsiness-score').textContent = 
            (drowsiness * 100).toFixed(0) + '%';
        
        // Update distraction
        const distraction = driverState.distraction?.type || 'none';
        document.getElementById('distraction-type').textContent = distraction;
    }
    
    updateRisks(riskAssessment) {
        const container = document.getElementById('risks-container');
        
        if (!riskAssessment.top_risks || riskAssessment.top_risks.length === 0) {
            container.innerHTML = '<div class="loading">No risks detected</div>';
            return;
        }
        
        container.innerHTML = '';
        
        riskAssessment.top_risks.forEach(risk => {
            const riskEl = document.createElement('div');
            riskEl.className = `risk-item ${risk.urgency}`;
            
            const hazard = risk.hazard;
            const score = (risk.contextual_score * 100).toFixed(0);
            const ttc = hazard.ttc ? hazard.ttc.toFixed(1) : '--';
            const aware = risk.driver_aware ? '👁️ Aware' : '⚠️ Unaware';
            
            riskEl.innerHTML = `
                <div class="risk-header">
                    <span class="risk-type">${hazard.type.toUpperCase()}</span>
                    <span class="risk-score">${score}%</span>
                </div>
                <div class="risk-details">
                    Zone: ${hazard.zone} | TTC: ${ttc}s | ${aware}
                </div>
            `;
            
            container.appendChild(riskEl);
        });
    }
    
    showAlerts(alerts) {
        const container = document.getElementById('alerts-container');
        
        alerts.forEach(alert => {
            const alertEl = document.createElement('div');
            alertEl.className = `alert ${alert.urgency}`;
            
            alertEl.innerHTML = `
                <div class="alert-header">${alert.urgency.toUpperCase()}</div>
                <div class="alert-message">${alert.message}</div>
            `;
            
            container.appendChild(alertEl);
            
            // Auto-dismiss after 5 seconds
            setTimeout(() => {
                alertEl.style.opacity = '0';
                setTimeout(() => alertEl.remove(), 300);
            }, 5000);
        });
    }
    
    updatePerformance(performance) {
        const container = document.getElementById('perf-metrics');
        
        const metrics = [
            { label: 'FPS', value: performance.fps?.toFixed(1) || '--' },
            { label: 'Total Latency', value: performance.latency?.total?.toFixed(1) + 'ms' || '--' },
            { label: 'BEV', value: performance.latency?.bev?.toFixed(1) + 'ms' || '--' },
            { label: 'Segmentation', value: performance.latency?.segmentation?.toFixed(1) + 'ms' || '--' },
            { label: 'Detection', value: performance.latency?.detection?.toFixed(1) + 'ms' || '--' },
            { label: 'DMS', value: performance.latency?.dms?.toFixed(1) + 'ms' || '--' },
            { label: 'Intelligence', value: performance.latency?.intelligence?.toFixed(1) + 'ms' || '--' },
            { label: 'CPU', value: performance.cpu_percent?.toFixed(1) + '%' || '--' },
        ];
        
        if (performance.gpu_memory_mb) {
            metrics.push({
                label: 'GPU Memory',
                value: (performance.gpu_memory_mb / 1024).toFixed(2) + 'GB'
            });
        }
        
        container.innerHTML = metrics.map(m => `
            <div class="perf-item">
                <span class="perf-label">${m.label}:</span>
                <span class="perf-value">${m.value}</span>
            </div>
        `).join('');
    }
    
    switchView(view) {
        this.currentView = view;
        
        // Update tabs
        document.querySelectorAll('.tab').forEach(tab => {
            tab.classList.remove('active');
            if (tab.dataset.view === view) {
                tab.classList.add('active');
            }
        });
        
        // Update panels
        document.querySelectorAll('.view-panel').forEach(panel => {
            panel.classList.remove('active');
        });
        document.getElementById(`${view}-view`).classList.add('active');
        
        // Redraw current view
        if (this.latestData) {
            if (view === 'bev' && this.latestData.bev) {
                this.updateBEV(this.latestData.bev, this.latestData.segmentation);
            }
        }
    }
}

// Initialize dashboard when page loads
window.addEventListener('DOMContentLoaded', () => {
    console.log('Starting SENTINEL Dashboard...');
    new SentinelDashboard();
});
