# SENTINEL Quick Start Guide

Get SENTINEL up and running in 5 minutes!

## 🚀 Installation (One-Time Setup)

```bash
# 1. Navigate to project directory
cd /home/user/Sentinel-Autonomy-Phase1

# 2. Install dependencies
pip3 install -r requirements.txt
```

## 🎮 Running the Application

### GUI Application (Recommended)

```bash
python3 src/gui_main.py
```

Then:
1. Press `F5` to start the system
2. Cameras will automatically connect
3. View real-time monitoring in the interface

### Console Mode (Headless)

```bash
python3 run_sentinel.py
```

---

## 🎯 Key Features & Shortcuts

### GUI Navigation

| Action | Shortcut | Menu |
|--------|----------|------|
| Start System | `F5` | System → Start |
| Stop System | `F6` | System → Stop |
| Fullscreen | `F11` | View → Fullscreen |
| Settings | - | Tools → Settings |
| Quit | `Ctrl+Q` | File → Quit |

### Accessing New Features

#### 📊 Analytics Dashboard
- **Menu**: `Analytics → Analytics Dashboard`
- **Features**: Historical trips, safety trends, performance metrics
- **Data Location**: `data/trips/`

#### 🎬 Incident Review
- **Menu**: `Analytics → Incident Review`
- **Features**: Browse and replay recorded scenarios
- **Data Location**: `scenarios/`

#### 🌐 GPS Tracking
- **Location**: Advanced Features dock → GPS tab
- **Features**: Position, speed limits, violations
- **Note**: Enable in config or use simulation mode

#### 🔮 Interaction Prediction
- **Status**: Auto-enabled
- **Features**: Predicts pedestrian crossings, lane changes, merges
- **Viewing**: Main display shows warnings

---

## ⚙️ Configuration (Optional)

Edit `configs/default.yaml`:

### Camera Settings
```yaml
cameras:
  interior:
    device: 0  # Change camera index if needed
  front_left:
    device: 1
  front_right:
    device: 2
```

### Enable GPS
```yaml
features:
  gps:
    enabled: true  # Set to true
    simulation: true  # Use simulated GPS
```

### CPU/GPU Mode
```yaml
models:
  segmentation:
    device: "cuda"  # or "cpu"
```

---

## 🔧 Testing Without Hardware

### No Cameras?
The system will gracefully handle missing cameras and log warnings. Some features require at least one camera.

### No GPS?
```yaml
features:
  gps:
    enabled: true
    simulation: true  # Simulated GPS data
```

---

## 📊 GUI Layout

```
┌─────────────────────────────────────────────────────┐
│  Menu Bar: File | System | View | Tools | Analytics │
├───────────┬───────────────────────────┬─────────────┤
│           │                           │  Driver     │
│  Camera   │    Live Monitor           │  State      │
│  Viewer   │    - BEV Display          │  ────────── │
│  ──────── │    - Detection Overlay    │  Risk       │
│  Advanced │    - Alerts               │  Assessment │
│  Features │                           │  ────────── │
│  • Safety │                           │  Alerts     │
│  • Score  │                           │             │
│  • Trip   │                           │             │
│  • Road   │                           │             │
│  • Signs  │                           │             │
│  • GPS ⭐ │                           │             │
├───────────┴───────────────────────────┴─────────────┤
│  Performance Metrics: FPS | CPU | Memory | Latency  │
└─────────────────────────────────────────────────────┘
```

⭐ = New in this release

---

## 🐛 Common Issues

### Import Error: `No module named 'PyQt6'`
```bash
pip3 install -r requirements.txt
```

### Camera Not Found
```bash
# Check available cameras (Linux)
ls -l /dev/video*

# Update device number in configs/default.yaml
```

### Low FPS
- Enable GPU: Set `device: "cuda"` in config
- Reduce resolution in camera settings
- Disable non-essential features

### GPU Out of Memory
- Set `device: "cpu"` in config
- Or reduce `num_hypotheses` in trajectory prediction

---

## 📦 Directory Structure

```
Sentinel-Autonomy-Phase1/
├── configs/          # Configuration files
│   └── default.yaml  # Main config ⚙️
├── data/            # Runtime data
│   └── trips/       # Trip analytics 📊
├── scenarios/       # Recorded incidents 🎬
├── models/          # AI model weights
├── src/             # Source code
│   ├── gui_main.py  # GUI entry point 🎮
│   └── main.py      # Console entry point
├── run_sentinel.py  # Launcher script
└── RUNNING.md       # Detailed guide 📖
```

---

## 🚦 System Status Indicators

| Indicator | Status | Action |
|-----------|--------|--------|
| 🟢 Green | Normal | No action needed |
| 🟡 Yellow | Caution | Review warnings |
| 🟠 Orange | Warning | Check risk panel |
| 🔴 Red | Critical | Immediate attention |

---

## 📈 Performance Targets

| Metric | Target | Typical |
|--------|--------|---------|
| FPS | ≥30 | 30-60 |
| Latency (p95) | <100ms | 70-90ms |
| CPU Usage | ≤60% | 40-55% |
| GPU Memory | ≤8GB | 3-5GB |

---

## 🎓 Next Steps

1. **Run the System**: `python3 src/gui_main.py`
2. **Start Processing**: Press `F5`
3. **Explore Features**: Check all dock tabs
4. **View Analytics**: Go to Analytics menu
5. **Read Full Docs**: See `RUNNING.md` for detailed guide

---

## 📞 Need Help?

- **Full Documentation**: `RUNNING.md`
- **Changelog**: `CHANGELOG.md`
- **Data Formats**: `data/README.md`, `scenarios/README.md`
- **Configuration**: See inline comments in `configs/default.yaml`

---

**Happy Monitoring! 🚗💨**
