# Contextual Intelligence Engine

The Contextual Intelligence Engine is the core risk assessment module of SENTINEL that correlates environmental hazards with driver awareness state to generate context-aware safety interventions.

## Overview

Unlike traditional ADAS systems that only monitor the environment, the Contextual Intelligence Engine understands both what's happening around the vehicle AND whether the driver is aware of those threats. This enables more effective safety interventions by accounting for driver cognitive state.

## Architecture

The engine consists of several integrated components:

### 1. Scene Graph Builder (`scene_graph.py`)
- Builds spatial representation of all detected objects
- Calculates relationships and proximity between objects
- Creates grid-based spatial map for efficient queries

### 2. Attention Mapper (`attention.py`)
- Maps driver gaze to 8 spatial zones around vehicle
- Zone boundaries:
  - Front: -30° to 30°
  - Front-left: 30° to 75°
  - Left: 75° to 105°
  - Rear-left: 105° to 150°
  - Rear: 150° to -150° (wraps around)
  - Rear-right: -150° to -105°
  - Right: -105° to -75°
  - Front-right: -75° to -30°
- Determines which zones driver is attending to

### 3. TTC Calculator (`ttc.py`)
- Calculates Time-To-Collision for each detected object
- Uses constant velocity model
- Includes configurable safety margin (default: 1.5m)

### 4. Trajectory Predictor (`trajectory.py`)
- Predicts object trajectories up to 3 seconds ahead
- Uses linear prediction with 0.1s time steps
- Generates 31 waypoints per trajectory
- Calculates trajectory conflict scores

### 5. Risk Calculator (`risk.py`)
- Calculates base risk from environmental factors:
  - TTC (weight: 0.4)
  - Trajectory conflict (weight: 0.3)
  - Object vulnerability (weight: 0.2)
  - Relative speed (weight: 0.1)
- Calculates contextual risk considering driver state:
  - Awareness penalty: 2.0x if driver not looking, 1.0x if looking
  - Capacity factor: 2.0 - (readiness/100)
  - Contextual risk = base_risk × awareness_penalty × capacity_factor
- Categorizes urgency levels: low, medium, high, critical

### 6. Risk Prioritizer (`prioritization.py`)
- Sorts risks by contextual score
- Selects top 3 threats
- Detects attention-risk mismatches

### 7. Contextual Intelligence Engine (`engine.py`)
- Main class that integrates all components
- Implements `IContextualIntelligence` interface
- Optimized for <10ms processing time
- Outputs `RiskAssessment` dataclass

## Usage

```python
from src.intelligence import ContextualIntelligence
from src.core.config import ConfigManager

# Load configuration
config = ConfigManager('configs/default.yaml').config

# Initialize engine
intelligence = ContextualIntelligence(config)

# Assess risks
risk_assessment = intelligence.assess(
    detections=detections_3d,      # List[Detection3D]
    driver_state=driver_state,      # DriverState
    bev_seg=bev_segmentation        # SegmentationOutput
)

# Access results
print(f"Top risks: {len(risk_assessment.top_risks)}")
for risk in risk_assessment.top_risks:
    print(f"  {risk.hazard.type}: {risk.contextual_score:.2f} ({risk.urgency})")
    print(f"  Driver aware: {risk.driver_aware}")
```

## Configuration

Risk assessment parameters are configured in `configs/default.yaml`:

```yaml
risk_assessment:
  ttc_calculation:
    method: "constant_velocity"
    safety_margin: 1.5  # meters
  
  trajectory_prediction:
    horizon: 3.0  # seconds
    dt: 0.1       # time step
    method: "linear"
  
  zone_mapping:
    num_zones: 8
  
  base_risk_weights:
    ttc: 0.4
    trajectory_conflict: 0.3
    vulnerability: 0.2
    relative_speed: 0.1
  
  thresholds:
    hazard_detection: 0.3
    intervention: 0.7
    critical: 0.9
```

## Output

The engine returns a `RiskAssessment` object containing:

- **scene_graph**: Spatial representation of all objects and relationships
- **hazards**: List of identified hazards with TTC, trajectory, zone, and base risk
- **attention_map**: Driver attention zones and gaze information
- **top_risks**: Top 3 prioritized risks with contextual scores and urgency levels

## Performance

- Target processing time: <10ms
- Typical processing time: 2-5ms for 5-10 objects
- Scales linearly with number of detected objects

## Testing

Run tests:
```bash
python3 -m pytest tests/test_intelligence.py -v
```

Run example:
```bash
python3 examples/intelligence_example.py
```

## Key Features

1. **Contextual Risk Assessment**: Combines environmental hazards with driver state
2. **Attention-Risk Mismatch Detection**: Identifies when driver is not looking at critical threats
3. **Adaptive Risk Scoring**: Accounts for driver readiness and cognitive load
4. **Multi-Factor Risk Calculation**: Considers TTC, trajectory, vulnerability, and speed
5. **Real-Time Performance**: Optimized for <10ms processing time

## Requirements

See `requirements.md` in the spec for detailed requirements (6.1-6.7).
