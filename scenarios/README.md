# SENTINEL Scenarios Directory

This directory stores recorded safety scenarios captured by the scenario recording system.

## Structure

```
scenarios/
├── scenario_YYYYMMDD_HHMMSS_<trigger>/  # Individual scenario recordings
│   ├── metadata.json                    # Scenario metadata
│   ├── frame_0000.jpg                   # Recorded frames
│   ├── frame_0001.jpg
│   └── ...
└── README.md                            # This file
```

## Scenario Recording

Scenarios are automatically recorded when:
- High-risk situations are detected (collision warnings, critical events)
- Manual recording is triggered by the user
- Specific safety events occur (hard braking, lane departures, etc.)

## Metadata Format

Each scenario directory contains a `metadata.json` file:

```json
{
  "scenario_id": "20241120_143022_collision_warning",
  "timestamp": 1700494222.5,
  "trigger": "collision_warning",
  "severity": "high",
  "duration": 10.0,
  "frame_count": 300,
  "fps": 30,
  "risk_level": "critical",
  "driver_attention": "distracted",
  "events": [
    {
      "type": "forward_collision_warning",
      "timestamp": 1700494223.0,
      "ttc": 1.2
    }
  ],
  "outcome": "collision_avoided"
}
```

## Viewing Scenarios

Use the Incident Review widget to browse and replay recorded scenarios:
1. Open SENTINEL GUI
2. Go to **Analytics → Incident Review**
3. Select a scenario from the list
4. Use playback controls to review

## Storage Management

- Scenarios can consume significant disk space (frames are stored as JPG)
- Consider implementing retention policies based on available storage
- Critical scenarios should be backed up or archived
- Low-severity scenarios can be periodically cleaned up

## Frame Format

- Frames are stored as JPEG images (quality: 85)
- Resolution matches the camera feed resolution
- Filename format: `frame_NNNN.jpg` (zero-padded, 4 digits)
- Frames are stored in chronological order

## Privacy and Security

**IMPORTANT**: Recorded scenarios may contain:
- Video footage of drivers (interior camera)
- Video footage of surrounding vehicles and pedestrians
- GPS location data (if GPS tracking is enabled)
- Sensitive driving behavior data

Ensure compliance with:
- Privacy regulations (GDPR, CCPA, etc.)
- Data retention policies
- Consent requirements for video recording
- Secure storage and access controls

## Export and Sharing

Scenarios can be exported for:
- Insurance claims
- Driver training
- Safety analysis
- System debugging

When sharing scenarios:
- Anonymize personal information if required
- Obtain necessary consents
- Follow data protection guidelines
