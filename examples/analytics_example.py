"""
Analytics Module Example

Demonstrates trip analytics, risk heatmaps, and report generation.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import time
from datetime import datetime
from src.analytics import (TripAnalytics, RiskHeatmap, ReportExporter, 
                           BehaviorReportGenerator)
from src.core.data_structures import Alert


def simulate_trip():
    """Simulate a trip with analytics tracking."""
    
    # Configuration
    config = {
        'analytics': {
            'segment_duration': 30.0,
            'high_risk_threshold': 0.7,
            'heatmap': {
                'grid_size': 2.0,
                'max_range': 100.0,
                'decay_factor': 0.95
            },
            'output_dir': 'reports/'
        }
    }
    
    print("=" * 60)
    print("SENTINEL Analytics Example")
    print("=" * 60)
    
    # Initialize analytics components
    print("\n1. Initializing analytics components...")
    trip_analytics = TripAnalytics(config)
    risk_heatmap = RiskHeatmap(config)
    exporter = ReportExporter(config)
    report_gen = BehaviorReportGenerator(config)
    
    # Start trip
    print("\n2. Starting trip...")
    driver_id = "driver_001"
    trip_id = trip_analytics.start_trip(driver_id=driver_id)
    print(f"   Trip ID: {trip_id}")
    
    # Simulate trip data
    print("\n3. Simulating trip (60 seconds)...")
    start_time = time.time()
    position = [0.0, 0.0]
    
    for i in range(60):
        timestamp = start_time + i
        
        # Simulate movement
        position[0] += np.random.uniform(5, 15)  # Move forward
        position[1] += np.random.uniform(-2, 2)  # Slight lateral movement
        
        # Simulate speed (20-30 m/s)
        speed = np.random.uniform(20, 30)
        
        # Simulate risk score (higher risk occasionally)
        if i % 15 == 0:
            risk_score = np.random.uniform(0.7, 0.9)  # High risk
        else:
            risk_score = np.random.uniform(0.1, 0.5)  # Normal risk
        
        # Simulate alerts
        alerts = []
        if risk_score > 0.7:
            alert = Alert(
                timestamp=timestamp,
                urgency='critical' if risk_score > 0.85 else 'warning',
                modalities=['visual', 'audio'],
                message=f"High risk detected: {risk_score:.2f}",
                hazard_id=i,
                dismissed=False
            )
            alerts.append(alert)
        
        # Update analytics
        trip_analytics.update(
            timestamp=timestamp,
            position=tuple(position),
            speed=speed,
            risk_score=risk_score,
            alerts=alerts
        )
        
        # Update heatmap
        risk_heatmap.add_risk_point(tuple(position), risk_score, radius=5.0)
        
        if (i + 1) % 15 == 0:
            print(f"   Progress: {i+1}/60 seconds")
    
    # End trip
    print("\n4. Ending trip and generating summary...")
    summary = trip_analytics.end_trip()
    
    print(f"\n   Trip Summary:")
    print(f"   - Duration: {summary.duration:.1f} seconds")
    print(f"   - Distance: {summary.distance:.1f} meters")
    print(f"   - Avg Speed: {summary.avg_speed*3.6:.1f} km/h")
    print(f"   - Safety Score: {summary.safety_score:.1f}/100")
    print(f"   - Alerts: {sum(summary.alert_counts.values())}")
    print(f"     - Info: {summary.alert_counts.get('info', 0)}")
    print(f"     - Warning: {summary.alert_counts.get('warning', 0)}")
    print(f"     - Critical: {summary.alert_counts.get('critical', 0)}")
    print(f"   - High-risk segments: {len(summary.high_risk_segments)}")
    
    # Heatmap statistics
    print("\n5. Risk heatmap statistics...")
    heatmap_stats = risk_heatmap.get_statistics()
    print(f"   - Total cells with risk: {heatmap_stats['total_cells']}")
    print(f"   - Max risk: {heatmap_stats['max_risk']:.3f}")
    print(f"   - Mean risk: {heatmap_stats['mean_risk']:.3f}")
    print(f"   - High-risk cells: {heatmap_stats['high_risk_cells']}")
    
    # Get high-risk locations
    high_risk_locs = risk_heatmap.get_high_risk_locations(threshold=0.7)
    print(f"   - High-risk locations: {len(high_risk_locs)}")
    
    # Export data
    print("\n6. Exporting data...")
    
    # Export trip CSV
    try:
        csv_path = exporter.export_trips_csv([summary.__dict__])
        print(f"   ✓ Trip CSV exported: {csv_path}")
    except Exception as e:
        print(f"   ✗ Trip CSV export failed: {e}")
    
    # Export heatmap image
    try:
        heatmap_array = risk_heatmap.get_heatmap(normalize=True)
        heatmap_path = 'reports/risk_heatmap.png'
        risk_heatmap.export_heatmap_image(heatmap_path, colormap='hot')
        print(f"   ✓ Heatmap image exported: {heatmap_path}")
    except Exception as e:
        print(f"   ✗ Heatmap export failed: {e}")
    
    # Create mock driver profile
    driver_profile = {
        'driver_id': driver_id,
        'driving_style': 'normal',
        'total_distance': summary.distance,
        'total_time': summary.duration,
        'safety_score': summary.safety_score,
        'attention_score': 85.0,
        'eco_score': 78.0,
        'metrics': {
            'reaction_time': {'mean': 1.2, 'std': 0.3, 'count': 10},
            'following_distance': {'mean': 25.0, 'std': 5.0, 'count': 50},
            'lane_change_frequency': 4.5,
            'near_miss_count': 1,
            'risk_tolerance': 0.4
        }
    }
    
    # Export driver profile CSV
    try:
        profile_path = exporter.export_driver_profile_csv(driver_profile)
        print(f"   ✓ Driver profile CSV exported: {profile_path}")
    except Exception as e:
        print(f"   ✗ Driver profile CSV export failed: {e}")
    
    # Generate PDF report (if reportlab available)
    print("\n7. Generating reports...")
    try:
        pdf_path = 'reports/driver_report.pdf'
        success = report_gen.generate_pdf_report(
            driver_profile=driver_profile,
            trip_summaries=[summary.__dict__],
            output_path=pdf_path
        )
        if success:
            print(f"   ✓ PDF report generated: {pdf_path}")
    except Exception as e:
        print(f"   ✗ PDF report generation failed: {e}")
    
    # Generate Excel report (if pandas available)
    try:
        excel_path = 'reports/driver_report.xlsx'
        success = report_gen.generate_excel_report(
            driver_profile=driver_profile,
            trip_summaries=[summary.__dict__],
            output_path=excel_path
        )
        if success:
            print(f"   ✓ Excel report generated: {excel_path}")
    except Exception as e:
        print(f"   ✗ Excel report generation failed: {e}")
    
    # Export complete bundle
    try:
        bundle_dir = exporter.export_analytics_bundle(
            driver_profile=driver_profile,
            trip_summaries=[summary.__dict__],
            heatmap=heatmap_array
        )
        print(f"   ✓ Complete analytics bundle exported: {bundle_dir}")
    except Exception as e:
        print(f"   ✗ Bundle export failed: {e}")
    
    # Display available export formats
    print("\n8. Available export formats:")
    formats = exporter.get_export_formats()
    for fmt in formats:
        print(f"   - {fmt}")
    
    print("\n" + "=" * 60)
    print("Analytics example completed!")
    print("=" * 60)


if __name__ == "__main__":
    simulate_trip()
