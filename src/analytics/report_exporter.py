"""
Report Exporter

Exports analytics data to various formats including CSV, PDF, and PNG.
"""

import numpy as np
import csv
from typing import Dict, List, Optional
from datetime import datetime
import logging
import os

logger = logging.getLogger(__name__)


class ReportExporter:
    """
    Exports analytics data to various formats.
    
    Capabilities:
    - Export data to CSV
    - Export reports to PDF
    - Export visualizations to PNG
    """
    
    def __init__(self, config: dict):
        """
        Initialize report exporter.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Output directory
        self.output_dir = config.get('analytics', {}).get('output_dir', 'reports/')
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info(f"ReportExporter initialized, output_dir: {self.output_dir}")
    
    def export_trips_csv(self, trip_summaries: List[Dict], filepath: Optional[str] = None) -> str:
        """
        Export trip summaries to CSV.
        
        Args:
            trip_summaries: List of trip summary dictionaries
            filepath: Optional output file path
        
        Returns:
            Path to exported file
        """
        if filepath is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = os.path.join(self.output_dir, f'trips_{timestamp}.csv')
        
        try:
            with open(filepath, 'w', newline='') as csvfile:
                fieldnames = [
                    'trip_id', 'start_time', 'end_time', 'duration_min',
                    'distance_km', 'avg_speed_kmh', 'max_speed_kmh',
                    'safety_score', 'info_alerts', 'warning_alerts',
                    'critical_alerts', 'high_risk_segments', 'driver_id'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for trip in trip_summaries:
                    row = {
                        'trip_id': trip.get('trip_id', ''),
                        'start_time': str(trip.get('start_time', '')),
                        'end_time': str(trip.get('end_time', '')),
                        'duration_min': f"{trip.get('duration', 0)/60:.2f}",
                        'distance_km': f"{trip.get('distance', 0)/1000:.2f}",
                        'avg_speed_kmh': f"{trip.get('avg_speed', 0)*3.6:.2f}",
                        'max_speed_kmh': f"{trip.get('max_speed', 0)*3.6:.2f}",
                        'safety_score': f"{trip.get('safety_score', 0):.2f}",
                        'info_alerts': trip.get('alert_counts', {}).get('info', 0),
                        'warning_alerts': trip.get('alert_counts', {}).get('warning', 0),
                        'critical_alerts': trip.get('alert_counts', {}).get('critical', 0),
                        'high_risk_segments': len(trip.get('high_risk_segments', [])),
                        'driver_id': trip.get('driver_id', '')
                    }
                    writer.writerow(row)
            
            logger.info(f"Trips exported to CSV: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to export trips to CSV: {e}")
            raise
    
    def export_driver_profile_csv(self, driver_profile: Dict, filepath: Optional[str] = None) -> str:
        """
        Export driver profile to CSV.
        
        Args:
            driver_profile: Driver profile dictionary
            filepath: Optional output file path
        
        Returns:
            Path to exported file
        """
        if filepath is None:
            driver_id = driver_profile.get('driver_id', 'unknown')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = os.path.join(self.output_dir, f'driver_{driver_id}_{timestamp}.csv')
        
        try:
            with open(filepath, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                # Header
                writer.writerow(['Driver Profile Export'])
                writer.writerow(['Generated:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
                writer.writerow([])
                
                # Basic info
                writer.writerow(['Basic Information'])
                writer.writerow(['Driver ID', driver_profile.get('driver_id', '')])
                writer.writerow(['Driving Style', driver_profile.get('driving_style', '')])
                writer.writerow(['Total Distance (km)', f"{driver_profile.get('total_distance', 0)/1000:.2f}"])
                writer.writerow(['Total Time (hours)', f"{driver_profile.get('total_time', 0)/3600:.2f}"])
                writer.writerow([])
                
                # Scores
                writer.writerow(['Scores'])
                writer.writerow(['Safety Score', f"{driver_profile.get('safety_score', 0):.2f}"])
                writer.writerow(['Attention Score', f"{driver_profile.get('attention_score', 0):.2f}"])
                writer.writerow(['Eco Score', f"{driver_profile.get('eco_score', 0):.2f}"])
                writer.writerow([])
                
                # Metrics
                writer.writerow(['Metrics'])
                metrics = driver_profile.get('metrics', {})
                for key, value in metrics.items():
                    if isinstance(value, dict):
                        writer.writerow([key, 'Mean', 'Std', 'Count'])
                        writer.writerow(['', 
                                       f"{value.get('mean', 0):.4f}",
                                       f"{value.get('std', 0):.4f}",
                                       value.get('count', 0)])
                    else:
                        writer.writerow([key, value])
            
            logger.info(f"Driver profile exported to CSV: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to export driver profile to CSV: {e}")
            raise
    
    def export_risk_heatmap_csv(self, heatmap: np.ndarray, filepath: Optional[str] = None) -> str:
        """
        Export risk heatmap data to CSV.
        
        Args:
            heatmap: Heatmap array
            filepath: Optional output file path
        
        Returns:
            Path to exported file
        """
        if filepath is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = os.path.join(self.output_dir, f'heatmap_{timestamp}.csv')
        
        try:
            np.savetxt(filepath, heatmap, delimiter=',', fmt='%.6f')
            logger.info(f"Heatmap exported to CSV: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to export heatmap to CSV: {e}")
            raise
    
    def export_visualization_png(self, image: np.ndarray, filepath: Optional[str] = None) -> str:
        """
        Export visualization to PNG.
        
        Args:
            image: Image array (H, W, 3) or (H, W)
            filepath: Optional output file path
        
        Returns:
            Path to exported file
        """
        if filepath is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = os.path.join(self.output_dir, f'visualization_{timestamp}.png')
        
        try:
            import cv2
            
            # Convert RGB to BGR if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            cv2.imwrite(filepath, image)
            logger.info(f"Visualization exported to PNG: {filepath}")
            return filepath
            
        except ImportError:
            logger.error("OpenCV not available for PNG export")
            raise
        except Exception as e:
            logger.error(f"Failed to export visualization to PNG: {e}")
            raise
    
    def export_summary_report_pdf(self,
                                  driver_profile: Dict,
                                  trip_summaries: List[Dict],
                                  filepath: Optional[str] = None) -> str:
        """
        Export summary report to PDF.
        
        Args:
            driver_profile: Driver profile dictionary
            trip_summaries: List of trip summaries
            filepath: Optional output file path
        
        Returns:
            Path to exported file
        """
        if filepath is None:
            driver_id = driver_profile.get('driver_id', 'unknown')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = os.path.join(self.output_dir, f'report_{driver_id}_{timestamp}.pdf')
        
        try:
            # Import behavior report generator
            from .behavior_report import BehaviorReportGenerator
            
            generator = BehaviorReportGenerator(self.config)
            success = generator.generate_pdf_report(driver_profile, trip_summaries, filepath)
            
            if success:
                logger.info(f"Summary report exported to PDF: {filepath}")
                return filepath
            else:
                raise Exception("PDF generation failed")
                
        except ImportError as e:
            logger.error(f"Required library not available for PDF export: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to export summary report to PDF: {e}")
            raise
    
    def export_summary_report_excel(self,
                                    driver_profile: Dict,
                                    trip_summaries: List[Dict],
                                    filepath: Optional[str] = None) -> str:
        """
        Export summary report to Excel.
        
        Args:
            driver_profile: Driver profile dictionary
            trip_summaries: List of trip summaries
            filepath: Optional output file path
        
        Returns:
            Path to exported file
        """
        if filepath is None:
            driver_id = driver_profile.get('driver_id', 'unknown')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = os.path.join(self.output_dir, f'report_{driver_id}_{timestamp}.xlsx')
        
        try:
            # Import behavior report generator
            from .behavior_report import BehaviorReportGenerator
            
            generator = BehaviorReportGenerator(self.config)
            success = generator.generate_excel_report(driver_profile, trip_summaries, filepath)
            
            if success:
                logger.info(f"Summary report exported to Excel: {filepath}")
                return filepath
            else:
                raise Exception("Excel generation failed")
                
        except ImportError as e:
            logger.error(f"Required library not available for Excel export: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to export summary report to Excel: {e}")
            raise
    
    def export_analytics_bundle(self,
                               driver_profile: Dict,
                               trip_summaries: List[Dict],
                               heatmap: Optional[np.ndarray] = None,
                               output_dir: Optional[str] = None) -> str:
        """
        Export complete analytics bundle with all formats.
        
        Args:
            driver_profile: Driver profile dictionary
            trip_summaries: List of trip summaries
            heatmap: Optional risk heatmap array
            output_dir: Optional output directory
        
        Returns:
            Path to output directory
        """
        if output_dir is None:
            driver_id = driver_profile.get('driver_id', 'unknown')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = os.path.join(self.output_dir, f'bundle_{driver_id}_{timestamp}')
        
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Export trips CSV
            trips_csv = os.path.join(output_dir, 'trips.csv')
            self.export_trips_csv(trip_summaries, trips_csv)
            
            # Export driver profile CSV
            profile_csv = os.path.join(output_dir, 'driver_profile.csv')
            self.export_driver_profile_csv(driver_profile, profile_csv)
            
            # Export PDF report
            try:
                report_pdf = os.path.join(output_dir, 'report.pdf')
                self.export_summary_report_pdf(driver_profile, trip_summaries, report_pdf)
            except Exception as e:
                logger.warning(f"Could not export PDF: {e}")
            
            # Export Excel report
            try:
                report_excel = os.path.join(output_dir, 'report.xlsx')
                self.export_summary_report_excel(driver_profile, trip_summaries, report_excel)
            except Exception as e:
                logger.warning(f"Could not export Excel: {e}")
            
            # Export heatmap if provided
            if heatmap is not None:
                heatmap_csv = os.path.join(output_dir, 'risk_heatmap.csv')
                self.export_risk_heatmap_csv(heatmap, heatmap_csv)
            
            logger.info(f"Analytics bundle exported to: {output_dir}")
            return output_dir
            
        except Exception as e:
            logger.error(f"Failed to export analytics bundle: {e}")
            raise
    
    def get_export_formats(self) -> List[str]:
        """
        Get list of available export formats.
        
        Returns:
            List of format names
        """
        formats = ['csv']
        
        try:
            import reportlab
            formats.append('pdf')
        except ImportError:
            pass
        
        try:
            import pandas
            formats.append('excel')
        except ImportError:
            pass
        
        try:
            import cv2
            formats.append('png')
        except ImportError:
            pass
        
        return formats
