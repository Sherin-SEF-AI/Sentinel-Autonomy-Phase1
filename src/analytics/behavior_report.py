"""
Driver Behavior Report Generator

Generates comprehensive PDF and Excel reports with charts,
safety scores, trends, and recommendations.
"""

import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import logging
import io

logger = logging.getLogger(__name__)


class BehaviorReportGenerator:
    """
    Generates driver behavior reports in PDF and Excel formats.
    
    Capabilities:
    - Generate PDF reports with charts
    - Include safety scores and trends
    - Add personalized recommendations
    - Export to Excel format
    """
    
    def __init__(self, config: dict):
        """
        Initialize behavior report generator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Check for optional dependencies
        self.has_reportlab = self._check_reportlab()
        self.has_matplotlib = self._check_matplotlib()
        self.has_pandas = self._check_pandas()
        
        logger.info(f"BehaviorReportGenerator initialized "
                   f"(PDF: {self.has_reportlab}, Charts: {self.has_matplotlib}, "
                   f"Excel: {self.has_pandas})")
    
    def _check_reportlab(self) -> bool:
        """Check if reportlab is available."""
        try:
            import reportlab
            return True
        except ImportError:
            return False
    
    def _check_matplotlib(self) -> bool:
        """Check if matplotlib is available."""
        try:
            import matplotlib
            return True
        except ImportError:
            return False
    
    def _check_pandas(self) -> bool:
        """Check if pandas is available."""
        try:
            import pandas
            return True
        except ImportError:
            return False
    
    def generate_pdf_report(self,
                           driver_profile: Dict,
                           trip_summaries: List[Dict],
                           output_path: str) -> bool:
        """
        Generate PDF report with charts and analysis.
        
        Args:
            driver_profile: Driver profile dictionary
            trip_summaries: List of trip summary dictionaries
            output_path: Output PDF file path
        
        Returns:
            True if successful, False otherwise
        """
        if not self.has_reportlab:
            logger.error("reportlab not available for PDF generation")
            return False
        
        try:
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
            from reportlab.lib import colors
            from reportlab.lib.enums import TA_CENTER, TA_LEFT
            
            # Create PDF document
            doc = SimpleDocTemplate(output_path, pagesize=letter)
            story = []
            styles = getSampleStyleSheet()
            
            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                textColor=colors.HexColor('#1f77b4'),
                spaceAfter=30,
                alignment=TA_CENTER
            )
            story.append(Paragraph("Driver Behavior Report", title_style))
            story.append(Spacer(1, 0.2*inch))
            
            # Driver Information
            story.append(Paragraph("Driver Information", styles['Heading2']))
            driver_data = [
                ['Driver ID:', driver_profile.get('driver_id', 'Unknown')],
                ['Report Date:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                ['Driving Style:', driver_profile.get('driving_style', 'Unknown').upper()],
                ['Total Distance:', f"{driver_profile.get('total_distance', 0)/1000:.2f} km"],
                ['Total Time:', f"{driver_profile.get('total_time', 0)/3600:.2f} hours"]
            ]
            driver_table = Table(driver_data, colWidths=[2*inch, 4*inch])
            driver_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(driver_table)
            story.append(Spacer(1, 0.3*inch))
            
            # Safety Scores
            story.append(Paragraph("Safety Scores", styles['Heading2']))
            metrics = driver_profile.get('metrics', {})
            scores_data = [
                ['Metric', 'Score', 'Status'],
                ['Safety Score', f"{driver_profile.get('safety_score', 0):.1f}/100", 
                 self._get_score_status(driver_profile.get('safety_score', 0))],
                ['Attention Score', f"{driver_profile.get('attention_score', 0):.1f}/100",
                 self._get_score_status(driver_profile.get('attention_score', 0))],
                ['Eco Score', f"{driver_profile.get('eco_score', 0):.1f}/100",
                 self._get_score_status(driver_profile.get('eco_score', 0))]
            ]
            scores_table = Table(scores_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
            scores_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(scores_table)
            story.append(Spacer(1, 0.3*inch))
            
            # Generate charts if matplotlib available
            if self.has_matplotlib:
                chart_path = self._generate_score_chart(driver_profile, trip_summaries)
                if chart_path:
                    story.append(Paragraph("Performance Trends", styles['Heading2']))
                    story.append(Image(chart_path, width=6*inch, height=3*inch))
                    story.append(Spacer(1, 0.3*inch))
            
            # Trip Statistics
            if trip_summaries:
                story.append(Paragraph("Recent Trip Statistics", styles['Heading2']))
                trip_data = [['Date', 'Duration', 'Distance', 'Safety Score', 'Alerts']]
                for trip in trip_summaries[-10:]:  # Last 10 trips
                    trip_data.append([
                        trip.get('start_time', datetime.now()).strftime('%Y-%m-%d'),
                        f"{trip.get('duration', 0)/60:.1f} min",
                        f"{trip.get('distance', 0)/1000:.2f} km",
                        f"{trip.get('safety_score', 0):.1f}",
                        str(sum(trip.get('alert_counts', {}).values()))
                    ])
                trip_table = Table(trip_data, colWidths=[1.2*inch, 1*inch, 1*inch, 1.2*inch, 0.8*inch])
                trip_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 9),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(trip_table)
                story.append(Spacer(1, 0.3*inch))
            
            # Recommendations
            recommendations = self._generate_recommendations(driver_profile, trip_summaries)
            if recommendations:
                story.append(Paragraph("Recommendations", styles['Heading2']))
                for i, rec in enumerate(recommendations, 1):
                    story.append(Paragraph(f"{i}. {rec}", styles['Normal']))
                    story.append(Spacer(1, 0.1*inch))
            
            # Build PDF
            doc.build(story)
            logger.info(f"PDF report generated: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to generate PDF report: {e}")
            return False
    
    def generate_excel_report(self,
                             driver_profile: Dict,
                             trip_summaries: List[Dict],
                             output_path: str) -> bool:
        """
        Generate Excel report with multiple sheets.
        
        Args:
            driver_profile: Driver profile dictionary
            trip_summaries: List of trip summary dictionaries
            output_path: Output Excel file path
        
        Returns:
            True if successful, False otherwise
        """
        if not self.has_pandas:
            logger.error("pandas not available for Excel generation")
            return False
        
        try:
            import pandas as pd
            
            # Create Excel writer
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # Driver Summary sheet
                driver_data = {
                    'Metric': ['Driver ID', 'Driving Style', 'Total Distance (km)', 
                              'Total Time (hours)', 'Safety Score', 'Attention Score', 'Eco Score'],
                    'Value': [
                        driver_profile.get('driver_id', 'Unknown'),
                        driver_profile.get('driving_style', 'Unknown'),
                        f"{driver_profile.get('total_distance', 0)/1000:.2f}",
                        f"{driver_profile.get('total_time', 0)/3600:.2f}",
                        f"{driver_profile.get('safety_score', 0):.1f}",
                        f"{driver_profile.get('attention_score', 0):.1f}",
                        f"{driver_profile.get('eco_score', 0):.1f}"
                    ]
                }
                df_driver = pd.DataFrame(driver_data)
                df_driver.to_excel(writer, sheet_name='Driver Summary', index=False)
                
                # Trip History sheet
                if trip_summaries:
                    trip_data = []
                    for trip in trip_summaries:
                        trip_data.append({
                            'Trip ID': trip.get('trip_id', ''),
                            'Date': trip.get('start_time', datetime.now()).strftime('%Y-%m-%d %H:%M'),
                            'Duration (min)': f"{trip.get('duration', 0)/60:.1f}",
                            'Distance (km)': f"{trip.get('distance', 0)/1000:.2f}",
                            'Avg Speed (km/h)': f"{trip.get('avg_speed', 0)*3.6:.1f}",
                            'Safety Score': f"{trip.get('safety_score', 0):.1f}",
                            'Info Alerts': trip.get('alert_counts', {}).get('info', 0),
                            'Warning Alerts': trip.get('alert_counts', {}).get('warning', 0),
                            'Critical Alerts': trip.get('alert_counts', {}).get('critical', 0)
                        })
                    df_trips = pd.DataFrame(trip_data)
                    df_trips.to_excel(writer, sheet_name='Trip History', index=False)
                
                # Metrics sheet
                metrics = driver_profile.get('metrics', {})
                metrics_data = {
                    'Metric': [],
                    'Mean': [],
                    'Std': [],
                    'Count': []
                }
                for key, value in metrics.items():
                    if isinstance(value, dict):
                        metrics_data['Metric'].append(key)
                        metrics_data['Mean'].append(value.get('mean', 0))
                        metrics_data['Std'].append(value.get('std', 0))
                        metrics_data['Count'].append(value.get('count', 0))
                
                if metrics_data['Metric']:
                    df_metrics = pd.DataFrame(metrics_data)
                    df_metrics.to_excel(writer, sheet_name='Metrics', index=False)
            
            logger.info(f"Excel report generated: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to generate Excel report: {e}")
            return False
    
    def _generate_score_chart(self,
                             driver_profile: Dict,
                             trip_summaries: List[Dict]) -> Optional[str]:
        """Generate score trend chart."""
        if not self.has_matplotlib:
            return None
        
        try:
            import matplotlib.pyplot as plt
            import tempfile
            
            # Extract safety scores from trips
            if not trip_summaries:
                return None
            
            dates = [t.get('start_time', datetime.now()) for t in trip_summaries[-20:]]
            safety_scores = [t.get('safety_score', 0) for t in trip_summaries[-20:]]
            
            # Create figure
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(dates, safety_scores, marker='o', linewidth=2, markersize=6)
            ax.set_xlabel('Date')
            ax.set_ylabel('Safety Score')
            ax.set_title('Safety Score Trend')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 100)
            
            # Rotate x-axis labels
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # Save to temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            plt.savefig(temp_file.name, dpi=150, bbox_inches='tight')
            plt.close()
            
            return temp_file.name
            
        except Exception as e:
            logger.error(f"Failed to generate chart: {e}")
            return None
    
    def _get_score_status(self, score: float) -> str:
        """Get status label for score."""
        if score >= 85:
            return 'Excellent'
        elif score >= 70:
            return 'Good'
        elif score >= 50:
            return 'Fair'
        else:
            return 'Needs Improvement'
    
    def _generate_recommendations(self,
                                 driver_profile: Dict,
                                 trip_summaries: List[Dict]) -> List[str]:
        """Generate personalized recommendations."""
        recommendations = []
        
        safety_score = driver_profile.get('safety_score', 0)
        attention_score = driver_profile.get('attention_score', 0)
        eco_score = driver_profile.get('eco_score', 0)
        driving_style = driver_profile.get('driving_style', 'normal')
        
        # Safety recommendations
        if safety_score < 70:
            recommendations.append(
                "Focus on improving safety by maintaining safer following distances "
                "and reducing near-miss events."
            )
        
        # Attention recommendations
        if attention_score < 70:
            recommendations.append(
                "Work on maintaining consistent attention levels. "
                "Take breaks during long drives to stay alert."
            )
        
        # Eco-driving recommendations
        if eco_score < 70:
            recommendations.append(
                "Improve fuel efficiency by maintaining more consistent speeds "
                "and reducing unnecessary lane changes."
            )
        
        # Style-specific recommendations
        if driving_style == 'aggressive':
            recommendations.append(
                "Consider adopting a more relaxed driving approach for improved "
                "safety and fuel efficiency."
            )
        
        # Positive reinforcement
        if safety_score >= 85 and attention_score >= 85:
            recommendations.append(
                "Excellent driving performance! Continue maintaining your safe "
                "and attentive behavior."
            )
        
        return recommendations
