"""
Driver Report Generator

Generates comprehensive driver behavior reports with scores and recommendations.
"""

import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime
from .style_classifier import DrivingStyle
import logging

logger = logging.getLogger(__name__)


class DriverReportGenerator:
    """
    Generates driver behavior reports.
    
    Calculates:
    - Safety score (0-100)
    - Attention score (0-100)
    - Eco-driving score (0-100)
    - Recommendations for improvement
    - Trend analysis over time
    """
    
    def __init__(self, config: dict):
        """
        Initialize report generator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Historical reports for trend analysis
        self.report_history: List[Dict] = []
        self.max_history = 50
        
        logger.info("DriverReportGenerator initialized")
    
    def generate_report(self,
                       metrics: Dict,
                       driving_style: DrivingStyle,
                       driver_id: str) -> Dict:
        """
        Generate comprehensive driver report.
        
        Args:
            metrics: Driver metrics from MetricsTracker
            driving_style: Classified driving style
            driver_id: Driver identifier
        
        Returns:
            Dictionary with complete report
        """
        # Calculate scores
        safety_score = self._calculate_safety_score(metrics, driving_style)
        attention_score = self._calculate_attention_score(metrics)
        eco_score = self._calculate_eco_score(metrics)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            metrics, driving_style, safety_score, attention_score, eco_score
        )
        
        # Analyze trends
        trends = self._analyze_trends(safety_score, attention_score, eco_score)
        
        # Create report
        report = {
            'driver_id': driver_id,
            'timestamp': datetime.now().isoformat(),
            'driving_style': driving_style.value,
            'scores': {
                'safety': safety_score,
                'attention': attention_score,
                'eco_driving': eco_score,
                'overall': (safety_score + attention_score + eco_score) / 3.0
            },
            'metrics_summary': {
                'session_duration': metrics.get('session_duration', 0.0),
                'total_distance': metrics.get('total_distance', 0.0),
                'reaction_time': metrics.get('reaction_time', {}).get('mean', 0.0),
                'following_distance': metrics.get('following_distance', {}).get('mean', 0.0),
                'lane_change_frequency': metrics.get('lane_change_frequency', 0.0),
                'near_miss_count': metrics.get('near_miss_count', 0),
                'risk_tolerance': metrics.get('risk_tolerance', 0.5)
            },
            'recommendations': recommendations,
            'trends': trends
        }
        
        # Add to history
        self.report_history.append(report)
        if len(self.report_history) > self.max_history:
            self.report_history.pop(0)
        
        logger.info(f"Report generated for {driver_id}: "
                   f"Safety={safety_score:.1f}, Attention={attention_score:.1f}, Eco={eco_score:.1f}")
        
        return report
    
    def _calculate_safety_score(self, metrics: Dict, driving_style: DrivingStyle) -> float:
        """
        Calculate safety score (0-100).
        
        Based on:
        - Near-miss events (fewer is better)
        - Risk tolerance (lower is better)
        - Following distance (appropriate for style)
        - Reaction time (faster is better, but not too fast)
        
        Args:
            metrics: Driver metrics
            driving_style: Driving style
        
        Returns:
            Safety score (0-100)
        """
        score = 100.0
        
        # Penalize near-miss events
        near_miss_count = metrics.get('near_miss_count', 0)
        session_hours = metrics.get('session_duration', 0.0) / 3600.0
        if session_hours > 0:
            near_miss_rate = near_miss_count / session_hours
            score -= min(30, near_miss_rate * 10)  # Up to -30 points
        
        # Penalize high risk tolerance
        risk_tolerance = metrics.get('risk_tolerance', 0.5)
        if risk_tolerance > 0.6:
            score -= (risk_tolerance - 0.6) * 50  # Up to -20 points
        
        # Check following distance appropriateness
        following_distance = metrics.get('following_distance', {}).get('mean', 25.0)
        if following_distance < 15.0:
            score -= (15.0 - following_distance) * 2  # Penalty for too close
        
        # Check reaction time
        reaction_time = metrics.get('reaction_time', {}).get('mean', 1.0)
        if reaction_time > 2.0:
            score -= (reaction_time - 2.0) * 10  # Penalty for slow reactions
        
        # Bonus for cautious driving
        if driving_style == DrivingStyle.CAUTIOUS:
            score += 5
        elif driving_style == DrivingStyle.AGGRESSIVE:
            score -= 5
        
        return float(np.clip(score, 0, 100))
    
    def _calculate_attention_score(self, metrics: Dict) -> float:
        """
        Calculate attention score (0-100).
        
        Based on:
        - Reaction time consistency
        - Alert response rate
        
        Args:
            metrics: Driver metrics
        
        Returns:
            Attention score (0-100)
        """
        score = 100.0
        
        # Check reaction time
        reaction_stats = metrics.get('reaction_time', {})
        reaction_mean = reaction_stats.get('mean', 1.0)
        reaction_std = reaction_stats.get('std', 0.0)
        reaction_count = reaction_stats.get('count', 0)
        
        if reaction_count > 0:
            # Penalize slow reactions
            if reaction_mean > 1.5:
                score -= (reaction_mean - 1.5) * 20
            
            # Penalize inconsistent reactions
            if reaction_std > 0.5:
                score -= (reaction_std - 0.5) * 15
        else:
            # No reaction data - assume moderate score
            score = 70.0
        
        return float(np.clip(score, 0, 100))
    
    def _calculate_eco_score(self, metrics: Dict) -> float:
        """
        Calculate eco-driving score (0-100).
        
        Based on:
        - Speed variance (smooth driving is better)
        - Lane change frequency (fewer is better)
        - Average speed (moderate is better)
        
        Args:
            metrics: Driver metrics
        
        Returns:
            Eco-driving score (0-100)
        """
        score = 100.0
        
        # Penalize high speed variance
        speed_stats = metrics.get('speed_profile', {})
        speed_std = speed_stats.get('std', 0.0)
        if speed_std > 3.0:
            score -= (speed_std - 3.0) * 5  # Up to -15 points
        
        # Penalize frequent lane changes
        lane_change_freq = metrics.get('lane_change_frequency', 0.0)
        if lane_change_freq > 6.0:
            score -= (lane_change_freq - 6.0) * 3  # Up to -12 points
        
        # Check average speed (penalize very high speeds)
        avg_speed = speed_stats.get('mean', 0.0)
        if avg_speed > 30.0:  # > 108 km/h
            score -= (avg_speed - 30.0) * 2
        
        return float(np.clip(score, 0, 100))
    
    def _generate_recommendations(self,
                                 metrics: Dict,
                                 driving_style: DrivingStyle,
                                 safety_score: float,
                                 attention_score: float,
                                 eco_score: float) -> List[str]:
        """
        Generate personalized recommendations.
        
        Args:
            metrics: Driver metrics
            driving_style: Driving style
            safety_score: Safety score
            attention_score: Attention score
            eco_score: Eco-driving score
        
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Safety recommendations
        if safety_score < 70:
            near_miss_count = metrics.get('near_miss_count', 0)
            if near_miss_count > 0:
                recommendations.append(
                    f"Reduce near-miss events by maintaining safer following distances. "
                    f"You had {near_miss_count} near-miss events this session."
                )
            
            following_distance = metrics.get('following_distance', {}).get('mean', 0.0)
            if following_distance < 20.0:
                recommendations.append(
                    f"Increase following distance. Current average: {following_distance:.1f}m. "
                    f"Recommended: 25-30m for safer driving."
                )
            
            risk_tolerance = metrics.get('risk_tolerance', 0.5)
            if risk_tolerance > 0.6:
                recommendations.append(
                    "Consider adopting a more cautious driving approach to reduce risk exposure."
                )
        
        # Attention recommendations
        if attention_score < 70:
            reaction_time = metrics.get('reaction_time', {}).get('mean', 0.0)
            if reaction_time > 1.5:
                recommendations.append(
                    f"Improve reaction time by staying more alert. "
                    f"Current average: {reaction_time:.2f}s. Target: <1.5s."
                )
            
            reaction_std = metrics.get('reaction_time', {}).get('std', 0.0)
            if reaction_std > 0.5:
                recommendations.append(
                    "Work on maintaining consistent attention levels throughout your drive."
                )
        
        # Eco-driving recommendations
        if eco_score < 70:
            speed_std = metrics.get('speed_profile', {}).get('std', 0.0)
            if speed_std > 4.0:
                recommendations.append(
                    "Maintain more consistent speeds for better fuel efficiency and smoother driving."
                )
            
            lane_change_freq = metrics.get('lane_change_frequency', 0.0)
            if lane_change_freq > 8.0:
                recommendations.append(
                    f"Reduce unnecessary lane changes. Current rate: {lane_change_freq:.1f} per hour."
                )
        
        # Style-specific recommendations
        if driving_style == DrivingStyle.AGGRESSIVE:
            recommendations.append(
                "Your driving style is classified as aggressive. "
                "Consider adopting a more relaxed approach for improved safety and fuel efficiency."
            )
        
        # Positive reinforcement
        if safety_score >= 85 and attention_score >= 85:
            recommendations.append(
                "Excellent driving! Keep up the safe and attentive behavior."
            )
        
        return recommendations
    
    def _analyze_trends(self,
                       current_safety: float,
                       current_attention: float,
                       current_eco: float) -> Dict[str, str]:
        """
        Analyze score trends over time.
        
        Args:
            current_safety: Current safety score
            current_attention: Current attention score
            current_eco: Current eco score
        
        Returns:
            Dictionary with trend descriptions
        """
        if len(self.report_history) < 3:
            return {
                'safety': 'insufficient_data',
                'attention': 'insufficient_data',
                'eco_driving': 'insufficient_data'
            }
        
        # Get recent scores
        recent_safety = [r['scores']['safety'] for r in self.report_history[-5:]]
        recent_attention = [r['scores']['attention'] for r in self.report_history[-5:]]
        recent_eco = [r['scores']['eco_driving'] for r in self.report_history[-5:]]
        
        # Calculate trends
        safety_trend = self._calculate_trend(recent_safety, current_safety)
        attention_trend = self._calculate_trend(recent_attention, current_attention)
        eco_trend = self._calculate_trend(recent_eco, current_eco)
        
        return {
            'safety': safety_trend,
            'attention': attention_trend,
            'eco_driving': eco_trend
        }
    
    def _calculate_trend(self, history: List[float], current: float) -> str:
        """
        Calculate trend direction.
        
        Args:
            history: Historical scores
            current: Current score
        
        Returns:
            Trend description: 'improving', 'stable', or 'declining'
        """
        if len(history) < 2:
            return 'stable'
        
        # Calculate average of history vs current
        avg_history = np.mean(history[:-1])
        
        if current > avg_history + 5:
            return 'improving'
        elif current < avg_history - 5:
            return 'declining'
        else:
            return 'stable'
    
    def get_report_history(self, driver_id: str, limit: int = 10) -> List[Dict]:
        """
        Get historical reports for a driver.
        
        Args:
            driver_id: Driver identifier
            limit: Maximum number of reports to return
        
        Returns:
            List of historical reports
        """
        driver_reports = [
            r for r in self.report_history 
            if r.get('driver_id') == driver_id
        ]
        return driver_reports[-limit:]
    
    def export_report_text(self, report: Dict) -> str:
        """
        Export report as formatted text.
        
        Args:
            report: Report dictionary
        
        Returns:
            Formatted text report
        """
        lines = []
        lines.append("=" * 60)
        lines.append("DRIVER BEHAVIOR REPORT")
        lines.append("=" * 60)
        lines.append(f"Driver ID: {report['driver_id']}")
        lines.append(f"Generated: {report['timestamp']}")
        lines.append(f"Driving Style: {report['driving_style'].upper()}")
        lines.append("")
        
        lines.append("SCORES:")
        lines.append(f"  Safety Score:      {report['scores']['safety']:.1f}/100")
        lines.append(f"  Attention Score:   {report['scores']['attention']:.1f}/100")
        lines.append(f"  Eco-Driving Score: {report['scores']['eco_driving']:.1f}/100")
        lines.append(f"  Overall Score:     {report['scores']['overall']:.1f}/100")
        lines.append("")
        
        lines.append("METRICS SUMMARY:")
        metrics = report['metrics_summary']
        lines.append(f"  Session Duration:    {metrics['session_duration']/60:.1f} minutes")
        lines.append(f"  Total Distance:      {metrics['total_distance']/1000:.2f} km")
        lines.append(f"  Reaction Time:       {metrics['reaction_time']:.2f} seconds")
        lines.append(f"  Following Distance:  {metrics['following_distance']:.1f} meters")
        lines.append(f"  Lane Changes/Hour:   {metrics['lane_change_frequency']:.1f}")
        lines.append(f"  Near-Miss Events:    {metrics['near_miss_count']}")
        lines.append("")
        
        if report['recommendations']:
            lines.append("RECOMMENDATIONS:")
            for i, rec in enumerate(report['recommendations'], 1):
                lines.append(f"  {i}. {rec}")
            lines.append("")
        
        lines.append("TRENDS:")
        trends = report['trends']
        lines.append(f"  Safety:      {trends['safety']}")
        lines.append(f"  Attention:   {trends['attention']}")
        lines.append(f"  Eco-Driving: {trends['eco_driving']}")
        lines.append("=" * 60)
        
        return "\n".join(lines)
