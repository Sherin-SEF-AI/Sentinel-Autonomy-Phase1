"""Risk prioritization and attention-risk mismatch detection."""

import logging
from typing import List, Dict, Any

from src.core.data_structures import Risk


class RiskPrioritizer:
    """Prioritizes risks and detects attention-risk mismatches."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize risk prioritizer.
        
        Args:
            config: Optional configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.top_n = 3  # Number of top risks to return
    
    def prioritize(self, risks: List[Risk]) -> List[Risk]:
        """
        Sort risks by contextual score and select top threats.
        
        Args:
            risks: List of Risk objects
            
        Returns:
            List of top N risks sorted by contextual score (highest first)
        """
        if not risks:
            return []
        
        # Sort by contextual score (descending)
        sorted_risks = sorted(
            risks,
            key=lambda r: r.contextual_score,
            reverse=True
        )
        
        # Return top N
        top_risks = sorted_risks[:self.top_n]
        
        self.logger.debug(
            f"Prioritized {len(risks)} risks, returning top {len(top_risks)}"
        )
        
        return top_risks
    
    def detect_attention_mismatches(
        self,
        risks: List[Risk],
        attention_map: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Detect attention-risk mismatches where driver is not looking at hazards.
        
        Args:
            risks: List of Risk objects
            attention_map: Attention map with attended zones
            
        Returns:
            List of mismatch dictionaries with hazard and zone information
        """
        mismatches = []
        
        if not attention_map.get('attention_valid', False):
            # Cannot determine mismatches without valid attention data
            return mismatches
        
        attended_zones = attention_map.get('attended_zones', [])
        
        for risk in risks:
            hazard_zone = risk.hazard.zone
            
            # Check if driver is not looking at this hazard's zone
            if hazard_zone not in attended_zones:
                mismatch = {
                    'hazard_id': risk.hazard.object_id,
                    'hazard_type': risk.hazard.type,
                    'hazard_zone': hazard_zone,
                    'attended_zones': attended_zones,
                    'contextual_score': risk.contextual_score,
                    'urgency': risk.urgency
                }
                mismatches.append(mismatch)
        
        self.logger.debug(
            f"Detected {len(mismatches)} attention-risk mismatches"
        )
        
        return mismatches
    
    def filter_by_threshold(
        self,
        risks: List[Risk],
        threshold: float = 0.3
    ) -> List[Risk]:
        """
        Filter risks by minimum contextual score threshold.
        
        Args:
            risks: List of Risk objects
            threshold: Minimum contextual score
            
        Returns:
            Filtered list of risks
        """
        filtered = [r for r in risks if r.contextual_score >= threshold]
        
        self.logger.debug(
            f"Filtered {len(risks)} risks with threshold {threshold}, "
            f"remaining: {len(filtered)}"
        )
        
        return filtered
    
    def get_highest_risk(self, risks: List[Risk]) -> Risk:
        """
        Get the highest priority risk.
        
        Args:
            risks: List of Risk objects
            
        Returns:
            Risk with highest contextual score, or None if empty
        """
        if not risks:
            return None
        
        return max(risks, key=lambda r: r.contextual_score)
