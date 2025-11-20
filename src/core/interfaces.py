"""Abstract base classes defining module interfaces."""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Tuple
import numpy as np

from .data_structures import (
    CameraBundle, BEVOutput, SegmentationOutput,
    Detection2D, Detection3D, DriverState,
    RiskAssessment, Alert
)


class ICameraManager(ABC):
    """Camera management interface."""
    
    @abstractmethod
    def start(self) -> None:
        """Start camera capture threads."""
        pass
    
    @abstractmethod
    def stop(self) -> None:
        """Stop camera capture."""
        pass
    
    @abstractmethod
    def get_frame_bundle(self) -> Optional[CameraBundle]:
        """Get synchronized frame bundle."""
        pass
    
    @abstractmethod
    def is_healthy(self) -> bool:
        """Check if all cameras are operational."""
        pass


class IBEVGenerator(ABC):
    """BEV generation interface."""
    
    @abstractmethod
    def generate(self, frames: List[np.ndarray]) -> BEVOutput:
        """Transform camera views to BEV."""
        pass


class ISemanticSegmentor(ABC):
    """Semantic segmentation interface."""
    
    @abstractmethod
    def segment(self, bev_image: np.ndarray) -> SegmentationOutput:
        """Segment BEV image."""
        pass


class IObjectDetector(ABC):
    """Object detection interface."""
    
    @abstractmethod
    def detect(self, frames: Dict[int, np.ndarray]) -> Tuple[Dict[int, List[Detection2D]], List[Detection3D]]:
        """Detect and fuse objects from multiple views."""
        pass


class IDMS(ABC):
    """Driver monitoring interface."""
    
    @abstractmethod
    def analyze(self, frame: np.ndarray) -> DriverState:
        """Analyze driver state."""
        pass


class IContextualIntelligence(ABC):
    """Contextual intelligence interface."""
    
    @abstractmethod
    def assess(self, detections: List[Detection3D], driver_state: DriverState, 
               bev_seg: SegmentationOutput) -> RiskAssessment:
        """Assess contextual risks."""
        pass


class IAlertSystem(ABC):
    """Alert system interface."""
    
    @abstractmethod
    def process(self, risks: RiskAssessment, driver: DriverState) -> List[Alert]:
        """Generate and dispatch alerts."""
        pass
