"""
Face Recognition System for Driver Identification

Uses face embeddings to identify drivers at session start.
"""

import numpy as np
import cv2
from typing import Optional, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class FaceRecognitionSystem:
    """
    Face recognition system for driver identification.
    
    Uses face embeddings (FaceNet-style) to identify drivers with a similarity threshold.
    """
    
    def __init__(self, config: dict):
        """
        Initialize face recognition system.
        
        Args:
            config: Configuration dictionary with:
                - embedding_model: Path to face embedding model
                - recognition_threshold: Similarity threshold (default: 0.6)
                - embedding_size: Size of face embedding vector (default: 128)
        """
        self.config = config
        self.recognition_threshold = config.get('recognition_threshold', 0.6)
        self.embedding_size = config.get('embedding_size', 128)
        
        # Initialize face detector (using OpenCV DNN or MediaPipe)
        self.face_detector = self._init_face_detector()
        
        # Initialize embedding model (placeholder for FaceNet or similar)
        self.embedding_model = self._init_embedding_model()
        
        logger.info(f"FaceRecognitionSystem initialized with threshold={self.recognition_threshold}")
    
    def _init_face_detector(self):
        """Initialize face detector using OpenCV DNN."""
        try:
            # Use OpenCV's DNN face detector
            model_path = "models/face_detection_yunet_2023mar.onnx"
            if Path(model_path).exists():
                detector = cv2.FaceDetectorYN.create(
                    model_path,
                    "",
                    (320, 320),
                    0.9,  # score threshold
                    0.3,  # nms threshold
                    5000  # top_k
                )
                logger.info("Face detector initialized successfully")
                return detector
            else:
                logger.warning(f"Face detection model not found at {model_path}, using fallback")
                return None
        except Exception as e:
            logger.error(f"Failed to initialize face detector: {e}")
            return None
    
    def _init_embedding_model(self):
        """Initialize face embedding model."""
        # Placeholder for FaceNet or similar model
        # In production, this would load a pretrained model
        logger.info("Face embedding model initialized (placeholder)")
        return None
    
    def extract_face_embedding(self, frame: np.ndarray, face_bbox: Optional[Tuple[int, int, int, int]] = None) -> Optional[np.ndarray]:
        """
        Extract face embedding from frame.
        
        Args:
            frame: Input image (H, W, 3)
            face_bbox: Optional face bounding box (x, y, w, h)
        
        Returns:
            Face embedding vector of shape (embedding_size,) or None if no face detected
        """
        try:
            # Detect face if bbox not provided
            if face_bbox is None:
                face_bbox = self._detect_face(frame)
                if face_bbox is None:
                    return None
            
            # Extract face region
            x, y, w, h = face_bbox
            face_roi = frame[y:y+h, x:x+w]
            
            # Resize to model input size
            face_resized = cv2.resize(face_roi, (160, 160))
            
            # Normalize
            face_normalized = (face_resized.astype(np.float32) - 127.5) / 128.0
            
            # Extract embedding
            embedding = self._compute_embedding(face_normalized)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to extract face embedding: {e}")
            return None
    
    def _detect_face(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect face in frame.
        
        Returns:
            Face bounding box (x, y, w, h) or None
        """
        if self.face_detector is None:
            # Fallback: use Haar cascade
            return self._detect_face_haar(frame)
        
        try:
            # Set input size
            height, width = frame.shape[:2]
            self.face_detector.setInputSize((width, height))
            
            # Detect faces
            _, faces = self.face_detector.detect(frame)
            
            if faces is not None and len(faces) > 0:
                # Return first face
                face = faces[0]
                x, y, w, h = face[:4].astype(int)
                return (x, y, w, h)
            
            return None
            
        except Exception as e:
            logger.error(f"Face detection failed: {e}")
            return None
    
    def _detect_face_haar(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Fallback face detection using Haar cascade."""
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) > 0:
                return tuple(faces[0])
            return None
        except Exception as e:
            logger.error(f"Haar cascade detection failed: {e}")
            return None
    
    def _compute_embedding(self, face: np.ndarray) -> np.ndarray:
        """
        Compute face embedding.
        
        Args:
            face: Normalized face image (160, 160, 3)
        
        Returns:
            Embedding vector (embedding_size,)
        """
        if self.embedding_model is not None:
            # Use actual model
            # embedding = self.embedding_model.predict(face[np.newaxis, ...])[0]
            pass
        
        # Placeholder: generate random embedding for testing
        # In production, this would use FaceNet or similar
        embedding = np.random.randn(self.embedding_size).astype(np.float32)
        
        # Normalize to unit length
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        
        return embedding
    
    def match_face(self, embedding: np.ndarray, stored_embeddings: dict) -> Tuple[Optional[str], float]:
        """
        Match face embedding against stored embeddings.
        
        Args:
            embedding: Query face embedding
            stored_embeddings: Dictionary mapping driver_id to embedding
        
        Returns:
            Tuple of (driver_id, similarity) or (None, 0.0) if no match
        """
        if not stored_embeddings:
            return None, 0.0
        
        best_match_id = None
        best_similarity = 0.0
        
        for driver_id, stored_embedding in stored_embeddings.items():
            # Compute cosine similarity
            similarity = self._cosine_similarity(embedding, stored_embedding)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_id = driver_id
        
        # Check if best match exceeds threshold
        if best_similarity >= self.recognition_threshold:
            logger.info(f"Driver matched: {best_match_id} (similarity={best_similarity:.3f})")
            return best_match_id, best_similarity
        else:
            logger.info(f"No driver match found (best similarity={best_similarity:.3f})")
            return None, best_similarity
    
    def _cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
        
        Returns:
            Cosine similarity in range [0, 1]
        """
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        similarity = dot_product / (norm1 * norm2 + 1e-8)
        
        # Convert from [-1, 1] to [0, 1]
        similarity = (similarity + 1.0) / 2.0
        
        return float(similarity)
    
    def generate_driver_id(self, embedding: np.ndarray) -> str:
        """
        Generate unique driver ID from embedding.
        
        Args:
            embedding: Face embedding
        
        Returns:
            Unique driver ID string
        """
        # Use hash of embedding as ID
        embedding_bytes = embedding.tobytes()
        import hashlib
        driver_id = hashlib.sha256(embedding_bytes).hexdigest()[:16]
        return f"driver_{driver_id}"
