"""Advanced trajectory prediction with LSTM and physics-based models."""

import logging
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from dataclasses import dataclass
import time

# Initialize logger
logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
    logger.info("PyTorch available: LSTM trajectory prediction enabled")
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available: LSTM trajectory prediction disabled")

from src.core.data_structures import Detection3D


@dataclass
class Trajectory:
    """Predicted trajectory with uncertainty."""
    points: List[Tuple[float, float, float]]  # (x, y, z) positions
    timestamps: List[float]  # Time for each point
    uncertainty: List[np.ndarray]  # Covariance matrices (3x3)
    confidence: float  # Overall confidence score
    model: str  # 'lstm', 'cv', 'ca', 'ct'


class LSTMTrajectoryModel(nn.Module):
    """LSTM-based trajectory prediction model."""
    
    def __init__(
        self,
        input_size: int = 6,  # x, y, z, vx, vy, vz
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 3,  # x, y, z
        dropout: float = 0.2
    ):
        """
        Initialize LSTM trajectory model.
        
        Args:
            input_size: Size of input features
            hidden_size: Size of hidden state
            num_layers: Number of LSTM layers
            output_size: Size of output (x, y, z)
            dropout: Dropout probability
        """
        super().__init__()
        
        logger.debug(
            f"Initializing LSTM trajectory model: input_size={input_size}, "
            f"hidden_size={hidden_size}, num_layers={num_layers}, "
            f"output_size={output_size}, dropout={dropout}"
        )
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
        
        # Uncertainty estimation layer
        self.uncertainty_fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size)  # Log variance
        )
        
        logger.info(
            f"LSTM trajectory model initialized: parameters={sum(p.numel() for p in self.parameters())}"
        )
    
    def forward(self, x: torch.Tensor, hidden: Optional[Tuple] = None):
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, seq_len, input_size)
            hidden: Hidden state tuple (h, c)
            
        Returns:
            Tuple of (output, uncertainty, hidden_state)
        """
        start_time = time.time()
        
        # LSTM forward
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Use last time step output
        last_output = lstm_out[:, -1, :]
        
        # Predict position
        position = self.fc(last_output)
        
        # Predict uncertainty (log variance)
        log_var = self.uncertainty_fc(last_output)
        
        duration = (time.time() - start_time) * 1000
        logger.debug(f"LSTM forward pass completed: duration={duration:.2f}ms, batch_size={x.shape[0]}")
        
        return position, log_var, hidden
    
    def predict_sequence(
        self,
        history: torch.Tensor,
        num_steps: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict future trajectory sequence.
        
        Args:
            history: Historical trajectory (batch, seq_len, input_size)
            num_steps: Number of future steps to predict
            
        Returns:
            Tuple of (predicted_positions, uncertainties)
        """
        start_time = time.time()
        logger.debug(
            f"Predicting trajectory sequence: batch_size={history.shape[0]}, "
            f"history_len={history.shape[1]}, num_steps={num_steps}"
        )
        
        self.eval()
        with torch.no_grad():
            batch_size = history.shape[0]
            
            # Initialize outputs
            predictions = []
            uncertainties = []
            
            # Get initial hidden state from history
            _, _, hidden = self.forward(history)
            
            # Last known state
            current_input = history[:, -1:, :]
            
            # Predict future steps
            for step in range(num_steps):
                pos, log_var, hidden = self.forward(current_input, hidden)
                
                predictions.append(pos)
                uncertainties.append(torch.exp(log_var))  # Convert log var to variance
                
                # Update input for next step
                # Assume constant velocity for velocity components
                velocity = current_input[:, -1, 3:6]
                next_input = torch.cat([pos, velocity], dim=1).unsqueeze(1)
                current_input = next_input
            
            # Stack predictions
            predictions = torch.stack(predictions, dim=1)  # (batch, num_steps, 3)
            uncertainties = torch.stack(uncertainties, dim=1)  # (batch, num_steps, 3)
            
            duration = (time.time() - start_time) * 1000
            logger.debug(
                f"Trajectory sequence prediction completed: duration={duration:.2f}ms, "
                f"batch_size={batch_size}, num_steps={num_steps}"
            )
            
            return predictions, uncertainties


def train_lstm_model(
    model: LSTMTrajectoryModel,
    train_data: List[Tuple[np.ndarray, np.ndarray]],
    num_epochs: int = 100,
    learning_rate: float = 0.001,
    device: str = 'cuda'
) -> LSTMTrajectoryModel:
    """
    Train LSTM trajectory model.
    
    Args:
        model: LSTM model to train
        train_data: List of (history, future) trajectory pairs
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device to train on
        
    Returns:
        Trained model
    """
    if not TORCH_AVAILABLE:
        logger.error("Training failed: PyTorch not available")
        raise RuntimeError("PyTorch not available for training")
    
    logger.info(
        f"Training LSTM model started: num_epochs={num_epochs}, "
        f"learning_rate={learning_rate}, device={device}, "
        f"train_samples={len(train_data)}"
    )
    
    training_start = time.time()
    
    model = model.to(device)
    model.train()
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Loss function (negative log likelihood with uncertainty)
    def nll_loss(pred, target, log_var):
        """Negative log likelihood loss."""
        precision = torch.exp(-log_var)
        loss = 0.5 * (precision * (pred - target) ** 2 + log_var)
        return loss.mean()
    
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        total_loss = 0.0
        
        for batch_idx, (history, future) in enumerate(train_data):
            # Convert to tensors
            history_tensor = torch.FloatTensor(history).unsqueeze(0).to(device)
            future_tensor = torch.FloatTensor(future).unsqueeze(0).to(device)
            
            # Forward pass
            optimizer.zero_grad()
            
            # Predict sequence
            predictions, log_vars = model.predict_sequence(
                history_tensor,
                future_tensor.shape[1]
            )
            
            # Calculate loss
            loss = nll_loss(predictions, future_tensor, log_vars)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_data)
        epoch_duration = time.time() - epoch_start
        
        # Track best loss
        if avg_loss < best_loss:
            best_loss = avg_loss
            logger.debug(f"New best loss achieved: {best_loss:.4f}")
        
        if (epoch + 1) % 10 == 0:
            logger.info(
                f"Epoch {epoch + 1}/{num_epochs} completed: "
                f"loss={avg_loss:.4f}, best_loss={best_loss:.4f}, "
                f"duration={epoch_duration:.2f}s"
            )
    
    training_duration = time.time() - training_start
    logger.info(
        f"Training completed: total_duration={training_duration:.2f}s, "
        f"final_loss={avg_loss:.4f}, best_loss={best_loss:.4f}"
    )
    
    return model


def save_lstm_model(model: LSTMTrajectoryModel, path: str):
    """Save LSTM model to file."""
    if not TORCH_AVAILABLE:
        logger.error("Model save failed: PyTorch not available")
        raise RuntimeError("PyTorch not available")
    
    try:
        logger.info(f"Saving LSTM model to: {path}")
        torch.save(model.state_dict(), path)
        logger.info(f"LSTM model saved successfully: {path}")
    except Exception as e:
        logger.error(f"Failed to save LSTM model: {e}", exc_info=True)
        raise


def load_lstm_model(path: str, **model_kwargs) -> LSTMTrajectoryModel:
    """Load LSTM model from file."""
    if not TORCH_AVAILABLE:
        logger.error("Model load failed: PyTorch not available")
        raise RuntimeError("PyTorch not available")
    
    try:
        logger.info(f"Loading LSTM model from: {path}")
        load_start = time.time()
        
        model = LSTMTrajectoryModel(**model_kwargs)
        model.load_state_dict(torch.load(path))
        model.eval()
        
        load_duration = (time.time() - load_start) * 1000
        logger.info(
            f"LSTM model loaded successfully: path={path}, "
            f"duration={load_duration:.2f}ms, "
            f"parameters={sum(p.numel() for p in model.parameters())}"
        )
        
        return model
    except Exception as e:
        logger.error(f"Failed to load LSTM model from {path}: {e}", exc_info=True)
        raise



class PhysicsBasedPredictor:
    """Physics-based trajectory prediction models."""
    
    def __init__(self, dt: float = 0.1):
        """
        Initialize physics-based predictor.
        
        Args:
            dt: Time step in seconds
        """
        self.dt = dt
        self.logger = logging.getLogger(__name__)
    
    def predict_constant_velocity(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        num_steps: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict trajectory using constant velocity model.
        
        Args:
            position: Current position (x, y, z)
            velocity: Current velocity (vx, vy, vz)
            num_steps: Number of future steps
            
        Returns:
            Tuple of (positions, uncertainties)
        """
        positions = []
        uncertainties = []
        
        current_pos = position.copy()
        
        for step in range(num_steps):
            # Update position
            current_pos = current_pos + velocity * self.dt
            positions.append(current_pos.copy())
            
            # Uncertainty grows linearly with time
            # Assume 0.5 m/s velocity uncertainty
            velocity_uncertainty = 0.5
            position_uncertainty = velocity_uncertainty * (step + 1) * self.dt
            
            # Covariance matrix (diagonal, isotropic uncertainty)
            cov = np.eye(3) * (position_uncertainty ** 2)
            uncertainties.append(cov)
        
        return np.array(positions), uncertainties
    
    def predict_constant_acceleration(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        acceleration: np.ndarray,
        num_steps: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict trajectory using constant acceleration model.
        
        Args:
            position: Current position (x, y, z)
            velocity: Current velocity (vx, vy, vz)
            acceleration: Current acceleration (ax, ay, az)
            num_steps: Number of future steps
            
        Returns:
            Tuple of (positions, uncertainties)
        """
        positions = []
        uncertainties = []
        
        current_pos = position.copy()
        current_vel = velocity.copy()
        
        for step in range(num_steps):
            # Update velocity and position
            current_vel = current_vel + acceleration * self.dt
            current_pos = current_pos + current_vel * self.dt + 0.5 * acceleration * (self.dt ** 2)
            
            positions.append(current_pos.copy())
            
            # Uncertainty grows quadratically with time
            accel_uncertainty = 1.0  # m/s^2
            t = (step + 1) * self.dt
            position_uncertainty = 0.5 * accel_uncertainty * (t ** 2)
            
            cov = np.eye(3) * (position_uncertainty ** 2)
            uncertainties.append(cov)
        
        return np.array(positions), uncertainties
    
    def predict_constant_turn_rate(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        yaw: float,
        yaw_rate: float,
        num_steps: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict trajectory using constant turn rate model.
        
        Args:
            position: Current position (x, y, z)
            velocity: Current velocity magnitude and direction
            yaw: Current yaw angle (radians)
            yaw_rate: Yaw rate (radians/second)
            num_steps: Number of future steps
            
        Returns:
            Tuple of (positions, uncertainties)
        """
        positions = []
        uncertainties = []
        
        current_pos = position.copy()
        current_yaw = yaw
        speed = np.linalg.norm(velocity[:2])  # 2D speed
        
        for step in range(num_steps):
            # Update yaw
            current_yaw = current_yaw + yaw_rate * self.dt
            
            # Update position based on current heading
            dx = speed * np.cos(current_yaw) * self.dt
            dy = speed * np.sin(current_yaw) * self.dt
            dz = velocity[2] * self.dt  # Vertical velocity unchanged
            
            current_pos = current_pos + np.array([dx, dy, dz])
            positions.append(current_pos.copy())
            
            # Uncertainty grows with time and turn rate
            t = (step + 1) * self.dt
            turn_uncertainty = abs(yaw_rate) * 0.1  # Proportional to turn rate
            position_uncertainty = (0.5 + turn_uncertainty) * t
            
            cov = np.eye(3) * (position_uncertainty ** 2)
            uncertainties.append(cov)
        
        return np.array(positions), uncertainties
    
    def select_model(
        self,
        detection: Detection3D,
        history: Optional[List[Detection3D]] = None
    ) -> str:
        """
        Select appropriate physics model based on object type and motion history.
        
        Args:
            detection: Current detection
            history: Historical detections for this object
            
        Returns:
            Model name: 'cv', 'ca', or 'ct'
        """
        # Default to constant velocity
        model = 'cv'
        
        # Check object type
        if detection.class_name in ['vehicle', 'cyclist']:
            # Vehicles and cyclists can turn
            if history and len(history) >= 3:
                # Estimate if turning based on velocity direction changes
                velocities = [np.array(d.velocity[:2]) for d in history[-3:]]
                
                # Calculate angle changes
                angles = []
                for i in range(len(velocities) - 1):
                    v1 = velocities[i]
                    v2 = velocities[i + 1]
                    
                    if np.linalg.norm(v1) > 0.1 and np.linalg.norm(v2) > 0.1:
                        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                        cos_angle = np.clip(cos_angle, -1.0, 1.0)
                        angle_change = np.arccos(cos_angle)
                        angles.append(angle_change)
                
                # If significant turning detected
                if angles and np.mean(angles) > 0.1:  # ~5.7 degrees
                    model = 'ct'
                
                # Check for acceleration/deceleration
                speeds = [np.linalg.norm(v) for v in velocities]
                speed_changes = [speeds[i+1] - speeds[i] for i in range(len(speeds) - 1)]
                
                if speed_changes and abs(np.mean(speed_changes)) > 0.5:  # m/s change
                    model = 'ca'
        
        elif detection.class_name == 'pedestrian':
            # Pedestrians typically have more erratic motion, use CV
            model = 'cv'
        
        return model



class AdvancedTrajectoryPredictor:
    """Advanced trajectory predictor with LSTM and physics models."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize advanced trajectory predictor.
        
        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.enabled = config.get('enabled', True)
        self.horizon = config.get('horizon', 5.0)  # seconds
        self.dt = config.get('dt', 0.1)
        self.num_steps = int(self.horizon / self.dt)
        self.num_hypotheses = config.get('num_hypotheses', 3)
        self.use_lstm = config.get('use_lstm', True) and TORCH_AVAILABLE
        self.uncertainty_estimation = config.get('uncertainty_estimation', True)
        
        # Physics-based predictor
        self.physics_predictor = PhysicsBasedPredictor(dt=self.dt)
        
        # LSTM model
        self.lstm_model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu' if TORCH_AVAILABLE else None
        
        if self.use_lstm:
            lstm_model_path = config.get('lstm_model', 'models/trajectory_lstm.pth')
            try:
                if TORCH_AVAILABLE:
                    self.lstm_model = load_lstm_model(lstm_model_path)
                    self.lstm_model = self.lstm_model.to(self.device)
                    self.lstm_model.eval()
                    self.logger.info(f"Loaded LSTM model from {lstm_model_path}")
            except Exception as e:
                self.logger.warning(f"Failed to load LSTM model: {e}, using physics models only")
                self.use_lstm = False
        
        # Motion history buffer (track_id -> list of detections)
        self.motion_history: Dict[int, List[Detection3D]] = {}
        self.max_history_length = 30  # Keep last 30 frames
        
        self.logger.info(
            f"Advanced Trajectory Predictor initialized: "
            f"horizon={self.horizon}s, dt={self.dt}s, "
            f"hypotheses={self.num_hypotheses}, lstm={self.use_lstm}"
        )
    
    def update_history(self, detections: List[Detection3D]):
        """
        Update motion history with new detections.
        
        Args:
            detections: List of current detections
        """
        # Update history for each detection
        for detection in detections:
            track_id = detection.track_id
            
            if track_id not in self.motion_history:
                self.motion_history[track_id] = []
            
            self.motion_history[track_id].append(detection)
            
            # Keep only recent history
            if len(self.motion_history[track_id]) > self.max_history_length:
                self.motion_history[track_id].pop(0)
        
        # Clean up old tracks
        current_track_ids = {d.track_id for d in detections}
        old_tracks = [tid for tid in self.motion_history.keys() if tid not in current_track_ids]
        for tid in old_tracks:
            del self.motion_history[tid]
    
    def extract_motion_features(self, detection: Detection3D) -> np.ndarray:
        """
        Extract motion features from detection.
        
        Args:
            detection: Detection3D object
            
        Returns:
            Feature vector (x, y, z, vx, vy, vz)
        """
        x, y, z, w, h, l, theta = detection.bbox_3d
        vx, vy, vz = detection.velocity
        
        return np.array([x, y, z, vx, vy, vz], dtype=np.float32)
    
    def extract_scene_context(
        self,
        detection: Detection3D,
        all_detections: List[Detection3D]
    ) -> Dict[str, Any]:
        """
        Extract scene context features.
        
        Args:
            detection: Target detection
            all_detections: All detections in scene
            
        Returns:
            Context dictionary
        """
        context = {
            'num_nearby_objects': 0,
            'closest_distance': float('inf'),
            'object_density': 0.0
        }
        
        x, y, z, _, _, _, _ = detection.bbox_3d
        position = np.array([x, y, z])
        
        # Find nearby objects
        nearby_threshold = 10.0  # meters
        nearby_objects = []
        
        for other in all_detections:
            if other.track_id == detection.track_id:
                continue
            
            ox, oy, oz, _, _, _, _ = other.bbox_3d
            other_pos = np.array([ox, oy, oz])
            
            distance = np.linalg.norm(position - other_pos)
            
            if distance < nearby_threshold:
                nearby_objects.append(other)
                context['closest_distance'] = min(context['closest_distance'], distance)
        
        context['num_nearby_objects'] = len(nearby_objects)
        context['object_density'] = len(nearby_objects) / (np.pi * nearby_threshold ** 2)
        
        return context
    
    def predict_lstm(
        self,
        detection: Detection3D,
        history: List[Detection3D]
    ) -> Optional[Trajectory]:
        """
        Predict trajectory using LSTM model.
        
        Args:
            detection: Current detection
            history: Historical detections
            
        Returns:
            Predicted trajectory or None if prediction fails
        """
        if not self.use_lstm or self.lstm_model is None:
            return None
        
        # Need at least 10 frames of history
        if len(history) < 10:
            return None
        
        try:
            # Extract features from history
            history_features = []
            for det in history[-10:]:  # Use last 10 frames
                features = self.extract_motion_features(det)
                history_features.append(features)
            
            history_tensor = torch.FloatTensor(history_features).unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                predictions, uncertainties = self.lstm_model.predict_sequence(
                    history_tensor,
                    self.num_steps
                )
            
            # Convert to numpy
            predictions = predictions.squeeze(0).cpu().numpy()
            uncertainties = uncertainties.squeeze(0).cpu().numpy()
            
            # Build trajectory
            points = [(p[0], p[1], p[2]) for p in predictions]
            timestamps = [i * self.dt for i in range(len(points))]
            
            # Build covariance matrices
            covariances = []
            for unc in uncertainties:
                cov = np.diag(unc)  # Diagonal covariance
                covariances.append(cov)
            
            # Calculate confidence based on uncertainty
            avg_uncertainty = np.mean(uncertainties)
            confidence = 1.0 / (1.0 + avg_uncertainty)
            
            return Trajectory(
                points=points,
                timestamps=timestamps,
                uncertainty=covariances,
                confidence=float(confidence),
                model='lstm'
            )
        
        except Exception as e:
            self.logger.warning(f"LSTM prediction failed: {e}")
            return None
    
    def predict_physics(
        self,
        detection: Detection3D,
        history: Optional[List[Detection3D]] = None
    ) -> List[Trajectory]:
        """
        Predict trajectories using physics-based models.
        
        Args:
            detection: Current detection
            history: Historical detections
            
        Returns:
            List of predicted trajectories
        """
        trajectories = []
        
        x, y, z, w, h, l, theta = detection.bbox_3d
        position = np.array([x, y, z])
        velocity = np.array(detection.velocity)
        
        # Select appropriate model
        model_type = self.physics_predictor.select_model(detection, history)
        
        # Constant velocity model
        if model_type == 'cv' or True:  # Always include CV
            positions, uncertainties = self.physics_predictor.predict_constant_velocity(
                position, velocity, self.num_steps
            )
            
            points = [(p[0], p[1], p[2]) for p in positions]
            timestamps = [i * self.dt for i in range(len(points))]
            
            trajectories.append(Trajectory(
                points=points,
                timestamps=timestamps,
                uncertainty=uncertainties,
                confidence=0.7,
                model='cv'
            ))
        
        # Constant acceleration model (if history available)
        if model_type == 'ca' and history and len(history) >= 2:
            # Estimate acceleration from velocity change
            prev_velocity = np.array(history[-1].velocity)
            acceleration = (velocity - prev_velocity) / self.dt
            
            positions, uncertainties = self.physics_predictor.predict_constant_acceleration(
                position, velocity, acceleration, self.num_steps
            )
            
            points = [(p[0], p[1], p[2]) for p in positions]
            timestamps = [i * self.dt for i in range(len(points))]
            
            trajectories.append(Trajectory(
                points=points,
                timestamps=timestamps,
                uncertainty=uncertainties,
                confidence=0.6,
                model='ca'
            ))
        
        # Constant turn rate model (if turning detected)
        if model_type == 'ct' and history and len(history) >= 2:
            # Estimate yaw and yaw rate
            current_yaw = theta
            prev_theta = history[-1].bbox_3d[6]
            yaw_rate = (current_yaw - prev_theta) / self.dt
            
            positions, uncertainties = self.physics_predictor.predict_constant_turn_rate(
                position, velocity, current_yaw, yaw_rate, self.num_steps
            )
            
            points = [(p[0], p[1], p[2]) for p in positions]
            timestamps = [i * self.dt for i in range(len(points))]
            
            trajectories.append(Trajectory(
                points=points,
                timestamps=timestamps,
                uncertainty=uncertainties,
                confidence=0.65,
                model='ct'
            ))
        
        return trajectories
    
    def predict(
        self,
        detection: Detection3D,
        all_detections: List[Detection3D]
    ) -> List[Trajectory]:
        """
        Generate multiple trajectory hypotheses for a detection.
        
        Args:
            detection: Detection to predict trajectory for
            all_detections: All detections in scene (for context)
            
        Returns:
            List of trajectory hypotheses (up to num_hypotheses)
        """
        if not self.enabled:
            return []
        
        trajectories = []
        
        # Get motion history
        history = self.motion_history.get(detection.track_id, [])
        
        # Extract scene context
        context = self.extract_scene_context(detection, all_detections)
        
        # Try LSTM prediction
        if self.use_lstm:
            lstm_traj = self.predict_lstm(detection, history)
            if lstm_traj is not None:
                trajectories.append(lstm_traj)
        
        # Physics-based predictions
        physics_trajs = self.predict_physics(detection, history)
        trajectories.extend(physics_trajs)
        
        # Sort by confidence and return top hypotheses
        trajectories.sort(key=lambda t: t.confidence, reverse=True)
        
        return trajectories[:self.num_hypotheses]
    
    def predict_all(
        self,
        detections: List[Detection3D]
    ) -> Dict[int, List[Trajectory]]:
        """
        Predict trajectories for all detections.
        
        Args:
            detections: List of detections
            
        Returns:
            Dictionary mapping track_id to list of trajectory hypotheses
        """
        # Update motion history
        self.update_history(detections)
        
        # Predict for each detection
        all_trajectories = {}
        
        for detection in detections:
            trajectories = self.predict(detection, detections)
            if trajectories:
                all_trajectories[detection.track_id] = trajectories
        
        return all_trajectories



class UncertaintyEstimator:
    """Estimate and propagate trajectory uncertainty."""
    
    def __init__(self):
        """Initialize uncertainty estimator."""
        self.logger = logging.getLogger(__name__)
    
    def calculate_covariance(
        self,
        trajectory: Trajectory,
        detection: Detection3D,
        model_type: str
    ) -> List[np.ndarray]:
        """
        Calculate covariance matrices for trajectory.
        
        Args:
            trajectory: Predicted trajectory
            detection: Source detection
            model_type: Model type used for prediction
            
        Returns:
            List of 3x3 covariance matrices
        """
        covariances = []
        
        # Base uncertainty depends on model type
        if model_type == 'lstm':
            base_uncertainty = 0.3  # meters
        elif model_type == 'cv':
            base_uncertainty = 0.5
        elif model_type == 'ca':
            base_uncertainty = 0.7
        elif model_type == 'ct':
            base_uncertainty = 0.8
        else:
            base_uncertainty = 1.0
        
        # Uncertainty grows with time
        for i, t in enumerate(trajectory.timestamps):
            # Linear growth with time
            time_factor = 1.0 + t
            
            # Velocity uncertainty contribution
            velocity = np.array(detection.velocity)
            speed = np.linalg.norm(velocity)
            velocity_factor = 1.0 + speed * 0.1
            
            # Combined uncertainty
            uncertainty = base_uncertainty * time_factor * velocity_factor
            
            # Create covariance matrix (isotropic)
            cov = np.eye(3) * (uncertainty ** 2)
            covariances.append(cov)
        
        return covariances
    
    def estimate_confidence(
        self,
        trajectory: Trajectory,
        history_length: int,
        scene_complexity: float
    ) -> float:
        """
        Estimate confidence for trajectory prediction.
        
        Args:
            trajectory: Predicted trajectory
            history_length: Length of motion history
            scene_complexity: Scene complexity measure (0-1)
            
        Returns:
            Confidence score (0-1)
        """
        # Base confidence from model
        confidence = trajectory.confidence
        
        # Adjust based on history length
        if history_length < 5:
            confidence *= 0.5
        elif history_length < 10:
            confidence *= 0.7
        elif history_length < 20:
            confidence *= 0.9
        
        # Adjust based on scene complexity
        # More complex scenes = lower confidence
        confidence *= (1.0 - scene_complexity * 0.3)
        
        # Adjust based on trajectory uncertainty
        if trajectory.uncertainty:
            avg_uncertainty = np.mean([np.trace(cov) for cov in trajectory.uncertainty])
            uncertainty_factor = 1.0 / (1.0 + avg_uncertainty)
            confidence *= uncertainty_factor
        
        return float(np.clip(confidence, 0.0, 1.0))
    
    def propagate_uncertainty(
        self,
        initial_cov: np.ndarray,
        velocity: np.ndarray,
        num_steps: int,
        dt: float
    ) -> List[np.ndarray]:
        """
        Propagate uncertainty through time.
        
        Args:
            initial_cov: Initial covariance matrix
            velocity: Velocity vector
            num_steps: Number of time steps
            dt: Time step size
            
        Returns:
            List of covariance matrices
        """
        covariances = []
        
        # Process noise (uncertainty in velocity)
        Q = np.eye(3) * 0.1  # Process noise covariance
        
        current_cov = initial_cov.copy()
        
        for step in range(num_steps):
            # State transition (constant velocity)
            # P(k+1) = F * P(k) * F^T + Q
            # For constant velocity, F = I
            current_cov = current_cov + Q * dt
            
            covariances.append(current_cov.copy())
        
        return covariances
    
    def merge_uncertainties(
        self,
        trajectories: List[Trajectory],
        weights: Optional[List[float]] = None
    ) -> Trajectory:
        """
        Merge multiple trajectory hypotheses with uncertainty.
        
        Args:
            trajectories: List of trajectory hypotheses
            weights: Optional weights for each trajectory
            
        Returns:
            Merged trajectory with combined uncertainty
        """
        if not trajectories:
            raise ValueError("No trajectories to merge")
        
        if len(trajectories) == 1:
            return trajectories[0]
        
        # Default to equal weights
        if weights is None:
            weights = [1.0 / len(trajectories)] * len(trajectories)
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        # Merge points (weighted average)
        num_steps = min(len(t.points) for t in trajectories)
        merged_points = []
        merged_uncertainties = []
        
        for step in range(num_steps):
            # Weighted average of positions
            positions = np.array([t.points[step] for t in trajectories])
            merged_pos = np.average(positions, axis=0, weights=weights)
            merged_points.append(tuple(merged_pos))
            
            # Merge uncertainties (weighted sum of covariances + spread)
            covariances = [t.uncertainty[step] for t in trajectories if step < len(t.uncertainty)]
            
            if covariances:
                # Weighted average of covariances
                merged_cov = np.average(covariances, axis=0, weights=weights[:len(covariances)])
                
                # Add uncertainty from spread between hypotheses
                spread = np.cov(positions.T)
                merged_cov = merged_cov + spread
                
                merged_uncertainties.append(merged_cov)
            else:
                merged_uncertainties.append(np.eye(3))
        
        # Merged confidence (weighted average)
        merged_confidence = np.average(
            [t.confidence for t in trajectories],
            weights=weights
        )
        
        # Use timestamps from first trajectory
        timestamps = trajectories[0].timestamps[:num_steps]
        
        return Trajectory(
            points=merged_points,
            timestamps=timestamps,
            uncertainty=merged_uncertainties,
            confidence=float(merged_confidence),
            model='merged'
        )



class CollisionProbabilityCalculator:
    """Calculate collision probability between trajectories."""
    
    def __init__(self, vehicle_dimensions: Tuple[float, float, float] = (4.5, 2.0, 1.5)):
        """
        Initialize collision probability calculator.
        
        Args:
            vehicle_dimensions: Ego vehicle dimensions (length, width, height) in meters
        """
        self.logger = logging.getLogger(__name__)
        self.vehicle_length = vehicle_dimensions[0]
        self.vehicle_width = vehicle_dimensions[1]
        self.vehicle_height = vehicle_dimensions[2]
    
    def mahalanobis_distance(
        self,
        pos1: np.ndarray,
        pos2: np.ndarray,
        cov1: np.ndarray,
        cov2: np.ndarray
    ) -> float:
        """
        Calculate Mahalanobis distance between two positions with uncertainty.
        
        Args:
            pos1: First position (x, y, z)
            pos2: Second position (x, y, z)
            cov1: Covariance matrix for first position
            cov2: Covariance matrix for second position
            
        Returns:
            Mahalanobis distance
        """
        # Difference vector
        diff = pos1 - pos2
        
        # Combined covariance
        combined_cov = cov1 + cov2
        
        # Add small regularization to avoid singular matrix
        combined_cov = combined_cov + np.eye(3) * 1e-6
        
        try:
            # Mahalanobis distance: sqrt(diff^T * Cov^-1 * diff)
            inv_cov = np.linalg.inv(combined_cov)
            distance = np.sqrt(diff.T @ inv_cov @ diff)
            return float(distance)
        except np.linalg.LinAlgError:
            # Fallback to Euclidean distance
            return float(np.linalg.norm(diff))
    
    def collision_probability_from_distance(
        self,
        mahal_distance: float,
        object_size: float = 2.0
    ) -> float:
        """
        Convert Mahalanobis distance to collision probability.
        
        Args:
            mahal_distance: Mahalanobis distance
            object_size: Size of object (meters)
            
        Returns:
            Collision probability (0-1)
        """
        # Collision threshold based on vehicle and object size
        collision_threshold = (self.vehicle_width + object_size) / 2.0
        
        # Probability decreases with distance
        # Use chi-squared distribution (3 DOF for 3D)
        # P(collision) = 1 - CDF(distance^2)
        
        # Simplified: exponential decay
        # P = exp(-distance / threshold)
        probability = np.exp(-mahal_distance / collision_threshold)
        
        return float(np.clip(probability, 0.0, 1.0))
    
    def calculate_trajectory_collision_probability(
        self,
        ego_trajectory: Trajectory,
        object_trajectory: Trajectory,
        object_size: float = 2.0
    ) -> Tuple[float, int]:
        """
        Calculate collision probability between ego and object trajectories.
        
        Args:
            ego_trajectory: Ego vehicle trajectory
            object_trajectory: Object trajectory
            object_size: Object size (meters)
            
        Returns:
            Tuple of (max_probability, time_step_of_max)
        """
        max_probability = 0.0
        max_step = -1
        
        # Check each time step
        num_steps = min(len(ego_trajectory.points), len(object_trajectory.points))
        
        for step in range(num_steps):
            ego_pos = np.array(ego_trajectory.points[step])
            obj_pos = np.array(object_trajectory.points[step])
            
            # Get uncertainties
            ego_cov = ego_trajectory.uncertainty[step] if step < len(ego_trajectory.uncertainty) else np.eye(3)
            obj_cov = object_trajectory.uncertainty[step] if step < len(object_trajectory.uncertainty) else np.eye(3)
            
            # Calculate Mahalanobis distance
            mahal_dist = self.mahalanobis_distance(ego_pos, obj_pos, ego_cov, obj_cov)
            
            # Convert to probability
            probability = self.collision_probability_from_distance(mahal_dist, object_size)
            
            if probability > max_probability:
                max_probability = probability
                max_step = step
        
        return max_probability, max_step
    
    def calculate_all_collision_probabilities(
        self,
        ego_trajectory: Trajectory,
        object_trajectories: Dict[int, List[Trajectory]],
        object_sizes: Optional[Dict[int, float]] = None
    ) -> Dict[int, Tuple[float, int, int]]:
        """
        Calculate collision probabilities for all objects.
        
        Args:
            ego_trajectory: Ego vehicle trajectory
            object_trajectories: Dictionary of object trajectories (track_id -> list of hypotheses)
            object_sizes: Optional dictionary of object sizes
            
        Returns:
            Dictionary mapping track_id to (max_probability, time_step, hypothesis_index)
        """
        collision_probs = {}
        
        for track_id, trajectories in object_trajectories.items():
            max_prob = 0.0
            max_step = -1
            max_hyp = -1
            
            # Check all hypotheses
            for hyp_idx, obj_traj in enumerate(trajectories):
                # Get object size
                obj_size = object_sizes.get(track_id, 2.0) if object_sizes else 2.0
                
                # Calculate collision probability
                prob, step = self.calculate_trajectory_collision_probability(
                    ego_trajectory,
                    obj_traj,
                    obj_size
                )
                
                if prob > max_prob:
                    max_prob = prob
                    max_step = step
                    max_hyp = hyp_idx
            
            if max_prob > 0.01:  # Only store significant probabilities
                collision_probs[track_id] = (max_prob, max_step, max_hyp)
        
        return collision_probs
    
    def check_uncertainty_ellipse_overlap(
        self,
        pos1: np.ndarray,
        cov1: np.ndarray,
        pos2: np.ndarray,
        cov2: np.ndarray,
        confidence_level: float = 0.95
    ) -> bool:
        """
        Check if uncertainty ellipses overlap.
        
        Args:
            pos1: First position
            cov1: First covariance matrix
            pos2: Second position
            cov2: Second covariance matrix
            confidence_level: Confidence level for ellipse (default 95%)
            
        Returns:
            True if ellipses overlap
        """
        # Chi-squared value for confidence level (3 DOF)
        from scipy import stats
        chi2_val = stats.chi2.ppf(confidence_level, df=3)
        
        # Mahalanobis distance
        mahal_dist = self.mahalanobis_distance(pos1, pos2, cov1, cov2)
        
        # Ellipses overlap if Mahalanobis distance < chi-squared threshold
        return mahal_dist < np.sqrt(chi2_val)
