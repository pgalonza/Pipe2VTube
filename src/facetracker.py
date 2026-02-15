"""
Module for face tracking using MediaPipe.
Handles landmark detection and parameter extraction for Live2D control.
"""

from typing import Optional, Dict, Any, Tuple
import logging
import numpy as np
import time

# Use new MediaPipe Tasks API
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import mediapipe as mp

from src.parameter_mapper import transform_mediapipe_to_vtubestudio

# Import the calibrator instance from the new eye_calibrator module
from src.eye_calibrator import calibrator

logger = logging.getLogger(__name__)


class FaceTracker:
    """
    Face tracker using MediaPipe Tasks API.
    """

    def __init__(self, model_path: str = "face_landmarker.task"):
        """
        Initialize the face tracker.

        Args:
            model_path: Path to the MediaPipe face landmarker model.
        """
        # Use the new Tasks API correctly
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1  # Track only one face
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)
        logger.info("Face tracker initialized with model: %s", model_path)
        
        # No smoothing - handled by VTube Studio
        self._prev_vtube_params = {}
        self._prev_fps = 30.0  # Assume 30 FPS initially

    def process_frame(self, frame: np.ndarray, draw_landmarks: bool = False) -> Tuple[Optional[Dict[str, Any]], np.ndarray]:
        """
        Process a video frame and extract face landmarks and blendshapes.

        Args:
            frame: Input image as numpy array (BGR).
            draw_landmarks: If True, draw landmarks on the frame for debugging.

        Returns:
            Tuple of (face_data, output_frame) where face_data is None if no face detected,
            and output_frame is the original frame or frame with landmarks drawn.
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        detection_result = self.detector.detect(image)

        if not detection_result.face_landmarks:
            if draw_landmarks:
                return None, frame
            return None, frame

        # Use first (and only) detected face
        landmarks = detection_result.face_landmarks[0]
        blendshapes = detection_result.face_blendshapes[0] if detection_result.face_blendshapes else None
        transformation_matrix = detection_result.facial_transformation_matrixes[0] if detection_result.facial_transformation_matrixes else None

        # Extract head pose from transformation matrix
        pose = self._extract_pose(transformation_matrix) if transformation_matrix is not None else None

        # Convert blendshapes to dictionary
        blendshapes_dict = {}
        if blendshapes:
            for bs in blendshapes:
                # Debug: Print all available blendshape categories
                # print(f"Available blendshape: {bs.category_name}")
                blendshapes_dict[bs.category_name] = bs.score

        # Update eye calibration if in progress
        if calibrator.is_calibrating:
            calibrator.update(landmarks)

        # Calculate current FPS for display
        current_fps = self._calculate_fps()
        
        # Update FPS for any potential future use
        if current_fps > 0:
            self._prev_fps = current_fps

        # Transform to VTube Studio parameters
        vtube_params = transform_mediapipe_to_vtubestudio({
            "landmarks": landmarks,
            "blendshapes": blendshapes_dict,
            "pose": pose
        })
        
        # Debug: Log raw parameters to verify data flow
        if 'FaceAngleX' in vtube_params:
            logger.debug(f"Raw pose data - X: {vtube_params['FaceAngleX']:.2f}, Y: {vtube_params.get('FaceAngleY', 0):.2f}, Z: {vtube_params.get('FaceAngleZ', 0):.2f}")
        if 'MouthOpen' in vtube_params:
            logger.debug(f"Raw mouth open: {vtube_params['MouthOpen']:.2f}")
        if 'EyeOpenLeft' in vtube_params:
            logger.debug(f"Raw eye openness - Left: {vtube_params['EyeOpenLeft']:.2f}, Right: {vtube_params.get('EyeOpenRight', 0):.2f}")
        
        # VTube Studio handles smoothing internally, pass through parameters directly
        output_params = vtube_params.copy()

        # Update previous values for parameter continuity
        self._prev_vtube_params = output_params

        result = {
            "landmarks": landmarks,
            "blendshapes": blendshapes_dict,
            "pose": pose,
            "vtube_params": output_params
        }

        # Draw landmarks if requested
        output_frame = frame.copy()
        if draw_landmarks:
            debug_frame = output_frame  # Use the already copied frame for drawing
            # Draw all landmarks
            for i, landmark in enumerate(landmarks):
                h, w = frame.shape[:2]
                x, y = int(landmark.x * w), int(landmark.y * h)
                # Draw landmark point with moderate size for visibility
                cv2.circle(debug_frame, (x, y), 2, (0, 255, 0), -1)
            
            # Draw connection lines using the detector's built-in connections
            try:
                face_connections = []
                for connection in self.detector.get_landmarks_connections():
                    start_idx = connection.start
                    end_idx = connection.end
                    face_connections.append((start_idx, end_idx))
            except AttributeError:
                # Fallback to manual connections
                face_connections = [
                    # Jawline
                    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), 
                    (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16), (16, 17),
                    # Left eyebrow
                    (17, 18), (18, 19), (19, 20), (20, 21),
                    # Right eyebrow
                    (22, 23), (23, 24), (24, 25), (25, 26),
                    # Nose
                    (27, 28), (28, 29), (29, 30),
                    # Nose bridge
                    (31, 32), (32, 33), (33, 34), (34, 35),
                    # Left eye
                    (36, 37), (37, 38), (38, 39), (39, 40), (40, 41), (41, 36),
                    # Right eye
                    (42, 43), (43, 44), (44, 45), (45, 46), (46, 47), (47, 42),
                    # Left to right face connection
                    (27, 39), (27, 42),
                    # Mouth
                    (48, 49), (49, 50), (50, 51), (51, 52), (52, 53), (53, 54), 
                    (54, 55), (55, 56), (56, 57), (57, 58), (58, 59), (59, 48),
                ]
            


            # Add text overlay
            cv2.putText(debug_frame, "Face Tracking Debug View", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(debug_frame, f"FPS: {current_fps:.1f}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            output_frame = debug_frame

        return result, output_frame
    
    def _calculate_fps(self) -> float:
        """Calculate current FPS."""
        now = time.time()
        if hasattr(self, '_last_time') and self._last_time > 0:
            fps = 1.0 / (now - self._last_time)
        else:
            fps = 0.0
        self._last_time = now
        return fps

    def _extract_pose(self, matrix: np.ndarray) -> Dict[str, float]:
        """
        Extract pitch, yaw, roll from transformation matrix.

        Args:
            matrix: 4x4 transformation matrix.

        Returns:
            Dictionary with pitch, yaw, roll in degrees.
        """
        # Extract rotation matrix (3x3) from 4x4 matrix
        rot_matrix = matrix[:3, :3]

        # Convert rotation matrix to Euler angles
        sy = np.sqrt(rot_matrix[0, 0] * rot_matrix[0, 0] + rot_matrix[1, 0] * rot_matrix[1, 0])
        singular = sy < 1e-6

        if not singular:
            x = np.arctan2(rot_matrix[2, 1], rot_matrix[2, 2])
            y = np.arctan2(-rot_matrix[2, 0], sy)
            z = np.arctan2(rot_matrix[1, 0], rot_matrix[0, 0])
        else:
            x = np.arctan2(-rot_matrix[1, 2], rot_matrix[1, 1])
            y = np.arctan2(-rot_matrix[2, 0], sy)
            z = 0

        # Convert to degrees
        # Note: MediaPipe's coordinate system may need axis remapping for VTube Studio
        pitch = np.degrees(y)  # Was y, now mapped to pitch (up/down)
        yaw = np.degrees(x)   # Was x, now mapped to yaw (left/right)
        roll = np.degrees(z)  # Roll remains the same (tilting)

        # Apply VTube Studio coordinate system mapping with corrected axes
        # VTube Studio uses: X=up/down (pitch), Y=left/right (yaw), Z=tilt (roll)
        # Invert Y and Z axes as requested
        return {
            "pitch": float(pitch),      # X: up/down (no inversion)
            "yaw": -float(yaw),         # Y: left/right (inverted)
            "roll": -float(roll)        # Z: tilt (inverted)
        }
