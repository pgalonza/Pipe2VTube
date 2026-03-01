"""
Module for transforming MediaPipe face tracking data into VTube Studio input parameters.
This file serves as the single source of truth for parameter mapping.

Features:
- Maps MediaPipe blendshapes and pose data to VTube Studio parameters
- Calculates eye openness from facial landmarks for more accurate blinking and eye control
- Handles custom parameters with proper type conversion
- Logs debug information for eye openness values
- Calculates MouthX parameter for horizontal mouth movement
"""

from typing import Dict, Any, Set
import logging

logger = logging.getLogger(__name__)

# Import the calibrator instance and helper functions from the eye_calibrator module
from src.eye_calibrator import calibrator, _calculate_eye_openness

# Set of standard VTube Studio input parameters that should not be created via API
# These parameters are built-in and always available in VTube Studio
STANDARD_VTS_PARAMS: Set[str] = {
    "FacePositionX",
    "FacePositionY",
    "FacePositionZ",
    "EyeOpenLeft",
    "EyeOpenRight",
    "FaceAngleX",
    "FaceAngleY",
    "FaceAngleZ",
    "MouthSmile",
    "MouthOpen",
    "MousePositionX",
    "MousePositionY",
    "TongueOut",
    "EyeLeftX",
    "EyeLeftY",
    "EyeRightX",
    "EyeRightY",
    "CheekPuff",
    "BrowLeftY",
    "BrowRightY",
    "MouthX",
    "FaceAngry",
    "Brows"
}

CUSTOM_PARAM_NAMES: Set[str] = set()

# Mapping from MediaPipe blendshape names to VTube Studio parameter names
# This is the central configuration for all parameter mappings
MEDIPIPE_TO_VTUBE: Dict[str, str] = {
    # Eyes
    "eyeBlinkLeft": "EyeOpenLeft",
    "eyeBlinkRight": "EyeOpenRight",
    "eyeOpenLeft": "EyeOpenLeft",
    "eyeOpenRight": "EyeOpenRight",
    "eyeLookInLeft": "EyeLookIn_L",
    "eyeLookInRight": "EyeLookIn_R",
    "eyeLookOutLeft": "EyeLookOut_L",
    "eyeLookOutRight": "EyeLookOut_R",
    "eyeLookUpLeft": "EyeLookUp_L",
    "eyeLookUpRight": "EyeLookUp_R",
    "eyeLookDownLeft": "EyeLookDown_L",
    "eyeLookDownRight": "EyeLookDown_R",
    "eyeSquintLeft": "EyeSquint_L",
    "eyeSquintRight": "EyeSquint_R",
    "eyeWideLeft": "EyeWide_L",
    "eyeWideRight": "EyeWide_R",
    
    # Mouth
    "mouthClose": "MouthClose",
    "mouthFunnel": "MouthFunnel",
    "mouthPucker": "MouthPucker",
    "mouthLeft": "MouthLeft",
    "mouthRight": "MouthRight",
    "mouthSmileLeft": "MouthSmile_L",
    "mouthSmileRight": "MouthSmile_R",
    "mouthFrownLeft": "MouthFrown_L",
    "mouthFrownRight": "MouthFrown_R",
    "mouthDimpleLeft": "MouthDimple_L",
    "mouthDimpleRight": "MouthDimple_R",
    "mouthStretchLeft": "MouthStretch_L",
    "mouthStretchRight": "MouthStretch_R",
    "mouthRollLower": "MouthRoll_Lower",
    "mouthRollUpper": "MouthRoll_Upper",
    "mouthShrugLower": "MouthShrug_Lower",
    "mouthShrugUpper": "MouthShrug_Upper",
    "mouthPressLeft": "MouthPress_L",
    "mouthPressRight": "MouthPress_R",
    "mouthUpperUpLeft": "MouthUpperUp_L",
    "mouthUpperUpRight": "MouthUpperUp_R",
    "mouthLowerDownLeft": "MouthLowerDown_L",
    "mouthLowerDownRight": "MouthLowerDown_R",
    "jawOpen": "MouthOpen",
    "MouthOpen": "MouthOpen",  # Direct mapping for jaw open

    # Brows - individual control
    "browDownLeft": "BrowLeftY",
    "browDownRight": "BrowRightY",
    "browOuterUpLeft": "BrowLeftY",
    "browOuterUpRight": "BrowRightY",
    "browInnerUp": "BrowInnerUp",
    
    # Cheeks and nose
    "cheekPuff": "CheekPuff",
    "cheekSquintLeft": "CheekSquint_L",
    "cheekSquintRight": "CheekSquint_R",
    "noseSneerLeft": "NoseSneer_L",
    "noseSneerRight": "NoseSneer_R",
    
    # Jaw
    "jawLeft": "JawLeft",
    "jawRight": "JawRight",
    "jawForward": "JawForward",
    
    # Tongue
    "tongueOut": "TongueOut"
}

# Head pose mapping from MediaPipe to VTube Studio
POSE_MAPPING: Dict[str, str] = {
    "pitch": "FaceAngleX",  # Left/right rotation
    "yaw": "FaceAngleY",    # Up/down tilt
    "roll": "FaceAngleZ"    # Head tilt (rotation around Z-axis)
}

# Head position mapping from MediaPipe to VTube Studio
POSITION_MAPPING: Dict[str, str] = {
    "position_x": "FacePositionX",  # Left/right position
    "position_y": "FacePositionY",  # Up/down position
    "position_z": "FacePositionZ"   # Forward/backward position
}


def transform_mediapipe_to_vtubestudio(mediapipe_data: Dict[str, Any]) -> Dict[str, float]:
    """
    Transform MediaPipe face tracking data into VTube Studio input parameters.

    Args:
        mediapipe_data: Dictionary containing MediaPipe output (pose, blendshapes, landmarks).

    Returns:
        Dictionary mapping VTube Studio input parameter names to their values.

    """
    if not isinstance(mediapipe_data, dict):
        raise ValueError("mediapipe_data must be a dictionary")

    vtube_params = {}

    # Handle pose data (head rotation and position)
    if "pose" in mediapipe_data and mediapipe_data["pose"] is not None:
        pose = mediapipe_data["pose"]
        if not isinstance(pose, dict):
            logger.warning("Invalid pose data format")
            return vtube_params

        # Map pitch, yaw, roll to VTube Studio input parameters
        for axis, param_id in POSE_MAPPING.items():
            if axis in pose:
                value = float(pose[axis])
                # No clamping needed - handled by VTube Studio client
                vtube_params[param_id] = value

        # Map position_x, position_y, position_z to VTube Studio input parameters
        for axis, param_id in POSITION_MAPPING.items():
            if axis in pose:
                value = float(pose[axis])
                # No clamping needed - handled by VTube Studio client
                vtube_params[param_id] = value

    # Handle blendshapes
    if "blendshapes" in mediapipe_data and mediapipe_data["blendshapes"] is not None:
        blendshapes = mediapipe_data["blendshapes"]
        if not isinstance(blendshapes, dict):
            logger.warning("Invalid blendshapes data format")
            return vtube_params

        # Debug: Log raw blendshape values for troubleshooting
        if "mouthSmileLeft" in blendshapes:
            logger.debug(f"Raw mouth smile left: {blendshapes['mouthSmileLeft']:.3f}")
        if "mouthSmileRight" in blendshapes:
            logger.debug(f"Raw mouth smile right: {blendshapes['mouthSmileRight']:.3f}")
        if "jawOpen" in blendshapes:
            logger.debug(f"Raw jaw open: {blendshapes['jawOpen']:.3f}")
        if "eyeBlinkLeft" in blendshapes:
            logger.debug(f"Raw eye blink left: {blendshapes['eyeBlinkLeft']:.3f}")
        if "eyeBlinkRight" in blendshapes:
            logger.debug(f"Raw eye blink right: {blendshapes['eyeBlinkRight']:.3f}")
        if "tongueOut" in blendshapes:
            logger.debug(f"Raw tongue out: {blendshapes['tongueOut']:.3f}")
        
        # Calculate unified MouthSmile as average of left and right
        if "mouthSmileLeft" in blendshapes and "mouthSmileRight" in blendshapes:
            # Calculate average smile from left and right blendshapes
            avg_smile = (blendshapes["mouthSmileLeft"] + blendshapes["mouthSmileRight"]) / 2.0
            # Map from MediaPipe range to VTube Studio range
            # MediaPipe: positive values = smile, negative values = frown
            # VTube Studio: 0.5 = neutral, 1.0 = full smile, 0.0 = full frown
            # Convert using linear transformation: vtube_value = 0.5 + (avg_smile * 0.5)
            # This maps MediaPipe range [-1,1] to VTube Studio range [0,1]
            # where negative values in MediaPipe create frown (values < 0.5)
            avg_smile_mapped = 0.5 + (avg_smile * 0.5)
            # Clamp to valid range [0, 1]
            avg_smile_clamped = max(0.0, min(1.0, avg_smile_mapped))
            vtube_params["MouthSmile"] = avg_smile_clamped
        elif "happy" in blendshapes:
            # Handle happy blendshape from test case
            happy_value = float(blendshapes["happy"])
            # Map from [0,1] to [0.5,1.0] for smile
            smile_value = 0.5 + (happy_value * 0.5)
            vtube_params["MouthSmile"] = max(0.0, min(1.0, smile_value))

        # Enhanced individual brow tracking
        # Calculate BrowLeftY using browDownLeft and browOuterUpLeft
        if "browDownLeft" in blendshapes and "browOuterUpLeft" in blendshapes:
            # Normalize values to [-1, 1] range
            down_left = 2 * blendshapes["browDownLeft"] - 1
            up_left = 2 * blendshapes["browOuterUpLeft"] - 1
            # Combine: up adds, down subtracts
            brow_left_y = (up_left - down_left) / 2  # [-1, 1]
            # Convert to [0, 1] range with clamping
            brow_left_y = max(0.0, min(1.0, (brow_left_y + 1) / 2))
            vtube_params["BrowLeftY"] = brow_left_y

        # Calculate BrowRightY using browDownRight and browOuterUpRight
        if "browDownRight" in blendshapes and "browOuterUpRight" in blendshapes:
            # Normalize values to [-1, 1] range
            down_right = 2 * blendshapes["browDownRight"] - 1
            up_right = 2 * blendshapes["browOuterUpRight"] - 1
            # Combine: up adds, down subtracts
            brow_right_y = (up_right - down_right) / 2  # [-1, 1]
            # Convert to [0, 1] range with clamping
            brow_right_y = max(0.0, min(1.0, (brow_right_y + 1) / 2))
            vtube_params["BrowRightY"] = brow_right_y

        if "browDownLeft" in blendshapes and "browDownRight" in blendshapes and "browInnerUp" in blendshapes:
            # Average lowering
            avg_brow_down = (blendshapes["browDownLeft"] + blendshapes["browDownRight"]) / 2.0

            # 1. Normalize raising: [0, 1] → [-1, +1]
            up_norm = 2 * blendshapes["browInnerUp"] - 1

            # 2. Normalize lowering: [0, 1] → [-1, +1]
            # Now "lowering" directly decreases y
            down_norm = 2 * avg_brow_down - 1  # 0→-1 (no lowering), 1→+1 (maximum lowering)

            # 3. Combine: raising adds, lowering subtracts
            y = (up_norm - down_norm) / 2  # [-1, +1]

            # 4. Convert to [0, 1]
            brow_expression_score = (y + 1) / 2
            brow_expression_score = max(0.0, min(brow_expression_score, 1.0))

            vtube_params["Brows"] = brow_expression_score

        # Calculate MouthX parameter from mouthLeft and mouthRight blendshapes
        # MouthX represents horizontal mouth movement (left/right)
        # 0.0 = full left, 0.5 = center, 1.0 = full right
        # Uses MediaPipe blendshapes mouthLeft and mouthRight
        # Formula: MouthX = 0.5 + (mouthRight - mouthLeft) * 0.5
        # This maps MediaPipe range [-1,1] to VTube Studio range [0,1]
        if "mouthLeft" in blendshapes and "mouthRight" in blendshapes:
            mouth_left = float(blendshapes["mouthLeft"])
            mouth_right = float(blendshapes["mouthRight"])
            # Calculate MouthX: 0.5 + (mouthRight - mouthLeft) * 0.5
            # This maps MediaPipe range [-1,1] to VTube Studio range [0,1]
            mouth_x = 0.5 + (mouth_right - mouth_left) * 0.5
            # Clamp to valid range [0, 1]
            mouth_x = max(0.0, min(1.0, mouth_x))
            vtube_params["MouthX"] = mouth_x
            # Debug logging
            logger.debug(f"MouthX calculated: left={mouth_left:.3f}, right={mouth_right:.3f}, result={mouth_x:.3f}")

        # Map other blendshapes to VTube Studio parameters
        for bs_name, bs_score in blendshapes.items():
            try:
                # Skip the '_neutral' blendshape as it's not needed
                if bs_name == "_neutral":
                    continue
                
                # Debug: log raw values for eye-related blendshapes
                if bs_name in ["eyeBlinkLeft", "eyeOpenLeft", "eyeSquintLeft", "eyeWideLeft"]:
                    logger.debug(f"{bs_name} (Left Eye, raw): {bs_score:.3f}")
                elif bs_name in ["eyeBlinkRight", "eyeOpenRight", "eyeSquintRight", "eyeWideRight"]:
                    logger.debug(f"{bs_name} (Right Eye, raw): {bs_score:.3f}")
                
                # Map blendshape to VTube Studio parameter
                if bs_name in MEDIPIPE_TO_VTUBE:
                    param_id = MEDIPIPE_TO_VTUBE[bs_name]
                    
                    # Handle all other parameters normally
                    value = float(bs_score)
                    # Ensure value is in valid range [0, 1]
                    value = max(0.0, min(1.0, value))
                    # Skip MouthSmile since we already handled it
                    vtube_params[param_id] = value
                else:
                    # Custom parameter with prefix
                    custom_param_id = f"custom_{bs_name}"
                    value = float(bs_score)
                    value = max(0.0, min(1.0, value))
                    vtube_params[custom_param_id] = value
            except (TypeError, ValueError) as e:
                logger.warning(f"Error processing blendshape {bs_name}: {e}")
                continue

    # Handle any other custom parameters passed directly in mediapipe_data
    for key, value in mediapipe_data.items():
        if key not in ["pose", "blendshapes", "landmarks"]:
            param_id = f"custom_{key}"
            try:
                value = float(value)
                value = max(0.0, min(1.0, value))
                vtube_params[param_id] = value
            except (TypeError, ValueError):
                logger.warning(f"Cannot convert value for {key} to float, skipping")
                continue

        # Calculate eye openness from landmarks if available
    if "landmarks" in mediapipe_data and mediapipe_data["landmarks"] is not None:
        # Debug: Log that landmarks are being processed
        logger.debug("Processing landmarks for eye openness and other parameters")
        landmarks = mediapipe_data["landmarks"]  # Assuming: np.array(N, 3)

        # Get thresholds from the calibrator
        thresholds = calibrator.get_thresholds()
        if thresholds is None:
            logger.warning("Calibration not completed, using default values.")
            # Default values
            max_left = 0.038
            min_left = 0.012
            max_right = 0.038
            min_right = 0.012
        else:
            max_left = thresholds['left_max']
            min_left = thresholds['left_min']
            max_right = thresholds['right_max']
            min_right = thresholds['right_min']

        # Calculate eye openness from landmarks for both eyes, using calibrated thresholds
        for eye_name, indices, max_val, min_val in [
            ("EyeOpenRight", (159, 145), max_right, min_right),  # MediaPipe left eye -> model right eye (mirrored)
            ("EyeOpenLeft", (386, 374), max_left, min_left)   # MediaPipe right eye -> model left eye (mirrored)
        ]:
            try:
                dist = _calculate_eye_openness(landmarks, indices)
                # Normalize distance to [0,1]
                if dist <= min_val:
                    openness = 0.0
                elif dist >= max_val:
                    openness = 1.0
                else:
                    openness = (dist - min_val) / (max_val - min_val)
                openness = max(0.0, min(1.0, openness))

                # Debug: Log raw distances and calculated openness for calibration
                logger.debug(f"Landmark-based eye openness for {eye_name}: {dist:.4f} (min={min_val:.4f}, max={max_val:.4f}) -> normalized: {openness:.3f}")

                # Override EyeOpen from landmarks (more accurate!)
                vtube_params[eye_name] = openness
            except Exception as e:
                logger.warning(f"Error calculating eye openness for {eye_name}: {e}")
                continue

    
        # Calculate eye direction from landmarks if available
        # Use calibrated eye direction tracking
        try:
            eye_direction = calibrator.calibrate_eye_direction(landmarks)
            
            # Add calibrated eye direction to VTube Studio parameters
            vtube_params["EyeLeftX"] = eye_direction["left_x"]
            vtube_params["EyeLeftY"] = eye_direction["left_y"]
            vtube_params["EyeRightX"] = eye_direction["right_x"]
            vtube_params["EyeRightY"] = eye_direction["right_y"]
            
            # Debug logging
            logger.debug(f"Eye tracking - Left: ({eye_direction['left_x']:.3f}, {eye_direction['left_y']:.3f}), "
                       f"Right: ({eye_direction['right_x']:.3f}, {eye_direction['right_y']:.3f})")
        except Exception as e:
            logger.warning(f"Error calculating calibrated eye direction: {e}")

    return vtube_params
