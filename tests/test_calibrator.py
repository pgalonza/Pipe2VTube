from src.eye_calibrator import calibrator

print("Calibrator attributes:")
print(f"EYE_OPEN_CALIBRATED_MAX_LEFT: {getattr(calibrator, 'EYE_OPEN_CALIBRATED_MAX_LEFT', 'NOT SET')}")
print(f"EYE_OPEN_CALIBRATED_MIN_LEFT: {getattr(calibrator, 'EYE_OPEN_CALIBRATED_MIN_LEFT', 'NOT SET')}")
print(f"EYE_OPEN_CALIBRATED_MAX_RIGHT: {getattr(calibrator, 'EYE_OPEN_CALIBRATED_MAX_RIGHT', 'NOT SET')}")
print(f"EYE_OPEN_CALIBRATED_MIN_RIGHT: {getattr(calibrator, 'EYE_OPEN_CALIBRATED_MIN_RIGHT', 'NOT SET')}")

print("\nHas attributes:")
print(f"hasattr EYE_OPEN_CALIBRATED_MAX_LEFT: {hasattr(calibrator, 'EYE_OPEN_CALIBRATED_MAX_LEFT')}")
print(f"hasattr EYE_OPEN_CALIBRATED_MIN_LEFT: {hasattr(calibrator, 'EYE_OPEN_CALIBRATED_MIN_LEFT')}")
print(f"hasattr EYE_OPEN_CALIBRATED_MAX_RIGHT: {hasattr(calibrator, 'EYE_OPEN_CALIBRATED_MAX_RIGHT')}")
print(f"hasattr EYE_OPEN_CALIBRATED_MIN_RIGHT: {hasattr(calibrator, 'EYE_OPEN_CALIBRATED_MIN_RIGHT')}")

print("\nGet thresholds:")
thresholds = calibrator.get_thresholds()
print(f"Thresholds: {thresholds}")