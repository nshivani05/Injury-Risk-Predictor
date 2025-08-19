import numpy as np

def angle(a, b, c):
    """Returns angle ABC (in degrees) between points a, b, c."""
    ba = a - b
    bc = c - b
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cosang = np.clip(cosang, -1.0, 1.0)
    return np.degrees(np.arccos(cosang))

def knee_valgus_angle(hip, knee, ankle):
    """Measures inward knee collapse (valgus)."""
    return angle(np.array(hip), np.array(knee), np.array(ankle))
