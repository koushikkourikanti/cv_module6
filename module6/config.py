from pathlib import Path
import numpy as np


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

VIDEO_DIR = PROJECT_ROOT / "data" / "videos"
SFM_DIR = PROJECT_ROOT / "data" / "sfm_images"

OUTPUT_A = PROJECT_ROOT / "outputs" / "partA"
OUTPUT_B = PROJECT_ROOT / "outputs" / "partB"


# Create folders automatically if they do not exist
for folder in [
    VIDEO_DIR,
    SFM_DIR,
    OUTPUT_A,
    OUTPUT_B,
]:
    folder.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# CAMERA CALIBRATION
# ---------------------------------------------------------
#
# These values are from the previous camera calibration.
#
# IMPORTANT:
# Use them only if Assignment 6 images are captured using
# the SAME camera, SAME resolution, and SAME camera mode.
#
# If we use a different camera/resolution later,
# we will replace these values.
# ---------------------------------------------------------

K = np.array(
    [
        [1160.2162, 0.0, 605.62396],
        [0.0, 1147.1919, 645.47437],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


DIST_COEFFS = np.array(
    [
        -0.00597957,
        0.12974463,
        -0.00750799,
        0.0014783,
        -0.11945499,
    ],
    dtype=np.float64,
)


# ---------------------------------------------------------
# EXPECTED INPUT FILES
# ---------------------------------------------------------

VIDEO_FILES = [
    VIDEO_DIR / "video1.mp4",
    VIDEO_DIR / "video2.mp4",
]


SFM_FILES = [
    SFM_DIR / "view1.jpg",
    SFM_DIR / "view2.jpg",
    SFM_DIR / "view3.jpg",
    SFM_DIR / "view4.jpg",
]