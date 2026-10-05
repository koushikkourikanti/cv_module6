"""
CSc 8830 - Assignment 6

Part A:
Lucas-Kanade Feature Tracking

This program:

1. Selects two consecutive frames.
2. Detects corner features.
3. Tracks the features using Lucas-Kanade.
4. Calculates displacement:

       u = x2 - x1
       v = y2 - y1

5. Calculates predicted next position:

       x2 = x1 + u
       y2 = y1 + v

6. Performs a forward-backward consistency check.
7. Saves results as CSV and image.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def read_two_frames(
    video_path,
    frame_index
):

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        raise FileNotFoundError(
            video_path
        )


    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        frame_index
    )


    success1, frame1 = cap.read()

    success2, frame2 = cap.read()

    cap.release()


    if not success1 or not success2:

        raise RuntimeError(
            "Could not read two consecutive frames."
        )


    return frame1, frame2


def validate_tracking(
    video_path,
    output_dir,
    frame_index=0,
    max_points=100,
):

    video_path = Path(video_path)

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    # -----------------------------------------------------
    # LOAD FRAMES
    # -----------------------------------------------------

    frame1, frame2 = read_two_frames(
        video_path,
        frame_index
    )


    gray1 = cv2.cvtColor(
        frame1,
        cv2.COLOR_BGR2GRAY
    )

    gray2 = cv2.cvtColor(
        frame2,
        cv2.COLOR_BGR2GRAY
    )


    # -----------------------------------------------------
    # FEATURE DETECTION
    # -----------------------------------------------------

    points_frame1 = cv2.goodFeaturesToTrack(
        gray1,
        mask=None,
        maxCorners=max_points,
        qualityLevel=0.01,
        minDistance=12,
        blockSize=7,
    )


    if points_frame1 is None:

        raise RuntimeError(
            "No trackable features were found."
        )


    # -----------------------------------------------------
    # LUCAS-KANADE OPTICAL FLOW
    # -----------------------------------------------------

    points_frame2, status, lk_error = (
        cv2.calcOpticalFlowPyrLK(
            gray1,
            gray2,
            points_frame1,
            None,

            winSize=(21, 21),

            maxLevel=3,

            criteria=(
                cv2.TERM_CRITERIA_EPS
                |
                cv2.TERM_CRITERIA_COUNT,
                30,
                0.01,
            ),
        )
    )


    valid = (
        status.flatten() == 1
    )


    old_points = (
        points_frame1[valid]
        .reshape(-1, 2)
    )

    new_points = (
        points_frame2[valid]
        .reshape(-1, 2)
    )

    valid_lk_error = (
        lk_error[valid]
        .reshape(-1)
    )


    # -----------------------------------------------------
    # FORWARD-BACKWARD CHECK
    # -----------------------------------------------------

    back_points, back_status, _ = (
        cv2.calcOpticalFlowPyrLK(
            gray2,
            gray1,

            new_points
            .reshape(-1, 1, 2)
            .astype(np.float32),

            None,

            winSize=(21, 21),

            maxLevel=3,

            criteria=(
                cv2.TERM_CRITERIA_EPS
                |
                cv2.TERM_CRITERIA_COUNT,
                30,
                0.01,
            ),
        )
    )


    back_points = (
        back_points.reshape(-1, 2)
    )


    forward_backward_error = (
        np.linalg.norm(
            old_points -
            back_points,
            axis=1,
        )
    )


    # -----------------------------------------------------
    # CREATE RESULTS
    # -----------------------------------------------------

    results = []

    visualization = frame2.copy()


    for index, (
        old_point,
        new_point,
        internal_error,
        fb_error,
    ) in enumerate(
        zip(
            old_points,
            new_points,
            valid_lk_error,
            forward_backward_error,
        ),
        start=1,
    ):

        x1 = float(
            old_point[0]
        )

        y1 = float(
            old_point[1]
        )

        x2 = float(
            new_point[0]
        )

        y2 = float(
            new_point[1]
        )


        # -------------------------------------------------
        # MOTION VECTOR
        # -------------------------------------------------

        u = x2 - x1

        v = y2 - y1


        # Theoretical next position
        predicted_x = x1 + u

        predicted_y = y1 + v


        results.append(
            {
                "point_id": index,

                "x_frame_t": x1,

                "y_frame_t": y1,

                "u_pixels_per_frame": u,

                "v_pixels_per_frame": v,

                "predicted_x_frame_t1":
                    predicted_x,

                "predicted_y_frame_t1":
                    predicted_y,

                "tracked_x_frame_t1":
                    x2,

                "tracked_y_frame_t1":
                    y2,

                "forward_backward_error_pixels":
                    float(fb_error),

                "lucas_kanade_error":
                    float(internal_error),
            }
        )


        # -------------------------------------------------
        # DRAW TRACK
        # -------------------------------------------------

        start = (
            int(round(x1)),
            int(round(y1))
        )

        end = (
            int(round(x2)),
            int(round(y2))
        )


        cv2.arrowedLine(
            visualization,
            start,
            end,
            (0, 255, 0),
            2,
            tipLength=0.25,
        )


        cv2.circle(
            visualization,
            end,
            4,
            (0, 0, 255),
            -1,
        )


    # -----------------------------------------------------
    # SAVE FILES
    # -----------------------------------------------------

    video_name = video_path.stem


    csv_output = (
        output_dir /
        f"{video_name}_tracking_validation.csv"
    )


    image_output = (
        output_dir /
        f"{video_name}_tracking_validation.jpg"
    )


    pd.DataFrame(
        results
    ).to_csv(
        csv_output,
        index=False
    )


    cv2.imwrite(
        str(image_output),
        visualization
    )


    print(
        "\nTracking completed."
    )

    print(
        f"CSV: {csv_output}"
    )

    print(
        f"Image: {image_output}"
    )


    return (
        str(csv_output),
        str(image_output),
    )


def main():

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--input",
        required=True,
    )


    parser.add_argument(
        "--frame",
        type=int,
        default=0,
    )


    parser.add_argument(
        "--output-dir",
        default="outputs/partA",
    )


    args = parser.parse_args()


    validate_tracking(
        args.input,
        args.output_dir,
        args.frame,
    )


if __name__ == "__main__":
    main()