"""
CSc 8830 - Assignment 6
Part A: Dense Optical Flow

This program:

1. Reads a video.
2. Calculates dense optical flow using Farneback.
3. Calculates motion direction and magnitude.
4. Creates:
      - HSV optical-flow video
      - Optical-flow arrow video
      - CSV containing motion statistics

Example:

python -m module6.partA_optical_flow --input data/videos/video1.mp4
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def create_video_writer(path, fps, width, height):
    """
    Create an MP4 video writer.
    """

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(path),
        fourcc,
        fps,
        (width, height),
    )

    return writer


def compute_dense_optical_flow(input_video, output_dir):

    input_video = Path(input_video)
    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # -----------------------------------------------------
    # OPEN VIDEO
    # -----------------------------------------------------

    cap = cv2.VideoCapture(str(input_video))

    if not cap.isOpened():
        raise FileNotFoundError(
            f"Could not open video: {input_video}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30.0

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )


    # -----------------------------------------------------
    # READ FIRST FRAME
    # -----------------------------------------------------

    success, previous_frame = cap.read()

    if not success:
        raise RuntimeError(
            "Could not read first frame."
        )

    previous_gray = cv2.cvtColor(
        previous_frame,
        cv2.COLOR_BGR2GRAY
    )


    # -----------------------------------------------------
    # OUTPUT FILES
    # -----------------------------------------------------

    video_name = input_video.stem

    hsv_output = (
        output_dir /
        f"{video_name}_flow_hsv.mp4"
    )

    arrow_output = (
        output_dir /
        f"{video_name}_flow_arrows.mp4"
    )

    csv_output = (
        output_dir /
        f"{video_name}_flow_statistics.csv"
    )


    hsv_writer = create_video_writer(
        hsv_output,
        fps,
        width,
        height
    )

    arrow_writer = create_video_writer(
        arrow_output,
        fps,
        width,
        height
    )


    # -----------------------------------------------------
    # HSV IMAGE
    # -----------------------------------------------------

    hsv = np.zeros_like(previous_frame)

    # Full saturation
    hsv[..., 1] = 255


    # Store numerical results
    statistics = []

    frame_number = 1


    # -----------------------------------------------------
    # PROCESS VIDEO
    # -----------------------------------------------------

    while True:

        success, frame = cap.read()

        if not success:
            break

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )


        # -------------------------------------------------
        # FARNEBACK OPTICAL FLOW
        # -------------------------------------------------

        flow = cv2.calcOpticalFlowFarneback(
            previous_gray,
            gray,
            None,
            pyr_scale=0.5,
            levels=4,
            winsize=21,
            iterations=3,
            poly_n=7,
            poly_sigma=1.5,
            flags=0,
        )


        # Horizontal component
        flow_x = flow[..., 0]

        # Vertical component
        flow_y = flow[..., 1]


        # -------------------------------------------------
        # MAGNITUDE AND DIRECTION
        # -------------------------------------------------

        magnitude, angle = cv2.cartToPolar(
            flow_x,
            flow_y,
            angleInDegrees=True,
        )


        # -------------------------------------------------
        # HSV VISUALIZATION
        # -------------------------------------------------
        #
        # Hue = direction
        # Value = magnitude
        # -------------------------------------------------

        hsv[..., 0] = (
            angle / 2
        ).astype(np.uint8)

        normalized_magnitude = cv2.normalize(
            magnitude,
            None,
            0,
            255,
            cv2.NORM_MINMAX,
        )

        hsv[..., 2] = (
            normalized_magnitude
        ).astype(np.uint8)


        optical_flow_color = cv2.cvtColor(
            hsv,
            cv2.COLOR_HSV2BGR
        )

        hsv_writer.write(
            optical_flow_color
        )


        # -------------------------------------------------
        # VECTOR / ARROW VISUALIZATION
        # -------------------------------------------------

        arrow_frame = frame.copy()

        step = max(
            15,
            min(width, height) // 30
        )

        arrow_scale = 4.0


        for y in range(
            step // 2,
            height,
            step
        ):

            for x in range(
                step // 2,
                width,
                step
            ):

                dx = flow[y, x, 0]
                dy = flow[y, x, 1]

                motion = np.sqrt(
                    dx ** 2 +
                    dy ** 2
                )


                # Only display visible motion
                if motion > 0.75:

                    start_point = (
                        int(x),
                        int(y)
                    )

                    end_point = (
                        int(x + arrow_scale * dx),
                        int(y + arrow_scale * dy),
                    )

                    cv2.arrowedLine(
                        arrow_frame,
                        start_point,
                        end_point,
                        (0, 255, 0),
                        1,
                        tipLength=0.25,
                    )


        arrow_writer.write(
            arrow_frame
        )


        # -------------------------------------------------
        # SAVE MOTION STATISTICS
        # -------------------------------------------------

        statistics.append(
            {
                "frame": frame_number,
                "time_seconds":
                    frame_number / fps,

                "mean_magnitude_pixels_per_frame":
                    float(
                        np.mean(magnitude)
                    ),

                "median_magnitude_pixels_per_frame":
                    float(
                        np.median(magnitude)
                    ),

                "maximum_magnitude_pixels_per_frame":
                    float(
                        np.max(magnitude)
                    ),
            }
        )


        previous_gray = gray

        frame_number += 1


    # -----------------------------------------------------
    # CLEANUP
    # -----------------------------------------------------

    cap.release()

    hsv_writer.release()

    arrow_writer.release()


    # -----------------------------------------------------
    # SAVE CSV
    # -----------------------------------------------------

    dataframe = pd.DataFrame(
        statistics
    )

    dataframe.to_csv(
        csv_output,
        index=False
    )


    print(
        "\nOptical flow processing complete."
    )

    print(
        f"HSV video: {hsv_output}"
    )

    print(
        f"Arrow video: {arrow_output}"
    )

    print(
        f"Statistics: {csv_output}"
    )


    return {
        "hsv_video": str(hsv_output),

        "arrow_video": str(arrow_output),

        "statistics_csv": str(csv_output),
    }


# ---------------------------------------------------------
# COMMAND LINE
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="Input MP4 video",
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/partA",
    )

    args = parser.parse_args()


    compute_dense_optical_flow(
        args.input,
        args.output_dir,
    )


if __name__ == "__main__":
    main()