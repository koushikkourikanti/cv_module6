"""
Manual Pixel Tracking Validation

The user clicks:

1. A feature in Frame t
2. The SAME physical feature in Frame t+1

Lucas-Kanade predicts where the point should move.

The program compares:

Predicted pixel position

vs.

Actual manually selected pixel position.

Validation error:

e = sqrt(
    (x_actual - x_predicted)^2
    +
    (y_actual - y_predicted)^2
)
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def read_frame_pair(
    video_path,
    frame_index
):

    cap = cv2.VideoCapture(
        str(video_path)
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
            "Unable to read selected frame pair."
        )


    return frame1, frame2


def select_pixel(
    image,
    window_name
):

    selected = []


    def mouse_callback(
        event,
        x,
        y,
        flags,
        parameter,
    ):

        if (
            event ==
            cv2.EVENT_LBUTTONDOWN
        ):

            selected.append(
                (
                    float(x),
                    float(y)
                )
            )


    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL
    )


    cv2.setMouseCallback(
        window_name,
        mouse_callback
    )


    while True:

        display = image.copy()


        if selected:

            x = int(
                selected[-1][0]
            )

            y = int(
                selected[-1][1]
            )


            cv2.circle(
                display,
                (x, y),
                7,
                (0, 0, 255),
                -1,
            )


        cv2.imshow(
            window_name,
            display
        )


        key = (
            cv2.waitKey(30)
            &
            0xFF
        )


        if selected:

            cv2.waitKey(300)

            cv2.destroyWindow(
                window_name
            )

            return selected[-1]


        if key == 27:

            cv2.destroyWindow(
                window_name
            )

            return None


def run_manual_validation(
    video_path,
    frame_index,
    output_csv,
):

    frame1, frame2 = (
        read_frame_pair(
            video_path,
            frame_index,
        )
    )


    gray1 = cv2.cvtColor(
        frame1,
        cv2.COLOR_BGR2GRAY
    )


    gray2 = cv2.cvtColor(
        frame2,
        cv2.COLOR_BGR2GRAY
    )


    results = []

    point_number = 1


    while True:

        point_frame1 = select_pixel(

            frame1,

            (
                f"Frame {frame_index}: "
                f"click point {point_number}"
            ),
        )


        if point_frame1 is None:
            break


        actual_point_frame2 = select_pixel(

            frame2,

            (
                f"Frame {frame_index + 1}: "
                f"click SAME point {point_number}"
            ),
        )


        if actual_point_frame2 is None:
            break


        # -------------------------------------------------
        # LUCAS-KANADE PREDICTION
        # -------------------------------------------------

        initial_point = np.array(
            [[point_frame1]],
            dtype=np.float32,
        )


        predicted_point, status, lk_error = (
            cv2.calcOpticalFlowPyrLK(

                gray1,
                gray2,

                initial_point,

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


        if status[0][0] != 1:

            print(
                "Tracking failed for this point."
            )

            continue


        predicted_x = float(
            predicted_point[0][0][0]
        )

        predicted_y = float(
            predicted_point[0][0][1]
        )


        x1 = point_frame1[0]

        y1 = point_frame1[1]


        actual_x = (
            actual_point_frame2[0]
        )

        actual_y = (
            actual_point_frame2[1]
        )


        u = predicted_x - x1

        v = predicted_y - y1


        error = np.sqrt(

            (
                actual_x -
                predicted_x
            ) ** 2

            +

            (
                actual_y -
                predicted_y
            ) ** 2
        )


        results.append(
            {
                "point_id":
                    point_number,

                "x_frame_t":
                    x1,

                "y_frame_t":
                    y1,

                "u":
                    u,

                "v":
                    v,

                "predicted_x_frame_t1":
                    predicted_x,

                "predicted_y_frame_t1":
                    predicted_y,

                "actual_x_frame_t1":
                    actual_x,

                "actual_y_frame_t1":
                    actual_y,

                "validation_error_pixels":
                    float(error),

                "lucas_kanade_error":
                    float(
                        lk_error[0][0]
                    ),
            }
        )


        print(
            f"\nPoint {point_number}"
        )

        print(
            "Predicted:",
            predicted_x,
            predicted_y,
        )

        print(
            "Actual:",
            actual_x,
            actual_y,
        )

        print(
            "Pixel error:",
            error,
        )


        point_number += 1


    # -----------------------------------------------------
    # SAVE RESULTS
    # -----------------------------------------------------

    if results:

        output_csv = Path(
            output_csv
        )


        output_csv.parent.mkdir(
            parents=True,
            exist_ok=True
        )


        dataframe = pd.DataFrame(
            results
        )


        dataframe.to_csv(
            output_csv,
            index=False
        )


        print(
            f"\nSaved results to {output_csv}"
        )

    else:

        print(
            "No points were saved."
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

        "--output",

        default=(
            "outputs/partA/"
            "manual_tracking_validation.csv"
        ),
    )


    args = parser.parse_args()


    run_manual_validation(
        args.input,
        args.frame,
        args.output,
    )


if __name__ == "__main__":
    main()