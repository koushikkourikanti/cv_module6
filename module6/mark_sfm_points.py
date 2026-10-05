"""
Mark the same four object boundary corners
in all four SfM images.

Click in this exact order:

1. Top-left
2. Top-right
3. Bottom-right
4. Bottom-left

The program saves the pixel coordinates.
"""

import json

import cv2

from module6.config import (
    SFM_FILES,
    SFM_DIR,
)


def collect_four_points(
    image,
    title,
):

    points = []


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

            and

            len(points) < 4
        ):

            points.append(
                [
                    float(x),
                    float(y)
                ]
            )


            print(
                f"Point {len(points)}:",
                x,
                y,
            )


    cv2.namedWindow(
        title,
        cv2.WINDOW_NORMAL
    )


    cv2.setMouseCallback(
        title,
        mouse_callback
    )


    while True:

        display = image.copy()


        for index, point in enumerate(
            points,
            start=1,
        ):

            x = int(point[0])

            y = int(point[1])


            cv2.circle(
                display,
                (x, y),
                7,
                (0, 0, 255),
                -1,
            )


            cv2.putText(
                display,
                str(index),
                (
                    x + 10,
                    y - 10,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )


        cv2.imshow(
            title,
            display
        )


        key = (
            cv2.waitKey(30)
            &
            0xFF
        )


        if len(points) == 4:

            cv2.waitKey(500)

            cv2.destroyWindow(
                title
            )

            return points


        if key == 27:

            cv2.destroyWindow(
                title
            )

            raise SystemExit(
                "Point selection cancelled."
            )


def main():

    collected_points = {}


    for image_path in SFM_FILES:

        if not image_path.exists():

            raise FileNotFoundError(
                f"Missing {image_path}"
            )


        image = cv2.imread(
            str(image_path)
        )


        title = (

            f"{image_path.name}: "

            "click TL, TR, BR, BL"
        )


        points = collect_four_points(
            image,
            title,
        )


        collected_points[
            image_path.name
        ] = points


    output_file = (

        SFM_DIR

        /

        "boundary_points.json"
    )


    output_file.write_text(

        json.dumps(
            collected_points,
            indent=4,
        )
    )


    print(
        "\nBoundary points saved:"
    )

    print(
        output_file
    )


if __name__ == "__main__":

    main()