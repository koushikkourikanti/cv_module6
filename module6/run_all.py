"""
Run all Assignment 6 processing
for input files that currently exist.

Missing files are skipped automatically.
"""

from module6.config import (
    VIDEO_FILES,
    SFM_FILES,
    OUTPUT_A,
    OUTPUT_B,
)

from module6.partA_optical_flow import (
    compute_dense_optical_flow,
)

from module6.partA_tracking_validation import (
    validate_tracking,
)

from module6.partB_sfm import (
    run_sfm,
)


def main():

    print(
        "\n===================================="
    )

    print(
        "CSc 8830 - Assignment 6"
    )

    print(
        "====================================\n"
    )


    # -----------------------------------------------------
    # PART A
    # -----------------------------------------------------

    for video in VIDEO_FILES:

        if video.exists():

            print(
                f"\nProcessing {video.name}"
            )


            compute_dense_optical_flow(

                video,

                OUTPUT_A,
            )


            validate_tracking(

                video,

                OUTPUT_A,

                frame_index=0,
            )


        else:

            print(
                f"Skipping missing video: "
                f"{video}"
            )


    # -----------------------------------------------------
    # PART B
    # -----------------------------------------------------

    all_images_present = all(

        image.exists()

        for image in SFM_FILES
    )


    if all_images_present:

        print(
            "\nRunning Structure from Motion..."
        )


        run_sfm(

            SFM_FILES,

            OUTPUT_B,
        )


    else:

        print(
            "\nSfM skipped."
        )

        print(
            "Add view1.jpg, view2.jpg, "
            "view3.jpg and view4.jpg."
        )


    print(
        "\nAvailable processing complete."
    )


if __name__ == "__main__":

    main()