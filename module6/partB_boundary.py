import json

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from module6.config import (
    K,
    DIST_COEFFS,
    SFM_DIR,
    OUTPUT_B,
)


# =========================================================
# UNDISTORT PIXEL POINTS
# =========================================================

def undistort_pixel_points(points):

    points = np.asarray(
        points,
        dtype=np.float64,
    )

    points = points.reshape(
        -1,
        1,
        2
    )

    corrected = cv2.undistortPoints(
        points,
        K,
        DIST_COEFFS,
        P=K,
    )

    return corrected.reshape(
        -1,
        2
    )


# =========================================================
# TRIANGULATION
# =========================================================

def triangulate(
    points1,
    points2,
    R,
    t,
):

    projection1 = (
        K
        @
        np.hstack(
            [
                np.eye(3),
                np.zeros(
                    (3, 1)
                ),
            ]
        )
    )

    projection2 = (
        K
        @
        np.hstack(
            [
                R,
                t.reshape(
                    3,
                    1
                ),
            ]
        )
    )

    homogeneous = cv2.triangulatePoints(
        projection1,
        projection2,
        points1.T.astype(
            np.float64
        ),
        points2.T.astype(
            np.float64
        ),
    )

    points_3d = (
        homogeneous[:3]
        /
        homogeneous[3]
    ).T

    return points_3d


# =========================================================
# REPROJECT 3D POINTS
# =========================================================

def project_points(
    points_3d,
    R,
    t,
):

    rotation_vector, _ = cv2.Rodrigues(
        R
    )

    projected, _ = cv2.projectPoints(
        points_3d.astype(
            np.float64
        ),
        rotation_vector,
        t.reshape(
            3,
            1
        ).astype(
            np.float64
        ),
        K,
        np.zeros(
            5
        ),
    )

    return projected.reshape(
        -1,
        2
    )


# =========================================================
# MAIN BOUNDARY RECONSTRUCTION
# =========================================================

def run_boundary_reconstruction():

    # -----------------------------------------------------
    # FILE PATHS
    # -----------------------------------------------------

    boundary_file = (
        SFM_DIR
        /
        "boundary_points.json"
    )

    camera_pose_file = (
        OUTPUT_B
        /
        "camera_poses.json"
    )


    # -----------------------------------------------------
    # CHECK REQUIRED FILES
    # -----------------------------------------------------

    if not boundary_file.exists():

        raise FileNotFoundError(
            "boundary_points.json does not exist.\n"
            "Run:\n"
            "python -m module6.mark_sfm_points"
        )


    if not camera_pose_file.exists():

        raise FileNotFoundError(
            "camera_poses.json does not exist.\n"
            "Run:\n"
            "python -m module6.partB_sfm"
        )


    # -----------------------------------------------------
    # LOAD BOUNDARY CORRESPONDENCES
    # -----------------------------------------------------

    boundary_data = json.loads(
        boundary_file.read_text()
    )


    reference_image = "view1.jpg"


    if reference_image not in boundary_data:

        raise RuntimeError(
            "Boundary points for view1.jpg "
            "were not found."
        )


    points1 = np.array(
        boundary_data[
            reference_image
        ],
        dtype=np.float64,
    )


    points1 = undistort_pixel_points(
        points1
    )


    # -----------------------------------------------------
    # LOAD CAMERA POSES
    # -----------------------------------------------------

    pose_data = json.loads(
        camera_pose_file.read_text()
    )


    # -----------------------------------------------------
    # CHOOSE BEST AVAILABLE SECONDARY VIEW
    # -----------------------------------------------------
    #
    # View 4 had the strongest feature matching,
    # so prefer View 4 first.
    #
    # If View 4 is unavailable:
    # use View 3, then View 2.
    # -----------------------------------------------------

    selected_pose = None


    for preferred_view in [
        4,
        3,
        2,
    ]:

        for pose in pose_data:

            if pose.get(
                "view"
            ) == preferred_view:

                image_name = pose.get(
                    "image"
                )

                if (
                    image_name
                    in
                    boundary_data
                ):

                    selected_pose = pose

                    break


        if selected_pose is not None:

            break


    if selected_pose is None:

        raise RuntimeError(
            "No usable secondary camera pose "
            "was found."
        )


    selected_view = (
        selected_pose[
            "view"
        ]
    )


    selected_image = (
        selected_pose[
            "image"
        ]
    )


    print(
        "\nUsing the following image pair:"
    )

    print(
        f"Reference: {reference_image}"
    )

    print(
        f"Secondary: {selected_image}"
    )

    print(
        f"Recovered camera view: "
        f"{selected_view}"
    )


    # -----------------------------------------------------
    # LOAD SECONDARY VIEW BOUNDARY POINTS
    # -----------------------------------------------------

    points2 = np.array(
        boundary_data[
            selected_image
        ],
        dtype=np.float64,
    )


    points2 = undistort_pixel_points(
        points2
    )


    # -----------------------------------------------------
    # LOAD CAMERA POSE
    # -----------------------------------------------------

    R = np.array(
        selected_pose[
            "R"
        ],
        dtype=np.float64,
    )


    t = np.array(
        selected_pose[
            "t"
        ],
        dtype=np.float64,
    )


    # -----------------------------------------------------
    # TRIANGULATE 4 BOUNDARY CORNERS
    # -----------------------------------------------------

    points_3d = triangulate(
        points1,
        points2,
        R,
        t,
    )


    # -----------------------------------------------------
    # REPROJECT INTO REFERENCE CAMERA
    # -----------------------------------------------------

    reference_R = np.eye(3)

    reference_t = np.zeros(3)


    reprojection_view1 = project_points(
        points_3d,
        reference_R,
        reference_t,
    )


    # -----------------------------------------------------
    # REPROJECT INTO SECONDARY CAMERA
    # -----------------------------------------------------

    reprojection_view2 = project_points(
        points_3d,
        R,
        t,
    )


    # -----------------------------------------------------
    # CALCULATE REPROJECTION ERRORS
    # -----------------------------------------------------

    error_view1 = np.linalg.norm(
        reprojection_view1
        -
        points1,
        axis=1,
    )


    error_view2 = np.linalg.norm(
        reprojection_view2
        -
        points2,
        axis=1,
    )


    # -----------------------------------------------------
    # CORNER LABELS
    # -----------------------------------------------------

    labels = [
        "Top Left",
        "Top Right",
        "Bottom Right",
        "Bottom Left",
    ]


    rows = []


    for index in range(4):

        rows.append(
            {
                "corner":
                    labels[index],

                "reference_image":
                    reference_image,

                "secondary_image":
                    selected_image,

                "X_relative":
                    float(
                        points_3d[
                            index,
                            0
                        ]
                    ),

                "Y_relative":
                    float(
                        points_3d[
                            index,
                            1
                        ]
                    ),

                "Z_relative":
                    float(
                        points_3d[
                            index,
                            2
                        ]
                    ),

                "view1_reprojection_error_pixels":
                    float(
                        error_view1[
                            index
                        ]
                    ),

                "secondary_view_reprojection_error_pixels":
                    float(
                        error_view2[
                            index
                        ]
                    ),
            }
        )


    # -----------------------------------------------------
    # SAVE BOUNDARY POINTS CSV
    # -----------------------------------------------------

    dataframe = pd.DataFrame(
        rows
    )


    csv_output = (
        OUTPUT_B
        /
        "boundary_3d_points.csv"
    )


    dataframe.to_csv(
        csv_output,
        index=False
    )


    # -----------------------------------------------------
    # PLOT 3D BOUNDARY
    # -----------------------------------------------------

    boundary_order = [
        0,
        1,
        2,
        3,
        0,
    ]


    figure = plt.figure(
        figsize=(9, 7)
    )


    axis = figure.add_subplot(
        111,
        projection="3d"
    )


    axis.scatter(
        points_3d[:, 0],
        points_3d[:, 1],
        points_3d[:, 2],
        s=70,
    )


    axis.plot(
        points_3d[
            boundary_order,
            0
        ],
        points_3d[
            boundary_order,
            1
        ],
        points_3d[
            boundary_order,
            2
        ],
        linewidth=2,
    )


    for index, label in enumerate(
        labels
    ):

        axis.text(
            points_3d[
                index,
                0
            ],
            points_3d[
                index,
                1
            ],
            points_3d[
                index,
                2
            ],
            label,
        )


    axis.set_title(
        "Reconstructed Planar Object Boundary"
    )


    axis.set_xlabel(
        "X - Relative Units"
    )


    axis.set_ylabel(
        "Y - Relative Units"
    )


    axis.set_zlabel(
        "Z - Relative Units"
    )


    figure.tight_layout()


    image_output = (
        OUTPUT_B
        /
        "boundary_reconstruction.png"
    )


    figure.savefig(
        image_output,
        dpi=180,
    )


    plt.close(
        figure
    )


    # -----------------------------------------------------
    # TERMINAL SUMMARY
    # -----------------------------------------------------

    print(
        "\n======================================"
    )

    print(
        "BOUNDARY RECONSTRUCTION COMPLETE"
    )

    print(
        "======================================"
    )


    print(
        f"\nReference Image: "
        f"{reference_image}"
    )


    print(
        f"Secondary Image: "
        f"{selected_image}"
    )


    print(
        "\n3D Boundary Points:"
    )


    print(
        dataframe
    )


    mean_reference_error = float(
        np.mean(
            error_view1
        )
    )


    mean_secondary_error = float(
        np.mean(
            error_view2
        )
    )


    print(
        "\nMean Reference View "
        "Reprojection Error:"
    )

    print(
        f"{mean_reference_error:.4f} pixels"
    )


    print(
        "\nMean Secondary View "
        "Reprojection Error:"
    )

    print(
        f"{mean_secondary_error:.4f} pixels"
    )


    print(
        "\nFiles saved:"
    )

    print(
        csv_output
    )

    print(
        image_output
    )


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    run_boundary_reconstruction()