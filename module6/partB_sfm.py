"""
CSc 8830 Assignment 6
Part B - Structure from Motion

Four-view SfM pipeline:

1. Read four images.
2. Undistort images.
3. Detect SIFT features.
4. Match features.
5. Estimate Essential Matrix.
6. Recover camera rotation R and translation t.
7. Triangulate 3D points.
8. Save:
      - feature matching images
      - camera poses
      - 3D point cloud CSV
      - 3D point cloud graph

NOTE:

Monocular Structure from Motion recovers geometry only
up to an unknown scale unless a known physical distance
is provided.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from module6.config import (
    K,
    DIST_COEFFS,
    SFM_FILES,
    OUTPUT_B,
)


def undistort_image(image):

    return cv2.undistort(
        image,
        K,
        DIST_COEFFS,
    )


def create_detector():

    if hasattr(
        cv2,
        "SIFT_create"
    ):

        detector = (
            cv2.SIFT_create(
                nfeatures=5000
            )
        )

        norm_type = (
            cv2.NORM_L2
        )

        detector_name = "SIFT"

    else:

        detector = (
            cv2.ORB_create(
                nfeatures=5000
            )
        )

        norm_type = (
            cv2.NORM_HAMMING
        )

        detector_name = "ORB"


    return (
        detector,
        norm_type,
        detector_name,
    )


def detect_features(
    image,
    detector,
):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )


    keypoints, descriptors = (
        detector.detectAndCompute(
            gray,
            None
        )
    )


    return (
        keypoints,
        descriptors
    )


def match_features(
    descriptors1,
    descriptors2,
    norm_type,
):

    matcher = cv2.BFMatcher(
        norm_type
    )


    raw_matches = (
        matcher.knnMatch(
            descriptors1,
            descriptors2,
            k=2
        )
    )


    good_matches = []


    for pair in raw_matches:

        if len(pair) != 2:
            continue


        first, second = pair


        # Lowe ratio test
        if (
            first.distance
            <
            0.75
            *
            second.distance
        ):

            good_matches.append(
                first
            )


    return good_matches


def triangulate_points(
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
                t.reshape(3, 1),
            ]
        )
    )


    homogeneous_points = (
        cv2.triangulatePoints(

            projection1,
            projection2,

            points1.T.astype(
                np.float64
            ),

            points2.T.astype(
                np.float64
            ),
        )
    )


    points_3d = (

        homogeneous_points[:3]

        /

        homogeneous_points[3]

    ).T


    return points_3d


def camera_center(
    R,
    t,
):

    center = (
        -R.T
        @
        t.reshape(3, 1)
    )


    return center.reshape(3)


def run_sfm(
    image_paths=None,
    output_dir=OUTPUT_B,
):

    if image_paths is None:

        image_paths = SFM_FILES


    image_paths = [
        Path(path)
        for path in image_paths
    ]


    output_dir = Path(
        output_dir
    )


    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    # -----------------------------------------------------
    # CHECK IMAGES
    # -----------------------------------------------------

    for path in image_paths:

        if not path.exists():

            raise FileNotFoundError(
                f"Missing image: {path}"
            )


    # -----------------------------------------------------
    # LOAD IMAGES
    # -----------------------------------------------------

    images = []


    for path in image_paths:

        image = cv2.imread(
            str(path)
        )


        if image is None:

            raise RuntimeError(
                f"Could not read {path}"
            )


        image = undistort_image(
            image
        )


        images.append(
            image
        )


    # -----------------------------------------------------
    # FEATURE DETECTION
    # -----------------------------------------------------

    detector, norm_type, detector_name = (
        create_detector()
    )


    features = []


    for image in images:

        keypoints, descriptors = (
            detect_features(
                image,
                detector
            )
        )


        features.append(
            (
                keypoints,
                descriptors,
            )
        )


    # Use View 1 as reference camera
    reference_keypoints = (
        features[0][0]
    )

    reference_descriptors = (
        features[0][1]
    )


    if reference_descriptors is None:

        raise RuntimeError(
            "No features found in View 1."
        )


    # -----------------------------------------------------
    # CAMERA 1 = ORIGIN
    # -----------------------------------------------------

    camera_poses = [

        {
            "view": 1,

            "image":
                image_paths[0].name,

            "R":
                np.eye(3).tolist(),

            "t":
                [0.0, 0.0, 0.0],

            "camera_center_relative":
                [0.0, 0.0, 0.0],

            "note":
                "Reference camera",
        }
    ]


    reconstructed_points = []


    # -----------------------------------------------------
    # MATCH VIEW 1 WITH VIEWS 2, 3, 4
    # -----------------------------------------------------

    for view_index in range(
        1,
        len(images)
    ):

        keypoints2 = (
            features[view_index][0]
        )

        descriptors2 = (
            features[view_index][1]
        )


        if descriptors2 is None:

            print(
                f"No features in View "
                f"{view_index + 1}"
            )

            continue


        matches = match_features(

            reference_descriptors,

            descriptors2,

            norm_type,
        )


        print(
            f"View 1 -> View "
            f"{view_index + 1}: "
            f"{len(matches)} matches"
        )


        if len(matches) < 12:

            print(
                "Not enough matches. "
                "Skipping this view."
            )

            continue


        points1 = np.float64(

            [
                reference_keypoints[
                    match.queryIdx
                ].pt

                for match in matches
            ]
        )


        points2 = np.float64(

            [
                keypoints2[
                    match.trainIdx
                ].pt

                for match in matches
            ]
        )


        # -------------------------------------------------
        # ESSENTIAL MATRIX
        # -------------------------------------------------

        essential_matrix, mask = (
            cv2.findEssentialMat(

                points1,

                points2,

                K,

                method=cv2.RANSAC,

                prob=0.999,

                threshold=1.5,
            )
        )


        if essential_matrix is None:

            print(
                "Essential matrix failed."
            )

            continue


        # -------------------------------------------------
        # RECOVER CAMERA POSE
        # -------------------------------------------------

        _, R, t, pose_mask = (
            cv2.recoverPose(

                essential_matrix,

                points1,

                points2,

                K,
            )
        )


        inliers = (
            pose_mask.ravel() > 0
        )


        points1_inliers = (
            points1[inliers]
        )

        points2_inliers = (
            points2[inliers]
        )


        if (
            len(points1_inliers)
            <
            8
        ):

            print(
                "Too few pose inliers."
            )

            continue


        # -------------------------------------------------
        # TRIANGULATION
        # -------------------------------------------------

        points_3d = triangulate_points(

            points1_inliers,

            points2_inliers,

            R,

            t,
        )


        # Keep finite points
        finite = (
            np.isfinite(
                points_3d
            ).all(axis=1)
        )


        # Positive Z in reference camera
        positive_depth = (
            points_3d[:, 2] > 0
        )


        valid_points = (
            points_3d[
                finite
                &
                positive_depth
            ]
        )


        for point in valid_points:

            reconstructed_points.append(

                {
                    "reference_view":
                        1,

                    "paired_view":
                        view_index + 1,

                    "X_relative":
                        float(
                            point[0]
                        ),

                    "Y_relative":
                        float(
                            point[1]
                        ),

                    "Z_relative":
                        float(
                            point[2]
                        ),
                }
            )


        # -------------------------------------------------
        # CAMERA CENTER
        # -------------------------------------------------

        center = camera_center(
            R,
            t,
        )


        camera_poses.append(

            {
                "view":
                    view_index + 1,

                "image":
                    image_paths[
                        view_index
                    ].name,

                "R":
                    R.tolist(),

                "t":
                    t.reshape(-1).tolist(),

                "camera_center_relative":
                    center.tolist(),

                "good_matches":
                    len(matches),

                "pose_inliers":
                    int(
                        np.count_nonzero(
                            inliers
                        )
                    ),

                "detector":
                    detector_name,
            }
        )


        # -------------------------------------------------
        # MATCH VISUALIZATION
        # -------------------------------------------------

        matches_to_draw = matches[
            :
            min(
                80,
                len(matches)
            )
        ]


        match_image = (
            cv2.drawMatches(

                images[0],

                reference_keypoints,

                images[view_index],

                keypoints2,

                matches_to_draw,

                None,

                flags=(
                    cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                ),
            )
        )


        match_output = (

            output_dir

            /

            (
                f"matches_view1_"
                f"view{view_index + 1}.jpg"
            )
        )


        cv2.imwrite(
            str(match_output),
            match_image
        )


    # -----------------------------------------------------
    # CHECK RECONSTRUCTION
    # -----------------------------------------------------

    if not reconstructed_points:

        raise RuntimeError(

            "No 3D points were reconstructed. "
            "Use images with more texture, "
            "overlap, and viewpoint change."
        )


    # -----------------------------------------------------
    # SAVE POINT CLOUD
    # -----------------------------------------------------

    dataframe = pd.DataFrame(
        reconstructed_points
    )


    point_cloud_csv = (

        output_dir

        /

        "sparse_point_cloud.csv"
    )


    dataframe.to_csv(
        point_cloud_csv,
        index=False
    )


    # -----------------------------------------------------
    # SAVE CAMERA POSES
    # -----------------------------------------------------

    camera_pose_file = (

        output_dir

        /

        "camera_poses.json"
    )


    camera_pose_file.write_text(

        json.dumps(
            camera_poses,
            indent=4,
        )
    )


    # -----------------------------------------------------
    # CREATE 3D GRAPH
    # -----------------------------------------------------

    xyz = dataframe[
        [
            "X_relative",
            "Y_relative",
            "Z_relative",
        ]
    ].to_numpy()


    # Remove extreme points for display
    lower = np.percentile(
        xyz,
        2,
        axis=0,
    )


    upper = np.percentile(
        xyz,
        98,
        axis=0,
    )


    display_mask = np.all(

        (
            xyz >= lower
        )

        &

        (
            xyz <= upper
        ),

        axis=1,
    )


    display_points = (
        xyz[display_mask]
    )


    figure = plt.figure(
        figsize=(9, 7)
    )


    axis = figure.add_subplot(
        111,
        projection="3d"
    )


    axis.scatter(

        display_points[:, 0],

        display_points[:, 1],

        display_points[:, 2],

        s=4,
    )


    axis.set_title(
        "Four-View Structure from Motion"
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


    point_cloud_plot = (

        output_dir

        /

        "sfm_point_cloud.png"
    )


    figure.savefig(

        point_cloud_plot,

        dpi=180,
    )


    plt.close(
        figure
    )


    print(
        "\nStructure from Motion completed."
    )


    print(
        "Point cloud:",
        point_cloud_csv,
    )


    print(
        "Camera poses:",
        camera_pose_file,
    )


    print(
        "3D graph:",
        point_cloud_plot,
    )


    return {

        "point_cloud_csv":
            str(point_cloud_csv),

        "camera_poses_json":
            str(camera_pose_file),

        "point_cloud_plot":
            str(point_cloud_plot),
    }


if __name__ == "__main__":

    run_sfm()