import json

import pandas as pd
import streamlit as st

from module6.config import (
    VIDEO_FILES,
    SFM_FILES,
    OUTPUT_A,
    OUTPUT_B,
    K,
    DIST_COEFFS,
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


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="CSc 8830 - Assignment 6",
    layout="wide",
)

st.title(
    "CSc 8830: Computer Vision"
)

st.subheader(
    "Assignment 6"
)

st.caption(
    "Optical Flow, Motion Tracking, "
    "Bilinear Interpolation and "
    "Structure from Motion"
)


# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3 = st.tabs(
    [
        "Part A - Optical Flow",
        "Part B - Structure from Motion",
        "Theory and Equations",
    ]
)


# =========================================================
# PART A
# =========================================================

with tab1:

    st.header(
        "Part A - Optical Flow and Motion Tracking"
    )

    st.write(
        "Two videos are processed using dense optical flow "
        "and Lucas-Kanade feature tracking."
    )

    for index, video_path in enumerate(
        VIDEO_FILES,
        start=1,
    ):

        st.divider()

        st.subheader(
            f"Video {index}"
        )

        if video_path.exists():

            st.markdown(
                "### Original Video"
            )

            st.video(
                str(video_path)
            )

            column1, column2 = st.columns(2)

            # ---------------------------------------------
            # OPTICAL FLOW BUTTON
            # ---------------------------------------------

            with column1:

                if st.button(
                    f"Compute Optical Flow - Video {index}",
                    key=f"flow_button_{index}",
                ):

                    with st.spinner(
                        "Calculating dense optical flow..."
                    ):

                        compute_dense_optical_flow(
                            video_path,
                            OUTPUT_A,
                        )

                    st.success(
                        "Optical flow calculation complete."
                    )

                    st.info(
                        "Run `python convert_videos.py` "
                        "if browser-compatible videos "
                        "have not yet been generated."
                    )

            # ---------------------------------------------
            # TRACKING BUTTON
            # ---------------------------------------------

            with column2:

                frame_number = st.number_input(
                    f"Select first frame for Video {index}",
                    min_value=0,
                    value=0,
                    step=1,
                    key=f"frame_number_{index}",
                )

                if st.button(
                    f"Track Consecutive Frames - Video {index}",
                    key=f"tracking_button_{index}",
                ):

                    with st.spinner(
                        "Tracking features using Lucas-Kanade..."
                    ):

                        validate_tracking(
                            video_path,
                            OUTPUT_A,
                            int(frame_number),
                        )

                    st.success(
                        "Feature tracking complete."
                    )

            # =================================================
            # FILE PATHS
            # =================================================

            hsv_video_web = (
                OUTPUT_A
                /
                f"{video_path.stem}_flow_hsv_web.mp4"
            )

            arrow_video_web = (
                OUTPUT_A
                /
                f"{video_path.stem}_flow_arrows_web.mp4"
            )

            hsv_video_original = (
                OUTPUT_A
                /
                f"{video_path.stem}_flow_hsv.mp4"
            )

            arrow_video_original = (
                OUTPUT_A
                /
                f"{video_path.stem}_flow_arrows.mp4"
            )

            statistics_csv = (
                OUTPUT_A
                /
                f"{video_path.stem}_flow_statistics.csv"
            )

            tracking_csv = (
                OUTPUT_A
                /
                f"{video_path.stem}_tracking_validation.csv"
            )

            tracking_image = (
                OUTPUT_A
                /
                f"{video_path.stem}_tracking_validation.jpg"
            )

            # =================================================
            # OPTICAL FLOW RESULTS
            # =================================================

            st.markdown(
                "## Optical Flow Results"
            )

            optical_column1, optical_column2 = st.columns(2)

            # ---------------------------------------------
            # DENSE OPTICAL FLOW
            # ---------------------------------------------

            with optical_column1:

                st.markdown(
                    "### Dense Optical Flow"
                )

                if hsv_video_web.exists():

                    st.video(
                        str(hsv_video_web)
                    )

                elif hsv_video_original.exists():

                    st.warning(
                        "Dense optical flow was generated, "
                        "but the browser-compatible version "
                        "has not been created yet."
                    )

                    st.caption(
                        "Run: python convert_videos.py"
                    )

                else:

                    st.info(
                        "Click Compute Optical Flow first."
                    )

            # ---------------------------------------------
            # MOTION VECTOR VIDEO
            # ---------------------------------------------

            with optical_column2:

                st.markdown(
                    "### Motion Vectors"
                )

                if arrow_video_web.exists():

                    st.video(
                        str(arrow_video_web)
                    )

                elif arrow_video_original.exists():

                    st.warning(
                        "Motion-vector video was generated, "
                        "but the browser-compatible version "
                        "has not been created yet."
                    )

                    st.caption(
                        "Run: python convert_videos.py"
                    )

                else:

                    st.info(
                        "Click Compute Optical Flow first."
                    )

            # =================================================
            # FLOW STATISTICS
            # =================================================

            if statistics_csv.exists():

                st.markdown(
                    "## Optical Flow Statistics"
                )

                try:

                    flow_dataframe = pd.read_csv(
                        statistics_csv
                    )

                    st.dataframe(
                        flow_dataframe,
                        use_container_width=True,
                        height=350,
                    )

                    if not flow_dataframe.empty:

                        mean_motion = (
                            flow_dataframe[
                                "mean_magnitude_pixels_per_frame"
                            ]
                            .mean()
                        )

                        max_motion = (
                            flow_dataframe[
                                "maximum_magnitude_pixels_per_frame"
                            ]
                            .max()
                        )

                        summary1, summary2 = st.columns(2)

                        summary1.metric(
                            "Average Motion Magnitude",
                            f"{mean_motion:.4f} pixels/frame",
                        )

                        summary2.metric(
                            "Maximum Motion Magnitude",
                            f"{max_motion:.4f} pixels/frame",
                        )

                except Exception as error:

                    st.error(
                        f"Could not read statistics: {error}"
                    )

            # =================================================
            # FEATURE TRACKING RESULTS
            # =================================================

            st.markdown(
                "## Feature Tracking"
            )

            if tracking_image.exists():

                st.image(
                    str(tracking_image),
                    caption=(
                        "Lucas-Kanade feature tracking "
                        "between two consecutive frames"
                    ),
                )

            else:

                st.info(
                    "Click Track Consecutive Frames "
                    "to generate the tracking result."
                )

            if tracking_csv.exists():

                try:

                    tracking_dataframe = pd.read_csv(
                        tracking_csv
                    )

                    st.markdown(
                        "### Tracking Data"
                    )

                    st.dataframe(
                        tracking_dataframe,
                        use_container_width=True,
                        height=350,
                    )

                    if (
                        not tracking_dataframe.empty
                        and
                        "forward_backward_error_pixels"
                        in tracking_dataframe.columns
                    ):

                        average_error = (
                            tracking_dataframe[
                                "forward_backward_error_pixels"
                            ]
                            .mean()
                        )

                        st.metric(
                            "Average Forward-Backward Tracking Error",
                            f"{average_error:.4f} pixels",
                        )

                except Exception as error:

                    st.error(
                        f"Could not read tracking data: {error}"
                    )

            else:

                st.caption(
                    "Tracking CSV has not been generated yet."
                )

        else:

            st.warning(
                f"`{video_path.name}` has not been added yet."
            )


# =========================================================
# PART B
# =========================================================

with tab2:

    st.header(
        "Part B - Structure from Motion"
    )

    st.write(
        "Four images captured from different viewpoints "
        "are used to estimate camera motion and reconstruct "
        "3D points."
    )

    image_columns = st.columns(4)

    all_images_present = True

    for index, image_path in enumerate(
        SFM_FILES
    ):

        if image_path.exists():

            image_columns[index].image(
                str(image_path),
                caption=image_path.name,
                use_container_width=True,
            )

        else:

            image_columns[index].warning(
                f"Missing {image_path.name}"
            )

            all_images_present = False

    if all_images_present:

        st.success(
            "All four images are available."
        )

        if st.button(
            "Run Structure from Motion"
        ):

            with st.spinner(
                "Estimating camera poses and "
                "triangulating 3D points..."
            ):

                run_sfm(
                    SFM_FILES,
                    OUTPUT_B,
                )

            st.success(
                "Structure from Motion complete."
            )

    else:

        st.info(
            "Add view1.jpg, view2.jpg, "
            "view3.jpg and view4.jpg "
            "inside data/sfm_images."
        )

    # =====================================================
    # PART B OUTPUT FILES
    # =====================================================

    point_cloud_image = (
        OUTPUT_B
        /
        "sfm_point_cloud.png"
    )

    point_cloud_csv = (
        OUTPUT_B
        /
        "sparse_point_cloud.csv"
    )

    camera_pose_file = (
        OUTPUT_B
        /
        "camera_poses.json"
    )

    boundary_image = (
        OUTPUT_B
        /
        "boundary_reconstruction.png"
    )

    boundary_csv = (
        OUTPUT_B
        /
        "boundary_3d_points.csv"
    )

    # ---------------------------------------------
    # POINT CLOUD IMAGE
    # ---------------------------------------------

    if point_cloud_image.exists():

        st.markdown(
            "## Sparse 3D Reconstruction"
        )

        st.image(
            str(point_cloud_image),
            caption=(
                "Sparse Structure-from-Motion "
                "3D point cloud"
            ),
        )

    # ---------------------------------------------
    # POINT CLOUD TABLE
    # ---------------------------------------------

    if point_cloud_csv.exists():

        try:

            point_cloud_dataframe = pd.read_csv(
                point_cloud_csv
            )

            st.markdown(
                "### Reconstructed 3D Points"
            )

            st.dataframe(
                point_cloud_dataframe.head(500),
                use_container_width=True,
                height=350,
            )

        except Exception as error:

            st.error(
                f"Unable to read point cloud: {error}"
            )

    # ---------------------------------------------
    # CAMERA POSES
    # ---------------------------------------------

    if camera_pose_file.exists():

        try:

            st.markdown(
                "## Estimated Camera Poses"
            )

            camera_data = json.loads(
                camera_pose_file.read_text()
            )

            st.json(
                camera_data
            )

        except Exception as error:

            st.error(
                f"Unable to read camera poses: {error}"
            )

    # ---------------------------------------------
    # BOUNDARY RECONSTRUCTION
    # ---------------------------------------------

    if boundary_image.exists():

        st.markdown(
            "## Reconstructed Object Boundary"
        )

        st.image(
            str(boundary_image),
            caption=(
                "Triangulated planar-object boundary"
            ),
        )

    if boundary_csv.exists():

        try:

            boundary_dataframe = pd.read_csv(
                boundary_csv
            )

            st.markdown(
                "### Boundary 3D Coordinates"
            )

            st.dataframe(
                boundary_dataframe,
                use_container_width=True,
            )

        except Exception as error:

            st.error(
                f"Unable to read boundary data: {error}"
            )

    # ---------------------------------------------
    # CAMERA CALIBRATION
    # ---------------------------------------------

    st.markdown(
        "## Camera Calibration Parameters"
    )

    st.write(
        "The following intrinsic camera matrix and "
        "distortion coefficients are used for "
        "Structure from Motion."
    )

    st.code(
        f"""
Intrinsic Matrix K:

{K}

Distortion Coefficients:

{DIST_COEFFS}
"""
    )

    st.warning(
        "Use these calibration parameters only when "
        "the same camera, resolution and camera mode "
        "used during calibration are used for the "
        "Structure-from-Motion images."
    )


# =========================================================
# THEORY
# =========================================================

with tab3:

    st.header(
        "Theory and Mathematical Equations"
    )

    st.markdown(
        r"""

# Part A — Optical Flow

## 1. Brightness Constancy Assumption

The intensity of a moving image point is assumed to remain
approximately constant between two consecutive frames.

$$
I(x,y,t)
=
I(x+\Delta x,\,
y+\Delta y,\,
t+\Delta t)
$$

Using a first-order Taylor expansion:

$$
I_x\Delta x
+
I_y\Delta y
+
I_t\Delta t
=
0
$$

Dividing by $\Delta t$:

$$
I_x
\frac{\Delta x}{\Delta t}
+
I_y
\frac{\Delta y}{\Delta t}
+
I_t
=
0
$$

Define:

$$
u =
\frac{dx}{dt}
$$

and

$$
v =
\frac{dy}{dt}
$$

Therefore:

$$
\boxed{
I_xu + I_yv + I_t = 0
}
$$

This is the optical-flow constraint equation.


---

## 2. Motion Magnitude

For a motion vector:

$$
(u,v)
$$

the motion magnitude is:

$$
\boxed{
M =
\sqrt{
u^2 + v^2
}
}
$$

The direction can be represented as:

$$
\boxed{
\theta =
\tan^{-1}
\left(
\frac{v}{u}
\right)
}
$$


---

## 3. Lucas-Kanade Tracking

Lucas-Kanade assumes that neighboring pixels
inside a small image window have approximately
the same motion.

For multiple pixels:

$$
A
\begin{bmatrix}
u\\
v
\end{bmatrix}
=
b
$$

where:

$$
A =
\begin{bmatrix}
I_{x1} & I_{y1}\\
I_{x2} & I_{y2}\\
\vdots & \vdots\\
I_{xn} & I_{yn}
\end{bmatrix}
$$

and:

$$
b =
-
\begin{bmatrix}
I_{t1}\\
I_{t2}\\
\vdots\\
I_{tn}
\end{bmatrix}
$$

The least-squares solution is:

$$
\boxed{
\begin{bmatrix}
u\\
v
\end{bmatrix}
=
(A^TA)^{-1}A^Tb
}
$$


---

## 4. Two-Frame Motion Tracking

If a feature is located at:

$$
(x_t,y_t)
$$

and the estimated motion is:

$$
(u,v)
$$

then the predicted location is:

$$
\boxed{
x_{t+1}=x_t+u
}
$$

$$
\boxed{
y_{t+1}=y_t+v
}
$$

The validation error is:

$$
\boxed{
e =
\sqrt{
(x_{actual}-x_{predicted})^2
+
(y_{actual}-y_{predicted})^2
}
}
$$


---

## 5. Bilinear Interpolation

Suppose a sub-pixel location lies between
four neighboring pixels.

Let:

$$
\alpha=x-x_0
$$

and:

$$
\beta=y-y_0
$$

Then:

$$
\boxed{
I(x,y)
=
(1-\alpha)(1-\beta)I_{00}
+
\alpha(1-\beta)I_{10}
+
(1-\alpha)\beta I_{01}
+
\alpha\beta I_{11}
}
$$


# Part B — Structure from Motion


## 6. Perspective Camera Projection

A 3D point is projected onto an image using:

$$
s
\begin{bmatrix}
u\\
v\\
1
\end{bmatrix}
=
K[R|t]
\begin{bmatrix}
X\\
Y\\
Z\\
1
\end{bmatrix}
$$

where:

- $K$ = camera intrinsic matrix
- $R$ = camera rotation matrix
- $t$ = camera translation vector
- $(X,Y,Z)$ = 3D point
- $(u,v)$ = image pixel coordinates


---

## 7. Essential Matrix

For corresponding normalized image points:

$$
x_2^T E x_1 = 0
$$

where:

$$
\boxed{
E = [t]_{\times}R
}
$$

The essential matrix is used to recover
relative camera rotation and translation direction.


---

## 8. Triangulation

After the projection matrices are known:

$$
P_1
$$

and:

$$
P_2
$$

the 3D point satisfies:

$$
x_1 \times P_1X = 0
$$

and:

$$
x_2 \times P_2X = 0
$$

The resulting homogeneous system is solved
to reconstruct the 3D point.


---

## 9. Structure-from-Motion Scale Ambiguity

With a single moving camera, the recovered
translation and reconstructed geometry have
an unknown global scale.

Therefore, unless a known physical distance
is introduced, the reconstructed coordinates
are reported in relative units.

"""
    )

    st.info(
        "All external references used in the final "
        "assignment report should be properly cited."
    )