import subprocess
from pathlib import Path

import imageio_ffmpeg


ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

input_folder = Path("outputs/partA")

videos = [
    (
        input_folder / "video1_flow_hsv.mp4",
        input_folder / "video1_flow_hsv_web.mp4",
    ),
    (
        input_folder / "video1_flow_arrows.mp4",
        input_folder / "video1_flow_arrows_web.mp4",
    ),
    (
        input_folder / "video2_flow_hsv.mp4",
        input_folder / "video2_flow_hsv_web.mp4",
    ),
    (
        input_folder / "video2_flow_arrows.mp4",
        input_folder / "video2_flow_arrows_web.mp4",
    ),
]


for source, destination in videos:

    if not source.exists():
        print(f"Skipping missing file: {source}")
        continue

    print(f"\nConverting {source.name}")

    command = [
        ffmpeg,
        "-y",
        "-i",
        str(source),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(destination),
    ]

    subprocess.run(
        command,
        check=True,
    )

    print(f"Created: {destination}")


print("\nAll available videos converted.")