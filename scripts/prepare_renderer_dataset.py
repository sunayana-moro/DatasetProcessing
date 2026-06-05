
#!/usr/bin/env python3

"""
Prepare renderer dataset from raw videos.

Repository layout:
------------------

repo_root/
├── scripts/
│   └── prepare_renderer_dataset.py
├── external/
│   └── datasetprocess/
│       ├── extract_scenes.py
│       ├── filter_videos_rough.py
│       ├── extract_cropped_faces.py
│       ├── extract_raw_video_data.py
│       └── extract_frame_landmarks.py
└── ...

How to run:
-----------

Recommended:

    cd <repo_root>

    python scripts/prepare_renderer_dataset.py \
        --input_dataset /path/to/raw_videos \
        --output_dataset /path/to/renderer_dataset

Example:

    cd ~/DatasetProcessing

    python scripts/prepare_renderer_dataset.py \
        --input_dataset ../datasettest \
        --output_dataset datasets/renderer_dataset

Output:
-------

renderer_dataset/
├── video_frame/
│   ├── video_0001/
│   ├── video_0002/
│   └── ...
└── lmd/
    ├── video_0001.txt
    ├── video_0002.txt
    └── ...

Notes:
------

1. Intermediate files are stored in a temporary directory and deleted
   automatically after completion.

2. Dataset processing scripts are expected to live under:

       external/datasetprocess/

3. Paths are resolved relative to this script's location, so the script
   can be launched from any working directory.
"""
import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path


# ============================================================================
# EDIT THIS
# ============================================================================

DATASET_PROCESS_ROOT = Path("external/talking_face_preprocessing")

# ============================================================================


def run(cmd):
    print("\n>>>", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_dataset",
        required=True,
    )

    parser.add_argument(
        "--output_dataset",
        required=True,
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dataset).resolve()
    output_dir = Path(args.output_dataset).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    video_frame_dir = output_dir / "video_frame"
    lmd_dir = output_dir / "lmd"

    video_frame_dir.mkdir(exist_ok=True)
    lmd_dir.mkdir(exist_ok=True)

    root = DATASET_PROCESS_ROOT.resolve()

    extract_scenes = root / "extract_scenes.py"
    filter_videos = root / "filter_videos_rough.py"
    crop_faces = root / "extract_cropped_faces.py"
    extract_raw = root / "extract_raw_video_data.py"
    extract_lmd = root / "extract_frame_landmarks.py"

    with tempfile.TemporaryDirectory() as tmp:

        tmp = Path(tmp)

        scene_dir = tmp / "scene_detected"
        filtered_dir = tmp / "filtered"
        cropped_dir = tmp / "cropped"

        standardized_root = tmp / "standardized"

        # ------------------------------------------------------
        # Scene detection
        # ------------------------------------------------------

        run([
            "python",
            str(extract_scenes),
            "--from_directory",
            str(input_dir),
            "--output_directory",
            str(scene_dir),
        ])

        # ------------------------------------------------------
        # Filtering
        # ------------------------------------------------------

        run([
            "python",
            str(filter_videos),
            "--before_filtering_dir",
            str(scene_dir),
            "--after_filtering_dir",
            str(filtered_dir),
            "--min_duration",
            "2",
            "--min_size",
            "10",
        ])

        # ------------------------------------------------------
        # Face crops
        # ------------------------------------------------------

        run([
            "python",
            str(crop_faces),
            "--from_dir_prefix",
            str(filtered_dir),
            "--output_dir_prefix",
            str(cropped_dir),
            "--expanded_ratio",
            "0.6",
        ])

        # ------------------------------------------------------
        # Frames
        # ------------------------------------------------------

        run([
            "python",
            str(extract_raw),
            "--source_folder",
            str(cropped_dir),
            "--video_target_folder",
            str(standardized_root / "videos_25fps"),
            "--audio_target_folder",
            str(standardized_root / "audios_16k"),
            "--frames_target_folder",
            str(video_frame_dir),
            "--convert_video",
            "True",
            "--convert_audio",
            "True",
            "--extract_frames",
            "True",
        ])

        # ------------------------------------------------------
        # Landmarks
        # ------------------------------------------------------

        run([
            "python",
            str(extract_lmd),
            "--from_dir",
            str(video_frame_dir),
            "--lmd_output_dir",
            str(lmd_dir),
            "--skip_existing",
        ])

    print()
    print("Dataset created:")
    print(output_dir)


if __name__ == "__main__":
    main()