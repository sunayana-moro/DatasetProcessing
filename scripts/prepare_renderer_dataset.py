#!/usr/bin/env python3
"""Prepare videos for renderer-style talking-face training data.

Example:
    python scripts/prepare_renderer_dataset.py \
      --input_dir /path/to/input_videos \
      --output_dir /tmp/renderer_dataset_test

Given a directory of videos, this script creates:

    output_dir/
      video_frame/
        video_0001/
          image_001.jpg
          image_002.jpg
      lmd/
        video_0001.txt

The landmark extraction step delegates to external/talking_face_preprocessing.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a directory of videos into video_frame/ and lmd/ renderer dataset format."
    )
    parser.add_argument("--input_dir", required=True, type=Path, help="Directory containing source videos.")
    parser.add_argument("--output_dir", required=True, type=Path, help="Directory to write the renderer dataset.")
    parser.add_argument(
        "--preprocessing_dir",
        type=Path,
        default=None,
        help="Path to talking_face_preprocessing. Defaults to ../external/talking_face_preprocessing.",
    )
    parser.add_argument("--fps", type=int, default=25, help="Frame rate to extract. Default: 25.")
    parser.add_argument("--jpeg_quality", type=int, default=2, help="ffmpeg JPEG qscale:v value. Lower is better.")
    parser.add_argument("--frame_digits", type=int, default=3, help="Minimum frame number width. Default: 3.")
    parser.add_argument("--video_digits", type=int, default=4, help="Minimum video directory number width. Default: 4.")
    parser.add_argument(
        "--preserve_names",
        action="store_true",
        help="Use source video stems as output clip names instead of video_0001, video_0002, ...",
    )
    parser.add_argument(
        "--crop_faces",
        action="store_true",
        help="Run talking_face_preprocessing/extract_cropped_faces.py before extracting frames.",
    )
    parser.add_argument(
        "--expanded_ratio",
        type=float,
        default=0.6,
        help="Face crop expansion ratio passed to extract_cropped_faces.py.",
    )
    parser.add_argument(
        "--skip_per_frame",
        type=int,
        default=25,
        help="Frame interval used by extract_cropped_faces.py for face box estimation.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Reuse existing extracted frames and landmark files when present.",
    )
    parser.add_argument(
        "--fail_on_bad_video",
        action="store_true",
        help="Stop immediately when an input video cannot be read or remuxed.",
    )
    parser.add_argument(
        "--check_and_padding",
        action="store_true",
        help="Pass --check_and_padding to landmark extraction to pad missing detections.",
    )
    parser.add_argument(
        "--skip_landmarks",
        action="store_true",
        help="Only extract frames; do not generate lmd/*.txt files.",
    )
    parser.add_argument(
        "--keep_work_dir",
        action="store_true",
        help="Keep output_dir/_work after processing. Useful for debugging cropped videos.",
    )
    return parser.parse_args()


def run(command: list[str], cwd: Path | None = None) -> None:
    printable = " ".join(command)
    print(f"$ {printable}", flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def require_executable(name: str) -> None:
    if shutil.which(name) is None:
        raise SystemExit(f"Required executable not found on PATH: {name}")


def discover_videos(input_dir: Path) -> list[Path]:
    videos = [
        path
        for path in sorted(input_dir.iterdir())
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    ]
    if not videos:
        supported = ", ".join(sorted(VIDEO_EXTENSIONS))
        raise SystemExit(f"No videos found in {input_dir}. Supported extensions: {supported}")
    return videos


def default_preprocessing_dir(script_path: Path) -> Path:
    repo_root = script_path.resolve().parents[1]
    return repo_root / "external" / "talking_face_preprocessing"


def make_clip_name(index: int, source: Path, preserve_names: bool, digits: int) -> str:
    if preserve_names:
        return source.stem
    return f"video_{index:0{digits}d}"


def prepare_named_videos(
    videos: list[Path],
    work_raw_dir: Path,
    preserve_names: bool,
    video_digits: int,
    fail_on_bad_video: bool,
) -> tuple[list[dict], list[dict]]:
    work_raw_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    skipped = []
    clip_index = 1
    for source in videos:
        clip_name = make_clip_name(clip_index, source, preserve_names, video_digits)
        target = work_raw_dir / f"{clip_name}.mp4"
        if target.exists():
            target.unlink()
        # Normalize container/name for downstream scripts while avoiding a lossy transcode here.
        command = ["ffmpeg", "-i", str(source), "-c", "copy", str(target), "-y", "-loglevel", "error"]
        print(f"$ {' '.join(command)}", flush=True)
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            target.unlink(missing_ok=True)
            skipped_item = {
                "source": str(source.resolve()),
                "reason": f"ffmpeg failed with exit code {result.returncode}",
            }
            if fail_on_bad_video:
                raise subprocess.CalledProcessError(result.returncode, command)
            print(f"Skipping unreadable video: {source}", flush=True)
            skipped.append(skipped_item)
            continue
        manifest.append({"clip": clip_name, "source": str(source.resolve()), "working_video": str(target)})
        clip_index += 1
    if not manifest:
        raise SystemExit("No valid videos were prepared. Check input files or rerun with --fail_on_bad_video.")
    return manifest, skipped


def crop_faces(preprocessing_dir: Path, raw_dir: Path, cropped_dir: Path, expanded_ratio: float, skip_per_frame: int) -> Path:
    cropped_dir.mkdir(parents=True, exist_ok=True)
    run(
        [
            sys.executable,
            str(preprocessing_dir / "extract_cropped_faces.py"),
            "--from_dir_prefix",
            str(raw_dir),
            "--output_dir_prefix",
            str(cropped_dir),
            "--expanded_ratio",
            str(expanded_ratio),
            "--skip_per_frame",
            str(skip_per_frame),
        ],
        cwd=preprocessing_dir,
    )
    return cropped_dir


def extract_frames(video_dir: Path, frames_dir: Path, fps: int, frame_digits: int, jpeg_quality: int, skip_existing: bool) -> None:
    frames_dir.mkdir(parents=True, exist_ok=True)
    for video in sorted(video_dir.glob("*.mp4")):
        clip_dir = frames_dir / video.stem
        if skip_existing and clip_dir.exists() and any(clip_dir.glob("image_*.jpg")):
            print(f"Skipping existing frames: {clip_dir}")
            continue
        if clip_dir.exists():
            shutil.rmtree(clip_dir)
        clip_dir.mkdir(parents=True)
        frame_pattern = clip_dir / f"image_%0{frame_digits}d.jpg"
        run(
            [
                "ffmpeg",
                "-i",
                str(video),
                "-vf",
                f"fps={fps}",
                "-q:v",
                str(jpeg_quality),
                str(frame_pattern),
                "-y",
                "-loglevel",
                "error",
            ]
        )


def extract_landmarks(
    preprocessing_dir: Path,
    frames_dir: Path,
    lmd_dir: Path,
    skip_existing: bool,
    check_and_padding: bool,
) -> None:
    lmd_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(preprocessing_dir / "extract_frame_landmarks.py"),
        "--from_dir",
        str(frames_dir),
        "--lmd_output_dir",
        str(lmd_dir),
    ]
    if skip_existing:
        command.append("--skip_existing")
    if check_and_padding:
        command.append("--check_and_padding")
    run(command, cwd=preprocessing_dir)


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    preprocessing_dir = (
        args.preprocessing_dir.resolve()
        if args.preprocessing_dir
        else default_preprocessing_dir(Path(__file__))
    )

    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")
    if not preprocessing_dir.is_dir():
        raise SystemExit(f"talking_face_preprocessing directory does not exist: {preprocessing_dir}")

    require_executable("ffmpeg")

    videos = discover_videos(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    work_dir = output_dir / "_work"
    raw_dir = work_dir / "raw_videos"
    cropped_dir = work_dir / "cropped_faces"
    frames_dir = output_dir / "video_frame"
    lmd_dir = output_dir / "lmd"

    manifest, skipped = prepare_named_videos(
        videos,
        raw_dir,
        args.preserve_names,
        args.video_digits,
        args.fail_on_bad_video,
    )
    videos_for_frames = (
        crop_faces(preprocessing_dir, raw_dir, cropped_dir, args.expanded_ratio, args.skip_per_frame)
        if args.crop_faces
        else raw_dir
    )

    extract_frames(videos_for_frames, frames_dir, args.fps, args.frame_digits, args.jpeg_quality, args.skip_existing)

    if not args.skip_landmarks:
        extract_landmarks(preprocessing_dir, frames_dir, lmd_dir, args.skip_existing, args.check_and_padding)

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    skipped_path = output_dir / "skipped_videos.json"
    if skipped:
        skipped_path.write_text(json.dumps(skipped, indent=2) + "\n", encoding="utf-8")

    if not args.keep_work_dir:
        shutil.rmtree(work_dir, ignore_errors=True)

    print(f"Done. Frames: {frames_dir}")
    if not args.skip_landmarks:
        print(f"Done. Landmarks: {lmd_dir}")
    print(f"Manifest: {manifest_path}")
    if skipped:
        print(f"Skipped videos: {skipped_path}")


if __name__ == "__main__":
    main()
