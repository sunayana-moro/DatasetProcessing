#!/usr/bin/env python3
"""Run motion, audio, and gaze extraction over a dataset in one pass.

Takes a renderer-style dataset (containing ``video_frame/<clip_id>/`` and, for
audio, ``audios_16k/``) and produces the generator dataset layout::

    <output_path>/
    ├── motion/<clip_id>.pt     (from extract_motion.py)
    ├── audio/<clip_id>.npy     (from extract_audio.py)
    └── gaze/<clip_id>.npy      (from extract_gaze.py)

Each stage is launched as its own subprocess so the three models load in
isolated processes (and a failure in one stage doesn't abort the others
unless --fail-fast is set).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_GENERATOR_DIR = Path(__file__).resolve().parent


def build_stage_cmd(
    script: str,
    dataset_path: str,
    output_path: str,
    device: str,
    overwrite: bool,
    extra: list[str],
) -> list[str]:
    cmd = [
        sys.executable,
        str(_GENERATOR_DIR / script),
        "--dataset_path",
        dataset_path,
        "--output_dir",
        output_path,
        "--device",
        device,
    ]
    if overwrite:
        cmd.append("--overwrite")
    cmd.extend(extra)
    return cmd


def run_stage(name: str, cmd: list[str]) -> int:
    print(f"\n{'=' * 70}\n[stage] {name}\n[cmd]   {' '.join(cmd)}\n{'=' * 70}", flush=True)
    result = subprocess.run(cmd)
    status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    print(f"[stage] {name}: {status}", flush=True)
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run motion + audio + gaze extraction to build a generator dataset.",
    )
    parser.add_argument(
        "dataset_path",
        help="Renderer dataset root (contains video_frame/ and audios_16k/).",
    )
    parser.add_argument(
        "output_path",
        help="Destination root; motion/, audio/, gaze/ are written here.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device passed to every stage (cuda auto-falls back to cpu if unusable).",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=["motion", "audio", "gaze"],
        default=["motion", "audio", "gaze"],
        help="Subset of stages to run (default: all three).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute outputs even if they already exist.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop at the first stage that exits non-zero.",
    )
    # Per-stage resource overrides (defaults live in each extract_*.py).
    parser.add_argument("--renderer_path", help="motion: renderer.ckpt path.")
    parser.add_argument("--wav2vec_model_path", help="audio: wav2vec2-base-960h dir.")
    parser.add_argument("--l2cs_weights", help="gaze: L2CS .pkl weights path.")
    parser.add_argument("--audio_dir", help="audio: per-clip audio dir (default <dataset>/audios_16k).")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    if not (dataset_path / "video_frame").is_dir():
        parser.error(f"video_frame/ not found under dataset_path: {dataset_path}")

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    stage_extra: dict[str, list[str]] = {"motion": [], "audio": [], "gaze": []}
    if args.renderer_path:
        stage_extra["motion"] += ["--renderer_path", args.renderer_path]
    if args.wav2vec_model_path:
        stage_extra["audio"] += ["--wav2vec_model_path", args.wav2vec_model_path]
    if args.audio_dir:
        stage_extra["audio"] += ["--audio_dir", args.audio_dir]
    if args.l2cs_weights:
        stage_extra["gaze"] += ["--l2cs_weights", args.l2cs_weights]

    stage_scripts = {
        "motion": "../external/Halfbody-new-identity-encoder/IMTalker/generator/extract_audio.py",
        "audio": "../external/Halfbody-new-identity-encoder/IMTalker/generator/extract_audio.py",
        "gaze": "../external/Halfbody-new-identity-encoder/IMTalker/generator/extract_gaze.py",
    }

    results: dict[str, int] = {}
    for stage in args.stages:
        cmd = build_stage_cmd(
            stage_scripts[stage],
            str(dataset_path),
            str(output_path),
            args.device,
            args.overwrite,
            stage_extra[stage],
        )
        rc = run_stage(stage, cmd)
        results[stage] = rc
        if rc != 0 and args.fail_fast:
            print(f"\n[abort] --fail-fast: stopping after failed stage '{stage}'.", flush=True)
            break

    print(f"\n{'=' * 70}\nSummary (output: {output_path})")
    for stage in args.stages:
        if stage in results:
            mark = "ok" if results[stage] == 0 else f"FAILED ({results[stage]})"
            print(f"  {stage:<7} -> {mark}")
        else:
            print(f"  {stage:<7} -> not run")
    print("=" * 70)

    if any(rc != 0 for rc in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
