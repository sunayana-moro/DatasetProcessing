# DatasetProcessing


Pipeline for preparing datasets used by faceshot training.

The workflow consists of two stages:

1. **Renderer Dataset Preparation**
2. **Generator Dataset Preparation**

All external dependencies are maintained as git submodules under `external/`.

---

# Repository Structure

```text
DatasetProcessing/
├── scripts/
│   ├── prepare_renderer_dataset.py
│   └── extract_generator.py
│
├── external/
│   ├── datasetprocess/
│   ├── Halfbody-new-identity-encoder/
│   └── ...
│
├── dataset/
│   ├── raw_videos/
│   ├── renderer_dataset/
│   └── generator_dataset/
│
└── README.md
```

---

# Setup

Clone the repository and initialize all submodules:

```bash
git submodule update --init --recursive
```

This downloads all dependencies stored under `external/`.

---

# Stage 1: Prepare Renderer Dataset

Converts raw videos into the renderer dataset used by IMTalker.

## Command

```bash
python scripts/prepare_renderer_dataset.py \
    --input_dataset /path/to/raw_videos \
    --output_dataset /path/to/renderer_dataset
```

## Example

```bash
cd ~/DatasetProcessing

python scripts/prepare_renderer_dataset.py \
    --input_dataset /teamspace/studios/this_studio/dataset/listner \
    --output_dataset /teamspace/studios/this_studio/dataset/renderer_dataset
```

---

## Input Format

```text
raw_videos/
├── video_0001.mp4
├── video_0002.mp4
└── ...
```

---

## Output Format

```text
renderer_dataset/
├── video_frame/
│   ├── video_0001/
│   │   ├── 000000.png
│   │   ├── 000001.png
│   │   └── ...
│   │
│   ├── video_0002/
│   └── ...
│
├── audios_16k/
│   ├── video_0001.wav
│   ├── video_0002.wav
│   └── ...
│
└── lmd/
    ├── video_0001.txt
    ├── video_0002.txt
    └── ...
```

---

## Processing Steps

The script internally runs the dataset processing pipeline:

```text
Raw Videos
    ↓
Scene Extraction
    ↓
Video Filtering
    ↓
Face Cropping
    ↓
Raw Data Extraction
    ↓
Landmark Extraction
    ↓
Renderer Dataset
```

using scripts located in:

```text
external/datasetprocess/
├── extract_scenes.py
├── filter_videos_rough.py
├── extract_cropped_faces.py
├── extract_raw_video_data.py
└── extract_frame_landmarks.py
```

---

## Notes

* Intermediate files are stored in a temporary directory.
* Temporary files are automatically deleted after completion.
* Output directories are created automatically if they do not exist.
* Paths are resolved relative to the repository root, so the script may be launched from any working directory.

---

# Stage 2: Prepare Generator Dataset

Generates motion, audio, smirk, and gaze features from the renderer dataset.

The renderer dataset produced in Stage 1 is used as input.

---

## Command

```bash
python scripts/extract_generator.py \
    /path/to/renderer_dataset \
    /path/to/generator_dataset \
    --wav2vec_model_path \
    /path/to/wav2vec2-base-960h
```

## Example

```bash
python scripts/extract_generator.py \
    /teamspace/studios/this_studio/dataset/test_render \
    /teamspace/studios/this_studio/dataset/test_ds \
    --wav2vec_model_path \
    /teamspace/studios/this_studio/DatasetProcessing/external/Halfbody-new-identity-encoder/IMTalker/checkpoints/wav2vec2-base-960h
```

---

## Expected Renderer Dataset Input

```text
renderer_dataset/
├── video_frame/
│   ├── video_0001/
│   ├── video_0002/
│   └── ...
│
└── audios_16k/
    ├── video_0001.wav
    ├── video_0002.wav
    └── ...
```

---

## Generator Dataset Output

```text
generator_dataset/
├── motion/
│   ├── video_0001.pt
│   ├── video_0002.pt
│   └── ...
│
├── audio/
│   ├── video_0001.npy
│   ├── video_0002.npy
│   └── ...
│
├── smirk/
│   ├── video_0001.pt
│   ├── video_0002.pt
│   └── ...
│
├── gaze/
│   ├── video_0001.npy
│   ├── video_0002.npy
│   └── ...
```

---

## Generated Features

### Motion

```text
motion/video_xxxx.pt
```

Motion representation extracted from the face sequence.

### Audio

```text
audio/video_xxxx.npy
```

Wav2Vec2 audio embeddings extracted from the corresponding audio file.

### Smirk

```text
smirk/video_xxxx.pt
```

SMIRK facial parameter representation.

### Gaze

```text
gaze/video_xxxx.npy
```

Per-frame gaze features.

---

# Complete Pipeline Example

```bash
# Clone repository
git clone <repo_url>
cd DatasetProcessing

# Initialize submodules
git submodule update --init --recursive

# Stage 1: Create renderer dataset
python scripts/prepare_renderer_dataset.py \
    --input_dataset /teamspace/studios/this_studio/dataset/listner \
    --output_dataset /teamspace/studios/this_studio/dataset/renderer_dataset

# Stage 2: Create generator dataset
python scripts/extract_generator.py \
    /teamspace/studios/this_studio/dataset/renderer_dataset \
    /teamspace/studios/this_studio/dataset/generator_dataset \
    --wav2vec_model_path \
    external/Halfbody-new-identity-encoder/IMTalker/checkpoints/wav2vec2-base-960h
```

---

# Final Dataset Flow

```text
Raw Videos
    ↓
prepare_renderer_dataset.py
    ↓
Renderer Dataset
(video_frame, audios_16k, lmd)
    ↓
extract_generator.py
    ↓
Generator Dataset
(motion, audio, smirk, gaze)
```
