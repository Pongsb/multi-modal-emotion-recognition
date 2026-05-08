# Multi-Modal Emotion Recognition

## Overview

This repository contains a full-stack multi-modal emotion recognition system that combines video and audio signals for emotion classification. The project includes dataset preprocessing utilities, feature extraction pipelines, deep learning training scripts, a FastAPI inference backend, a React/Vite frontend, and Docker-based deployment configuration.

The system is designed around a two-stream pipeline:

1. Extract visual representations from video frames.
2. Extract audio representations from speech signals.
3. Fuse both modalities using an attention-based cross-modal model.
4. Predict emotion probabilities across six standardized emotion classes.

The repository is organized for public technical review. Large datasets, generated feature files, model checkpoints, raw media files, and runtime artifacts should be excluded from version control unless they are intentionally provided as small reproducible samples.

## Technical Focus

Multi-modal emotion recognition combines computer vision, speech representation learning, deep learning, and web application deployment. This project demonstrates how a machine learning pipeline can be connected to a usable inference application through a backend API and frontend interface.

Core technical areas include:

- Video feature extraction from frame sequences
- Audio feature extraction from speech signals
- Cross-modal fusion for video-audio representation learning
- Emotion classification with deep neural networks
- Dataset preprocessing for RAVDESS and CREMA-D style datasets
- Face bounding-box extraction and audio conversion workflows
- FastAPI-based model inference service
- React/Vite frontend for video upload and prediction display
- Docker and Docker Compose deployment

## Skills & Technologies

![Python](https://img.shields.io/badge/Python-Deep%20Learning-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Model%20Training-orange)
![Transformers](https://img.shields.io/badge/Transformers-Feature%20Extraction-purple)
![FastAPI](https://img.shields.io/badge/FastAPI-Inference%20API-green)
![React](https://img.shields.io/badge/React-Frontend%20UI-blue)
![Vite](https://img.shields.io/badge/Vite-Frontend%20Build-yellow)
![Docker](https://img.shields.io/badge/Docker-Deployment-lightgrey)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-Video%20Features-darkgreen)
![Speech Processing](https://img.shields.io/badge/Speech%20Processing-Audio%20Features-red)

**Core skills:** `Python`, `PyTorch`, `Deep Learning`, `Multi-Modal Learning`, `Computer Vision`, `Speech Processing`, `Transformers`, `ViViT`, `Wav2Vec2`, `Cross-Modal Fusion`, `FastAPI`, `React`, `Vite`, `Tailwind CSS`, `Docker`, `Docker Compose`, `FFmpeg`, `Model Inference`, `Technical Documentation`

## System Pipeline

```text
Input Video
    │
    ├── Video stream
    │      └── Frame sampling / face-focused preprocessing
    │              └── ViViT-style video embedding extraction
    │
    ├── Audio stream
    │      └── Audio extraction with FFmpeg
    │              └── Wav2Vec2-based audio embedding extraction
    │
    └── Multi-modal model
           └── Attention-based cross-modal fusion
                   └── Emotion probability prediction
```

## Emotion Classes

The model uses a standardized six-class emotion label space.

| Class ID | Label | Emotion | Description |
|---:|---|---|---|
| 0 | `NEU` | Neutral | Calm or neutral emotional state |
| 1 | `HAP` | Happy | Positive or joyful emotional state |
| 2 | `SAD` | Sad | Sorrowful or unhappy emotional state |
| 3 | `ANG` | Angry | Angry or frustrated emotional state |
| 4 | `FEA` | Fearful | Fearful or anxious emotional state |
| 5 | `DIS` | Disgusted | Disgusted or repulsed emotional state |

### Dataset Label Mapping

| Dataset | Original Label Set | Standardization Strategy |
|---|---|---|
| RAVDESS | calm, happy, sad, angry, fearful, surprise, disgust, neutral | calm and neutral are mapped to `NEU`; surprise is excluded; remaining labels are mapped to the six-class schema |
| CREMA-D | happy, sad, anger, fear, disgust, neutral | labels are directly mapped to the six-class schema |

## Project Components

| Component | Location | Purpose | Main Technologies |
|---|---|---|---|
| Python feature extraction and training utilities | repository root | Prepare features, train the multi-modal model, and validate intermediate artifacts | `Python`, `PyTorch`, `Transformers`, `FFmpeg` |
| Backend API | `back-end/` | Serve model inference through HTTP endpoints | `FastAPI`, `Uvicorn`, `PyTorch` |
| Frontend application | `front-end/` | Upload videos and visualize emotion predictions | `React`, `Vite`, `Tailwind CSS` |
| Deployment configuration | root / service folders | Run services with Docker and Docker Compose | `Docker`, `Docker Compose`, `nginx` |

## Key Scripts

| Script | Purpose |
|---|---|
| `video_extractor.py` | Extracts video embeddings from frame sequences using a ViViT-style video representation pipeline |
| `voice_extractor.py` | Extracts audio emotion embeddings using a Wav2Vec2-based speech representation model |
| `train.py` | Trains the first version of the multi-modal emotion classifier |
| `train2.py` | Trains an enhanced model version with attention-based interpretability support |
| `test.py` | Validates extracted feature shapes and dataset consistency |
| `check.py` | Checks PyTorch and CUDA availability |
| `cremad_video_to_audio_converter.py` | Extracts audio from CREMA-D video files |
| `cremad_extract_bboxes.py` | Extracts face bounding boxes from CREMA-D videos |
| `cremad_bbox_converter.py` | Converts CREMA-D bounding-box outputs into a standardized training format |
| `ravdess_video_to_audio_converter.py` | Extracts audio from RAVDESS video files |
| `ravdess_extract_bboxes.py` | Extracts face bounding boxes from RAVDESS videos |
| `ravdess_bbox_converter.py` | Converts RAVDESS bounding-box outputs into a standardized training format |

## Model Overview

The project uses a multi-modal architecture that combines video and audio embeddings.

### Video Stream

- Extracts frame-level or sequence-level visual features.
- Uses a ViViT-style representation approach for spatio-temporal video modeling.
- Produces high-dimensional visual embeddings for downstream fusion.

### Audio Stream

- Extracts speech-based emotional representations from audio tracks.
- Uses a Wav2Vec2-based pretrained model for emotion-aware audio embeddings.
- Produces fixed-size audio vectors for fusion with video features.

### Fusion and Classification

- Combines video and audio features through a cross-modal fusion module.
- Uses attention-based fusion to integrate complementary modality information.
- Predicts confidence scores for all six emotion classes.
- Supports class-imbalance handling through focal-loss-style training.

## Repository Structure

```text
multi-modal-emotion-recognition/
├── README.md
├── back-end/
│   ├── app/
│   │   ├── main.py
│   │   ├── inference.py
│   │   ├── routers/
│   │   ├── libs/
│   │   └── api/
│   ├── requirements.txt
│   ├── Dockerfile
│   └── start.sh
├── front-end/
│   ├── src/
│   │   ├── App.tsx
│   │   ├── components/
│   │   ├── lib/
│   │   └── styles/
│   ├── package.json
│   ├── vite.config.ts
│   ├── tailwind.config.cjs
│   ├── Dockerfile
│   └── nginx.conf
├── video_extractor.py
├── voice_extractor.py
├── train.py
├── train2.py
├── test.py
├── check.py
├── cremad_extract_bboxes.py
├── cremad_video_to_audio_converter.py
├── cremad_bbox_converter.py
├── ravdess_extract_bboxes.py
├── ravdess_video_to_audio_converter.py
├── ravdess_bbox_converter.py
├── docker-compose.yml
├── docker-compose.dev.yml
├── nginx.conf
├── requirements.txt
└── pyproject.toml
```

## Excluded Runtime Artifacts

The following directories or file types are typically generated during preprocessing, training, or inference and should not be committed unless intentionally reduced to small reproducible samples:

```text
audio_features/
video_features/
extracted_audio/
extracted_bboxes/
extracted_faces_videos/
models/
training_runs/
training_runs_2/
*.pth
*.pt
*.npy
*.mp4
*.flv
*.mp3
```

## Backend Setup

### Requirements

- Python 3.9+
- FFmpeg
- PyTorch-compatible environment
- CUDA-capable GPU for accelerated inference or training, when available

### Install and Run

```bash
cd back-end
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API is available at:

```text
http://localhost:8000
```

### Main API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/ping` | Basic health check |
| `GET` | `/health` | Service status check |
| `POST` | `/infer` | Upload a video file for emotion inference |
| `POST` | `/infer/predict` | Upload a video file and return prediction scores |

Example request:

```bash
curl -F "file=@sample.mp4" http://localhost:8000/infer/predict
```

## Frontend Setup

### Requirements

- Node.js 18+
- npm or yarn

### Install and Run

```bash
cd front-end
npm ci
npm run dev
```

The development app is available at:

```text
http://localhost:5173
```

### Production Build

```bash
cd front-end
npm run build
```

Build artifacts are generated in:

```text
front-end/dist/
```

## Docker Deployment

### Production Compose

```bash
docker-compose up --build -d
```

Stop services:

```bash
docker-compose down
```

Typical local access points:

```text
Frontend: http://localhost
API route: http://localhost/api
Direct backend: http://localhost:8000
```

### Development Compose

```bash
docker-compose -f docker-compose.dev.yml up --build
```

View logs:

```bash
docker-compose -f docker-compose.dev.yml logs -f
```

## Reproducible Workflow

### 1. Prepare Dataset Assets

```bash
python ravdess_video_to_audio_converter.py
python ravdess_extract_bboxes.py
python ravdess_bbox_converter.py
```

For CREMA-D style preprocessing:

```bash
python cremad_video_to_audio_converter.py
python cremad_extract_bboxes.py
python cremad_bbox_converter.py
```

### 2. Extract Features

```bash
python video_extractor.py
python voice_extractor.py
```

### 3. Validate Feature Files

```bash
python test.py
```

### 4. Check GPU Environment

```bash
python check.py
```

### 5. Train the Model

```bash
python train2.py --epochs 50 --batch-size 32 --learning-rate 1e-4
```

### 6. Run the Application

```bash
docker-compose up --build
```

Then open:

```text
http://localhost
```

## Results Summary

The repository documents an end-to-end implementation pipeline for video-audio emotion recognition. It includes preprocessing scripts, feature extraction utilities, model training code, backend inference endpoints, frontend visualization, and Docker deployment support.

Quantitative results should be reported in project logs or result summaries when available. If model checkpoints or benchmark outputs are not included, they can be regenerated using the preprocessing and training workflow above.

## Limitations and Notes

- Large datasets such as RAVDESS and CREMA-D are not included.
- Generated `.npy` feature files and trained model weights should normally be excluded from Git.
- Local performance depends on GPU availability, CUDA configuration, batch size, and sampled frame count.
- Model predictions should be interpreted as confidence scores, not definitive psychological assessments.
- Dataset licensing and usage terms should be reviewed before redistributing any dataset-derived assets.

## Suggested `.gitignore`

```gitignore
# Python
__pycache__/
*.pyc
.env
.venv/
venv/

# Node / frontend
node_modules/
front-end/dist/

# Build and runtime files
build/
dist/
*.log
*.out
*.err

# Model and generated ML artifacts
*.pth
*.pt
*.onnx
*.npy
models/
training_runs/
training_runs_2/

# Dataset and media artifacts
audio_features/
video_features/
extracted_audio/
extracted_bboxes/
extracted_faces_videos/
*.mp4
*.flv
*.mp3
*.wav

# Docker / local override files
docker-compose.override.yml

# Editor / OS
.DS_Store
Thumbs.db
.vscode/
.idea/
*.swp
```

## Skills / Tags

`Python` `PyTorch` `Deep Learning` `Multi-Modal Learning` `Computer Vision` `Speech Processing` `Transformers` `ViViT` `Wav2Vec2` `Cross-Modal Fusion` `Focal Loss` `FastAPI` `Uvicorn` `React` `Vite` `Tailwind CSS` `Docker` `Docker Compose` `FFmpeg` `CUDA` `Model Inference` `Technical Documentation`
