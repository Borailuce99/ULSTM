# ULSTM — U-Net + ConvLSTM for Video Anomaly Detection

Implementation of a **U-Net with ConvLSTM** architecture for frame prediction and anomaly detection in video sequences, based on the paper [*Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting*](https://arxiv.org/pdf/1805.11247).

## Architecture

**ULSTM** combines a U-Net with ConvLSTM layers in the encoder skip connections:

```
Input: [B, T, 3, H, W]  (T = 5 frames)
  │
  ├─ Encoder (DownSample × 4)
  │   ├─ ConvLSTM → GroupNorm → Conv2D → Skip-connection
  │   └─ Down-conv (stride 2)
  │
  ├─ Bottleneck (DoubleConv 512 → 1024)
  │
  ├─ Decoder (UpSample × 4)
  │   ├─ Bilinear upsampling
  │   ├─ Skip-connection + ConvLSTM
  │   └─ DoubleConv
  │
  └─ Output: Conv2D(64 → 3) + Sigmoid → [B, T, 3, H, W]
```

### Components

| Module | Description |
|--------|-------------|
| `ConvLSTMCell` | ConvLSTM cell with input/forget/output gates and peephole connections |
| `ConvLSTM` | ConvLSTM layer processing the temporal sequence step by step |
| `DownSample` | ConvLSTM + GroupNorm + Conv2D + downsampling (stride 2) |
| `UpSample` | Bilinear upsampling + skip + ConvLSTM + DoubleConv |
| `DoubleConv` | Two 3×3 convolutions with GroupNorm and LeakyReLU |

## Project structure

```
├── config.py                    # Configuration (paths, hyperparameters)
├── train.py                     # Training with validation and test splits
├── inferenceAnomaly_ULSTM.py    # Inference on anomaly sequences
├── inferenceNormal_ULSTM.py     # Inference on normal sequences
├── Dockerfile                   # Docker image with CUDA 12.9 + PyTorch
├── docker-compose.yml           # Container orchestration
├── utils/
│   ├── __init__.py
│   ├── ConvLSTMCell.py          # ConvLSTM cell
│   ├── ConvLSTM.py              # ConvLSTM layer
│   └── ULSTM_Definitions.py     # ULSTM model, Dataset, utilities
└── .gitignore
```

## Requirements

- **NVIDIA GPU** with CUDA 12.9 support
- **Docker** with `nvidia-container-toolkit`

## Setup

Edit `config.py`:

```python
DATASET = "/path/to/dataset.csv"       # CSV with image paths
INF_DATASET = "/path/to/inference.csv"
cuda_device = "cuda:0"
num_epochs = 1000
batch_size = 2
lr = 1e-4
img_size = 256
```

### CSV format

The CSV must include the following columns:
- `input_frames`: list of paths to the T input frames
- `output_frames`: list of paths to the T target frames
- `anomaly`: anomaly flag (0/1)

## Usage with Docker

```bash
# Create .env file
echo "USER_ID=$(id -u)
USER_NAME=$(whoami)
GROUP_ID=$(id -g)
GROUP_NAME=$(id -gn)" > .env

# Build and run
docker compose up -d

# Enter the container
docker exec -it ulstm_borja bash

# Train
python train.py

# Inference
python inferenceAnomaly_ULSTM.py
python inferenceNormal_ULSTM.py
```

## Training

`train.py` performs:
- 80/10/10 train/val/test split
- AdamW optimizer with ReduceLROnPlateau scheduler
- Loss: SmoothL1Loss
- Mixed precision (AMP) for RTX 50 series
- Gradient clipping (max_norm=1.0)
- Best model saved to `models/best_model`
- Prediction visualization every epoch
- Bottleneck feature maps every epoch

## Inference

### `inferenceAnomaly_ULSTM.py`
- Computes **PSNR**, **SSIM** and **LPIPS** per sequence
- Saves ConvLSTM hidden states as grids and overlay heatmaps
- Exports results to CSV

### `inferenceNormal_ULSTM.py`
- Similar to anomaly script but filters out anomalous sequences
- Groups consecutive normal subsequences

## Metrics

- **PSNR** (Peak Signal-to-Noise Ratio)
- **SSIM** (Structural Similarity Index)
- **LPIPS** (Learned Perceptual Image Patch Similarity) — anomaly only

## License

Academic project.
