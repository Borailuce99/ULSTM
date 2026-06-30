# ULSTM — U-Net + ConvLSTM para Detección de Anomalías en Video

Implementación de una arquitectura **U-Net con ConvLSTM** para predicción de fotogramas y detección de anomalías en secuencias de video, basada en el paper [*Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting*](https://arxiv.org/pdf/1805.11247).

## Arquitectura

**ULSTM** combina una U-Net con capas ConvLSTM en los skip connections del encoder:

```
Entrada: [B, T, 3, H, W]  (T = 5 fotogramas)
  │
  ├─ Encoder (DownSample × 4)
  │   ├─ ConvLSTM → GroupNorm → Conv2D → Skip-connection
  │   └─ Down-conv (stride 2)
  │
  ├─ Bottleneck (DoubleConv 512 → 1024)
  │
  ├─ Decoder (UpSample × 4)
  │   ├─ Upsampling bilineal
  │   ├─ Skip-connection + ConvLSTM
  │   └─ DoubleConv
  │
  └─ Salida: Conv2D(64 → 3) + Sigmoid → [B, T, 3, H, W]
```

### Componentes

| Módulo | Descripción |
|--------|-------------|
| `ConvLSTMCell` | Celda ConvLSTM con compuertas (input, forget, output) y conexiones peephole |
| `ConvLSTM` | Capa ConvLSTM que procesa la secuencia temporal paso a paso |
| `DownSample` | ConvLSTM + GroupNorm + Conv2D + downsampling (stride 2) |
| `UpSample` | Upsampling bilineal + skip + ConvLSTM + DoubleConv |
| `DoubleConv` | Dos convoluciones 3×3 con GroupNorm y LeakyReLU |

## Estructura del proyecto

```
├── config.py                    # Configuración (rutas, hiperparámetros)
├── train.py                     # Entrenamiento con validación y test
├── inferenceAnomaly_ULSTM.py    # Inferencia en secuencias con anomalías
├── inferenceNormal_ULSTM.py     # Inferencia en secuencias normales
├── Dockerfile                   # Imagen Docker con CUDA 12.9 + PyTorch
├── docker-compose.yml           # Orquestación del contenedor
├── utils/
│   ├── __init__.py
│   ├── ConvLSTMCell.py          # Celda ConvLSTM
│   ├── ConvLSTM.py              # Capa ConvLSTM
│   └── ULSTM_Definitions.py     # Modelo ULSTM, Dataset, utilidades
└── .gitignore
```

## Requisitos

- **GPU NVIDIA** con soporte CUDA 12.9
- **Docker** con `nvidia-container-toolkit`

## Configuración

Edita `config.py`:

```python
DATASET = "/ruta/al/dataset.csv"       # CSV con rutas a imágenes
INF_DATASET = "/ruta/al/inferencia.csv"
cuda_device = "cuda:0"
num_epochs = 1000
batch_size = 2
lr = 1e-4
img_size = 256
```

### Formato del CSV

El CSV debe tener las columnas:
- `input_frames`: lista de rutas a los T fotogramas de entrada
- `output_frames`: lista de rutas a los T fotogramas objetivo
- `anomaly`: bandera de anomalía (0/1)

## Uso con Docker

```bash
# Crear archivo .env
echo "USER_ID=$(id -u)
USER_NAME=$(whoami)
GROUP_ID=$(id -g)
GROUP_NAME=$(id -gn)" > .env

# Construir y ejecutar
docker compose up -d

# Entrar al contenedor
docker exec -it ulstm_borja bash

# Entrenar
python train.py

# Inferencia
python inferenceAnomaly_ULSTM.py
python inferenceNormal_ULSTM.py
```

## Entrenamiento

`train.py` realiza:
- Split 80/10/10 (train/val/test) del dataset
- Optimizador AdamW con LR scheduler (ReduceLROnPlateau)
- Loss: SmoothL1Loss
- Mixed precision (AMP) para RTX 50
- Clipping de gradientes (max_norm=1.0)
- Guardado del mejor modelo en `models/best_model`
- Visualización de predicciones cada época
- Feature maps del bottleneck cada época

## Inferencia

### `inferenceAnomaly_ULSTM.py`
- Calcula **PSNR**, **SSIM** y **LPIPS** por secuencia
- Guarda hidden states de ConvLSTM como grids y heatmaps overlay
- Exporta resultados a CSV

### `inferenceNormal_ULSTM.py`
- Similar al de anomalías pero filtra secuencias anómalas
- Agrupa subsecuencias normales consecutivas

## Métricas

- **PSNR** (Peak Signal-to-Noise Ratio)
- **SSIM** (Structural Similarity Index)
- **LPIPS** (Learned Perceptual Image Patch Similarity) — solo anomalías

## Licencia

Proyecto académico.
