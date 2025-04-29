# Energy-Efficient AI Inference

A toolkit and pipeline for reducing the energy consumption of deep learning inference by combining model compression techniques—quantization, pruning, and knowledge distillation—on Vision Transformer (ViT) models. Benchmarked on CIFAR-100, this project uses Intel RAPL (CPU) and NVIDIA NVML (GPU) for energy measurement and provides reusable scripts for evaluation and training.

## Features

- **Energy measurement** via RAPL (CPU) and NVML (GPU)  
- **Quantization**: Post-training uint8 and float8 quantization using _quanto_  
- **Pruning**: Structural and unstructural pruning with architecture-aware options  
- **Knowledge distillation**: Recover accuracy lost to pruning  
- **Benchmarking** on CIFAR-100 with Vision Transformer (ViT)  
- Modular design for easy extension to other models and datasets  

## Repository Structure
- `energy-efficient-ai-inference/`
  - `src/`
    - `data/`
      - `dataset.py`
    - `energy/`
      - `energy_tracker.py`
      - `model_analysis.py`
    - `models/`
      - `pretrained.py`
      - `train.py`
      - `pruning/`
        - `structural_pruning.py`
        - `unstructural_pruning.py`
      - `train_utils/`
        - `loss.py`
        - `train_epoch.py`
        - `evaluate_epoch.py`
    - `evaluate_model.py`
    - `metrics.py`
    - `utils.py`
  - `requirements.txt`
  - `README.md`

## Installation

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/RaviShah1/energy-efficient-ai-inference.git
   cd energy-efficient-ai-inference
    ```
2. **Create Environment and Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```
3. **Machine Requirements**
    <details> <summary><strong>GPU & CUDA notes</strong></summary>
    The requirements.txt pins CUDA-enabled builds for PyTorch and related packages (e.g. torch==2.5.1+cu121). To run on GPU, ensure you have the matching NVIDIA driver and CUDA 12.1 toolkit installed. The GPU must be compatible with NVML power readings to get GPU energy measurements. 
    </details>
    <details> <summary><strong>CPU & RAPL notes</strong></summary>
    To measure CPU energy consumption, we use RAPL. Your machine must use an Intel Processor with RAPL energy tracker and permissions to effectively use our code.
    </details>

## Usage

### 1. Evaluate Inference & Measure Energy

```bash
# Basic FP32 (no pruning, no quantization)
python src/evaluate_model.py

# Structural pruning (10% channels)
python src/evaluate_model.py \
  --pruning_type structural \
  --pruning_ratio 0.1

# Unstructural pruning (10% weights)
python src/evaluate_model.py \
  --pruning_type unstructural \
  --pruning_ratio 0.1

# Quantize to 8-bit uint
python src/evaluate_model.py --quantization

# Load a saved checkpoint, prune & quantize
python src/evaluate_model.py \
  --weights path/to/weights.pth \
  --pruning_type structural \
  --pruning_ratio 0.1 \
  --quantization
```
**Flags:**
- `--pruning_type {structural,unstructural}` — type of pruning to apply
- `--pruning_ratio R` — fraction (0.0–1.0) to prune
- `--quantization` — enable uint8 quantization
- `--weights PATH` — path to .pth checkpoint

### 2. Training
```bash
# Standard training
python src/models/train.py \
  --mode train \
  --pruning_ratio 0.1 \
  --epochs 10 \
  --lr 3e-5 \
  --batch_size 16 \
  --save_prefix weights

# Knowledge distillation
python src/models/train.py \
  --mode distill \
  --pruning_ratio 0.1 \
  --epochs 10 \
  --lr 3e-5 \
  --batch_size 16 \
  --temperature 4.0 \
  --alpha 0.25 \
  --save_prefix distilled \
  --scheduler
```

**Flags:**
- `--mode {train,distill}`
- `--pruning_ratio R` — prune student before training
- `--epochs N` — number of epochs
- `--lr LR` — learning rate
- `--batch_size B` — batch size
- `--temperature T` — distillation softmax temperature
- -`-alpha A` — weight for distillation loss
- `--save_prefix PREFIX` — checkpoint filename prefix
- `--scheduler` — use cosine annealing LR scheduler

## Experiments and Results

We drafted a research paper based on our experiments

- [Reducing Energy Consumption in AI Inference (PDF)](EEC_Research_Paper.pdf)