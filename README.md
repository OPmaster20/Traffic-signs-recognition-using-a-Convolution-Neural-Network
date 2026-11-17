
### Directory / File Descriptions

| Item      | Description |
|-----------|-------------|
| `gpu/`    | Checks whether a CUDA-capable GPU is available locally, and handles device setup (CPU/GPU). |
| `model/`  | Stores model files such as trained weights and checkpoints. |
| `dmg/`    | Displays images from the dataset for inspection and visualization. |
| `app.py`  | Graphical user interface (GUI) that makes it easy to test the model interactively. |
| `main.py` | Main script used for training the ResNet18 model on the German traffic sign dataset. |

---

## Model

- **Architecture:** ResNet18
- **Framework:** PyTorch
- **Training environment:** CUDA

## Dataset

- **Name:** German Traffic Sign Recognition Benchmark (GTSRB)
- **Task:** Multi-class traffic sign classification

---

## Usage

### 1. Check GPU Availability

Run the script in `gpu/` to verify whether CUDA is available on your machine.

### 2. Preview Dataset Images

Use the scripts in `dmg/` to display sample images from the dataset and information about dataset.

### 3. Train the Model

```bash
python main.py
