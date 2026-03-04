# Brain MRI Segmentation Project

This project implements a Mask R-CNN model for segmentation tasks using PyTorch.

## Project Structure

- `train.py`: Main training script.
- `co2.ipynb`: Original research notebook.
- `requirements.txt`: Python dependencies.

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: On Windows, you might need a C++ compiler for `pycocotools`. If installation fails, try installing `git` and Visual Studio Build Tools, or look for pre-built wheels.*

2.  **Prepare Data**:
    Ensure your dataset is extracted in the root directory:
    - `train-300/`
    - `validation-300/`
    - `test-30/` (Note regarding test data: Ensure it follows the standard `data/` structure if running inference).

## Usage

**Train the model:**
```bash
python train.py --epochs 10 --batch-size 2
```

The script will automatically use CUDA if available, otherwise CPU.
Output models will be saved as `coco_model.pth`.

## Details

The model uses a pre-trained ResNet-50 backbone with FPN. It is fine-tuned on the provided custom dataset which follows the COCO format.
