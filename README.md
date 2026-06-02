# DefectFill – Assignment 3

Defect synthesis for industrial inspection using Stable Diffusion 2 inpainting with LoRA fine-tuning on the MVTec dataset.

---

## Requirements

### Environment
- Python 3.8+
- CUDA-capable GPU (16 GB+ VRAM recommended; tested on Kaggle P100)

### Install dependencies

```bash
pip install -r "DefectFill Repo/DefectFill/requirements.txt"
pip install lpips torchinfo torch-fidelity
```

---

## Dataset

The notebook expects the **curated MVTec** dataset at:

```
/kaggle/input/datasets/dahyuntw/curated-mvtec/curated_mvtec/
```

Directory structure:

```
curated_mvtec/
└── <object_class>/          # e.g. bottle
    ├── train/
    │   ├── good/
    │   └── defective/
    │       └── <defect_type>/   # e.g. broken_large
    └── test/
```

---

## Repository Setup

Add the `DefectFill` source directory to your Python path before importing modules:

```python
import sys
sys.path.append("/path/to/DefectFill Repo/DefectFill")
```

On Kaggle this path is:
```
/kaggle/input/datasets/dahyuntw/defectfill-model/DefectFill Repo/DefectFill
```

---

## Running the Notebook (`cv-a3.ipynb`)

Open `0. submission/cv-a3.ipynb` in Jupyter or Kaggle and run cells **top to bottom**. The sections are:

### 1. Configuration (Cell 4)
Edit the key variables before running anything else:

| Variable | Description | Default |
|---|---|---|
| `DATA_DIR` | Path to curated MVTec root | Kaggle path |
| `OUTPUT_DIR` | Where checkpoints & logs are saved | `/kaggle/working/` |
| `OBJ_CLASS` | Object category to train on | `"bottle"` |
| `DEFECT_TYPE` | Specific defect or `None` for all | `None` |
| `MAX_TRAIN_STEPS` | Total training steps | `500` |
| `BATCH_SIZE` | Images per batch | `2` |
| `RESUME_FROM` | Path to checkpoint to resume from | `None` |
| `ABLATE_DEFECT_LOSS` | Disable defect loss (ablation) | `False` |
| `ABLATE_OBJ_LOSS` | Disable object loss (ablation) | `False` |
| `ABLATE_ATTN_LOSS` | Disable attention loss (ablation) | `False` |

### 2. Training (Cells 5–23)
Cells set up data loaders, initialise the model with LoRA adapters, configure the DDPM noise scheduler, and run the training loop. Checkpoints are saved every `SAVE_STEPS` steps to `OUTPUT_DIR/checkpoints/`. TensorBoard logs go to `OUTPUT_DIR/tensorboard/`.

To resume training from a checkpoint, set `RESUME_FROM` to the `.pth` file path before running.

### 3. Inference (Cells 24–33)
Generates defective images using a saved checkpoint. Configure `args` in Cell 31:

```python
args = argparse.Namespace(
    checkpoint    = "/path/to/checkpoint.pth",
    output_dir    = "/kaggle/working/generated",
    object_class  = "bottle",
    defect_type   = "broken_large",   # or iterate over a list (Cell 33)
    data_dir      = DATA_DIR,
    num_samples   = 1,     # candidates per image (best is kept)
    total_images  = 25,    # total images to generate
    steps         = 50,    # diffusion steps
    guidance_scale= 7.5,
    lora_rank     = 8,
    lora_alpha    = 16,
)
```

Cell 33 loops over multiple defect types automatically:
```python
for defect in ['broken_large', 'broken_small', 'contamination']:
    args.defect_type = defect
    inference(args)
```

### 4. Benchmark / Evaluation (Cells 34–49)
Expects generated results at:
```
/kaggle/input/datasets/dahyuntw/gen-results/2. results/
└── <object>/
    └── <defect_type>/
        ├── defective/     # generated images
        ├── mask/          # defect masks
        └── org_defect/    # original defective images
```

Three metrics are computed:
- **FID / KID** – distribution-level similarity (Cells 35–39)
- **LPIPS** – perceptual distance within masked region (Cells 40–44), saved to `lpips_metrics.csv`
- **IC-LPIPS** – intra-cluster diversity (Cells 45–49)

---

## Outputs

| Path | Contents |
|---|---|
| `OUTPUT_DIR/checkpoints/` | Saved model checkpoints (`.pth`) |
| `OUTPUT_DIR/tensorboard/` | TensorBoard training logs |
| `OUTPUT_DIR/generated/` | Synthesised defect images |
| `OUTPUT_DIR/lpips_metrics.csv` | Per-defect LPIPS scores |

---

## TensorBoard

```bash
tensorboard --logdir /kaggle/working/tensorboard
```