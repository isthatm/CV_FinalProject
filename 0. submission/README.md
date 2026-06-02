# DefectFill – Assignment 3

Defect synthesis for industrial inspection using Stable Diffusion 2 inpainting with LoRA fine-tuning on the MVTec dataset.

---

## Requirements

### Environment
- Python 3.11.9
- CUDA-capable GPU (16 GB+ VRAM recommended; tested on Kaggle T4 x2)

### Install dependencies

```bash
pip install lpips
pip install torchinfo
pip install torch-fidelity
```

---

## Dataset

The notebook expects the **curated MVTec** dataset with the following directory structure:

```
curated_mvtec/
└── <object_class>/          # e.g. bottle
    ├── train/
    │   ├── defective_masks/
    │       └── <defect_type>/ 
    │   ├── defective/
    │       └── <defect_type>/   # e.g. broken_large
    └── test/
        ├── good/
```

---

## Repository Setup

If Kaggle or GG Colab is used, add the `DefectFill` source directory to your Python path before importing modules:

```python
import sys
sys.path.append("/path/to/DefectFill Repo/DefectFill")
```

On my Kaggle settings this path is:
```
/kaggle/input/datasets/dahyuntw/defectfill-model/DefectFill Repo/DefectFill
```

---

## Running the Notebook (`cv-a3.ipynb`)

Open `0. submission/cv-a3.ipynb` in Jupyter or Kaggle and run cells **top to bottom**. The sections are:

### 1. Configuration
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

### 2. Training
Cells set up data loaders, initialise the model with LoRA adapters, configure the DDPM noise scheduler, and run the training loop. Checkpoints are saved every `SAVE_STEPS` steps to `OUTPUT_DIR/checkpoints/`. TensorBoard logs go to `OUTPUT_DIR/tensorboard/`.

To resume training from a checkpoint, set `RESUME_FROM` to the `.pth` file path before running.

### 3. Inference
Generates defective images using a saved checkpoint. Configure `args`:

```python
args = argparse.Namespace(
    checkpoint    = "/path/to/checkpoint.pth",
    output_dir    = "/kaggle/working/generated",
    object_class  = "bottle",
    defect_type   = "broken_large",   # or iterate over a list
    data_dir      = DATA_DIR,
    num_samples   = 1,     # candidates per image (best is kept)
    total_images  = 25,    # total images to generate
    steps         = 50,    # diffusion steps
    guidance_scale= 7.5,
    lora_rank     = 8,
    lora_alpha    = 16,
)
```

Loop over multiple defect types automatically:
```python
for defect in ['broken_large', 'broken_small', 'contamination']:
    args.defect_type = defect
    inference(args)
```

### 4. Benchmark / Evaluation
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
- **FID / KID** – distribution-level similarity 
- **LPIPS** – perceptual distance within masked region, saved to `lpips_metrics.csv`
- **IC-LPIPS** – intra-cluster diversity

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
tensorboard --logdir <path to directory with tensorboard files>
```
