# MCB-U

Official PyTorch implementation of:

**Enhancing Occluded X-Ray Security Screening via Collaborative Optimization and Representation Learning**  
(*submitted to* **The Visual Computer**)

This repository targets **dual-view image-level multi-label classification** on **DvXray**.

---

## Overview

We propose a lightweight **closed-loop** framework that couples optimization and representation:

- **MCB-Convex**: difficulty-aware dynamic reweighting loss  
- **CoordAtt-U**: loss-conditioned coordinate attention applied **only at N3**

**Backbone & Fusion**
- Dual-view **shared-weight ConvNeXtV2-Tiny**
- Same-scale **ADD** fusion (stable + parameter-free)
- Multi-scale aggregation with **FPN–PAN**

---

## Core Methods (brief)

### 1) MCBLossConvex (MCB-Convex)
A difficulty-aware class-reweighting BCE loss.  
It estimates per-class difficulty using the **intra-mini-batch dispersion** of predicted probabilities (STD), then builds stable weights with:

- softmax weighting (temperature `tau`)
- lower-bound clamp (`w_min`)
- re-normalization
- EMA smoothing (`momentum`)
- stop-gradient on weight updates

This mitigates optimization bias toward easy/head categories and strengthens learning on hard / under-learned categories.

### 2) CoordAtt_U (CoordAtt-U)
CoordAtt-U is inserted **only at N3** (high-resolution pyramid feature).  
It extends Coordinate Attention by using a **loss-derived difficulty summary signal** (from MCB-Convex) to modulate attention strength, making shallow features focus more on challenging regions while keeping the model lightweight.

---

## Reproducibility: Tested Environment (The Visual Computer style)

To satisfy reproducibility expectations, we report the **tested runtime environment** (instead of pasting the full YAML file contents).

> Journal compliance note  
> Springer Nature journals (including *The Visual Computer*) typically request explicit **Data Availability** and **Code Availability** statements in the manuscript submission system. The repository sections “Code Availability” and “Data Availability” are written to match that expectation and can be reused verbatim in the paper.

### System
- **OS**: Ubuntu **24.04.3 LTS** (minimal install; `lsb` modules not required)
- **GPU**: NVIDIA **GeForce RTX 5080** (16 GB)
- **NVIDIA Driver**: **580.95.05**
- **CUDA (driver-reported)**: **13.0**
- **nvidia-smi timestamp**: **Fri Jan 23 13:05:25 2026**

### Conda / Python
- **Conda env file**: `environment.yml`
- **Env name** (as in YAML): `v2b384`
- **Python**: python==3.10.4

### Key packages (from `environment.yml`)
**Core**
- `python==3.10.4`
- `torch==2.8.0+cu128`
- `torchvision==0.23.0+cu128`
- `torchaudio==2.8.0+cu128`
- `timm==1.0.3`
- `opencv-python==4.12.0.88`
- `safetensors==0.6.2`  

**Common scientific stack**
- `numpy==2.1.2`
- `scipy==1.15.3`
- `scikit-learn==1.7.2`
- `pandas==2.3.3`
- `tensorboard==2.20.0`  

> Notes  
> 1) `environment.yml` pins a known-good set of versions for reproduction.  
> 2) If you use a different CUDA / PyTorch build, results may vary slightly.

---

## Installation (Conda YAML)

Create environment:
```bash
conda env create -f environment.yml
conda activate v2b384
```

Update environment:
```bash
conda env update -f environment.yml --prune
conda activate v2b384
```

---

## Project Layout (Key Components)

```text
MCB-U-code/
├── README.md                          # this file
├── environment.yml                    # reproducible Conda environment
├── main_finetune.py                   # main training entrypoint
├── convert_all_no_head.sh             # convert official weights to backbone only
├── annotations/                       # dataset splits & class names
│   ├── DvXray_train.txt
│   ├── DvXray_val.txt
│   ├── DvXray_test.txt
│   └── classes.txt
├── models/
│   ├── convnextv2_dual.py             # dual-view ConvNeXtV2 backbone
│   ├── fpn.py                         # FPN multi-scale aggregator
│   └── necks.py                       # FPN–PAN neck modules
├── models/modules/custom_losses/
│   ├── mcb_loss.py                    # MCB-Convex loss
│   └── attentions.py                  # CoordAtt-U attention
└── tools/
    └── map_official_v2_weights.py     # official weight mapping tool
```

> Notes  
> - `.idea/` is an IDE directory and can be ignored for reproduction.  
> - The two core algorithm implementations are located in `models/...`:
>   - `mcb_loss` (MCB-Convex)  
>   - `attentions` (CoordAtt-U)

---

## Dataset Preparation (DvXray)

- Dataset homepage: **DvXray** (see the paper / dataset repository)
- Default annotation paths (changeable via CLI args):
  - `annotations/DvXray_train.txt`
  - `annotations/DvXray_val.txt`
  - `annotations/classes.txt`

Each line in `DvXray_train.txt` / `DvXray_val.txt` should describe one sample (**dual-view pair + multi-label target**).  
Please ensure your dataset loader (e.g., `datasets.py`) matches the file format.

---

## Step 1: Convert Pretrained Weights (No-Head)

We first convert the official student weights into **backbone-only** `safetensors`.

1) Put weights here:
```text
student_weights/
```

2) Run conversion:
```bash
bash convert_all_no_head.sh
```

Output:
```text
student_weights_switch_no_head/
```

Example:
```text
student_weights_switch_no_head/convnextv2_tiny.mapped_to_backbone.safetensors
```

---

## Step 2: Train (MCB-Convex + CoordAtt-U)

**Recommended reproduction command (paper setting):**
```bash
python -u main_finetune.py \
  --model convnextv2_tiny \
  --finetune ./student_weights_switch_no_head/convnextv2_tiny.mapped_to_backbone.safetensors \
  --output_dir ./outputs/CoordAttU_MCBConvex \
  --dual_view true \
  --train_list annotations/DvXray_train.txt \
  --val_list annotations/DvXray_val.txt \
  --classes_file annotations/classes.txt \
  --num_classes 15 \
  --input_size 224 \
  --batch_size 32 \
  --epochs 180 \
  --lr 1e-4 \
  --weight_decay 0.05 \
  --warmup_epochs 5 \
  --drop_path 0.2 \
  --num_workers 8 \
  --seed 42 \
  --device cuda \
  --fuse_mode add \
  --head_type fpn_pan \
  --attention_config '{"N3":"coordatt_u"}' \
  --base_loss mcb_convex \
  --mcb_tau 1.0 \
  --mcb_momentum 0.9 \
  --mcb_wmin 1e-3 \
  --patience 25
```

---

## Batch Experiments (script)

Run the provided script:
```bash
bash main.sh
```

Useful options:
```bash
bash main.sh --start-from 3
bash main.sh --run-specific 1
bash main.sh --run-specific 1,3,5
bash main.sh --run-specific 2-6
bash main.sh --run-count 2
bash main.sh --no-skip-completed
```

---

## Logs and Checkpoints

Training outputs (inside `--output_dir`):
- `training_log.csv`
- `checkpoint_last.pth`
- `checkpoint_best.pth`

`main.sh` may create a `_FINISHED` flag file for completed runs.

---

## Code Availability (for journal compliance)

The source code for **MCB-Convex** and **CoordAtt-U** (and training scripts) will be made publicly available via this GitHub repository upon publication / acceptance, consistent with Springer Nature code availability recommendations.

---

## Data Availability

This repository does **not** redistribute the DvXray dataset.  
Please obtain the dataset from its official release and follow its license/terms.

---

## Repository Policies

- Please read **CODE_OF_CONDUCT.md** before participating.
- Contributions are welcome; see **CONTRIBUTING.md**.
- Licensing information is provided in **LICENSE**.

---

## Citation

If you use this work, please cite our paper (BibTeX will be provided after acceptance).

---

## License

Add your license here (e.g., MIT / Apache-2.0) before releasing the repository.
