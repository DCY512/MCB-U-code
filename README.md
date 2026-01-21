MCB-Convex + CoordAtt-U (DvXray Dual-View Multi-Label)

Official PyTorch implementation of:

MCB-Convex and CoordAtt-U: A Collaborative Optimization-Representation Framework for Occluded X-Ray Security Screening

This repo targets dual-view image-level multi-label classification on DvXray.
We propose a lightweight closed-loop framework coupling:

MCB-Convex: difficulty-aware dynamic reweighting loss

CoordAtt-U: loss-conditioned coordinate attention applied only at N3

✨ Key Features

Dual-view shared-weight ConvNeXtV2-Tiny

Same-scale ADD fusion (stable + parameter-free)

Multi-scale aggregation with FPN–PAN

MCB-Convex improves hard-category optimization stability

CoordAtt-U uses loss-derived difficulty statistics to modulate attention strength

🔧 Installation (Conda YAML)

Create environment:

conda env create -f environment.yml
conda activate <your_env_name>


Update environment:

conda env update -f environment.yml --prune
conda activate <your_env_name>

📁 Project Layout
.
├── environment.yml
├── convert_all_no_head.sh
├── main.sh
├── main_finetune.py
├── student_weights/                      # official pretrained weights (with head)
├── student_weights_switch_no_head/       # backbone-only weights (no head)
├── tools/
│   └── map_official_v2_weights.py
├── annotations/
│   ├── DvXray_train.txt
│   ├── DvXray_val.txt
│   └── classes.txt
└── models/modules/custom_losses/
    ├── mcb_loss.py                       # MCBLossConvex
    └── attentions.py                     # CoordAtt-U

📌 Dataset Preparation (DvXray)

Training uses these default paths (can be changed via CLI args): 

main_finetune

annotations/DvXray_train.txt

annotations/DvXray_val.txt

annotations/classes.txt

Each line in DvXray_train.txt / DvXray_val.txt should describe one sample (dual-view pair + multi-label target).
Make sure your dataset loader in datasets.py matches the file format.

🚀 Step 1: Convert Pretrained Weights (No-Head)

We first convert official student weights into backbone-only safetensors.

1) Put weights here
student_weights/

2) Run conversion
bash convert_all_no_head.sh


Output will be saved to:

student_weights_switch_no_head/


Example:

student_weights_switch_no_head/convnextv2_tiny.mapped_to_backbone.safetensors

🏋️ Step 2: Train (MCB-Convex + CoordAtt-U)
✅ Recommended reproduction command (Paper setting)
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

🧪 Run With Provided Script (Batch Experiments)

Directly run:

bash main.sh


This script will automatically load:

./student_weights_switch_no_head/convnextv2_tiny.mapped_to_backbone.safetensors


and perform multiple runs / experiments.

Useful options:

bash main.sh --start-from 3
bash main.sh --run-specific 1
bash main.sh --run-specific 1,3,5
bash main.sh --run-specific 2-6
bash main.sh --run-count 2
bash main.sh --no-skip-completed

📌 Core Methods (Brief)
1) MCB-Convex Loss

MCB-Convex is a difficulty-aware dynamic class reweighting BCE loss.
It computes class weights using intra-mini-batch prediction dispersion (std), then applies:

softmax weighting (tau)

lower-bound clamp (w_min)

normalization

EMA smoothing (momentum)

This improves optimization stability and strengthens learning on hard / under-learned categories.

2) CoordAtt-U

CoordAtt-U is applied only at N3 (high-resolution pyramid feature).
It extends Coordinate Attention by using a loss-derived difficulty summary signal (from MCB-Convex weights) to adaptively scale attention strength, improving focus on challenging regions while keeping the model lightweight.

📊 Logs and Checkpoints

Training outputs (inside --output_dir): 

main_finetune

training_log.csv (epoch-wise logs)

checkpoint_last.pth

checkpoint_best.pth

A _FINISHED flag file may be created by main.sh experiment script.
