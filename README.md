# StyleGAN2-ADA DDPM Fine-tuning

## Overview
DDPM fine-tuning on CelebA-HQ with DDIM sampling and multiple filtering strategies.

## Configuration
- Model: google/ddpm-celebahq-256
- Checkpoint: finetuned_gpu0_epoch14.pt
- Seed: 42 (reproducibility)
- Steps: 1000
- Scheduler: DDIM
- ETA: 0.8

## Files
- `generate_final.py`: Generate 2000 images
- `filter_balanced.py`: Balanced filtering
- `filter_brightness.py`: Brightness-weighted filtering
- `filter_sharpness.py`: Sharpness-weighted filtering
- `filter_clip.py`: CLIP-based filtering

## Best Result
- FID: 73.74
- IS: 3.43
- KID: 0.0363
- TOPPR: 0.7687

## Usage
python generate_final.py
python filter_clip.py
