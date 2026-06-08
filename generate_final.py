import torch
from diffusers import DDIMPipeline, DDIMScheduler
import os, numpy as np
from PIL import Image
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print(f"[생성] Seed={SEED} | 2000장")

os.makedirs("temp_2000", exist_ok=True)

pipe = DDIMPipeline.from_pretrained("google/ddpm-celebahq-256")
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.unet.load_state_dict(torch.load("finetuned_gpu0_epoch14.pt"))
pipe = pipe.to("cuda:0")
pipe.unet.eval()

for i in range(0, 2000, 4):
    imgs = pipe(batch_size=4, num_inference_steps=1000, eta=0.8).images
    for j, img in enumerate(imgs):
        if i + j < 2000:
            img.save(f"temp_2000/img_{i+j:04d}.png")
    if i % 200 == 0:
        print(f"  {min(i+4, 2000)}/2000")

print("생성 완료!")
