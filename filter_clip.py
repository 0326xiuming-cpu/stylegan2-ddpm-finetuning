import os, numpy as np
from PIL import Image
from shutil import copyfile
import torch
from transformers import CLIPProcessor, CLIPModel

print("[필터링 v1] CLIP 기반 (가장 강함)")

# CLIP 모델 로드
device = "cuda:0"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()

# 텍스트 임베딩 (한 번만)
text_input = processor(text=["a photo of a face"], return_tensors="pt", padding=True).to(device)

files = sorted(os.listdir("temp_2000"))
scores = []

print(f"CLIP 스코어 계산 중... ({len(files)}장)")

for idx, f in enumerate(files):
    try:
        img = Image.open(f"temp_2000/{f}").convert("RGB")
        img_input = processor(images=img, return_tensors="pt").to(device)
        
        with torch.no_grad():
            img_emb = model.get_image_features(**img_input)
            text_emb = model.get_text_features(**text_input)
            
            # 코사인 유사도
            similarity = torch.nn.functional.cosine_similarity(img_emb, text_emb).item()
        
        scores.append((similarity, f))
    except:
        pass
    
    if (idx + 1) % 100 == 0:
        print(f"  {idx + 1}/{len(files)}")

scores.sort(reverse=True)
top1000 = sorted(scores[:1000], key=lambda x: x[1])

os.makedirs("final_clip", exist_ok=True)
for idx, (score, fname) in enumerate(top1000):
    copyfile(f"temp_2000/{fname}", f"final_clip/img_{idx:04d}.png")

print(f"완료! final_clip에 {len(top1000)}장 저장됨")
