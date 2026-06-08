import os, numpy as np
from PIL import Image
from shutil import copyfile

print("[필터링 v4] 기본 균형형")

def score_image(path):
    img = np.array(Image.open(path).convert("L")).astype(float)
    brightness = img.mean()
    if brightness < 30 or brightness > 225:
        return -1
    sharpness = np.std(img)
    brightness_sc = 1.0 - abs(brightness - 128) / 128
    contrast_sc = (img.max() - img.min()) / 255.0
    return sharpness * 0.5 + brightness_sc * 100 * 0.3 + contrast_sc * 100 * 0.2

files = sorted(os.listdir("temp_2000"))
scores = [(score_image(f"temp_2000/{f}"), f) for f in files]
scores = [(s, f) for s, f in scores if s > 0]
scores.sort(reverse=True)
top1000 = sorted(scores[:1000], key=lambda x: x[1])

os.makedirs("final_balanced", exist_ok=True)
for idx, (_, fname) in enumerate(top1000):
    copyfile(f"temp_2000/{fname}", f"final_balanced/img_{idx:04d}.png")

print(f"완료! final_balanced에 {len(top1000)}장")
