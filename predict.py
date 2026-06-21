""""
python predict.py ^
  --model checkpoints/best_UNet_CBAM.pth ^
  --input data/test/images/01_test.png ^
  --output results/vessel_mask.png
"""
import torch
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
from model import UNet_CBAM as Model

# ========= 配置 =========
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 512
THRESH = 0.5
# =========================


def load_model(model_path):
    model = Model(in_channels=1, out_channels=1)
    ckpt = torch.load(model_path, map_location=DEVICE)
    state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model


def predict_one(model, img_path):
    # 读入灰度图
    img = Image.open(img_path).convert('L')
    img = np.array(img, dtype=np.float32)

    # resize + 归一化
    if img.shape[0] != IMG_SIZE or img.shape[1] != IMG_SIZE:
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    img = img / 255.0
    tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        prob = torch.sigmoid(logits)
        mask = (prob > THRESH).float()

    # 转成 0 / 255 单通道
    mask = (mask.cpu().numpy()[0, 0] * 255).astype(np.uint8)
    return mask


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="best_xxx.pth")
    parser.add_argument("--input", required=True, help="输入图像路径")
    parser.add_argument("--output", required=True, help="输出掩码路径")
    args = parser.parse_args()

    model = load_model(args.model)
    mask = predict_one(model, args.input)

    cv2.imwrite(args.output, mask)
    print(f"✅ 血管掩码已保存: {args.output}")


if __name__ == "__main__":
    main()