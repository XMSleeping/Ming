# python predict.py --model checkpoints/best_UNet_CBAM.pth --input data/val/images/01_test.png --output results/vessel_mask.png
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

def load_model(model_path):
    model = Model(in_channels=1, out_channels=1)
    ckpt = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=True 
    )
    
    state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model


def predict_one(model, img_path):
    img = Image.open(img_path).convert('L')
    img = np.array(img, dtype=np.float32)

    if img.shape[0] != IMG_SIZE or img.shape[1] != IMG_SIZE:
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    img = img / 255.0
    tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        prob = torch.sigmoid(logits)
        mask = (prob > THRESH).float()

    mask = (mask.cpu().numpy()[0, 0] * 255).astype(np.uint8)
    return mask


def main():
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    # ✅ 自动创建目录
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = load_model(args.model)
    mask = predict_one(model, args.input)

    ok = cv2.imwrite(str(out_path), mask)
    if not ok:
        raise RuntimeError(f"cv2.imwrite 失败: {out_path}")

    print(f"✅ 已保存：{out_path.resolve()}")


if __name__ == "__main__":
    main()
