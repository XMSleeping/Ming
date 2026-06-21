# 数据预处理：绿色通道+CLAHE+伽马矫正
import os
import sys
import argparse
import numpy as np
import cv2
from pathlib import Path
import shutil
import torch
from torch.utils.data import Dataset
from PIL import Image


# 预处理参数设置
DEFAULT_CLIP_LIMIT = 2.0
DEFAULT_TILE_SIZE = 8
DEFAULT_GAMMA = 0.8


def extract_green_channel(img_bgr: np.ndarray) -> np.ndarray:
    return img_bgr[:, :, 1]


def apply_clahe_gray(gray_img: np.ndarray, clip_limit: float, tile_grid: tuple) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return clahe.apply(gray_img)


def apply_gamma(gray_img: np.ndarray, gamma: float) -> np.ndarray:
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(gray_img, table)


def enhance_image(img_bgr: np.ndarray, clip_limit: float, tile_grid: tuple, gamma: float) -> np.ndarray:
    green = extract_green_channel(img_bgr)  # 提取绿色通道
    clahe = apply_clahe_gray(green, clip_limit, tile_grid)  # CLAHE
    gamma_corrected = apply_gamma(clahe, gamma)  # 伽马校正
    return gamma_corrected


def main():
    parser = argparse.ArgumentParser(description="训练集和验证集绿色通道增强 + 血管标注原样复制")
    parser.add_argument("--src", default="preDRIVE", help="输入根目录（默认：preDRIVE）")
    parser.add_argument("--dst", default="data", help="输出根目录（默认：data）")
    parser.add_argument("--clip", type=float, default=DEFAULT_CLIP_LIMIT, help="CLAHE clipLimit")
    parser.add_argument("--tile", type=int, default=DEFAULT_TILE_SIZE, help="CLAHE tile size")
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA, help="Gamma值（默认：0.8）")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    if not src.exists():
        print(f"[ERROR] 输入目录不存在: {src.resolve()}")
        sys.exit(1)

    # 使用局部变量
    clip_limit = args.clip
    tile_grid = (args.tile, args.tile)
    gamma = args.gamma

    print("=" * 60)
    print(f"  Source : {src.resolve()}")
    print(f"  Dest   : {dst.resolve()}")
    print(f"  处理规则：")
    print(f"  ✅ 训练集图像：提取绿色通道 → CLAHE → Gamma")
    print(f"  ✅ 验证集图像：提取绿色通道 → CLAHE → Gamma")
    print(f"  ❌ 血管标注（vessels）：原样复制（不处理）")
    print(f"  CLAHE  : clipLimit={clip_limit}, tileGridSize={tile_grid}")
    print(f"  Gamma  : {gamma}")
    print("=" * 60)

    # 处理训练和验证集的图像
    splits = ["train", "val"]
    
    for split in splits:
        img_src = src / split / "images"
        img_dst = dst / split / "images"
        img_dst.mkdir(parents=True, exist_ok=True)

        if img_src.exists():
            img_files = sorted(img_src.glob("*.png")) or sorted(img_src.glob("*.tif"))
            print(f"\n>>> {split} 集图像：发现 {len(img_files)} 张，开始处理...")

            for img_path in img_files:
                img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                if img_bgr is None:
                    print(f"  [WARN] 读取失败: {img_path.name}")
                    continue

                # 绿色通道+CLAHE+伽马
                enhanced = enhance_image(img_bgr, clip_limit, tile_grid, gamma)
                cv2.imwrite(str(img_dst / img_path.name), enhanced)
                print(f"  已处理: {img_path.name}")

        else:
            print(f"\n  [WARN] {split} 集图像目录不存在: {img_src}")

    # 复制所有血管标注
    for split in splits:
        ves_src = src / split / "vessels"
        ves_dst = dst / split / "vessels"
        ves_dst.mkdir(parents=True, exist_ok=True)

        if ves_src.exists():
            ves_files = sorted(ves_src.glob("*.png")) or sorted(ves_src.glob("*.tif"))
            print(f"\n>>> {split} 血管标注：发现 {len(ves_files)} 张，原样复制...")

            for ves_path in ves_files:
                shutil.copy(ves_path, ves_dst / ves_path.name)
                print(f"  已复制: {ves_path.name}")

        else:
            print(f"\n  [WARN] {split} 血管标注目录不存在: {ves_src}")

    print(f"\n✅ 全部完成！输出目录: {dst.resolve()}")


if __name__ == "__main__":
    main()

class SelfSupervisedDataset(Dataset):
    """自监督学习数据集 - 用于预训练"""
    def __init__(self, data_dir, split='train', pretext_task='rotation'):
        self.img_dir = os.path.join(data_dir, split, 'images')
        self.pretext_task = pretext_task
        
        self.images = sorted([f for f in os.listdir(self.img_dir) 
                            if f.endswith(('.png', '.jpg', '.tif'))])
        
        print(f"{split} 自监督数据集: 找到 {len(self.images)} 张图像")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        # 读取图像
        image = np.array(Image.open(img_path).convert('L'), dtype=np.float32) / 255.0
        
        if self.pretext_task == 'rotation':
            # 旋转预测任务
            rotations = [0, 90, 180, 270]
            rotation_idx = np.random.randint(0, 4)
            angle = rotations[rotation_idx]
            
            # 旋转图像
            h, w = image.shape
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h))
            
            image = torch.from_numpy(rotated).unsqueeze(0)
            label = torch.tensor(rotation_idx, dtype=torch.long)
            
            return image, label
        
        elif self.pretext_task == 'jigsaw':
            # 拼图任务
            h, w = image.shape
            patch_size = h // 2
            
            # 切割成4个patch
            patches = []
            positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
            
            for i, j in positions:
                patch = image[i*patch_size:(i+1)*patch_size, 
                           j*patch_size:(j+1)*patch_size]
                patches.append(patch)
            
            # 打乱顺序
            order = np.random.permutation(4)
            shuffled_patches = [patches[i] for i in order]
            
            # 重组图像
            top_row = np.concatenate([shuffled_patches[0], shuffled_patches[1]], axis=1)
            bottom_row = np.concatenate([shuffled_patches[2], shuffled_patches[3]], axis=1)
            jigsaw_image = np.concatenate([top_row, bottom_row], axis=0)
            
            image = torch.from_numpy(jigsaw_image).unsqueeze(0)
            label = torch.tensor(order, dtype=torch.long)
            
            return image, label
        
        else:  # 默认：简单重建任务
            # 添加噪声
            noise = np.random.normal(0, 0.1, image.shape).astype(np.float32)
            noisy_image = np.clip(image + noise, 0, 1)
            
            image = torch.from_numpy(image).unsqueeze(0)
            noisy_image = torch.from_numpy(noisy_image).unsqueeze(0)
            
            return noisy_image, image