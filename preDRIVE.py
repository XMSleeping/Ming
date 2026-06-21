# 数据扩充：训练集 --> 160张 测试集 --> 20张 图片尺寸 --> 512×512
import os
import sys
import math
import random
import numpy as np
import cv2
from pathlib import Path
from PIL import Image


DRIVE_ROOT = Path("DRIVE")      # 输入路径
OUT_ROOT   = Path("preDRIVE")    # 输出路径
TARGET     = 512
SEED       = 42
AUG_PER_ORIGIN = 8               # 20 × 8 = 160

def load_tif_rgb(path: Path) -> np.ndarray:
    """读取 .tif → RGB uint8 ndarray (H,W,3)"""
    return np.array(Image.open(path).convert("RGB"))


def load_vessel_gif(path: Path, thresh: int = 127) -> np.ndarray:
    """
    读取 1st_manual/*.gif（调色板二值图）
    返回 0/1 uint8 ndarray (H,W)
    """
    arr = np.array(Image.open(path).convert("L"))
    return (arr > thresh).astype(np.uint8)

def warp_affine_sync_two(
    img: np.ndarray,
    vessel: np.ndarray,
    M: np.ndarray,
    dsize,
):
    """dsize = (w, h) 两个 Python int"""
    w = int(dsize[0])
    h = int(dsize[1])

    img_w = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    ves_w = cv2.warpAffine(
        vessel.astype(np.uint8), M, (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return img_w, ves_w

def infer_vessel_name(img_stem: str) -> str:
    """
    把图像 stem 转成 vessel gif 文件名
    例: "21_training" -> "21_manual1.gif"
        "01_test"      -> "01_manual1.gif"
    """
    prefix = img_stem.split("_")[0] 
    return f"{prefix}_manual1.gif"

def make_output_dirs(out_root: Path):
    dirs = {}
    for split in ("train", "val"):
        for sub in ("images", "vessels"):
            p = out_root / split / sub
            p.mkdir(parents=True, exist_ok=True)
            dirs[(split, sub)] = p
    return dirs

def process_train_image(
    img_path: Path,
    manual_dir: Path,
    out_dirs: dict,
    target: int = 512,
    rng: random.Random | None = None,
):
    stem = img_path.stem  # e.g. "21_training"

    v_name = infer_vessel_name(stem)
    v_path = manual_dir / v_name
    if not v_path.exists():
        hits = list(manual_dir.glob(f"*{stem.split('_')[0]}*manual*.gif"))
        v_path = hits[0] if hits else None

    if v_path is None or not v_path.exists():
        print(f"  [SKIP] vessel not found for {img_path.name}")
        return

    img    = load_tif_rgb(img_path)
    vessel = load_vessel_gif(v_path)
    h, w   = img.shape[:2]

    rng_local = rng if rng is not None else random.Random(hash(img_path.name) & 0xFFFFFFFF)

    base_out_stem = stem

    for k in range(AUG_PER_ORIGIN):
        if k == 0:
            img_w   = cv2.resize(img,    (target, target), interpolation=cv2.INTER_LINEAR)
            vessel_w = cv2.resize(vessel, (target, target), interpolation=cv2.INTER_NEAREST)
        else:
            do_h = rng_local.random() < 0.5
            do_v = rng_local.random() < 0.5
            angle = rng_local.uniform(-15, 15)
            tx    = rng_local.randint(-40, 40)
            ty    = rng_local.randint(-30, 30)

            cx, cy = w / 2.0, h / 2.0

            cos_a = math.cos(math.radians(angle))
            sin_a = math.sin(math.radians(angle))
            R = np.array([[cos_a, -sin_a],
                          [sin_a,  cos_a]], dtype=np.float64)

            sx = -1 if do_h else 1
            sy = -1 if do_v else 1
            S = np.diag([sx, sy]).astype(np.float64)

            A = S @ R

            c_rot   = R @ np.array([cx, cy])
            c_final = S @ c_rot + np.array([tx, ty])
            b = (cx - c_final[0], cy - c_final[1])

            M_final = np.float32([
                [A[0, 0], A[0, 1], b[0]],
                [A[1, 0], A[1, 1], b[1]],
            ])

            img_w, vessel_w = warp_affine_sync_two(
                img, vessel, M_final, (target, target)
            )

        yield img_w, vessel_w

def main():
    drive = Path(DRIVE_ROOT)
    if not drive.exists():
        print(f"[ERROR] DRIVE_ROOT={drive} 不存在! 请修改脚本顶部的 DRIVE_ROOT")
        sys.exit(1)

    train_img_dir = drive / "training" / "images"
    train_man_dir = drive / "training" / "1st_manual"
    val_img_dir   = drive / "test"      / "images"
    val_man_dir   = drive / "test"      / "1st_manual"

    for d in [train_img_dir, train_man_dir, val_img_dir, val_man_dir]:
        if not d.exists():
            print(f"[WARN] expected dir missing: {d}")

    out_dirs = make_output_dirs(OUT_ROOT)
    rng = random.Random(SEED)

    print("=" * 60)
    print("  DRIVE → Augment (NO FOV/mask; keep vessel GT)")
    print(f"  Output : {TARGET}×{TARGET}")
    print(f"  Out    : {OUT_ROOT.resolve()}")
    print("=" * 60)

    # ============ 训练集：20 → 160 ============
    print("\n>>> Training set (augment 20 → 160)...")
    train_imgs = sorted(train_img_dir.glob("*.tif"))
    print(f"    Found {len(train_imgs)} training images")

    out_train_img  = out_dirs[("train", "images")]
    out_train_ves  = out_dirs[("train", "vessels")]

    out_idx = 0

    for p in train_imgs:
        stem = p.stem
        v_name = infer_vessel_name(stem)
        v_path = train_man_dir / v_name
        if not v_path.exists():
            hits = list(train_man_dir.glob(f"*{stem.split('_')[0]}*manual*.gif"))
            v_path = hits[0] if hits else None
        if v_path is None or not v_path.exists():
            print(f"  [SKIP] vessel not found for {p.name}")
            continue

        img    = load_tif_rgb(p)
        vessel = load_vessel_gif(v_path)
        h, w   = img.shape[:2]

        rng_local = random.Random(hash(p.name) & 0xFFFFFFFF)

        for k in range(AUG_PER_ORIGIN):
            if k == 0:
                img_w    = cv2.resize(img,    (TARGET, TARGET), interpolation=cv2.INTER_LINEAR)
                vessel_w = cv2.resize(vessel, (TARGET, TARGET), interpolation=cv2.INTER_NEAREST)
            else:
                do_h = rng_local.random() < 0.5
                do_v = rng_local.random() < 0.5
                angle = rng_local.uniform(-15, 15)
                tx    = rng_local.randint(-40, 40)
                ty    = rng_local.randint(-30, 30)

                cx, cy = w / 2.0, h / 2.0
                cos_a = math.cos(math.radians(angle))
                sin_a = math.sin(math.radians(angle))
                R = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float64)

                sx = -1 if do_h else 1
                sy = -1 if do_v else 1
                S = np.diag([sx, sy]).astype(np.float64)
                A = S @ R

                c_rot   = R @ np.array([cx, cy])
                c_final = S @ c_rot + np.array([tx, ty])
                b = (cx - c_final[0], cy - c_final[1])

                M_final = np.float32([
                    [A[0, 0], A[0, 1], b[0]],
                    [A[1, 0], A[1, 1], b[1]],
                ])
                img_w, vessel_w = warp_affine_sync_two(
                    img, vessel, M_final, (TARGET, TARGET)
                )

            # ---- 保存 ----
            name = f"{out_idx:03d}.png"  # 000.png ... 159.png
            Image.fromarray(img_w).save(out_train_img / name)
            Image.fromarray((vessel_w * 255).astype(np.uint8)).save(
                out_train_ves / name
            )
            out_idx += 1

    print(f"  Train written: {out_idx}  (expect 160)")

    print("\n>>> Validation set (resize only, 20 images)...")
    val_imgs = sorted(val_img_dir.glob("*.tif"))
    print(f"    Found {len(val_imgs)} test/val images")

    out_val_img = out_dirs[("val", "images")]
    out_val_ves = out_dirs[("val", "vessels")]

    for p in val_imgs:
        stem = p.stem
        v_name = infer_vessel_name(stem)
        v_path = val_man_dir / v_name
        if not v_path.exists():
            hits = list(val_man_dir.glob(f"*{stem.split('_')[0]}*manual*.gif"))
            v_path = hits[0] if hits else None

        img    = load_tif_rgb(p)
        img512 = cv2.resize(img, (TARGET, TARGET), interpolation=cv2.INTER_LINEAR)

        if v_path and v_path.exists():
            vessel = load_vessel_gif(v_path)
            ves512 = cv2.resize(vessel, (TARGET, TARGET), interpolation=cv2.INTER_NEAREST)
            Image.fromarray((ves512 * 255).astype(np.uint8)).save(
                out_val_ves / f"{stem}.png"
            )
        else:
            print(f"  [WARN] val vessel missing for {p.name}")

        Image.fromarray(img512).save(out_val_img / f"{stem}.png")

    nv = len(val_imgs)
    print(f"  Val   written: {nv}  (unchanged count)")

    print("\nDONE →", OUT_ROOT.resolve())


if __name__ == "__main__":
    main()