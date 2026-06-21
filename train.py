# 训练**个epoch，无预训练
# python train.py --epochs **
# 带**轮旋转预测预训练
#python train.py --epochs 50 --pretrain_epochs ** --pretext_task rotation
# 带**轮拼图预训练
#python train.py --epochs 50 --pretrain_epochs ** --pretext_task jigsaw
# 带**轮重建预训练
#python train.py --epochs 50 --pretrain_epochs ** --pretext_task reconstruction
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from PIL import Image
import argparse
from tqdm import tqdm
from datetime import datetime
import time
import random

from dataset import SelfSupervisedDataset

# 模型选择
# from model import LMBiSNet as Model
from model import UNet_CBAM as Model
from model import SelfSupervisedUNet  

# 设置随机种子
def set_seed(seed=42):
    """设置随机种子以确保实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"✓ 随机种子已设置为: {seed}")

# Worker初始化函数
def worker_init_fn(worker_id):
    """DataLoader worker初始化函数"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# 数据集类
class RetinaDataset(Dataset):
    def __init__(self, data_dir, split='train'):
        self.img_dir = os.path.join(data_dir, split, 'images')
        self.mask_dir = os.path.join(data_dir, split, 'vessels')
        
        self.images = sorted([f for f in os.listdir(self.img_dir) 
                            if f.endswith(('.png', '.jpg', '.tif'))])
        
        print(f"{split} 集: 找到 {len(self.images)} 张图像")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)
        
        image = np.array(Image.open(img_path).convert('L'), dtype=np.float32) / 255.0
        mask = np.array(Image.open(mask_path).convert('L'), dtype=np.float32) / 255.0
        
        image = torch.from_numpy(image).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)
        
        return image, mask

# 评价指标
def calculate_metrics(pred, target, threshold=0.5):
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()
    
    pred_flat = pred_binary.view(-1)
    target_flat = target_binary.view(-1)
    
    TP = (pred_flat * target_flat).sum()
    TN = ((1 - pred_flat) * (1 - target_flat)).sum()
    FP = (pred_flat * (1 - target_flat)).sum()
    FN = ((1 - pred_flat) * target_flat).sum()
    
    epsilon = 1e-7
    
    acc = (TP + TN) / (TP + TN + FP + FN + epsilon)
    sp = TN / (TN + FP + epsilon)
    dsc = (2 * TP) / (2 * TP + FP + FN + epsilon)
    iou_foreground = TP / (TP + FP + FN + epsilon)
    iou_background = TN / (TN + FN + FP + epsilon)
    miou = (iou_foreground + iou_background) / 2.0
    
    return acc.item(), sp.item(), dsc.item(), miou.item()

# 训练函数
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    
    progress_bar = tqdm(dataloader, desc="Training")
    for images, masks in progress_bar:
        images = images.to(device)
        masks = masks.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
    
    return total_loss / len(dataloader)

# 新增：自监督预训练函数（使用你提供的版本）
def pretrain_epoch(model, dataloader, criterion, optimizer, device, pretext_task):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc="Pre-training")
    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        
        if pretext_task in ['rotation', 'jigsaw']:
            loss = criterion(outputs, labels)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        else:
            loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if pretext_task in ['rotation', 'jigsaw']:
            progress_bar.set_postfix(loss=loss.item(), acc=f"{100.*correct/total:.2f}%")
        else:
            progress_bar.set_postfix(loss=loss.item())
    
    if pretext_task in ['rotation', 'jigsaw']:
        return total_loss / len(dataloader), 100. * correct / total
    return total_loss / len(dataloader), 0

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_sp = 0.0
    total_dsc = 0.0
    total_miou = 0.0
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation")
        for images, masks in progress_bar:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            acc, sp, dsc, miou = calculate_metrics(torch.sigmoid(outputs), masks)
            
            total_loss += loss.item()
            total_acc += acc
            total_sp += sp
            total_dsc += dsc
            total_miou += miou
            
            progress_bar.set_postfix(loss=loss.item(), dsc=f"{dsc:.4f}")
    
    num_batches = len(dataloader)
    return {
        'loss': total_loss / num_batches,
        'acc': total_acc / num_batches,
        'sp': total_sp / num_batches,
        'dsc': total_dsc / num_batches,
        'miou': total_miou / num_batches
    }

# 主函数
def main():
    parser = argparse.ArgumentParser(description="视网膜血管分割训练脚本")
    parser.add_argument("--data_dir", type=str, default="data", help="数据目录")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数（上限）")
    parser.add_argument("--pretrain_epochs", type=int, default=20, help="自监督预训练轮数")
    parser.add_argument("--batch_size", type=int, default=4, help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="模型保存目录")
    parser.add_argument("--log_file", type=str, default="training_log.txt", help="日志文件名")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader工作线程数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--pretext_task", type=str, default="rotation", 
                       choices=['rotation', 'jigsaw', 'reconstruction'],
                       help="自监督任务类型")
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # ========== 自监督预训练阶段 ==========
    if args.pretrain_epochs > 0:
        print("\n" + "="*60)
        print("开始自监督预训练...")
        print("="*60)
        
        # 创建自监督数据集
        pretrain_dataset = SelfSupervisedDataset(
            args.data_dir, split='train', 
            pretext_task=args.pretext_task
        )
        
        pretrain_loader = DataLoader(
            pretrain_dataset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=args.num_workers,
            pin_memory=True if device.type == 'cuda' else False,
            worker_init_fn=worker_init_fn
        )
        
        # 创建自监督模型
        print("创建自监督模型...")
        pretrain_model = SelfSupervisedUNet(
            in_channels=1, 
            out_channels=1,
            pretext_task=args.pretext_task
        ).to(device)
        
        # 损失函数和优化器
        if args.pretext_task in ['rotation', 'jigsaw']:
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()
        
        optimizer = optim.Adam(pretrain_model.parameters(), lr=args.lr)
        
        # 预训练循环
        best_acc = 0.0
        for epoch in range(1, args.pretrain_epochs + 1):
            print(f"\n{'='*20} Pre-train Epoch {epoch}/{args.pretrain_epochs} {'='*20}")
            
            loss, acc = pretrain_epoch(
                pretrain_model, pretrain_loader, 
                criterion, optimizer, device, args.pretext_task
            )
            
            print(f"预训练损失: {loss:.4f}", end="")
            if args.pretext_task in ['rotation', 'jigsaw']:
                print(f", 准确率: {acc:.2f}%")
                if acc > best_acc:
                    best_acc = acc
                    # 保存最佳预训练模型
                    pretrained_path = os.path.join(args.save_dir, "best_pretrained.pth")
                    torch.save(pretrain_model.state_dict(), pretrained_path)
                    print(f"✓ 保存最佳预训练模型到: {pretrained_path}")
            else:
                print("")
        
        print(f"\n预训练完成! 最佳准确率: {best_acc:.2f}%")
        
        # 加载预训练权重到主模型
        print("加载预训练权重到分割模型...")
        main_model = Model(in_channels=1, out_channels=1).to(device)
        
        # 加载backbone权重
        pretrained_path = os.path.join(args.save_dir, "best_pretrained.pth")
        if os.path.exists(pretrained_path):
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            model_dict = main_model.state_dict()
            
            # 过滤掉不匹配的键（主要是head层的参数）
            pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                             if k in model_dict and 'head' not in k}
            
            # 更新模型字典
            model_dict.update(pretrained_dict)
            main_model.load_state_dict(model_dict)
            
            print(f"✓ 从 {pretrained_path} 加载预训练权重成功!")
            print(f"  加载了 {len(pretrained_dict)}/{len(model_dict)} 个参数")
        else:
            print(f"⚠ 警告: 未找到预训练模型 {pretrained_path}")
    else:
        main_model = Model(in_channels=1, out_channels=1).to(device)
    
    # ========== 正常分割训练阶段 ==========
    print("\n" + "="*60)
    print("开始分割训练...")
    print("="*60)
    
    # 使用加载了预训练权重的模型
    model = main_model
    
    # 分割训练的损失函数和优化器
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # 加载数据集
    print("加载数据集...")
    train_dataset = RetinaDataset(args.data_dir, split='train')
    val_dataset = RetinaDataset(args.data_dir, split='val')
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False,
        worker_init_fn=worker_init_fn
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False,
        worker_init_fn=worker_init_fn
    )
    
    # 日志文件
    log_path = os.path.join(args.save_dir, args.log_file)
    with open(log_path, 'w') as f:
        f.write(f"训练开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"模型: {Model.__name__}\n")
        f.write(f"数据目录: {args.data_dir}\n")
        f.write(f"预训练轮数: {args.pretrain_epochs}\n")
        f.write(f"预训练任务: {args.pretext_task}\n")
        f.write(f"训练轮数: {args.epochs}\n")
        f.write(f"批次大小: {args.batch_size}\n")
        f.write(f"学习率: {args.lr}\n")
        f.write(f"随机种子: {args.seed}\n")
        f.write(f"DataLoader workers: {args.num_workers}\n")
        f.write("=" * 80 + "\n")
        f.write("Epoch\tTrain_Loss\tVal_Loss\tACC\t\tSP\t\tDSC\t\tMIoU\n")
        f.write("-" * 80 + "\n")
    
    best_dsc = 0.0
    best_epoch = 0
    
    print("开始训练...")
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*20} Epoch {epoch}/{args.epochs} {'='*20}")
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = validate_epoch(model, val_loader, criterion, device)
        
        scheduler.step(val_metrics['loss'])
        
        print(f"训练损失: {train_loss:.4f}")
        print(f"验证损失: {val_metrics['loss']:.4f}")
        print(f"ACC: {val_metrics['acc']:.4f}")
        print(f"SP: {val_metrics['sp']:.4f}")
        print(f"DSC: {val_metrics['dsc']:.4f}")
        print(f"MIoU: {val_metrics['miou']:.4f}")
        
        with open(log_path, 'a') as f:
            f.write(f"{epoch}\t{train_loss:.4f}\t\t{val_metrics['loss']:.4f}\t\t"
                   f"{val_metrics['acc']:.4f}\t{val_metrics['sp']:.4f}\t"
                   f"{val_metrics['dsc']:.4f}\t{val_metrics['miou']:.4f}\n")
        
        if val_metrics['dsc'] > best_dsc:
            best_dsc = val_metrics['dsc']
            best_epoch = epoch
            best_model_path = os.path.join(args.save_dir, f"best_{Model.__name__}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dsc': val_metrics['dsc'],
                'val_miou': val_metrics['miou'],
            }, best_model_path)
            print(f"✓ 保存最佳模型到: {best_model_path}")
        
        if epoch % 10 == 0:
            checkpoint_path = os.path.join(args.save_dir, f"checkpoint_epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)
            print(f"✓ 保存检查点到: {checkpoint_path}")
    
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"训练完成!")
    print(f"总训练时间: {total_time/3600:.2f} 小时")
    print(f"最佳模型在第 {best_epoch} 轮，DSC: {best_dsc:.4f}")
    print(f"日志文件: {log_path}")
    print(f"最佳模型: {os.path.join(args.save_dir, f'best_{Model.__name__}.pth')}")
    print(f"{'='*60}")
    
    with open(log_path, 'a') as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"训练结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总训练时间: {total_time/3600:.2f} 小时\n")
        f.write(f"最佳模型在第 {best_epoch} 轮，DSC: {best_dsc:.4f}\n")
        f.write("=" * 80 + "\n")

if __name__ == "__main__":
    main()