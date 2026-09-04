"""
【train_base.py】A1 基线训练脚本（云端 AutoDL 运行）
作用：
  从零训练干净对照组 base（yolo11x.pt 起点 + 30 轮 + 固定增强），
  产出 best.pt 和 val 混淆矩阵/逐类 AP。base 既是 A2/A3 的对照尺子，
  也是 A3 解耦重训的阶段一模型（加载它的权重）。
配置依据：《长尾路口检测研究-执行手册.md》A1 章节。
运行（云端）：
  cd /root/autodl-tmp && nohup python train_base.py > base.log 2>&1 &
"""
import os
# 指定 ultralytics 配置目录（云端固定位置，避免跟默认目录冲突）
os.environ["ULTRALYTICS_CONFIG_DIR"] = "/root/autodl-tmp/.ultralytics"
os.environ["OMP_NUM_THREADS"] = "8"

from ultralytics import YOLO

# ---------- 配置 ----------
DATA = "/root/autodl-tmp/dataset_clean/dataset_clean.yaml"  # 干净数据集（Linux 路径）
PROJECT = "/root/autodl-tmp/runs_longtail"                  # 所有长尾实验的统一输出根目录
NAME = "base"                                               # 本实验名


def main():
    # yolo11x.pt = COCO 预训练起点，所有实验同一起点（不是加载历史 V1~V3 权重）
    model = YOLO("yolo11x.pt")
    results = model.train(
        # --- 数据与训练量 ---
        data=DATA,
        epochs=30,
        patience=5,          # val 连续 5 轮不涨就早停
        batch=8,
        imgsz=1024,
        # --- 优化 ---
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=5,
        cos_lr=True,
        weight_decay=0.0005,
        # --- 可复现 ---
        seed=0,
        deterministic=True,
        # --- 增强（真实生效的部分） ---
        close_mosaic=10,     # 前 20 轮开 mosaic，最后 10 轮关闭（干净精修，复现 V2 的有效结构）
        degrees=90.0,        # 90° 随机旋转
        fliplr=0.5,          # 水平翻转概率
        flipud=0.5,          # 垂直翻转概率
        hsv_h=0.015,
        hsv_s=0.3,
        hsv_v=0.2,
        # --- 验证与保存 ---
        val=True,            # 每轮在干净 val 上验证
        save=True,
        plots=True,          # 出混淆矩阵/PR/逐类曲线图
        project=PROJECT,
        name=NAME,
        exist_ok=True,       # 允许覆盖重跑同名实验
    )
    # 打印整体分数，方便直接抄进记录表
    m = results.box
    print(f"[base] mAP50={m.map50:.4f} mAP50-95={m.map:.4f} P={m.mp:.4f} R={m.mr:.4f}")

    # 用训练得到的 best.pt 再完整验证一遍，专门出逐类指标（诊断 T 型病根用）
    best = YOLO(f"{PROJECT}/{NAME}/weights/best.pt")
    vm = best.val(data=DATA, imgsz=1024, batch=8, device=0, plots=True,
                  project=PROJECT, name=f"{NAME}_val", exist_ok=True)
    b = vm.box
    print(f"[base-val] mAP50={b.map50:.4f} mAP50-95={b.map:.4f}")
    names = list(vm.names.values())
    try:
        for i in range(len(names)):
            # ap50[i] = 该类别 mAP50，ap[i] = 该类别 mAP50-95
            print(f"  {names[i]}: mAP50={b.ap50[i]:.4f}  mAP50-95={b.ap[i]:.4f}")
    except Exception as e:
        # 不同 ultralytics 版本字段可能略有差异，失败不影响训练，看 results.csv 即可
        print("逐类指标打印失败(版本差异，看 results.csv):", e)


if __name__ == "__main__":
    main()
