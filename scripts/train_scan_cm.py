"""
【train_scan_cm.py】close_mosaic 扫描脚本（云端 AutoDL 运行）
作用：
  在 A1 base（close_mosaic=10）基础上，把 close_mosaic 换成 15 和 20 各训一组，
  每组都从 yolo11x.pt 从零训满 30 轮（patience=0 不早停，组间才可比），
  用来选出"满血 base*"，作为后续 A3 解耦重训的加载起点。
与 base 的唯一差异：close_mosaic 换值（15/20）+ patience 改为 0（跑满 30 轮）。
配置依据：《长尾路口检测研究-执行手册.md》第一层 close_mosaic 扫描。
运行（云端）：
  cd /root/autodl-tmp && nohup python train_scan_cm.py > scan_cm.log 2>&1 &
"""
import os
# 指定 ultralytics 配置目录（云端固定位置，避免跟默认目录冲突）
os.environ["ULTRALYTICS_CONFIG_DIR"] = "/root/autodl-tmp/.ultralytics"
os.environ["OMP_NUM_THREADS"] = "8"

from ultralytics import YOLO

# ---------- 配置 ----------
DATA = "/root/autodl-tmp/dataset_clean/dataset_clean.yaml"  # 干净数据集（Linux 路径）
PROJECT = "/root/autodl-tmp/runs_longtail"                  # 所有长尾实验的统一输出根目录
# 本次要扫的两组 close_mosaic 值（10 已有 base 结果，不重复训）
CLOSE_MOSAIC_LIST = [15, 20]


def train_and_val(cm_value):
    # 作用：训单个 close_mosaic 值的一组，训完自动 val 一遍并打印整体/逐类指标，方便直接抄表
    name = f"scan_cm{cm_value}"
    # 每组都从 COCO 预训练起点 yolo11x.pt 从头训，保证跟 base 同起点（受控实验第一要素）
    model = YOLO("yolo11x.pt")
    results = model.train(
        # --- 数据与训练量 ---
        data=DATA,
        epochs=30,
        patience=0,          # 扫描期不早停，每组都跑满 30 轮，组间才公平
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
        # --- 增强（除 close_mosaic 外，其余跟 base 一字不改） ---
        # 研究变量：值越大 = mosaic 关得越早、无 mosaic 精修期越长
        #   15：前 15 轮开 mosaic，后 15 轮关闭精修
        #   20：前 10 轮开 mosaic，后 20 轮关闭精修
        close_mosaic=cm_value,
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
        name=name,
        exist_ok=True,       # 允许覆盖重跑同名实验
    )
    # 打印整体分数，方便直接抄进记录表
    m = results.box
    print(f"[{name}] mAP50={m.map50:.4f} mAP50-95={m.map:.4f} P={m.mp:.4f} R={m.mr:.4f}")

    # 用训练得到的 best.pt 再完整验证一遍，专门出逐类指标（选满血 base* 主要看这里）
    best = YOLO(f"{PROJECT}/{name}/weights/best.pt")
    vm = best.val(data=DATA, imgsz=1024, batch=8, device=0, plots=True,
                  project=PROJECT, name=f"{name}_val", exist_ok=True)
    b = vm.box
    print(f"[{name}-val] mAP50={b.map50:.4f} mAP50-95={b.map:.4f}")
    names = list(vm.names.values())
    try:
        for i in range(len(names)):
            # ap50[i] = 该类别 mAP50，ap[i] = 该类别 mAP50-95
            print(f"  {names[i]}: mAP50={b.ap50[i]:.4f}  mAP50-95={b.ap[i]:.4f}")
    except Exception as e:
        # 不同 ultralytics 版本字段可能略有差异，失败不影响训练，看 results.csv 即可
        print("逐类指标打印失败(版本差异，看 results.csv):", e)


def main():
    # 作用：挨个训完 close_mosaic 列表里所有组（串行跑，一组结束自动开下一组）
    for cm_value in CLOSE_MOSAIC_LIST:
        print(f"===== 开始训练 close_mosaic={cm_value} =====")
        train_and_val(cm_value)
    print("===== 全部扫描完成 =====")


if __name__ == "__main__":
    main()
