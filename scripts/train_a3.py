"""
【train_a3.py】A3 系列 2×2 析因消融脚本（云端 AutoDL 运行）
作用：
  在 base*（scan_cm15/best.pt）上续训 10 轮，做四个受控实验，分离两个变量的贡献：
    变量1 冻结（结构层：冻结 model.0~22 特征提取，只训检测头 model.23）
    变量2 pos_weight 加权（损失层：按类别放大分类正样本损失，等价 class-balanced loss）
  四组（全部同起点、同轮数、同配置，轮数彻底不是变量）：
    A3-ref   = 不冻结 + 不加权    ← 底座，剥离"多训10轮"的收益
    A2       = 不冻结 + 加权      ← vs A3-ref 看加权的净贡献
    A3-ctrl  = 冻结 + 不加权      ← vs A3-ref 看冻结的净贡献
    A3       = 冻结 + 加权        ← vs A3-ctrl/A2 看组合完整效果
配置依据：《长尾路口检测研究-执行手册.md》A3 章节 3.3 设计定稿（2026-09-04）。
运行（云端）：
  cd /root/autodl-tmp && nohup python train_a3.py > a3.log 2>&1 &
"""
import os
# 指定 ultralytics 配置目录（云端固定位置，避免跟默认目录冲突）
os.environ["ULTRALYTICS_CONFIG_DIR"] = "/root/autodl-tmp/.ultralytics"
os.environ["OMP_NUM_THREADS"] = "8"

import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.utils.loss import v8DetectionLoss

# ---------- 配置 ----------
DATA = "/root/autodl-tmp/dataset_clean/dataset_clean.yaml"  # 干净数据集（Linux 路径）
PROJECT = "/root/autodl-tmp/runs_longtail"                  # 所有长尾实验的统一输出根目录
BASE_WEIGHTS = "/root/autodl-tmp/runs_longtail/scan_cm15/weights/best.pt"  # base* 权重

# 类别权重：w_c = (N_max / N_c)^0.5，N_c 是该类 train 框数
# 十字(20676): 1.0, T型(5504): ~1.94, 环形(149): ~11.79
CLASS_WEIGHTS = [1.0, 1.937, 11.789]

SMOKE_TEST = False   # True=只跑1轮冒烟验证(冻结/加权是否生效)；False=跑下面全部四组

# 四组实验定义：(实验名, 冻结?, 加权?)
# 每个都是受控变量组合，四组串行跑
EXPERIMENTS = [
    ("a3_ref",      False, False),  # 纯续训对照（底座）
    ("a2_weighted", False, True),   # 只加权
    ("a3_ctrl",     True,  False),  # 只冻结
    ("a3_decoupled", True, True),   # 冻结+加权（完整方法）
]


def patch_loss_with_pos_weight():
    # 作用：monkeypatch v8DetectionLoss.__init__，把 bce 换成带 pos_weight 的版本
    # 原理：bce 分类损失里加 pos_weight 会放大"真少类被判错"的损失，把模型注意力扳向少类。
    #       pos_weight 是 BCEWithLogitsLoss 原生参数，不用改损失公式，也不碰源码文件。
    # 返回一个恢复函数：跑完加权实验后调用，把原版 init 装回去，保证下一组不加权实验不被污染
    original_init = v8DetectionLoss.__init__

    def patched_init(self, model, *args, **kwargs):
        # 先跑原版初始化，确保该有的属性都在（原版签名还带 tal_topk/tal_topk2，用 *args/**kwargs 兜住）
        original_init(self, model, *args, **kwargs)
        # 原版 self.bce 也是 BCEWithLogitsLoss（reduction="none"），这里只补一个 pos_weight；
        # reduction 从原版对象继承，避免不同版本写法不同
        weight_tensor = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32,
                                     device=next(model.parameters()).device)
        self.bce = nn.BCEWithLogitsLoss(pos_weight=weight_tensor,
                                        reduction=self.bce.reduction)

    v8DetectionLoss.__init__ = patched_init

    # 恢复函数：把 __init__ 换回原版
    def unpatch():
        v8DetectionLoss.__init__ = original_init

    return unpatch


def get_freeze_list():
    # 作用：生成要冻结的层号列表
    # 原理：yolo11x 共 24 层（0~23），model.23 是 Detect 检测头（含 cls/cv2/dfl）。
    #       冻结 0~22 = 冻住全部特征提取（backbone+neck），只留检测头可训。
    #       用原生 freeze 白名单机制（模型重建后依然生效），比手动 requires_grad 可靠。
    return list(range(23))  # 0,1,...,22


def smoke_check(exp_name, freeze):
    # 作用：smoke test 的检查提示（冻结是否生效靠 ultralytics 自带 "Freezing layer" 日志确认）
    # 原理：freeze 由训练器内部 setup_model() 重建模型后应用，外部查原始 model 会误报，
    #       所以只打印提示，让用户去 a3.log 里 grep 确认
    if freeze:
        print(f"[smoke-{exp_name}] 请在 a3.log 中检查大量 'Freezing layer model.X.' 日志"
              f"（X=0~22 均出现=冻结生效），且应出现 'model.23.*' 的训练更新")
    else:
        print(f"[smoke-{exp_name}] 本组不冻结，无需检查 Freezing layer")


def train_one(exp_name, freeze, weighted, smoke=False):
    # 作用：训练单个实验组，训完自动 val 出逐类指标
    print(f"\n===== 开始 [{exp_name}] freeze={freeze} weighted={weighted} =====")
    # 加权补丁是全局的，用 try/finally 保证无论正常/异常结束都恢复原样，绝不污染下一组
    unpatch = None
    try:
        # 每组都从 base* 重新加载，互不污染（冻结是局部改 model，无需 undo）
        model = YOLO(BASE_WEIGHTS)

        if weighted:
            unpatch = patch_loss_with_pos_weight()
            print(f"  [配置] pos_weight 已生效: {CLASS_WEIGHTS}")

        # 冻结开关：用原生 freeze 参数（训练器内部模型重建后依然生效），冻结 0~22 层留检测头
        freeze_arg = get_freeze_list() if freeze else 0
        if freeze:
            print(f"  [配置] 冻结 model.0~22（特征提取），只训检测头 model.23")

        # 统一续训配置：四组一字不改，只有 freeze/weighted 两个开关在变
        results = model.train(
            data=DATA,
            epochs=1 if smoke else 10,   # smoke 只跑 1 轮验证逻辑
            patience=0,                  # 跑满轮数不早停，保证四组轮数一致可比
            freeze=freeze_arg,           # 冻结层列表（原生机制，重建模型后仍生效）
            batch=8,
            imgsz=1024,
            optimizer="AdamW",
            lr0=0.0001,                  # 微调用低 lr（base 是 0.001 的 1/10）
            lrf=0.01,
            warmup_epochs=0,             # 续训无需长 warmup
            cos_lr=True,
            weight_decay=0.0005,
            seed=0,
            deterministic=True,
            # 续训关强增强：不需要 mosaic/旋转，重点是分类头精修
            mosaic=0.0,
            degrees=0.0,
            fliplr=0.5,                  # 翻转保留（不影响类别语义）
            flipud=0.5,
            hsv_h=0.015,
            hsv_s=0.3,
            hsv_v=0.2,
            val=True,
            save=True,
            plots=True,
            project=PROJECT,
            name=exp_name,
            exist_ok=True,
        )

        if smoke:
            # smoke 模式：只验证冻结/加权是否生效，不打印正式指标
            smoke_check(exp_name, freeze)
            return

        # 打印整体分数，方便直接抄进记录表
        m = results.box
        print(f"[{exp_name}] mAP50={m.map50:.4f} mAP50-95={m.map:.4f} P={m.mp:.4f} R={m.mr:.4f}")

        # 用 best.pt 完整 val 一遍，出逐类指标（选最优模型看这里）
        best = YOLO(f"{PROJECT}/{exp_name}/weights/best.pt")
        vm = best.val(data=DATA, imgsz=1024, batch=8, device=0, plots=True,
                      project=PROJECT, name=f"{exp_name}_val", exist_ok=True)
        b = vm.box
        print(f"[{exp_name}-val] mAP50={b.map50:.4f} mAP50-95={b.map:.4f}")
        names = list(vm.names.values())
        try:
            for i in range(len(names)):
                # ap50[i] = 该类别 mAP50，ap[i] = 该类别 mAP50-95
                print(f"  {names[i]}: mAP50={b.ap50[i]:.4f}  mAP50-95={b.ap[i]:.4f}")
        except Exception as e:
            # 不同 ultralytics 版本字段可能略有差异，失败不影响训练，看 results.csv 即可
            print("逐类指标打印失败(版本差异，看 results.csv):", e)
    finally:
        # 加权实验无论正常还是报错结束，都立刻恢复损失函数原样，避免影响下一组
        if unpatch is not None:
            unpatch()
            print(f"  [配置] pos_weight 已恢复（{exp_name} 结束）")


def main():
    # 作用：按顺序跑完全部实验组（先 smoke 验证再正式四组）
    if SMOKE_TEST:
        # smoke 用第一组的配置（不冻结不加权）训 1 轮验证管线
        # 但冻结/加权的验证需要分别测，所以直接对 冻结组 和 加权组 各 smoke 一次成本高；
        # 折中：smoke 跑 a3_decoupled(冻结+加权) 一次，同时验证两个开关都生效
        train_one("a3_smoke", freeze=True, weighted=True, smoke=True)
        print("\n===== smoke test 完成，请检查上方输出后把 SMOKE_TEST 改成 False 再跑正式四组 =====")
        return

    for exp_name, freeze, weighted in EXPERIMENTS:
        train_one(exp_name, freeze, weighted)
    print("\n===== 全部实验完成 =====")


if __name__ == "__main__":
    main()
