# 长尾遥感路口检测 · 受控实验研究

> Long-tailed / Confusion Road-Junction Detection — A Controlled-Ablation Study on YOLO11x

面向 1024×1024 遥感影像的交通路口检测（十字 / T 型 / 环形 3 类），类别分布严重长尾（78.5% / 20.9% / 0.56%）。本项目不是单纯"调参冲分"，而是做了一份**受控、可归因**的长尾问题研究：先排查数据污染，再以单一变量原则扫描数据增强策略（mosaic 关闭时机 close_mosaic）定基线，最后用 2×2 析因消融验证两个方法干预的独立与组合效果。

**核心结论**：T 型路口弱 = "样本少 + 形态混淆"叠加；**冻结特征提取（结构层）× 分类损失类别加权（损失层）**两个干预单独用各有短板、组合后互补。最优解耦方案 mAP@0.5:0.95 0.317 → 0.376（相对 +18.7%），十字与 T 型同步提升、三类无回退。

---

## 主要结果（同一份 val，499 张，全部实测）

| 方法 | 冻结 | 类别加权 | 十字 | T型 | 环形 | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|---|---|---|
| base（对照，close_mosaic=10） | — | — | 0.503 | 0.226 | 0.223 | 0.530 | 0.317 |
| base\*（close_mosaic=15，最优基线） | — | — | 0.5235 | 0.2723 | 0.3001 | 0.6056 | 0.3653 |
| a3-ref（纯续训 10 轮 · 底座） | ❌ | ❌ | 0.5392 | 0.2851 | 0.2853 | 0.6203 | 0.3699 |
| a2（续训 + 加权） | ❌ | ✅ | 0.5484 | **0.2931** | 0.2619 | 0.6206 | 0.3678 |
| a3-ctrl（续训 + 冻结） | ✅ | ❌ | 0.5428 | 0.2837 | 0.2929 | 0.6205 | 0.3731 |
| **a3-decoupled（冻结 + 加权）** | ✅ | ✅ | **0.5439** | 0.2868 | **0.2982** | 0.6164 | **0.3763** |

**归因要点**（为什么必须做 2×2 + 一个纯续训底座）：
- `a3-ref` = 剥离"多训 10 轮"本身的收益（+0.005），不算方法功劳
- `a2` vs `a3-ref` → 加权的净贡献；`a3-ctrl` vs `a3-ref` → 冻结的净贡献
- `a3-decoupled` vs `a3-ctrl` → 冻结基础上加权的额外贡献（最终组合 0.3763）
- 环形 val 仅 27 框，其 mAP@0.5:0.95 波动大，只报告、不宣称"显著提升"

---

## 目录结构

```
├── scripts/                  # 训练脚本（云端 AutoDL 运行）
│   ├── train_base.py         # A1 基线（close_mosaic=10）
│   ├── train_scan_cm.py      # close_mosaic 扫描 {15, 20}
│   └── train_a3.py           # A3 系列 2×2 消融四组（含 smoke test 开关）
├── runs_longtail/            # 全部实验结果（best.pt / 混淆矩阵 / PR 曲线 / results.csv）
│   ├── base/ scan_cm15/ scan_cm20/     # A1 + 扫描
│   └── a3_ref/ a2_weighted/ a3_ctrl/ a3_decoupled/   # 2×2 消融
├── 长尾路口检测研究-执行手册.md  # 研究全流程：诊断→定稿→执行→汇总（实验记录总表）
```

> 数据集（dataset_clean）与历史污染数据属于竞赛材料，未包含在本仓库内。

---

## 复现

### 环境

- Python 3.10、PyTorch 2.1、`ultralytics==8.4.137`
- 训练：AutoDL RTX 3090（单卡 24G），脚本内路径为云端约定路径，本地跑需自行替换
- 数据：需要 YOLO 格式的 `dataset_clean`（train 4002 图 / val 499 图，零重叠），并准备 `dataset_clean.yaml`

### 第一步：A1 基线（可选，已有结果则跳过）

```bash
# 云端
cd /root/autodl-tmp && nohup python train_base.py > base.log 2>&1 &
```

### 第二步：close_mosaic 扫描（选最优基线 base*）

```bash
cd /root/autodl-tmp && nohup python train_scan_cm.py > scan_cm.log 2>&1 &
```

脚本串行训 `close_mosaic ∈ {15, 20}`，每组跑满 30 轮（patience=0 保证组间公平），训完自动 val 并打印逐类指标。以整体 mAP@0.5:0.95 为主、T 型为辅选 base\*（本项目选 cm=15）。

### 第三步：A3 系列 2×2 消融

```bash
cd /root/autodl-tmp && nohup python train_a3.py > a3.log 2>&1 &
```

- 脚本从 `BASE_WEIGHTS`（须改成你的 base*/best.pt 路径）续训 10 轮，串行跑四组：a3-ref → a2 → a3-ctrl → a3-decoupled
- 两个研究变量：
  1. **冻结**：ultralytics 原生 `freeze=[0..22]`，冻结特征提取（backbone+neck），只训检测头 model.23
  2. **类别加权**：monkeypatch `v8DetectionLoss.__init__`，给 `self.bce` 挂 `pos_weight=[1.0, 1.937, 11.789]`（等价 class-balanced loss，不碰源码）
- 先跑 smoke：把脚本里 `SMOKE_TEST=True`，验证 a3.log 出现 `Freezing layer 'model.0~22'` 后再改 `False` 跑正式四组

### 已知实现要点（踩过的坑）

1. 冻结**不要**用 `requires_grad=False` 手动改——ultralytics 训练器内部 `setup_model()` 会重建模型并强制解冻非白名单参数，手动冻结会被冲掉；必须用原生 `freeze` 参数
2. 加权 monkeypatch 会改类方法，务必用 `try/finally` 在每组结束后 `unpatch()`，否则污染后续不加权对照实验

---

## 一句话给面试官

> 长尾路口检测里 T 型 mAP@0.5:0.95 只有十字的一半，我用受控实验（同起点 / 同数据 / 同轮数）拆解病因：T 型弱 = 样本少 + 形态混淆叠加。单一手段——类别加权拉高 T 型但压崩环形，冻结特征稳但不进功；2×2 消融证明**冻结 × 加权互补**，组合方案 mAP@0.5:0.95 0.317→0.376 且三类不降。研究的价值不在绝对分数，而在每一步增益都可指认来源、可复现。
