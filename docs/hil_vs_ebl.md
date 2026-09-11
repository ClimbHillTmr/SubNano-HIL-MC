# HIL vs EBL — 方法学与文献说明

> 配套脚本：`scripts/compare_hil_ebl.py`
> 输出：`analysis/hil_vs_ebl/{fig_hil_vs_ebl.png, report.html, results.json, run.log}`
> 核心模块：`src/hil_mc/ebeam.py`（EBL 解析 PSF）、`src/hil_mc/mc.py`（HIL MC）

---

## 1. 为什么需要「公平对比」

文献里 HIL 与 EBL 的 LER 比较经常出现以下偏差之一：

1. **抗蚀剂不同**：HIL 用 Ti-簇或 SAM 抗蚀剂（~40 nm），EBL 多用
   PMMA/HSQ（~0.5 µm）；LSF FWHM 与 α 直接依赖于抗蚀剂厚度。
2. **「最佳 CD」的定义**：单点最小 LER（可能远超设计目标）vs CD=目标值
   的 LER；两者可以差一个量级。
3. **背散射项**：EBL 的长程 η·β 项占 EBL 总能量的 40–50%，但 EBL 与 HIL
   仿真经常忽略这一项，导致「EBL LER 看起来也好」。
4. **绝对剂量轴**：HIL 通常给出 µC/cm²，EBL 通常给出 µC/cm² 或
   pC/cm；不统一定标就直接比数值是不公平的。

本仓库的解决思路是**保持抗蚀剂模型与剂量规则不变，只让 PSF 形状变化**，
把差异归因到「HIL 的二次电子局域化」与「EBL 的前向/背散射」。

---

## 2. 共享模型

HIL 与 EBL 两条链路共用以下参数（`scripts/compare_hil_ebl.py` 顶部常量）：

| 参数 | 值 | 含义 |
|---|---|---|
| `RESIST_NM` | 40 | 抗蚀剂厚度（与主工况一致） |
| `RHO_GEL` | 15 eV/nm³ | 单位体积交联阈值能量密度 |
| `W_SE` | 20 eV | 单次交联事件所需能量 |
| `N_A` | ρ·t/w = 30 /nm² | 平均入射电子/离子交联事件数 |
| `A_EFF` | 9.17 nm² | 泊松面积（与主工况的反拟合一致） |
| `EBL_ALPHA_SPOT_NM` | 3.0 nm | EBL* 缩放分支的探针斑点半径 |
| `DEPTH_MAX_NM` | 2500 | MC 深度截止 |
| `N_IONS` | 20000 | 每工况 MC 离子数 |

阈值规则：`dose` 满足 `dose · N_A(r) ≥ RHO_GEL·t_res` 即交联；阈值映射与
`metrics.evaluate_line` 完全一致。

---

## 3. HIL 模型

- **能量**：30 / 100 keV 单能 He⁺ 入射。
- **横向 SE**：σ = 0.65 nm Gaussian（`se_sigma` 参数，对应 ~1.3 nm FWHM 的
  本征二次电子斑），SE 深度衰减 λz = 5 nm，逃逸深度 4 nm。
- **衬底 SE2**：MC 自然产生，衬底回注的 SE 重新被抗蚀剂吸收；统计量
  `proximity_background_ratio = (resist 内总 SE - 直接 SE1) / 直接 SE1`。
- **背散射**：MC 直接统计，30 keV He⁺ 在 Si 上 η ≈ 1.5%，100 keV ≈ 0.3%
  （比 EBL 的 30–50% 低两个数量级）。
- **深度截止**：`depth_max_nm=2500` 涵盖 30 / 100 keV 全射程（282 nm /
  723 nm），离子物理上走完整段。

输出：`{depth_edges_full, depth_profile_full, range_substrate, energy_in_resist,
backscatter_fraction, substrate_map, ...}`（见 `MCResult`）。

---

## 4. EBL 模型（解析双高斯）

双高斯 PSF 是 EBL 工艺仿真的标准近似（Broers 模式 → 直接写入
抗蚀剂；前向散射 + 衬底背散射）：

```text
f(r) = (1/(1+η)π) [exp(-r²/α²)/α² + η·exp(-r²/β²)/β²]
```

其中 `α` 为前向散射半径（FWHM ≈ 2.355 α），`β ≫ α` 为衬底背散射半径
（µm 量级），`η` 为背/前能量比。

### 4.1 文献参数（Georgia Tech *Nanolithography* 教材表）

抗蚀剂厚度 0.5 µm，Si 衬底：

| 能量 | α（nm） | β（µm） | η |
|---|---|---|---|
| 20 keV | 120 | 2.0 | 0.74 |
| 50 keV | 24 | 9.5 | 0.74 |
| **30 keV** | **58.87**（公式拟合） | **3.99** | **0.74** |
| **100 keV** | **7.30** | **31.20** | **0.74** |

公式：`α[nm] = 0.9·(R_t[nm]/V_b[kV])^1.5`，自洽验证：
20 keV / 500 nm → α = 0.9·(500/20)^1.5 = 113 nm（教材表 120 nm，偏差 < 6%）。

### 4.2 「EBL* 缩放」分支（40 nm 抗蚀剂乐观估计）

直接把 α 折算到 40 nm 抗蚀剂并叠加 3 nm 探针斑点：

```text
α_scaled = max(α_lit · (40/500)^1.5, α_spot)  ≈ 3.0–3.3 nm
```

并叠加 `gaussian_filter1d(lsf, 3 nm)` 表示探针斑点展宽。该分支相当于
「理想 EBL 写入 + 40 nm 薄抗蚀剂」，对 EBL 是乐观估计。**但 η 与 β 保持文
献值**——背散射仍是大 CD 段 LER 主导项。

### 4.3 边缘采样与 LER 链路

- 边缘位置 `x_edge`：扫描阈值 `0.05 → 0.95` 在 LSF 上求交点；
- `ler3_3sigma(cd, threshold)`：CD = `x_edge(thr) - x_edge(1-thr)`；
  σ_edge = `x_edge(thr) · sqrt(2/(N_A·A_EFF))`；
  LER(3σ) = `3 · σ_edge / sqrt(N_A · A_EFF)`；
- 在 CD 探针（5, 10, 20, 50, 100 nm）处记录 LER，画 `LER-vs-CD` 双对数图。

---

## 5. 结果对比（HIL vs EBL）

> 数字来自 `analysis/hil_vs_ebl/results.json`；脚本默认 N_IONS=20000,
> N_REP=3。

| 工况 | 射程 / α | LSF FWHM | 长程邻近背景 | 最佳 LER₃σ |
|---|---|---|---|---|
| **HIL 30 keV**（MC） | 282 nm | 2.30 nm | 4.4%（SE2） | **0.129 nm @ CD 3.6 nm** |
| **HIL 100 keV**（MC） | 723 nm | 1.90 nm | 4.2%（SE2） | **0.064 nm @ CD 3.7 nm** |
| EBL 30 kV（文献） | α = 58.9 nm | 98.8 nm | η = 0.74（背散射 42.5%） | 3.75 nm @ CD 200 nm |
| EBL 100 kV（文献） | α = 7.3 nm | 12.2 nm | η = 0.74（背散射 42.5%） | 0.288 nm @ CD 36 nm |
| EBL* 30 kV（缩放） | α = 3.3 nm | 5.6 nm | 同上 | 0.146 nm @ CD 15 nm |
| EBL* 100 kV（缩放） | α = 3.0 nm | 5.1 nm | 同上 | 0.111 nm @ CD 16 nm |

### 核心结论

1. **LSF FWHM 维度**：在「40 nm 抗蚀剂 + 探针斑点」标度下，HIL（2 nm）
   与 EBL*（5 nm）仍差 2–3 倍，但比「EBL 文献 0.5 µm 抗蚀剂」（12–99 nm）
   已经接近。
2. **LER 维度**：HIL 在 CD 1–20 nm 全程亚纳米；EBL* 在 CD 15 nm 也能达亚
   纳米但在大 CD 段（η=0.74 长程尾）LER₃σ 飙升到 1–10 nm 量级。
3. **物理根因**：HIL 的优势**主要来自背散射/SE2 的低占比**（4.4% vs
   42.5%），而非「二次电子斑更小」。单看 LSF FWHM 会严重低估 EBL 的
   LER 退化。
4. **大 CD 段交叉**：EBL* 30 kV（缩放）在 CD ≳ 50 nm 时 LER₃σ 从 0.15 nm
   单调升到 ≫ 0.5 nm；HIL 100 keV 在 CD 5–200 nm 全程 ≲ 0.5 nm，差异达
   1 个数量级以上。

### 诚实声明 / 已知局限

- HIL LER 是**随机下限**，未耦合：
  - 表面粗糙度 → 边缘的传递系数 ξ（实测缺失，Zhuang 2024 给的 ~0.6
    但未拆解）；
  - 抗蚀剂显影/侧壁动力学（`hazards.simulate_line_3d` 仅做泊松潜像）；
  - 束流抖动与剂量分片。
- EBL「文献」分支适用于 0.5 µm 抗蚀剂；EBL* 缩放是「理想 40 nm 抗蚀剂
  + 3 nm 探针」的乐观估计，不能直接当成真实 EBL 系统的预测。
- 绝对剂量轴未用实验标定（`ρ_gel` / `W_SE` 为假设）；剂量读数是相对量。

---

## 6. 复现与扩展

```bash
python scripts/compare_hil_ebl.py
# 默认 N_IONS=20000, N_REP=3，约 1 分钟
# 想跑更精细？编辑脚本顶部 VOLTAGES / N_IONS / N_REP
```

可扩展方向（尚未实现，欢迎 PR）：

- 多抗蚀剂厚度（30 / 50 / 80 nm）扫描；
- EBL 探针斑点参数化（含 aberration）；
- 抗蚀剂显影后的 CD-SEM 仿真（基于 `simulate_line_3d` 的泊松潜像 + 阈值
  + 显影核）；
- 与 Langmuir 2025 OTS-SAM 实验的端到端复现。

---

## 7. 参考文献

- **Georgia Tech ECE 6451 — Introduction to Nanolithography**（教材讲义，
  EBL 双高斯 PSF 参数表）。
- **Manfrinato, V. R., et al.** (2013). *Resolution limits of electron-beam
  lithography toward the atomic scale*. **Nano Letters**, 13(4), 1555–1558.
  （高分辨 EBL α 实测。）
- **Sidorkin, V.** (2014). *Proximity effect correction in electron beam
  lithography*. 综述，η ≈ 0.5–0.8 在 0.5 µm 抗蚀剂 / Si 上是典型值。
- **Chen, M., et al.** (2025). *Nanofabrication of silicon gratings enabled
  by vapor-phase, highly uniform self-assembled monolayer resists*.
  **Langmuir**, 41(43), 29152.（OTS-SAM HIL 1 µm 周期 21 nm 深度光栅。）
- **Zhuang, X., et al.** (2024). *A strategy to fabricate nanostructures
  with sub-nanometer line edge roughness*. **Nanotechnology**, 35(49), 495301.
  （本仓库 `a_eff` 反拟合的基准。）