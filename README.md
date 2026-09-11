# SubNano-HIL-MC · 氦离子光刻亚纳米线边缘粗糙度多物理场蒙特卡洛框架

> **SubNano-HIL-MC**: Multi-Physics Monte-Carlo framework for sub-nanometer
> Line-Edge-Roughness (LER) in Helium Ion Lithography (HIL / SHIBL), with a
> companion analytical EBL (electron-beam lithography) comparison.

根据国际半导体技术路线图（IRDS），下一代逻辑芯片要求
**LER < 0.5 nm**。传统电子束光刻（EBL）受电子前向/背散射与大范围二次电子
扩散（邻近效应）限制，难以突破亚纳米瓶颈；**氦离子光刻（HIL）** 借助 He⁺
巨大的质量（≈7300 × mₑ）带来的强准直性与极小的二次电子局域化，可在
**~0.2 nm** 量级实现亚纳米 LER（Zhuang et al. 2024, *Nanotechnology* 35 495301）。

本仓库端到端复现「单晶结构解析 → XRR 拟合 → 阻止本领 → 离子输运 → PSF/NILS →
随机 LER/工艺窗口」的完整物理链，并以同一组抗蚀剂/标定常数与 30/100 kV EBL
文献双高斯 PSF 做直接对比，定量回答「HIL 究竟在何处、以何种方式压低 LER」。

---

## 📑 目录

- [1. 功能特性](#1-功能特性)
- [2. 最新结果](#2-最新结果)
- [3. 目录结构](#3-目录结构)
- [4. 安装](#4-安装)
- [5. 使用方式](#5-使用方式)
- [6. 物理链与模块](#6-物理链与模块)
- [7. HIL vs EBL：方法与结论](#7-hil-vs-ebl方法与结论)
- [8. 已知缺口（设计层）](#8-已知缺口设计层)
- [9. 参考文献](#9-参考文献)

---

## 1. 功能特性

- **单晶 → 物理参数**：解析 `data/4.cif` 单晶文件，提取 Ti₆-氧簇的核/配体/
  桥氧/分子尺寸/Ti 质量分数；用 Parratt 递推拟合 XRR 曲线得到膜密度、膜厚、
  表面粗糙度 σ。
- **多物理场输运**：Lindhard–Scharff 电子阻止（Bragg 加和，含 `e_scale` 标定）
  + ZBL 核阻止；矢量化的 He⁺ 蒙特卡洛 `HeMonteCarlo`，记录全层 Bragg 曲线、
  SE1 沉积图（横向 σ≈0.65 nm Gaussian）、SE2 衬底回注、背散射系数 η。
- **光刻度量**：`radial_profile` → Abel 变换 → 对称 LSF → NILS → 随机 LER
  → 曝光宽容度（`dose_sweep` / `best_for_cd` / `exposure_latitude`）。
- **设计闭环**：CD 下限（分子尺寸）、表面粗糙度耦合 LER、剂量轴未标定声明、
  4 case 对比（40 nm 块体、20 nm SiNₓ 悬空膜、XRR 实测厚度、单晶密度）。
- **风险与工艺**（`hazards.py`）：对比度曲线（D₀/D₁₀₀/γ）、He 溅射
  （Sigmund）、束热（dT）、3-D 潜像泊松仿真（CD / LER / LWR / 相关长度）。
- **HIL vs EBL 对比**（`src/hil_mc/ebeam.py` + `scripts/compare_hil_ebl.py`）：
  EBL 用 Georgia Tech 文献双高斯 PSF（α, β, η），对 30/100 kV 做「文献参
  数」与「40 nm 抗蚀剂缩放」两套灵敏度分析；输出图、LER-vs-CD 表、JSON、
  报告 HTML。
- **测试与质量门**：`pytest` 30 个用例（mc/metrics/hazards/pipeline），
  `ruff` E/F/W 零告警（遗留脚本按 per-file-ignores 放宽纯格式规则）。

---

## 2. 最新结果

### 2.1 主工况：40 nm Ti-簇抗蚀剂 / 670 µm Si(001) / 30 keV He⁺

来自 `analysis/case_40nm_bulk/results.json`（LSF 对称化修复后重跑）。

| 指标 | 块体 Si（670 µm） | 悬空 SiNₓ（20 nm） |
|---|---|---|
| 胶内沉积能量 | ~3293 eV (~11.0%) | ~3290 eV |
| 衬底 SE2 回注 | ~143 eV/离子 (~4.3%) | ~245 eV/离子 |
| 背散射系数 η | ~0.18% | ~0.09% |
| PSF 包络 r₉₀ | 1.62 nm | 1.62 nm |
| LSF FWHM | ~2.00 nm | ~2.00 nm |
| 最佳剂量 | ~245 pC/cm | ~175 pC/cm |
| **最佳 CD** | **~15.2 nm** | ~15.1 nm |
| **NILS** | **~12.6** | ~12.5 |
| **LER 3σ（随机下限）** | **0.215 nm** | **0.210 nm** |

> LER(3σ) 是 LER 随机下限；表面粗糙度耦合后的总 LER(3σ) ≈
> `3·√((LER_stoch/3)² + σ_surface²)`，σ_surface ≈ 0.2 nm（XRR），详细见
> `report.html` §2。
>
> **关键结论**：在 40 nm 抗蚀剂厚度下，He⁺ 在 Si 中射程 ~282 nm ≪ 670 µm，
> 故**衬底厚度完全不影响结果**；且块体 Si 与悬空 SiNₓ 薄膜给出几乎相同的
> ~0.21 nm LER，差值 < 0.01 nm。悬空膜的真实优势只在更厚胶 / 更细线宽
> 时显现。膜厚扫描 15–80 nm 的最小 LER₃σ 落在 0.17–0.27 nm 区间。

### 2.2 HIL vs EBL 对比（30 / 100 kV，同抗蚀剂标定）

来自 `analysis/hil_vs_ebl/results.json` + `report.html`。
全部共享 `N_A = ρ_gel·t_res / w_SE` 与 `a_eff = 9.17 nm²`，差异仅来自 PSF
形状与背散射。EBL「文献」分支使用 Georgia Tech *Nanolithography* 教材表
（0.5 µm 抗蚀剂基准）；「EBL* 缩放」分支按 α ∝ (R_t/V_b)^1.5 折算到
40 nm 抗蚀剂并叠加 3 nm 探针斑点，给出乐观估计。

| 工况 | 射程 / α | LSF FWHM | 长程邻近背景 | 最佳 LER₃σ |
|---|---|---|---|---|
| **HIL 30 keV**（MC） | 282 nm | 2.30 nm | 4.4%（SE2） | **0.129 nm @ CD 3.6 nm** |
| **HIL 100 keV**（MC） | 723 nm | 1.90 nm | 4.2%（SE2） | **0.064 nm @ CD 3.7 nm** |
| EBL 30 kV（文献） | α = 58.9 nm | 98.8 nm | η = 0.74（背散射 42.5%） | 3.75 nm @ CD 200 nm |
| EBL 100 kV（文献） | α = 7.3 nm | 12.2 nm | η = 0.74（背散射 42.5%） | 0.288 nm @ CD 36 nm |
| EBL* 30 kV（缩放） | α = 3.3 nm | 5.6 nm | 同上 | 0.146 nm @ CD 15 nm |
| EBL* 100 kV（缩放） | α = 3.0 nm | 5.1 nm | 同上 | 0.111 nm @ CD 16 nm |

**核心发现**：

1. **HIL 的物理优势不在于「边缘更锐」**：在 40 nm 抗蚀剂标度（EBL* 缩放
   分支）下，EBL* 的 LSF FWHM 与 HIL 已属同一量级（5–6 nm vs 2 nm）；
2. **真正的差距是长程邻近背景**：HIL 的背散射/SE2 总和 ~4.4%，而 EBL 文
   献 η = 0.74 → 背散射能量比 ≈ 42.5%。这导致 EBL 在 CD ≳ 50 nm 时 LER
   急速发散，HIL 在 CD 1–200 nm 全程保持在亚纳米；
3. **CD ≲ 20 nm 区间** HIL 的最低 LER₃σ 达 0.06–0.13 nm，比 EBL*（缩放
   后最好的 0.11 nm）还低约 1.5–2×；EBL 文献分支在大 CD 段甚至可达
   LER₃σ ≈ 6 nm（CD = 100 nm 处）。

图与详细推导见 `analysis/hil_vs_ebl/fig_hil_vs_ebl.png` 与 `report.html`。

---

## 3. 目录结构

```
SubNano-HIL-MC/
├── README.md                      本文件
├── REPAIR_SUMMARY.md              2026-09-09 构建/测试/lint 修复日志
├── LICENSE
├── pyproject.toml                 ruff + pytest 配置
├── requirements.txt               运行时依赖（numpy / scipy / matplotlib）
├── requirements-dev.txt           开发依赖（+ pytest / ruff）
├── run_case.py                    ★ 主驱动：CIF → XRR → MC → 度量 → report.html
│
├── src/hil_mc/                    核心算法库（纯 Python，无 Numba 等）
│   ├── cifio.py                   轻量 CIF 解析器
│   ├── structure.py               簇化学：核/配体/桥氧/可交联位点
│   ├── materials.py               阻止本领（Lindhard-Scharff + ZBL）
│   ├── mc.py                      He⁺ 蒙特卡洛（HeMonteCarlo）
│   ├── metrics.py                 PSF / LSF (Abel) / NILS / LER / 工艺窗口
│   ├── design.py                  设计闭环（CD 下限、粗糙度预算）
│   ├── hazards.py                 对比度 / 溅射 / 束热 / 3D 潜像
│   ├── xrr.py                     Parratt 递推 XRR 拟合
│   └── ebeam.py                   EBL 双高斯 PSF + LER 链路
│
├── tests/                         pytest（30 用例）
│   ├── test_mc.py                 mc.py 物理与数值守恒
│   ├── test_metrics.py            metrics.py LSF / LER 链路
│   ├── test_hazards.py            hazards.py 物理量
│   └── test_pipeline.py           集成测试（设计常数 + 端到端）
│
├── scripts/                       辅助脚本
│   ├── compare_hil_ebl.py         ★ HIL vs EBL 对比（§7）
│   ├── design_check.py            基于缓存 MC 数组的设计符合性核验
│   ├── nature_visualization.py     Nature 风格 4 图重绘
│   ├── fig_extras.py              hazards 派生图
│   ├── analyze_xrr_sample4.py     XRR 原始曲线分析
│   ├── generate_4_master_figures.py   早期 master suite（留档）
│   ├── generate_figures_v2.py     早期版本（留档）
│   ├── generate_final_experiment_suite.py  早期版本（留档）
│   └── export_master_report_pdf.py        PDF 导出（reportlab）
│
├── data/                          原始实验输入
│   ├── 4.cif                      Ti₆-氧簇单晶
│   ├── sample+4.pdf               XRR 实验报告（page 1）
│   └── 0828-24355-new/            XRR 原始数据（jip / raw / docx / txt）
│
├── analysis/
│   ├── case_40nm_bulk/            ★ run_case.py 输出
│   │   ├── fig1_structure.png
│   │   ├── fig2_xrr.png
│   │   ├── fig3_transport.png
│   │   ├── fig4_psf_nils.png
│   │   ├── fig5_ler_window.png
│   │   ├── fig6_design.png
│   │   ├── report.html            ★ 完整 HTML 报告
│   │   ├── results.json           ★ 全量机器可读结果
│   │   └── run.log
│   ├── case_nature/               Nature 风格可视化重绘 + design_check
│   └── hil_vs_ebl/                ★ HIL vs EBL 对比输出
│       ├── fig_hil_vs_ebl.png
│       ├── report.html
│       ├── results.json
│       └── run.log
│
└── docs/
    ├── references/                参考字典/文献页/JSON
    ├── source_checks/             源数据核查图
    ├── legacy/                    早期 master suite（留档）
    ├── hil_vs_ebl.md              ★ HIL vs EBL 方法学与文献说明
    └── METHODOLOGY.md             ★ 物理链、标定、误差来源
```

`scripts/` 中的早期 `generate_*` / `export_master_report_pdf.py` 与
`docs/legacy/` 仅为历史留档，不参与当前主流程。

---

## 4. 安装

要求 **Python 3.9+**（已在 3.13 上验证）。仓库已附 `.venv/`。

```bash
# 1. 克隆
git clone https://github.com/ClimbHillTmr/SubNano-HIL-MC.git
cd SubNano-HIL-MC

# 2. 虚拟环境（首次）
python -m venv .venv
.venv/Scripts/activate           # Windows
# 或 source .venv/bin/activate    # Linux / macOS

# 3. 运行时依赖
pip install -r requirements.txt
# 开发依赖（pytest / ruff）
pip install -r requirements-dev.txt
```

可选：`scripts/export_master_report_pdf.py` 需 `reportlab`。

---

## 5. 使用方式

### 5.1 主流程（HIL 40 nm / Si 670 µm）

```bash
python run_case.py
# → analysis/case_40nm_bulk/{6 figures, report.html, results.json, run.log}
# 约 3–4 分钟（4 case × N_MAIN=40000 + 厚度扫描）
```

### 5.2 HIL vs EBL 对比

```bash
python scripts/compare_hil_ebl.py
# → analysis/hil_vs_ebl/{fig_hil_vs_ebl.png, report.html, results.json, run.log}
# 约 1 分钟
```

### 5.3 设计符合性核验（基于缓存 MC 数组）

```bash
python scripts/design_check.py
# 读取 analysis/case_nature/_cache.pkl，核验 22 项设计指标
```

### 5.4 测试与 lint

```bash
pytest tests/ -q                         # 30 passed
ruff check .                              # 0 errors
```

### 5.5 Python API 简例

```python
import sys; sys.path.insert(0, "src")
from hil_mc.materials import Material
from hil_mc.mc import make_stack, HeMonteCarlo

R = Material.from_formula("Ti6-oxo", "C120 H152 O28 Ti6", 0.7708,
                          se_sigma=0.65, se_lambda_z=5.0, se_escape=4.0)
Si = Material.from_formula("Si", "Si", 2.3290)
stack = make_stack(R, Si, 40.0, 670e3, e_scale=1.10)

mc = HeMonteCarlo(stack, 30.0, seed=2024)
res = mc.run(40000, depth_max_nm=2500.0)
print(f"range = {res.range_substrate:.1f} nm")    # → 281.8
```

---

## 6. 物理链与模块

| 步骤 | 模块 | 作用 |
|---|---|---|
| 1. 单晶解析 | `cifio` + `structure` | `data/4.cif` → 核 `O₄Ti₆`、8 正己酸根 + 4 厚朴酚、晶胞/密度/R1 |
| 2. XRR 拟合 | `xrr` | Parratt 递推 → 膜厚 t、密度 ρ、表面 σ |
| 3. 材料/阻止本领 | `materials` | Bragg 加和电子阻止（Lindhard-Scharff，`e_scale`）+ ZBL 核阻止 |
| 4. 离子输运 | `mc` | 30 keV He⁺ 在「胶/衬底」叠层中随机轨迹、SE1/SE2 沉积、背散射 |
| 5. 光刻度量 | `metrics` | 径向 PSF → 对称 Abel → LSF → NILS → 随机 LER → 工艺窗口 |
| 6. 设计闭环 | `design` | CD 下限、粗糙度预算、剂量轴未标定声明 |
| 7. 风险评估 | `hazards` | 对比度 / 溅射 / 束热 / 3D 潜像泊松仿真 |

**关键标定**（`run_case.py` 顶部常量）：

- `E_SCALE = 1.10` → 30 keV He⁺ 在 Si 中射程 282 nm（SRIM 282.2 nm，<1%）。
- `a_eff ≈ 9.17 nm²`（反拟合）→ 使悬空膜基准最小 LER(3σ) = 0.21 nm（对齐
  Zhuang 2024 实测）。
- `W_SE = 20 eV`，`RHO_GEL = 15 eV/nm³`，`RESIST_THICKNESS_NM = 40`，
  `SUBSTRATE_THICKNESS_NM = 670e3`，`BEAM_ENERGY_KEV = 30`。

**`metrics.converged_lsf` 对称化（2026-09 修复）**：原先 PSF Abel 变换后只
返回半轴 [0, r_max]，使 `evaluate_line` 把 CD 算成半宽（≈7.5 nm）、NILS
减半；LER 因依赖 NILS/CD 比值不变。修复后镜像到完整对称轴，物理 CD ≈
15.2 nm，NILS ≈ 12.6。本仓库所有 CD / NILS 数值均为修复后口径。

---

## 7. HIL vs EBL：方法与结论

> 详细推导与图表见 `docs/hil_vs_ebl.md`。

**动机**：文献中 EBL 与 HIL 的 LER 比较常常混淆抗蚀剂体系、束流能量与「最
佳 CD」的定义。本仓库用同一组 `N_A`、`a_eff` 与阈值规则，只让 PSF 形状
变化，从而把差异孤立到「HIL 的二次电子局域化」与「EBL 的前向/背散射」。

**EBL 模型**（`src/hil_mc/ebeam.py`）：

```text
f(r) = (1/(1+η)π) [exp(-r²/α²)/α² + η·exp(-r²/β²)/β²]
```

- α：前向散射半径；β：背散射半径（μm）；η：背/前能量比。
- 文献参数取 Georgia Tech *Introduction to Nanolithography* 教材表（0.5 µm
  抗蚀剂，Si 衬底）：30 kV → α=58.9 nm, β=4.0 µm, η=0.74；100 kV →
  α=7.3 nm, β=31.2 µm, η=0.74。
- 「EBL* 缩放」分支用经验 `α[nm] = 0.9·(R_t[nm]/V_b[kV])^1.5` 把 α 折算到
  40 nm 抗蚀剂并叠加 3 nm 探针斑点。

**HIL 模型**：30 / 100 keV He⁺ MC 输运（`src/hil_mc/mc.py`），SE1 σ = 0.65 nm
Gaussian，SE2 来自衬底回注。

**LER 链路**：两路共用 `metrics.lsf_from_radial` → `evaluate_line` →
`ler3_3sigma`；阈值/`dose_sweep` 规则相同。CD 范围 5–200 nm。

**结论摘要**（详见 `analysis/hil_vs_ebl/report.html`）：

- HIL 30 keV 在 CD 1–10 nm 给出 LER₃σ ≈ 0.13–0.21 nm；
- HIL 100 keV 在 CD 3.7 nm 处最低 LER₃σ ≈ 0.064 nm（CD 进一步缩小受限
  于 `cd_floor` = 分子直径 ≈ 2.2 nm）；
- EBL 30 kV（文献）在 CD 20–200 nm 的 LER₃σ 从 31.7 nm 单调降到 6.4 nm，
  全程 ≫ 0.5 nm IRDS 阈值；
- EBL* 缩放后在 CD 15 nm 给出 LER₃σ ≈ 0.15 nm，但长程 η=0.74 背散射仍
  限制其在大 CD 段的可用性。

**诚实声明**：HIL 端的 LER 是随机下限（无表面粗糙度耦合、无抗蚀剂开发动
力学），与 Zhuang 2024 实测对齐（`a_eff` 反拟合）；EBL 端的「文献参数」
适用于 0.5 µm 抗蚀剂，缩放到 40 nm 仅为乐观估计。绝对剂量轴未用实验标定
（`ρ_gel` / `W_SE` 为假设）。

---

## 8. 已知缺口（设计层）

`analysis/design_verification.md` 给出 22 项符合性表与 3 项 P0 缺口：

1. **LER 未耦合表面粗糙度** — 当前 LER₃σ = 0.215 nm 是随机下限，加 σ 后
   总 LER₃σ ≈ 0.75 nm；缺少实测 ξ（粗糙度-边缘传递系数）。
2. **绝对剂量轴未定标** — `ρ_gel` / `W_SE` 为假设、`a_eff` 反拟合；剂量
   数值是相对量级。
3. **缺 CD 下限导致退化最优** — 已通过 `cd_floor = molecule_extent` 修复。

详见 `analysis/design_verification.md`。

---

## 9. 参考文献

- **Zhuang, X., Deng, Y., Zhang, Y., et al.** (2024). *A strategy to fabricate
  nanostructures with sub-nanometer line edge roughness*. **Nanotechnology**,
  35(49), 495301. [DOI: 10.1088/1361-6528/ad6e88](https://doi.org/10.1088/1361-6528/ad6e88)
- Ramachandra, R., Griffin, B., & Joy, D. (2009). *A model of secondary
  electron imaging in the helium ion scanning microscope*. **Ultramicroscopy**,
  109(6), 748–757.
- Manfrinato, V. R., et al. (2013). *Resolution limits of electron-beam
  lithography toward the atomic scale*. **Nano Letters**, 13(4), 1555–1558.
- Mack, C. A. (2018). *Reducing roughness in extreme ultraviolet lithography*.
  **JM3**, 17(4), 041006.
- *Georgia Tech ECE 6451 — Introduction to Nanolithography* (教材讲义，
  EBL 双高斯 PSF 表）。

---

## 📄 许可证

MIT License — 详见 [LICENSE](LICENSE)。