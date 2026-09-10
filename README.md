# SubNano-HIL-MC · 氦离子光刻亚纳米线边缘粗糙度多物理场蒙特卡洛框架

> **SubNano-HIL-MC**: Multi-Physics Monte-Carlo framework for sub-nanometer Line-Edge-Roughness (LER) in Helium Ion Lithography.

根据国际半导体技术路线图（IRDS），下一代逻辑芯片对线边缘粗糙度要求 **LER < 0.5 nm**。
传统电子束光刻（EBL）受电子前向/背散射与大范围二次电子扩散（邻近效应）限制，难以突破亚纳米瓶颈；
**氦离子光刻（HIL / SHIBL）** 借助氦离子巨大的质量（≈7300×mₑ）带来的强准直性与极小的二次电子（SE）局域化，
可在 **~0.2 nm** 量级实现亚纳米 LER（Zhuang et al. 2024, *Nanotechnology* 35 495301）。

本仓库以一组**真实实验数据**为案例，端到端复现「单晶结构解析 → 阻止本领 → 离子输运 → PSF/NILS → 随机 LER/工艺窗口」
的完整物理链：

- **光刻胶**：Ti₆-氧簇分子胶（Ti-MOC），组成取自单晶文件 `data/4.cif`，膜密度取自 XRR 实测。
- **工况**：光刻胶 40 nm / 衬底 Si(001) 670 µm / 入射 He⁺ 30 keV。
- **结论**：在当前 40 nm 胶厚下，块体 Si 衬底与悬空 SiNₓ 薄膜给出几乎相同的 ~0.21 nm LER（详见 `analysis/case_40nm_bulk/report.html`）。

---

## 📁 目录结构

```
SubNano-HIL-MC/
├── README.md                     # 本文件
├── LICENSE
├── run_case.py                   # ★ 主驱动脚本：解析 4.cif → MC 输运 → 生成 4 图 + report.html + results.json
├── src/
│   └── hil_mc/                   # 核心算法库（纯 Python）
│       ├── cifio.py              # 单晶 CIF 解析（自定义轻量解析器）
│       ├── structure.py          # 簇化学解析：核/配体/桥氧/可交联位点
│       ├── materials.py          # 材料与阻止本领（Lindhard-Scharff 电子阻止 + ZBL 核阻止）
│       ├── mc.py                 # He⁺ 蒙特卡洛输运（HeMonteCarlo）
│       └── metrics.py            # 径向 PSF、Abel 变换求 LSF、NILS、随机 LER、工艺窗口
├── data/                        # 原始实验输入（勿改动）
│   ├── 4.cif                    # Ti6-氧簇单晶（Olex2/SHELXL 精修）
│   ├── sample+4.pdf
│   └── 0828-24355-new/          # XRR 原始数据（4.txt / 4.raw / 4.jip / sample 4.pdf 等）
├── analysis/
│   ├── case_40nm_bulk/          # ★ 当前工况输出（run_case.py 生成）
│   │   ├── fig1_structure.png           # 4.cif 单晶结构解析
│   │   ├── fig2_stack_energy.png        # 轨迹 / 深度剖面 / 能量收支
│   │   ├── fig3_psf_nils.png            # PSF(95% CI) / LSF 分解 / NILS
│   │   ├── fig4_ler_window.png          # LER(剂量) U 型曲线 / CD(剂量) / 2D 工艺窗口
│   │   ├── report.html                  # 完整 HTML 报告（含 4.cif 解读与结论）
│   │   ├── results.json                 # 全部数值结果
│   │   └── run.log
│   └── case_nature/             # Nature 风格可视化重绘 + 审查报告
│       ├── fig1..4_*.png               # 优化后的四张图
│       ├── visualization_optimization.md   # 可视化审查与优化报告
│       └── _cache.pkl                   # 蒙特卡洛数组缓存（内部）
├── docs/                        # 文档与参考资料
│   ├── legacy/                  # 早期「master suite」图与报告（已被 run_case.py 取代，仅留档）
│   │   ├── Master_Visualization_Report.pdf
│   │   ├── master_fig1..5_*.png
│   │   ├── final_experiment_master_suite.png
│   │   └── xrr_sample4_analysis.png
│   ├── references/              # 参考字典/文献页/参考 JSON（CIF core dic、NIST ASTAR、Parratt、LER 参考等）
│   └── source_checks/           # 源数据核查图（sample4 报告页）
└── scripts/                     # 早期辅助/被取代脚本（存档，非主流程）
    ├── analyze_xrr_sample4.py
    ├── generate_4_master_figures.py
    ├── generate_figures_v2.py
    ├── generate_final_experiment_suite.py
    └── export_master_report_pdf.py
```

> 说明：当前**唯一权威流程**是 `run_case.py` + `src/hil_mc/`。`scripts/` 与 `docs/legacy/` 为早期版本，仅供追溯，不再维护。

---

## 🚀 快速开始

依赖：Python 3.9+，`numpy` / `scipy` / `matplotlib`（`scripts/export_master_report_pdf.py` 另需 `reportlab`）。

```bash
# 1) 创建并激活虚拟环境（项目内已含 .venv，可跳过）
python -m venv .venv
.venv/Scripts/activate        # Windows  (或 source .venv/bin/activate on Linux/macOS)

# 2) 安装依赖
pip install numpy scipy matplotlib

# 3) 运行主流程
python run_case.py
```

运行后在 `analysis/case_40nm_bulk/` 生成 4 张图、`report.html` 与 `results.json`（约 2 分钟）。

---

## 🔬 物理链与模块

| 步骤 | 模块 | 作用 |
|---|---|---|
| 1. 单晶解析 | `cifio` + `structure` | 解析 `data/4.cif`：核 `O₄Ti₆`、8 个正己酸根 + 4 个厚朴酚配体、晶胞/密度/R1 |
| 2. 材料/阻止本领 | `materials` | Bragg 加和电子阻止（Lindhard-Scharff，`e_scale=1.40` 标定射程）+ ZBL 核阻止 |
| 3. 离子输运 | `mc` | 30 keV He⁺ 在「胶/衬底」叠层中的随机轨迹、SE1/SE2 能量沉积 |
| 4. 光刻度量 | `metrics` | 径向 PSF → Abel 变换求线扩散函数(LSF) → NILS → 随机 LER → 工艺窗口 |

关键标定（`run_case.py` 顶部常量）：

- `e_scale = 1.40` → 30 keV He⁺ 在 Si 中射程 282 nm（SRIM 282.2 nm，误差 <1%）。
- `a_eff = 9.33 nm²` → 使悬空膜基准最小 LER(3σ) = 0.21 nm（对齐 Zhuang 2024 实测）。
- `w_SE = 20 eV/事件`，`ρ_gel = 15 eV/nm³`，`RESIST_THICKNESS_NM = 40`，`SUBSTRATE_THICKNESS_NM = 670e3`。

---

## 📊 最新结果（工况：40 nm 胶 / 670 µm Si）

来自 `analysis/case_40nm_bulk/results.json`：

**单晶 `data/4.cif`**：`C120 H152 O28 Ti6`，三斜 `P -1`，`Z=1`，`ρ_calc = 1.354 g/cm³`，`R1 = 0.0448`；
无机核 `O₄Ti₆`（2 μ₃-O + 2 μ₂-O），8 个正己酸根 + 4 个厚朴酚双阴离子，Ti 质量分数 12.3%，分子尺寸 ~2.24 nm。

| 指标 | 40 nm / 670 µm Si（本工况） | 40 nm / 20 nm SiNₓ 悬空膜（基准） |
|---|---|---|
| 胶内沉积能量 | 3293 eV（占 11.0%） | 3290 eV |
| 衬底 SE2 回注 | 143 eV/离子（~4.3%） | 245 eV/离子 |
| 背散射系数 η | 0.18%（EBL 为 30–50%） | 0.09% |
| PSF 包络 r50 / r90 / r99 | 0.62 / 1.62 / 3.87 nm | 0.62 / 1.62 / 4.37 nm |
| 最佳剂量 | 245 pC/cm | 188 pC/cm |
| 最佳 CD | 7.58 nm | 7.72 nm |
| NILS | 6.32 | 6.59 |
| **LER 3σ（最佳）** | **0.215 nm** | **0.210 nm** |

**关键结论**

1. He⁺ 在 Si 中射程约 282 nm ≪ 670 µm，故**衬底厚度完全不影响结果**；且在 40 nm 胶厚下，
   **衬底类型（块体 Si vs 悬空膜）也不是 LER 的决定因素**——两者最佳 LER 基本持平（差值 <0.01 nm）。
2. 胶厚扫描（15–80 nm）显示最优 LER 在 20–60 nm 区间基本持平（~0.18–0.21 nm），仅 80 nm 才回升至 ~0.27 nm；
   40 nm 并非明显吃亏。致密化在此模型下对随机 LER 收益有限（且受 `a_eff` 标定锚定影响）。
3. 悬空 SiNₓ 薄膜的真正优势只在**更厚胶 / 更细线宽**时显现。

> 完整推导、图表与 4.cif 逐配体解读见 `analysis/case_40nm_bulk/report.html`。

---

## 📚 参考

- **Zhuang, X., Deng, Y., Zhang, Y., et al.** (2024). *A strategy to fabricate nanostructures with sub-nanometer line edge roughness*. **Nanotechnology**, 35(49), 495301. [DOI: 10.1088/1361-6528/ad6e88](https://doi.org/10.1088/1361-6528/ad6e88)
- Ramachandra, R., Griffin, B., & Joy, D. (2009). *A model of secondary electron imaging in the helium ion scanning microscope*. **Ultramicroscopy**, 109(6), 748–757.
- Manfrinato, V. R., et al. (2013). *Resolution limits of electron-beam lithography toward the atomic scale*. **Nano Letters**, 13(4), 1555–1558.
- Mack, C. A. (2018). *Reducing roughness in extreme ultraviolet lithography*. **JM3**, 17(4), 041006.

---

## 📄 许可证

MIT License — 详见 [LICENSE](LICENSE)。
