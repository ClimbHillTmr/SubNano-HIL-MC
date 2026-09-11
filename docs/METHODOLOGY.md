# METHODOLOGY — 物理链、标定与误差来源

> 配套主驱动：`run_case.py`
> 输出：`analysis/case_40nm_bulk/`
> 核心模块：`src/hil_mc/{cifio, structure, materials, mc, metrics, design, hazards, xrr}`

---

## 1. 端到端物理链

```
┌────────────────────────────────────────────────────────────────────────┐
│  1. CIF 单晶解析  ──►  2. XRR 拟合  ──►  3. 阻止本领  ──►  4. MC 输运 │
│  cifio / structure    xrr              materials           mc          │
│                                                                        │
│                          5. 光刻度量  ──►  6. 设计闭环  ──►  7. 风险   │
│                          metrics          design             hazards    │
└────────────────────────────────────────────────────────────────────────┘
```

每一步的输入只依赖前一步的输出，所有常量集中在 `run_case.py` 顶部。

### 1.1 CIF 解析（`cifio`, `structure`）

- 解析 `data/4.cif`（Olex2/SHELXL 精修输出）。
- 提取核（Ti₆ + 桥氧）、配体（8 正己酸根 + 4 厚朴酚双阴离子）、晶胞
  (a, b, c, α, β, γ)、空间群、Z、R1。
- 计算分子尺寸 `molecule_extent_nm`（用于 CD 下限）、Ti 质量分数、可
  交联位点。

### 1.2 XRR 拟合（`xrr`）

- 读取 `data/sample+4.pdf` 的实验曲线（如有 `.xy` 则直接用）。
- Parratt 递推 + 自由层模型：t（膜厚）、ρ（密度）、σ（界面粗糙度）。
- 输出 `XRRFit.to_dict()` 进入 `results.json`。

### 1.3 阻止本领（`materials`）

- **电子阻止**：Lindhard–Scharff `S_e = k·√E`，Bragg 加和
  `S_e(mix) = Σ wᵢ S_e(i)`；含可调 `e_scale` 校准 He⁺ 在 Si 中的射程。
- **核阻止**：ZBL 普适曲线。
- 能量网格：`np.logspace(0, 3, 241)` keV，覆盖 1 keV → 1 MeV，足以支持
  30 / 50 / 100 keV He⁺。
- 校验：`E_SCALE = 1.10` → 30 keV He⁺ 在 Si 中射程 281.8 nm（SRIM
  282.2 nm，< 1% 误差）。

### 1.4 蒙特卡洛输运（`mc`）

- 类：`HeMonteCarlo(stack, energy_keV, seed=, box_nm=)`。
- 步进方式：电子阻止连续减速（步长自适应）+ ZBL 核散射（随机反弹）。
- **SE1 沉积**：横向 σ = 0.65 nm Gaussian（`se_sigma`），深度衰减
  λz = 5 nm，逃逸深度 4 nm。每步释放 `dE/λz` 给 2D 沉积图。
- **SE2 衬底回注**：离子进入衬底后，背散射或穿透产生的 SE 仍可能反向进
  入抗蚀剂层；MC 单独累计 `substrate_map`。
- **全层 Bragg 曲线**：`depth_profile_full`（`depth_edges_full` 对应）覆
  盖从 0 到 `depth_max_nm`（默认 2500 nm），是离子在抗蚀剂 + 衬底的全
  部能量沉积；Si 区 Bragg 峰位置 ≈ 282 nm。
- **横向盒**：仅为 2D 沉积图的边界，离子物理上不被截断（修复后）；终
  端统计 `n_alive_at_end` 应为 0。

### 1.5 光刻度量（`metrics`）

```
PSF(r)           =  radial_profile(se1_map + se2_map)
LSF(x)           =  Abel_transform(PSF)  （对称镜像到负半轴）
NILS             =  integrate(|dLSF/dx|) / max(LSF)  在 [x1, x2] 区间
CD               =  x2 - x1  （阈值 = RHO_GEL · t_res）
σ_LER(stoch)     =  CD / (NILS · sqrt(N_A · a_eff))
LER(3σ)          =  3 · σ_LER(stoch)
LER(3σ,total)    =  3 · sqrt(σ_LER(stoch)² + σ_surface²)
```

- `converged_lsf(maps, pixel_nm, r_min=8.0, symmetric=True)`：先对多张
  MC 沉积图取径向平均 → `smooth_tail` 抑制远尾噪声 → Abel 变换 → 对
  称镜像。**对称化是 2026-09 修复**——原版只返回半轴 [0, r_max]，把 CD
  算成半宽，NILS 减半（LER 因比值不变而不受影响）。
- `dose_sweep(x, lsf, doses, thresh, w_se, a_eff, sigma_rough_nm=)` 给出
  每个剂量下的 `LineMetrics(cd_nm, nils, ler_3sigma_nm, ler_total_3sigma_nm)`。
- `best_for_cd` / `exposure_latitude` 用于按 CD 目标反查剂量与曝光宽容度。

### 1.6 设计闭环（`design`）

- `cd_floor_nm(molecule_extent_nm)`：CD 下限 = 1 × 分子直径，避免剂量
  扫描在「CD=0」退化。
- `summarise_design(...)`：构建 `DesignSummary`（含 best / target /
  exposure_latitude 与 ler3_budget_*）。
- 4 个 case 对比：40 nm 块体、20 nm SiNₓ 悬空膜、XRR 实测厚度、单晶
  密度抗蚀剂。

### 1.7 风险评估（`hazards`）

- `contrast_curve(dp, de, rho_gel, doses, E_ion, ...)`：D₀ / D₁₀₀ / γ。
- `sputter_yield_he(energy, dose, ...)`：Sigmund 公式 → yield_atoms /
  removed_thickness（He → Ti-簇，Y ≈ 4e-3，几乎可忽略）。
- `beam_heating(dose, dwell_us, ...)`：dT_scan / dT_worst（K），通常安
  全。
- `simulate_line_3d(...)`：泊松潜像 + 阈值 → CD / LER / LWR / 相关长度
  ξ。

---

## 2. 标定常数与误差

| 常量 | 值 | 出处 | 误差 |
|---|---|---|---|
| `E_SCALE` | 1.10 | 反拟合 30 keV He⁺ 在 Si 中射程 | SRIM 282.2 nm → 偏差 < 1% |
| `a_eff` | ~9.17 nm² | 反拟合膜悬空基准最小 LER₃σ = 0.21 nm | 与文献对齐 ±10% |
| `W_SE` | 20 eV | 假设（典型 PMMA 20 eV；Ti-簇可能 15–25 eV） | ±25% |
| `RHO_GEL` | 15 eV/nm³ | 假设（PMMA ~10；HSQ ~30；Ti-簇待定） | ±50% |
| `se_sigma` | 0.65 nm | 文献 Ti-cluster 二次电子斑 | ±0.2 nm |
| `se_lambda_z` | 5 nm | 浅表 SE 主导深度 | ±2 nm |
| `rho_gel` 假设影响 | — | 绝对剂量轴未定标 → 剂量读数为相对量 | — |

**剂量轴未标定声明**：`a_eff` 反拟合、W_SE / RHO_GEL 假设 → 结果中的剂
量值是相对量级；与实测 µC/cm² 的对比需要一次剂量定标实验（dose-to-clear
或 SEM 切片）。

---

## 3. 误差来源分层

1. **物理模型误差**（最大）
   - He⁺ SE 产生截面：MC 用解析积分近似，缺少完整的 dE/dx → SE 角分布
     数据库；估计 10–20%。
   - Bragg 加和电子阻止：Lindhard-Scharff 适用 10 keV – 1 MeV，30 keV
     处于最佳段（误差 <5%）。
   - 双高斯 EBL PSF：忽略前向散射的角分布与显影非线性（这是 EBL 模型本
     身的局限）。

2. **统计误差**（可压低）
   - 主工况 N_MAIN=40000 离子，N_REP=15000；LER 统计不确定度 ≈
     1/√N ≈ 0.5%。
   - 增加 N_MAIN 到 100000 可再降 50%，但 4 case 跑时间翻倍。

3. **标定误差**
   - `a_eff` 反拟合的不唯一性：在固定 LER₃σ=0.21 nm 反拟合时，不同初
     始剂量范围可能给出 ±1 nm² 的差异（9.0–9.4）。
   - `RHO_GEL` / `W_SE` 假设 → 剂量轴整体平移。

4. **离散化误差**（次要）
   - 径向 PSF 网格 0.5 nm；CD 解析精度 ≈ 网格。
   - Abel 变换的 `r_min=8 nm` 远尾平滑会影响 NILS 0.5%。

---

## 4. 已知缺口（设计层）

详见 `analysis/design_verification.md`（22 项符合性表 + 3 项 P0 缺口）：

1. **LER 未耦合表面粗糙度**：本仓库 LER₃σ = 0.215 nm 是随机下限，加 σ
   后总 LER₃σ ≈ 0.75 nm；缺少实测 ξ（粗糙度-边缘传递系数）。
2. **绝对剂量轴未定标**：`ρ_gel` / `W_SE` 为假设、`a_eff` 反拟合；剂量
   数值是相对量级。
3. **缺 CD 下限导致退化最优**：已通过 `cd_floor = molecule_extent` 修
   复。

---

## 5. 修复历史（按日期）

- **2026-09-09** — `REPAIR_SUMMARY.md` 详述构建/测试/lint 全面修复；
  `mc.py` 补全全层 Bragg + 移除横向盒误杀；`metrics.py` 新增粗糙度耦
  合 LER / `best_for_cd` / `exposure_latitude`；`hazards.py` 新增。
- **2026-09-10** —
  - `run_case.py` 补回 `if __name__ == "__main__": main()` 入口
    （原版缺失 → `python run_case.py` 是静默 no-op）；
  - `metrics.converged_lsf` 对称化（CD 7.5 → 15.2 nm，NILS 6.3 → 12.6）；
  - 新增 `src/hil_mc/ebeam.py`（EBL 双高斯 PSF + LER 链路）；
  - 新增 `scripts/compare_hil_ebl.py`（HIL vs EBL 30/100 kV 对比）；
  - 新增 `analysis/hil_vs_ebl/{fig,report.html,results.json,run.log}`；
  - `run_case.py.write_report` 缺失函数补回（首版提交从未定义此函数，
    仅因入口缺失未暴露）。

---

## 6. 复现清单

```bash
# 安装
pip install -r requirements.txt

# 测试 + lint
pytest tests/ -q          # 30 passed
ruff check .               # 0 errors

# 主流程
python run_case.py         # ~3–4 min → analysis/case_40nm_bulk/

# HIL vs EBL
python scripts/compare_hil_ebl.py  # ~1 min → analysis/hil_vs_ebl/

# 设计核验
python scripts/design_check.py
```

每个 case 的 `run.log` 都包含完整的材料/MC/度量日志，方便与本方法学对
照核对。