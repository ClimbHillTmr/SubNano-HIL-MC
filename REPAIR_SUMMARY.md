# SubNano-HIL-MC — 修复工作总结 (Repair Summary)

**日期：** 2026-09-09
**范围：** 对 HIL 蒙特卡洛项目的构建/运行/测试/lint 进行全面排查与修复，并重新验证清零。
**设计符合性说明：** 本文件只覆盖"构建/测试/lint 修复"；设计目标是否达成是另一项独立审查，结论见 `analysis/design_verification.md`（判定：部分实现，3 项 P0 缺口）。

---

## 1. 验证总览（全部清零）

| 检查项 | 命令 | 结果 |
|---|---|---|
| 单元测试 | `.venv/Scripts/python.exe -m pytest tests/ -q` | **30 passed** |
| Lint | `.venv/Scripts/python.exe -m ruff check .` | **All checks passed (0 errors)** |
| 主流程 | `python run_case.py` | **成功**，369 s，4 图 + `results.json` + `report.html`，`a_eff=9.17 nm²` |
| Nature 图 | `python scripts/nature_visualization.py` | **成功**，fig1–4，DONE |
| Hazards 图 | `python scripts/fig_extras.py` | **成功**，exit 0，模块自洽 |

`src/` 与 `tests/` 在完整 E/F/W 规则下**零告警**；遗留分析脚本 (`scripts/*.py`) 仅放宽纯格式规则 (E501/E702/E701/E402)，F 类（真实错误）与其余 E 类仍强制。

---

## 2. 排查出的待修复项（按类别）

1. **编译/运行失败** — 仓库重组后 5 个遗留脚本硬编码旧路径（`.gemini` 产物目录、`docs/*.png`），在非根目录运行即报错；`export_master_report_pdf.py` 静默跳过缺失图片。
2. **Lint 告警 (259 处)** — 自动修复 195 处（W291/W293 空白、F401 未用导入、F541 空 f-string）；剩余 64 处均为遗留脚本的纯格式规则，经 per-file-ignores 放宽。
3. **测试不通过 (1 处)** — `test_trajectories_record_depth_correctly` 断言横向展宽 `<200 nm`，该约束编码了已被修复的旧行为（在 80 nm 盒边界杀掉离子），现已不适用。
4. **真实死代码 (F841, 2 处)** — `export_master_report_pdf.py` 未用的 `styles`、`generate_4_master_figures.py` 未用的 `im`。
5. **物理缺陷 (2 项，mc.py)** — 缺陷1：深度沉积直方图只覆盖光刻胶层，缺全层 Bragg 曲线；缺陷2：80 nm 横向盒边界提前终止离子，使 `range_substrate` 与轨迹末端深度不一致（偏差 ~79%）。
6. **功能缺口 (metrics.py / hazards.py)** — 缺少粗糙度耦合 LER、按 CD 求优、曝光宽容度、对比度曲线、溅射/加热/3D 潜像等。

---

## 3. 按文件汇总改动

### 配置 / 构建
- **`pyproject.toml`**（新增）：ruff `line-length=100`、`select=["E","F","W"]`、`ignore=["E741"]`；`per-file-ignores` 对 `run_case.py` 与 5 个遗留脚本放宽格式规则；`[tool.pytest.ini_options]` 设 `pythonpath=["src","."]`、`testpaths=["tests"]`。
- **`requirements.txt` / `requirements-dev.txt`**（新增）：numpy/scipy/matplotlib/pillow/reportlab；dev 加 pytest/ruff/pyflakes。

### 核心源码 `src/hil_mc/`
- **`mc.py`** — 缺陷1：新增 `depth_max_nm` 参数与 `depth_edges_full`/`depth_profile_full`（全层 eV/nm·ion 深度剂量/Bragg 曲线）。缺陷2：移除横向盒边界终止逻辑（盒仅为 2D 图域），新增诊断字段 `n_alive_at_end`/`n_escaped_lateral`/`energy_transmitted_out`，步数上限升至 200000。`range_substrate≈222 nm` 现已与轨迹末端深度一致；Si 区 Bragg 峰 ~107–112 nm。
- **`metrics.py`** — 新增关键字 `sigma_rough_nm` → `ler_total_3sigma_nm = 3·√(ler₁σ²+σ²)`（粗糙度耦合 LER，默认=随机下限）；新增纯函数 `best_for_cd()` 与 `exposure_latitude()`；位置参数旧调用保持兼容。
- **`hazards.py`**（新增）— `contrast_curve()`（D0=0.40/D100=338/γ=0.34）、`sputter_yield_he()`（Sigmund，Y=3.78e-3，可忽略）、`beam_heating()`（ΔT~1e-6 K，安全）、`simulate_line_3d()`（带种子的泊松随机潜像，CD=9.48 nm / LER₃σ=4.91 nm）。
- **`cifio.py` / `structure.py` / `materials.py`** — lint 修复：删未用变量/导入、重命名歧义名 `l`、拆分长行。

### 测试 `tests/`（全部新增）
- **`test_mc.py`**（8）：锁定缺陷1/2 修复（全层剖面、积分守恒、Bragg 峰在 Si 内、`n_alive_at_end==0`、末端深度一致、向后兼容、`depth_max_nm`）。
- **`test_metrics.py`**（14）：σ 耦合、按 CD 求优、曝光宽容度。
- **`test_hazards.py`**（7）：对比度/溅射/加热/3D 潜像。
- **`test_pipeline.py`**（集成）：设计常数、CIF 解析、输运/能量守恒、地图有限性、亚纳米 PSF、轨迹列序、LSF/NILS 一致性。

### 脚本 `scripts/`
- **`run_case.py`** — lint 修复（删未用导入/变量、拆 35 处分号行）；**修复轨迹深度 bug**：`ax.plot(tr[:,0], tr[:,2], ...)`（原误用 `tr[:,1]` vs `tr[:,0]`，把横向-横向误标为深度）。
- **`nature_visualization.py`** — 拆 17 处分号行、包裹长行；确认 4 图正常渲染。
- **`export_master_report_pdf.py`** — 删未用 `styles` 及其导入（F841）；缺失图片时改为抛错（不再静默跳过）；Agent 1 已修 CWD 路径。
- **`generate_4_master_figures.py`** — 删未用 `im =` 赋值（F841）；Agent 1 已修路径。
- **`analyze_xrr_sample4.py` / `generate_figures_v2.py` / `generate_final_experiment_suite.py`** — Agent 1 修 CWD/路径与 `.gemini` 死代码；纯格式规则经 per-file-ignores 放宽。
- **`design_check.py`**（新增）— 基于缓存 MC 数组的设计符合性核验（Si 止停深度、LSF FWHM、CD 约束最优、工艺窗口、粗糙度耦合 LER）。
- **`fig_extras.py`**（新增）— fig5_extras.png（320 dpi，基于 hazards 模块）。

### 文档 / 分析 `analysis/`
- **`design_verification.md`**（新增）— 22 项符合性表，判定"部分实现"，3 项 P0 缺口：LER 未含粗糙度、剂量轴未定标、缺 CD 下限导致退化最优。

---

## 4. 修复根因与验证要点

- **mc.py 缺陷2 根因**：80 nm 横向盒在每次步进后杀掉越界离子，导致约 79% 离子在止停前被截断，深度统计偏浅、`range_substrate` 与轨迹末端深度偏差。修复后盒仅作为 2D 沉积图域，离子物理上走完整个射程。
- **测试断言修正根因**：原 `np.abs(x).max() < 200.0` 隐含"离子在盒边界被截断"的旧假设；修复后横向展宽物理上无界（实测尾部 ~256 nm，仍 < 3×射程），改为 `np.abs(x).max() < 3.0*(RESIST_THICKNESS_NM + range_substrate)` 的物理合理上界。
- **全链路验证**：`run_case.py` 完整跑通 4 种 MC 工况 + 厚度扫描（15→80 nm，最小 LER₃σ 0.174–0.268 nm）并写出 `results.json`/`report.html`；`nature_visualization.py` 与 `fig_extras.py` 均成功产出图片。

---

## 5. 仍未解决的（非"修复"范畴，属设计层）

见 `analysis/design_verification.md`：① 10 nm 隔离线在当前"最优剂量"下不可达（CD 7.58 nm）；② LER 0.198 nm 仅为随机下限，粗糙度耦合预测为 0.753 nm；③ 绝对剂量轴未用实验定标（ρ_gel=15 eV/nm³ 假设、a_eff=9.33 反拟合）；④ 未建 Si Bragg 峰物理模型（mc.py 现有曲线为 Lindhard-Scharff 前载电子止停所致，非 SRIM 282 nm 投影射程）。
