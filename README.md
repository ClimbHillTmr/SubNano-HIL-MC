# SubNano-HIL-MC: Multi-Physics Monte Carlo Framework for Sub-Nanometer Line Edge Roughness in Helium Ion Lithography

[![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://img.shields.io/badge/Reference-Nanotechnology%2035%20(2024)%20495301-success.svg)](https://doi.org/10.1088/1361-6528/ad6e88)

A comprehensive, publication-grade **Multi-Physics Monte Carlo Simulation Framework** dedicated to modeling particle scattering, secondary electron (SE) delocalization, point spread functions (PSF), normalized image log-slope (NILS), and stochastic line edge roughness (LER) in **Scanning Helium Ion Beam Lithography (SHIBL/HIL)** versus **Electron Beam Lithography (EBL)**.

---

## 🌟 Overview & Key Innovations

Sub-nanometer line edge roughness ($< 0.5\text{ nm}$) is a crucial benchmark outlined by the International Roadmap for Devices and Systems (IRDS) for next-generation logic chips. While traditional Electron Beam Lithography (EBL) suffers from severe forward/backscattering and broad secondary electron diffusion (the proximity effect), **Helium Ion Lithography (HIL)** on **suspended $\text{SiN}_x$ membranes** provides an unprecedented route to **$0.2\text{ nm}$ sub-nanometer LER**.

This framework models the complete physics-to-metrology chain:
1. **Microscopic Trajectory & SE Delocalization**: Full stochastic tracking of primary ions/electrons and low-energy secondary electrons ($\text{SE1}$ & $\text{SE2}$).
2. **Membrane Energy Filter**: Modeling the transmission of collimated $\text{He}^+$ ions through a $20\text{ nm}$ suspended membrane, eliminating backscattered $\text{SE2}$ background.
3. **Continuous Field & NILS Metrology**: Dual-scale Point Spread Function ($\text{PSF}$) extraction with Bootstrap 95% Confidence Intervals and Normalized Image Log-Slope ($\text{NILS}$) calculations.
4. **Stochastic LER & Polymer Percolation**: Synthesis of 2D/3D latent images with discrete Poisson shot noise, development threshold percolation, and Gaussian-fitted LER distribution analysis.
5. **Process Window Landscape**: Multi-parameter optimization across exposure dose ($p\text{C/cm}$), designed exposure linewidth ($\text{DEL}$), and resist film thickness (compatible with XRR-measured density).

---

## 📊 Master Visual Suite (The 4 Scientific Pillars)

### 1. Microscopic Interaction Volume & Statistical Energy Containment
Demonstrating how the huge mass of helium ions ($\approx 7300 \times m_e$) prevents lateral deviation and confines $99\%$ of deposited energy within a $< 1\text{ nm}$ core.

![Figure 1: Interaction Volume](master_fig1_interaction_volume.png)

* **EBL (30 kV & 100 kV)**: Large teardrop interaction bulb; $99\%$ energy envelope extends outward by tens to hundreds of nanometers (proximity effect haze).
* **HIL (30 kV & 100 kV)**: Collimated, laser-straight trajectory; $99\%$ energy containment tightly locked around the central axis, completely restricting electron diffusion.

---

### 2. Point Spread Function (PSF 95% CI) & Normalized Image Log-Slope (NILS)
Quantitative transfer function bridging microscopic energy deposition with spatial chemical contrast.

![Figure 2: PSF & NILS](master_fig2_psf_nils.png)

* **PSF Profile (a)**: On a logarithmic scale, HIL exhibits a needle-like core with a steep $4$-order-of-magnitude energy drop at $5\text{ nm}$, with zero backscattering baseline.
* **NILS Profile (b)**: Across a $10\text{ nm}$ isolated line, HIL achieves an ultra-steep $\text{NILS} \approx 9.5$, compared to $\text{NILS} \approx 1.8$ for EBL, delivering immense immunity to stochastic shot noise.

---

### 3. Line Edge Roughness (LER) Waveforms & Stochastic Deviation Distributions
Direct mathematical proof of sub-nanometer LER via dual-layer visual representation.

![Figure 3: LER Statistics](master_fig3_ler_statistics.png)

* **Physical Waveform (Top)**: Real-space developed boundary with $\pm 3\sigma$ confidence bands.
* **Probability Density (Bottom)**: Edge deviation histogram with Gaussian normal distribution fit. EBL shows a wide, flat variance ($\text{LER} \approx 5-15\text{ nm}$), whereas HIL converges to a sharp, needle-like distribution ($\text{LER} \approx 0.21\text{ nm}$).

---

### 4. Lithographic Process Window Landscape
Mapping the fundamental trade-off between **Shot Noise** and **PSF Tail Broadening** across resist thicknesses and feature dimensions.

![Figure 4: Process Window](master_fig4_process_window.png)

* **U-shaped LER Curve (a)**: Identifies the three distinct regimes—(1) Shot Noise Limited ($< 15\ p\text{C/cm}$), (2) Optimal Line Edge Polishing Window ($15–80\ p\text{C/cm}$, $\text{LER} \approx 0.2\text{ nm}$), and (3) PSF Tail Dominating ($> 80\ p\text{C/cm}$).
* **2D Process Landscape (b)**: High-resolution parameter contour map highlighting the sub-$0.25\text{ nm}$ LER sweet spot in $(\text{DEL}, \text{Dose})$ space.

---

## 🚀 Quick Start

### Installation
Ensure Python 3.9+ is installed with standard scientific computing packages:
```bash
git clone https://github.com/climbhilltmr/SubNano-HIL-MC.git
cd SubNano-HIL-MC
pip install numpy scipy matplotlib
```

### Running the Complete Simulation
Generate all 4 publication-grade figures with a single command:
```bash
python generate_4_master_figures.py
```

### Material Customization (XRR Measured Film Density)
You can directly customize the film density (measured via X-ray Reflectometry, XRR) and resist thickness in `generate_4_master_figures.py`:
```python
# In generate_4_master_figures.py
density = 1.40  # Replace with XRR measured HSQ film density (g/cm^3)
sim = ParticleSimulation(particle_type='helium', initial_energy_kev=30.0, density=density)
```

---

## 📖 Theoretical Formulations

1. **Screened Rutherford Scattering**:
   $$\frac{d\sigma}{d\Omega} = \frac{Z^2 e^4}{4 E^2 (1 - \cos\theta + 2\alpha)^2}$$
2. **Modified Bethe Energy Loss**:
   $$\frac{dE}{ds} = -7.85 \times 10^4 \frac{\rho Z}{A E} \ln\left( \frac{1.166(E + 0.85 J)}{J} \right)$$
3. **Analytical Stochastic LER Scaling (Mack Model)**:
   $$\sigma_{\text{LER}} \propto \frac{\sigma_{\text{dose}}}{\partial \text{Dose}/\partial x} = \frac{1}{\text{NILS} \cdot \sqrt{\text{Dose}}}$$

---

## 📚 References & Citation

If you find this simulation framework useful in your research, please cite the corresponding literature:

* **Zhuang, X., Deng, Y., Zhang, Y., Wang, K., Chen, Y., Gao, S., Xu, J., Wang, L., & Cheng, X.** (2024). *A strategy to fabricate nanostructures with sub-nanometer line edge roughness*. **Nanotechnology**, 35(49), 495301. [DOI: 10.1088/1361-6528/ad6e88](https://doi.org/10.1088/1361-6528/ad6e88)
* **Ramachandra, R., Griffin, B., & Joy, D.** (2009). *A model of secondary electron imaging in the helium ion scanning microscope*. **Ultramicroscopy**, 109(6), 748–757.
* **Manfrinato, V. R., et al.** (2013). *Resolution limits of electron-beam lithography toward the atomic scale*. **Nano Letters**, 13(4), 1555–1558.
* **Mack, C. A.** (2018). *Reducing roughness in extreme ultraviolet lithography*. **Journal of Micro/Nanolithography, MEMS, and MOEMS**, 17(4), 041006.

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
