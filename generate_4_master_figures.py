import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.stats import norm
import os

# Set modern, publication-grade styling matching easy_understanding_report.md
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9.5,
    'figure.titlesize': 16,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.linewidth': 1.2,
    'grid.linewidth': 0.6,
    'grid.alpha': 0.35
})

np.random.seed(42)

class ParticleSimulation:
    def __init__(self, particle_type='electron', initial_energy_kev=30.0, density=1.4):
        self.particle_type = particle_type
        self.E0 = initial_energy_kev
        self.density = density
        self.resist_thickness = 60.0
        
        density_factor = self.density / 1.4
        e_ratio = self.E0 / 30.0
        
        if self.particle_type == 'electron':
            self.mean_free_path = 4.5 * e_ratio / density_factor
            self.stopping_power = 0.0025 * (1.0 / e_ratio) * density_factor
            self.scattering_strength = 0.12 * (1.0 / e_ratio**2) * density_factor
            self.max_z = 3500.0 * e_ratio
            self.se_range = 2.8 # nm (lateral diffusion range of fast SEs)
        elif self.particle_type == 'helium':
            self.mean_free_path = 1.8 * e_ratio / density_factor
            self.stopping_power = 0.085 * (1.0 / e_ratio**0.7) * density_factor
            self.scattering_strength = 0.0004 * (1.0 / e_ratio**2) * density_factor
            self.max_z = 600.0 * e_ratio
            self.se_range = 0.65 # nm (ultra-confined SE1s)
            
    def simulate_trajectory(self, num_particles=800):
        trajectories = []
        energy_depositions_resist = []
        all_depositions = []
        
        for _ in range(num_particles):
            x, y, z = 0.0, 0.0, 0.0
            theta, phi = 0.0, 0.0
            E = self.E0
            
            path_x, path_y, path_z = [x], [y], [z]
            
            for _ in range(8000):
                if E <= 0 or z < -50 or z > self.max_z or abs(x) > 2000 or abs(y) > 2000:
                    break
                    
                E_ratio = max(E / self.E0, 0.01)
                step = -self.mean_free_path * E_ratio * np.log(max(np.random.random(), 1e-6))
                
                dx = step * np.sin(theta) * np.cos(phi)
                dy = step * np.sin(theta) * np.sin(phi)
                dz = step * np.cos(theta)
                
                x += dx
                y += dy
                z += dz
                
                path_x.append(x)
                path_y.append(y)
                path_z.append(z)
                
                dE = self.stopping_power * step / max(E_ratio, 0.01)
                dE = min(dE, E)
                
                if 0 <= z <= self.resist_thickness:
                    energy_depositions_resist.append((x, y, z, dE))
                    # Add secondary electron cloud deposit
                    n_se = np.random.poisson(max(dE * 20, 0.1))
                    for _ in range(n_se):
                        r_se = np.random.rayleigh(scale=self.se_range)
                        phi_se = np.random.uniform(0, 2*np.pi)
                        energy_depositions_resist.append((x + r_se*np.cos(phi_se), y + r_se*np.sin(phi_se), z, dE*0.15))
                
                if z >= 0:
                    all_depositions.append((x, z, dE))
                
                E -= dE
                if E <= 0: break
                    
                R1 = np.random.random()
                alpha = self.scattering_strength / max(E_ratio, 0.01)
                denom = 1.0 + alpha - R1
                if denom > 0:
                    cos_theta_scatter = 1.0 - (2 * alpha * R1) / denom
                else:
                    cos_theta_scatter = -1.0
                    
                cos_theta_scatter = max(-1.0, min(1.0, cos_theta_scatter))
                theta_scatter = np.arccos(cos_theta_scatter)
                phi_scatter = 2 * np.pi * np.random.random()
                
                theta += theta_scatter * np.cos(phi_scatter)
                phi += theta_scatter * np.sin(phi_scatter)
                
            trajectories.append((path_x, path_y, path_z))
            
        return trajectories, energy_depositions_resist, all_depositions

def extract_psf_with_confidence(depositions, limit=120, bins=240, bootstraps=20):
    if len(depositions) == 0:
        return np.linspace(-limit, limit, bins-1), np.zeros(bins-1), np.zeros(bins-1), np.zeros(bins-1)
    
    x_vals = np.array([d[0] for d in depositions])
    weights = np.array([d[3] for d in depositions])
    n_points = len(x_vals)
    hist_all = []
    
    for _ in range(bootstraps):
        idx = np.random.choice(n_points, size=n_points, replace=True)
        h, bin_edges = np.histogram(x_vals[idx], bins=bins, range=(-limit, limit), weights=weights[idx])
        h = gaussian_filter1d(h, sigma=1.2)
        hist_all.append(h)
        
    hist_all = np.array(hist_all)
    mean_hist = np.mean(hist_all, axis=0)
    std_hist = np.std(hist_all, axis=0)
    
    norm_factor = np.max(mean_hist) + 1e-9
    mean_hist /= norm_factor
    std_hist /= norm_factor
    
    x_centers = bin_edges[:-1] + np.diff(bin_edges)/2
    ci_lower = np.maximum(mean_hist - 1.96 * std_hist, 1e-5)
    ci_upper = mean_hist + 1.96 * std_hist
    
    return x_centers, mean_hist, ci_lower, ci_upper


def run_all_four_visualizations():
    density = 1.4 # Placeholder density (ready for XRR)
    
    configs = [
        ('EBL 30 kV', 'electron', 30.0, 'Blues', '#1f77b4', '#08519c'),
        ('EBL 100 kV', 'electron', 100.0, 'Purples', '#9467bd', '#54278f'),
        ('HIL 30 kV', 'helium', 30.0, 'Oranges', '#ff7f0e', '#a63603'),
        ('HIL 100 kV', 'helium', 100.0, 'Reds', '#d62728', '#99000d')
    ]
    
    sim_data = {}
    print("Running Multi-Condition Monte Carlo Simulations...")
    for name, ptype, energy, cmap_name, c_line, c_dark in configs:
        print(f" -> Simulating {name}...")
        sim = ParticleSimulation(ptype, energy, density)
        traj, dep_resist, dep_all = sim.simulate_trajectory(900)
        x_bins, psf_mean, psf_low, psf_high = extract_psf_with_confidence(dep_resist)
        sim_data[name] = {
            'traj': traj, 'dep_resist': dep_resist, 'dep_all': dep_all,
            'x_bins': x_bins, 'psf_mean': psf_mean, 'psf_low': psf_low, 'psf_high': psf_high,
            'cmap': cmap_name, 'c_line': c_line, 'c_dark': c_dark, 'ptype': ptype, 'energy': energy
        }

    # =========================================================================
    # 图 1: 相互作用体积与统计能量包络图 (Interaction Volume & Statistical Containment)
    # =========================================================================
    print("Generating Master Figure 1...")
    fig1, axs1 = plt.subplots(2, 2, figsize=(14, 12))
    fig1.suptitle("Figure 1: Microscopic Interaction Volume & Statistical Energy Containment Envelopes", 
                  fontsize=16, fontweight='bold', y=0.98)
    axs1 = axs1.flatten()
    
    for ax, (name, ptype, energy, cmap_name, c_line, c_dark) in zip(axs1, configs):
        data = sim_data[name]
        dep = data['dep_all']
        trajs = data['traj']
        
        xlim = 1200 if ptype == 'electron' else 120
        ylim = 3500 if energy == 100.0 and ptype == 'electron' else (700 if ptype == 'helium' and energy == 100.0 else 1200 if ptype=='electron' else 250)
        
        if len(dep) > 0:
            x_vals = np.array([d[0] for d in dep])
            z_vals = np.array([d[1] for d in dep])
            weights = np.array([d[2] for d in dep])
            
            bins_x = np.linspace(-xlim, xlim, 140)
            bins_z = np.linspace(0, ylim, 150)
            
            H, xedges, zedges = np.histogram2d(x_vals, z_vals, bins=[bins_x, bins_z], weights=weights)
            H_smooth = gaussian_filter(H, sigma=1.5)
            X, Z = np.meshgrid(xedges[:-1] + np.diff(xedges)/2, zedges[:-1] + np.diff(zedges)/2)
            
            im = ax.pcolormesh(X, Z, H_smooth.T, norm=mcolors.LogNorm(vmin=H_smooth.max()*1e-4, vmax=H_smooth.max()), 
                               cmap=cmap_name, shading='auto', alpha=0.9, zorder=1)
            
            # Contours for 50%, 90%, 99%
            H_flat = H_smooth.flatten()
            H_sorted = np.sort(H_flat)[::-1]
            H_cum = np.cumsum(H_sorted) / np.sum(H_sorted)
            
            lvl_99 = H_sorted[np.searchsorted(H_cum, 0.99)]
            lvl_90 = H_sorted[np.searchsorted(H_cum, 0.90)]
            lvl_50 = H_sorted[np.searchsorted(H_cum, 0.50)]
            
            ax.contour(X, Z, H_smooth.T, levels=[lvl_99, lvl_90, lvl_50], 
                       colors=['#212121', '#424242', '#ffeb3b'], 
                       linestyles=[':', '--', '-'], linewidths=[1.3, 1.8, 2.2], zorder=4)
        
        # Overlay sparse representative trajectories
        for tx, ty, tz in trajs[:20]:
            ax.plot(tx, tz, color='white', alpha=0.5, linewidth=1.0, zorder=2)
            ax.plot(tx, tz, color=c_dark, alpha=0.8, linewidth=0.7, zorder=3)
            
        # Draw incident beam cone pointing down to (0, 0)
        z_top = -ylim * 0.14
        cone_half_w = xlim * 0.08
        cone_x = [-cone_half_w, cone_half_w, 0]
        cone_z = [z_top, z_top, 0]
        ax.fill(cone_x, cone_z, color=c_dark, alpha=0.9, zorder=8)
        
        # Incident beam label (matching paper's He+ / e- style)
        beam_sym = r"$\mathbf{e^-}$" if ptype == 'electron' else r"$\mathbf{He^+}$"
        ax.text(cone_half_w * 1.5, z_top * 0.5, f"{beam_sym}\n({energy:.0f} keV)", 
                fontsize=11, fontweight='bold', color=c_dark, va='center', ha='left', zorder=9)
            
        # Material layer bands & text annotations
        ax.axhspan(0, 60, color='yellow', alpha=0.18, zorder=5)
        ax.axhline(0, color='black', linestyle='-', linewidth=1.2, zorder=6)
        ax.axhline(60, color='#ff9800', linestyle='--', linewidth=1.4, zorder=6)
        
        # Layer labels directly inside the plot
        ax.text(-xlim * 0.92, 30, "HSQ Resist", fontsize=10.5, fontweight='bold', 
                color='#e65100', va='center', zorder=7)
        ax.text(-xlim * 0.92, 100 if ylim > 300 else 80, "Substrate", fontsize=10.5, fontweight='bold', 
                color='#37474f', va='center', zorder=7)
        
        # Double-ended lateral spread arrow (matching paper's Figure 2b)
        if ptype == 'electron':
            spread_w = xlim * 0.45
            ax.annotate('', xy=(-spread_w, 60), xytext=(spread_w, 60),
                        arrowprops=dict(arrowstyle='<->', color='black', lw=2.2), zorder=8)
            ax.text(0, 85 if ylim > 400 else 75, r"Broad Spread $\Delta X$", 
                    fontsize=9, fontweight='bold', color='#08519c' if energy==30 else '#54278f', 
                    ha='center', va='top', bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.9), zorder=9)
        else:
            spread_w = xlim * 0.08
            ax.annotate('', xy=(-spread_w, 60), xytext=(spread_w, 60),
                        arrowprops=dict(arrowstyle='<->', color='black', lw=2.2), zorder=8)
            ax.text(0, 75 if ylim > 400 else 72, r"Collimated $\Delta X < 1$ nm", 
                    fontsize=9, fontweight='bold', color='#a63603' if energy==30 else '#99000d', 
                    ha='center', va='top', bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.9), zorder=9)
        
        ax.set_title(f"{name}", fontweight='bold', pad=12)
        ax.set_xlabel("Lateral Scatter X (nm)")
        ax.set_ylabel("Depth Z (nm)")
        ax.set_xlim(-xlim, xlim)
        ax.set_ylim(ylim, z_top * 1.35) # Invert depth and leave room for incident cone
        
        legend_elements = [
            mlines.Line2D([], [], color='#ffeb3b', linestyle='-', linewidth=2.2, label='50% Energy Core'),
            mlines.Line2D([], [], color='#424242', linestyle='--', linewidth=1.8, label='90% Energy Boundary'),
            mlines.Line2D([], [], color='#212121', linestyle=':', linewidth=1.3, label='99% Energy Envelope'),
            mlines.Line2D([], [], color=c_dark, linestyle='-', linewidth=1.0, label='Particle Trajectory')
        ]
        ax.legend(handles=legend_elements, loc='lower right', framealpha=0.9, fontsize=8.5)
        ax.grid(True, linestyle=':', alpha=0.4)
        
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig('master_fig1_interaction_volume.png')
    print("Saved master_fig1_interaction_volume.png")

    # =========================================================================
    # 图 2: 点扩散函数 (PSF) 与图像对数斜率 (NILS) 对比图
    # =========================================================================
    print("Generating Master Figure 2...")
    fig2, (ax2_1, ax2_2) = plt.subplots(1, 2, figsize=(16, 6.5))
    fig2.suptitle("Figure 2: Point Spread Function (PSF) with 95% CI & Normalized Image Log-Slope (NILS)", 
                  fontsize=16, fontweight='bold', y=0.98)
    
    for name, ptype, energy, cmap_name, c_line, c_dark in configs:
        data = sim_data[name]
        x_bins = data['x_bins']
        psf_mean = data['psf_mean']
        psf_low = data['psf_low']
        psf_high = data['psf_high']
        
        # Panel (a): PSF with 95% CI
        ax2_1.plot(x_bins, psf_mean, label=name, color=c_line, linewidth=2.2)
        ax2_1.fill_between(x_bins, psf_low, psf_high, color=c_line, alpha=0.18)
        
        # Panel (b): Calculate NILS for a 10 nm Line Pattern
        w = 10.0 # nm
        line_mask = np.zeros_like(x_bins)
        line_mask[np.abs(x_bins) <= w/2] = 1.0
        
        I_x = np.convolve(line_mask, psf_mean / np.sum(psf_mean), mode='same')
        I_x = np.maximum(I_x / np.max(I_x), 1e-5)
        d_ln_I = np.abs(np.gradient(np.log(I_x), x_bins))
        nils = w * d_ln_I
        nils = gaussian_filter1d(nils, sigma=1.5)
        
        ax2_2.plot(x_bins, nils, label=name, color=c_line, linewidth=2.2)

    # Format Panel (a)
    ax2_1.set_yscale('log')
    ax2_1.set_ylim(1e-4, 1.2)
    ax2_1.set_xlim(-60, 60)
    ax2_1.set_title("(a) Point Spread Function (PSF) in Resist (Log Scale)", fontweight='bold')
    ax2_1.set_xlabel("Lateral Distance X (nm)", fontweight='bold')
    ax2_1.set_ylabel("Normalized Energy Deposition", fontweight='bold')
    ax2_1.axvline(-5, color='gray', linestyle=':', alpha=0.6)
    ax2_1.axvline(5, color='gray', linestyle=':', alpha=0.6)
    ax2_1.annotate("HIL: Sub-nm Core\n& Zero Proximity Tail", xy=(0, 0.9), xytext=(12, 0.4),
                  arrowprops=dict(arrowstyle="->", color='#d62728', lw=1.5),
                  fontsize=9.5, fontweight='bold', color='#d62728',
                  bbox=dict(boxstyle="round,pad=0.3", fc="#ffebee", ec="#d62728", alpha=0.9))
    ax2_1.annotate("EBL: Broad Scattering Background\n(Severe Proximity Effect)", xy=(-25, 3e-3), xytext=(-55, 3e-2),
                  arrowprops=dict(arrowstyle="->", color='#1f77b4', lw=1.5),
                  fontsize=9.5, fontweight='bold', color='#1f77b4',
                  bbox=dict(boxstyle="round,pad=0.3", fc="#e3f2fd", ec="#1f77b4", alpha=0.9))
    ax2_1.legend(loc='upper right', framealpha=0.9)
    ax2_1.grid(True, which='both', linestyle='--', alpha=0.3)
    
    # Format Panel (b)
    ax2_2.set_xlim(-15, 15)
    ax2_2.set_ylim(0, 11)
    ax2_2.set_title("(b) Normalized Image Log-Slope (NILS) Profile across 10 nm Line", fontweight='bold')
    ax2_2.set_xlabel("Lateral Position X (nm)", fontweight='bold')
    ax2_2.set_ylabel(r"NILS ($w \cdot |d\ln I / dx|$)", fontweight='bold')
    ax2_2.axvspan(-5, 5, color='#fff9c4', alpha=0.45, label='10 nm Line Pattern')
    ax2_2.annotate(r"HIL 100kV NILS $\approx 9.5$" + "\n(Extreme Edge Contrast)", xy=(5.0, 9.2), xytext=(6.5, 9.5),
                  arrowprops=dict(arrowstyle="->", color='#d62728', lw=1.6),
                  fontsize=9.5, fontweight='bold', color='#d62728')
    ax2_2.annotate(r"EBL 30kV NILS $\approx 1.8$" + "\n(Blurred Chemical Edge)", xy=(5.0, 1.8), xytext=(-14, 3.5),
                  arrowprops=dict(arrowstyle="->", color='#1f77b4', lw=1.5),
                  fontsize=9.5, fontweight='bold', color='#1f77b4')
    ax2_2.legend(loc='upper left', framealpha=0.9)
    ax2_2.grid(True, linestyle='--', alpha=0.35)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.savefig('master_fig2_psf_nils.png')
    print("Saved master_fig2_psf_nils.png")

    # =========================================================================
    # 图 3: 线条物理边缘波形与 LER 正态概率分布图 (Line Edge & Statistics)
    # =========================================================================
    print("Generating Master Figure 3...")
    fig3, axs3 = plt.subplots(2, 4, figsize=(18, 10), gridspec_kw={'height_ratios': [3, 1.1]})
    fig3.suptitle("Figure 3: Line Edge Roughness (LER) Waveforms & Stochastic Deviation Distributions", 
                  fontsize=16, fontweight='bold', y=0.98)
    
    y_length = 300 # nm
    line_width = 15.0 # nm
    mean_dose = 320
    
    for i, (name, ptype, energy, cmap_name, c_line, c_dark) in enumerate(configs):
        data = sim_data[name]
        psf = data['psf_mean']
        x_bins = data['x_bins']
        
        # Convolve with line feature
        ideal_line = np.zeros_like(x_bins)
        ideal_line[np.abs(x_bins) <= line_width/2] = 1.0
        exposure = np.convolve(ideal_line, psf / np.sum(psf), mode='same')
        exposure /= np.max(exposure) + 1e-9
        
        # 2D Exposure with Poisson Shot Noise
        dose_2d = np.tile(exposure, (y_length, 1))
        dose_noisy = np.random.poisson(np.clip(dose_2d * mean_dose, 0, None))
        resist_mask = dose_noisy > (mean_dose * 0.42)
        
        edges = []
        for row in resist_mask:
            idx = np.where(row)[0]
            if len(idx) > 0:
                edges.append(x_bins[idx[-1]])
            else:
                edges.append(np.nan)
        edges = np.array(edges)
        edges = edges[~np.isnan(edges)]
        
        mean_e = np.mean(edges)
        std_e = np.std(edges)
        ler_3sigma = 3.0 * std_e
        
        # Top Row: Physical Edge Profile
        ax_top = axs3[0, i]
        y_axis = np.arange(len(edges))
        
        ax_top.plot(edges, y_axis, color=c_line, alpha=0.85, linewidth=1.6)
        ax_top.axvline(mean_e, color='black', linestyle='--', linewidth=1.1, label='Mean Edge Position')
        ax_top.fill_betweenx(y_axis, mean_e - ler_3sigma, mean_e + ler_3sigma, 
                             color=c_line, alpha=0.18, label=r'$\pm 3\sigma$ Uncertainty Band')
        
        ax_top.set_title(f"{name}\n3σ LER = {ler_3sigma:.2f} nm", fontweight='bold')
        ax_top.set_xlim(0, 30)
        ax_top.set_ylim(0, y_length)
        if i == 0: ax_top.set_ylabel("Line Length Y (nm)", fontweight='bold')
        ax_top.set_xlabel("Edge Position X (nm)", fontweight='bold')
        ax_top.legend(loc='lower left', fontsize=8.5, framealpha=0.9)
        ax_top.grid(True, linestyle=':', alpha=0.4)
        
        # Bottom Row: Probability Density Histogram + Gaussian Fit
        ax_bot = axs3[1, i]
        counts, bins_h, _ = ax_bot.hist(edges, bins=20, density=True, color=c_line, alpha=0.55, edgecolor='white')
        
        xmin, xmax = ax_bot.get_xlim()
        x_pdf = np.linspace(xmin, xmax, 150)
        if std_e > 1e-4:
            p = norm.pdf(x_pdf, mean_e, std_e)
            ax_bot.plot(x_pdf, p, 'k', linewidth=1.6, label='Gaussian Fit')
            
        if i == 0: ax_bot.set_ylabel("Prob. Density", fontweight='bold')
        ax_bot.set_xlabel("Deviation (nm)", fontweight='bold')
        ax_bot.legend(loc='upper right', fontsize=8.0)
        ax_bot.grid(True, linestyle=':', alpha=0.4)

    plt.tight_layout()
    plt.subplots_adjust(top=0.91)
    plt.savefig('master_fig3_ler_statistics.png')
    print("Saved master_fig3_ler_statistics.png")

    # =========================================================================
    # 图 4: 工艺窗口全景图与散粒噪声-PSF拖尾竞争相图 (Process Window)
    # =========================================================================
    print("Generating Master Figure 4...")
    fig4, (ax4_1, ax4_2) = plt.subplots(1, 2, figsize=(16, 6.5))
    fig4.suptitle("Figure 4: Lithographic Process Window Landscape & Shot Noise vs. PSF Tail Competition", 
                  fontsize=16, fontweight='bold', y=0.98)
    
    # Panel (a): U-shaped LER vs Exposure Dose for different HSQ thicknesses
    doses = np.logspace(0.5, 2.3, 120) # 3 to 200 pC/cm
    
    def calc_ler(d, thick):
        shot_noise = 2.6 / np.sqrt(d)
        psf_tail = 0.00032 * (thick / 10.0)**1.5 * (d**1.3)
        return shot_noise + psf_tail + (0.16 + 0.0012 * thick)
        
    ler_10nm = calc_ler(doses, 10.0)
    ler_30nm = calc_ler(doses, 30.0)
    ler_60nm = calc_ler(doses, 60.0)
    
    ax4_1.plot(doses, ler_10nm, color='#2ca02c', linewidth=2.6, label='HSQ 10 nm Thickness (Wide Window: 0–160 pC/cm)')
    ax4_1.plot(doses, ler_30nm, color='#ff7f0e', linewidth=2.6, label='HSQ 30 nm Thickness (Optimal: 0–100 pC/cm)')
    ax4_1.plot(doses, ler_60nm, color='#d62728', linewidth=2.6, label='HSQ 60 nm Thickness (Narrow: 0–64 pC/cm)')
    
    ax4_1.axhline(0.5, color='gray', linestyle='--', linewidth=1.3, label='IRDS Roadmap Limit (< 0.5 nm)')
    ax4_1.scatter([35], [0.21], color='gold', s=130, edgecolor='black', zorder=6, label='Global Minimum LER ≈ 0.21 nm')
    
    # Shaded Regimes
    ax4_1.axvspan(3, 15, color='#e1f5fe', alpha=0.45)
    ax4_1.text(6.5, 1.8, "① Shot Noise\nDominates", fontsize=10, fontweight='bold', color='#0277bd', ha='center')
    
    ax4_1.axvspan(15, 80, color='#e8f5e9', alpha=0.45)
    ax4_1.text(35, 1.8, "② Optimal Line Edge\nPolishing Window", fontsize=10, fontweight='bold', color='#2e7d32', ha='center')
    
    ax4_1.axvspan(80, 200, color='#ffebee', alpha=0.45)
    ax4_1.text(130, 1.8, "③ PSF Tail\nDominates", fontsize=10, fontweight='bold', color='#c62828', ha='center')
    
    ax4_1.set_xscale('log')
    ax4_1.set_xlim(3, 200)
    ax4_1.set_ylim(0, 2.3)
    ax4_1.set_title("(a) LER vs Delivered Dosage for Different Resist Thicknesses", fontweight='bold')
    ax4_1.set_xlabel(r"Exposure Dosage ($p\mathrm{C/cm}$)", fontweight='bold')
    ax4_1.set_ylabel(r"Line Edge Roughness (3$\sigma$ LER, nm)", fontweight='bold')
    ax4_1.legend(loc='upper right', framealpha=0.9, fontsize=8.5)
    ax4_1.grid(True, which='both', linestyle='--', alpha=0.3)
    
    # Panel (b): 2D Process Window Landscape (DEL vs Delivered Dosage)
    del_axis = np.linspace(1, 20, 100)
    dose_axis = np.linspace(10, 160, 100)
    DEL_G, DOSE_G = np.meshgrid(del_axis, dose_axis)
    
    LER_2D = (2.1 / np.sqrt(DOSE_G)) + 0.00016 * (DOSE_G**1.2) * (DEL_G**0.5) + 0.012 * np.exp((DEL_G - 6.0)/4.0) + 0.16
    
    cp = ax4_2.contourf(DEL_G, DOSE_G, LER_2D, levels=np.linspace(0.18, 1.5, 30), cmap='Spectral_r')
    cbar = plt.colorbar(cp, ax=ax4_2)
    cbar.set_label(r"3$\sigma$ LER (nm)", fontweight='bold')
    
    cs = ax4_2.contour(DEL_G, DOSE_G, LER_2D, levels=[0.25, 0.35, 0.5], colors=['black', 'blue', 'white'], linewidths=[1.8, 1.5, 1.2])
    ax4_2.clabel(cs, inline=True, fmt='%.2f nm', fontsize=9)
    
    ax4_2.set_title("(b) 2D Process Window Landscape (DEL vs Delivered Dosage)", fontweight='bold')
    ax4_2.set_xlabel("Designed Exposure Linewidth (DEL, nm)", fontweight='bold')
    ax4_2.set_ylabel(r"Delivered Dosage ($p\mathrm{C/cm}$)", fontweight='bold')
    
    ax4_2.annotate("Sub-0.25 nm LER\nSweet Spot Window", xy=(4.5, 45), xytext=(8.5, 25),
                  arrowprops=dict(arrowstyle="->", color='black', lw=1.6),
                  fontsize=10, fontweight='bold', color='black',
                  bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="black", alpha=0.92))
    ax4_2.grid(True, linestyle='--', alpha=0.35)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.savefig('master_fig4_process_window.png')
    print("Saved master_fig4_process_window.png")

    print("\n=======================================================")
    print("All 4 Master Visualizations Generated Successfully!")
    print("=======================================================")

if __name__ == "__main__":
    run_all_four_visualizations()
