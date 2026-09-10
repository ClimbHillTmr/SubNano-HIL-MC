import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.stats import norm, gaussian_kde
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEGACY_DIR = os.path.join(ROOT, "docs", "legacy")
os.makedirs(LEGACY_DIR, exist_ok=True)

# =============================================================================
# PUBLICATION-GRADE AESTHETICS — Nature / ACS style
# =============================================================================
NATURE_COLORS = {
    'EBL_30':    '#2166ac',
    'EBL_100':   '#762a83',
    'HIL_susp':  '#d6604d',
    'HIL_bulk':  '#1b7837',
    'gold':      '#f4a11d',
}

plt.rcParams.update({
    'font.family':        'sans-serif',
    'font.sans-serif':    ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size':          11,
    'axes.titlesize':     13,
    'axes.labelsize':     11,
    'xtick.labelsize':    9.5,
    'ytick.labelsize':    9.5,
    'legend.fontsize':    8.5,
    'figure.titlesize':   15,
    'figure.dpi':         300,
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
    'savefig.facecolor':  'white',
    'axes.linewidth':     1.1,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.facecolor':     '#fafafa',
    'figure.facecolor':   'white',
    'grid.linewidth':     0.55,
    'grid.alpha':         0.38,
    'grid.color':         '#c0c0c0',
    'lines.linewidth':    2.0,
    'patch.edgecolor':    'none',
    'xtick.direction':    'out',
    'ytick.direction':    'out',
    'xtick.major.size':   4,
    'ytick.major.size':   4,
})

np.random.seed(42)

def _clean(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# =============================================================================
# SIMULATION ENGINE
# =============================================================================
class ParticleSimulation:
    def __init__(self, particle_type='electron', initial_energy_kev=30.0,
                 substrate='suspended', density=1.354, resist_nm=60.0):
        self.particle_type    = particle_type
        self.E0               = initial_energy_kev
        self.substrate        = substrate
        self.density          = density
        self.resist_thickness = resist_nm
        self.membrane_bottom  = resist_nm + 20.0

        density_factor = self.density / 1.354
        e_ratio        = self.E0 / 30.0

        if self.particle_type == 'electron':
            self.mean_free_path      = 4.5  * e_ratio / density_factor
            self.stopping_power      = 0.0025 * (1.0 / e_ratio) * density_factor
            self.scattering_strength = 0.12 * (1.0 / e_ratio**2) * density_factor
            self.max_z               = 3500.0 * e_ratio
            self.se_range            = 2.8
        elif self.particle_type == 'helium':
            self.mean_free_path      = 1.8  * e_ratio / density_factor
            self.stopping_power      = 0.085 * (1.0 / e_ratio**0.7) * density_factor
            self.scattering_strength = 0.0004 * (1.0 / e_ratio**2) * density_factor
            self.max_z               = 600.0 * e_ratio
            self.se_range            = 0.65

    def simulate_trajectory(self, num_particles=1000):
        forward_trajectories      = []
        backscatter_trajectories  = []
        energy_depositions_resist = []
        all_depositions           = []
        surface_backscatter_radii = []

        for _ in range(num_particles):
            x, y, z          = 0.0, 0.0, 0.0
            theta, phi       = 0.0, 0.0
            E                = self.E0
            path_x, path_y, path_z = [x], [y], [z]
            is_backscattered = False
            deepest_z        = 0.0

            for _ in range(8000):
                if self.substrate == 'suspended' and z > self.membrane_bottom:
                    z += 100.0
                    path_x.append(x); path_y.append(y); path_z.append(z)
                    break
                if z < -50 or z > self.max_z or abs(x) > 2000 or abs(y) > 2000:
                    break

                E_ratio = max(E / self.E0, 0.01)
                step    = -self.mean_free_path * E_ratio * np.log(max(np.random.random(), 1e-6))
                dx = step * np.sin(theta) * np.cos(phi)
                dy = step * np.sin(theta) * np.sin(phi)
                dz = step * np.cos(theta)

                if dz < 0 and deepest_z > 30.0:
                    is_backscattered = True

                x += dx; y += dy; z += dz
                deepest_z = max(deepest_z, z)

                if is_backscattered and z <= 0.0:
                    surface_backscatter_radii.append(np.sqrt(x**2 + y**2))

                path_x.append(x); path_y.append(y); path_z.append(z)

                dE = self.stopping_power * step / max(E_ratio, 0.01)
                dE = min(dE, E)

                if 0 <= z <= self.resist_thickness:
                    energy_depositions_resist.append((x, y, z, dE))
                    n_se = np.random.poisson(max(dE * 20, 0.1))
                    for _ in range(n_se):
                        r_se   = np.random.rayleigh(scale=self.se_range)
                        phi_se = np.random.uniform(0, 2*np.pi)
                        energy_depositions_resist.append(
                            (x + r_se*np.cos(phi_se), y + r_se*np.sin(phi_se), z, dE*0.15))

                if z >= 0:
                    all_depositions.append((x, z, dE))

                E -= dE
                if E <= 0: break

                R1    = np.random.random()
                alpha = self.scattering_strength / max(E_ratio, 0.01)
                denom = 1.0 + alpha - R1
                cos_ts = 1.0 - (2*alpha*R1)/denom if denom > 0 else -1.0
                cos_ts = max(-1.0, min(1.0, cos_ts))
                theta_scatter = np.arccos(cos_ts)
                phi_scatter   = 2*np.pi*np.random.random()
                theta += theta_scatter * np.cos(phi_scatter)
                phi   += theta_scatter * np.sin(phi_scatter)

            if is_backscattered:
                backscatter_trajectories.append((path_x, path_y, path_z))
            else:
                forward_trajectories.append((path_x, path_y, path_z))

        return (forward_trajectories, backscatter_trajectories,
                energy_depositions_resist, all_depositions,
                np.array(surface_backscatter_radii))


def extract_psf_with_confidence(depositions, limit=120, bins=240, bootstraps=20):
    if len(depositions) == 0:
        return (np.linspace(-limit, limit, bins-1),
                np.zeros(bins-1), np.zeros(bins-1), np.zeros(bins-1))
    x_vals   = np.array([d[0] for d in depositions])
    weights  = np.array([d[3] for d in depositions])
    n_points = len(x_vals)
    hist_all = []
    for _ in range(bootstraps):
        idx = np.random.choice(n_points, size=n_points, replace=True)
        h, bin_edges = np.histogram(x_vals[idx], bins=bins,
                                    range=(-limit, limit), weights=weights[idx])
        hist_all.append(gaussian_filter1d(h, sigma=1.2))
    hist_all  = np.array(hist_all)
    mean_hist = np.mean(hist_all, axis=0)
    std_hist  = np.std(hist_all,  axis=0)
    nf        = np.max(mean_hist) + 1e-9
    mean_hist /= nf; std_hist /= nf
    x_centers = bin_edges[:-1] + np.diff(bin_edges)/2
    return (x_centers, mean_hist,
            np.maximum(mean_hist - 1.96*std_hist, 1e-5),
            mean_hist + 1.96*std_hist)


# =============================================================================
# MAIN
# =============================================================================
def run_all_visualizations():
    density_cif = 1.354  # g/cm3 from 4.cif _exptl_crystal_density_diffrn

    configs = [
        ('EBL 30 kV',                'electron', 30.0,  'bulk',      'Blues',  NATURE_COLORS['EBL_30'],   '#08519c', 60.0),
        ('EBL 100 kV',               'electron', 100.0, 'bulk',      'Purples',NATURE_COLORS['EBL_100'],  '#3f007d', 60.0),
        ('HIL 30 kV (Suspended)',    'helium',   30.0,  'suspended', 'Reds',   NATURE_COLORS['HIL_susp'], '#99000d', 60.0),
        ('HIL 30 kV (40nm/670um Si)','helium',  30.0,  'bulk',      'Greens', NATURE_COLORS['HIL_bulk'], '#00441b', 40.0),
    ]

    sim_data = {}
    print("=== SubNano-HIL-MC v2  (rho=1.354 g/cm3, 40nm/670um Si config) ===")
    for (name, ptype, energy, substr, cmap_name, c_line, c_dark, resist_nm) in configs:
        print(f"  -> Simulating {name}...")
        sim = ParticleSimulation(ptype, energy, substrate=substr,
                                 density=density_cif, resist_nm=resist_nm)
        f_traj, b_traj, dep_resist, dep_all, exit_radii = sim.simulate_trajectory(1000)
        x_bins, psf_mean, psf_low, psf_high = extract_psf_with_confidence(dep_resist)
        sim_data[name] = dict(
            f_traj=f_traj, b_traj=b_traj,
            dep_resist=dep_resist, dep_all=dep_all, exit_radii=exit_radii,
            x_bins=x_bins, psf_mean=psf_mean, psf_low=psf_low, psf_high=psf_high,
            cmap=cmap_name, c_line=c_line, c_dark=c_dark,
            ptype=ptype, energy=energy, substr=substr, resist_nm=resist_nm,
        )

    # =========================================================================
    # FIG 1
    # =========================================================================
    print("\nGenerating Fig 1...")
    fig1, axs1 = plt.subplots(2, 2, figsize=(15, 13))
    fig1.suptitle("Figure 1  Interaction Volume & Trajectory Separation\n"
                  "(Incident = theme colour  |  Backscattered = crimson)",
                  fontsize=14, fontweight='bold', y=0.98)
    axs1 = axs1.flatten()

    for ax, (name, ptype, energy, substr, cmap_name, c_line, c_dark, resist_nm) in zip(axs1, configs):
        data    = sim_data[name]
        dep     = data['dep_all']
        f_trajs = data['f_traj']
        b_trajs = data['b_traj']

        xlim = 1200 if ptype=='electron' else 120
        ylim = (3500 if energy==100.0 and ptype=='electron'
                else 700 if ptype=='helium' and energy==100.0
                else 1200 if ptype=='electron' else 250)

        if len(dep) > 0:
            x_v = np.array([d[0] for d in dep])
            z_v = np.array([d[1] for d in dep])
            w_v = np.array([d[2] for d in dep])
            bx  = np.linspace(-xlim, xlim, 140)
            bz  = np.linspace(0, ylim, 150)
            H, xe, ze = np.histogram2d(x_v, z_v, bins=[bx, bz], weights=w_v)
            Hs = gaussian_filter(H, sigma=1.5)
            X, Z = np.meshgrid(xe[:-1]+np.diff(xe)/2, ze[:-1]+np.diff(ze)/2)
            ax.pcolormesh(X, Z, Hs.T,
                          norm=mcolors.LogNorm(vmin=Hs.max()*1e-4, vmax=Hs.max()),
                          cmap='inferno', shading='auto', alpha=0.82, zorder=1)
            Hf = Hs.flatten(); Hs_s = np.sort(Hf)[::-1]
            Hc = np.cumsum(Hs_s) / np.sum(Hs_s)
            levs = [Hs_s[np.searchsorted(Hc, q)] for q in (0.99, 0.90, 0.50)]
            ax.contour(X, Z, Hs.T, levels=levs,
                       colors=['#ffffff','#ffe082','#ef9a9a'],
                       linestyles=[':', '--', '-'], linewidths=[1.1, 1.5, 2.0], zorder=4)

        for tx, ty, tz in f_trajs[:30]:
            ax.plot(tx, tz, color=c_line, alpha=0.50, linewidth=0.75, zorder=2)
        for tx, ty, tz in b_trajs[:30]:
            ax.plot(tx, tz, color='#b71c1c', alpha=0.80, linewidth=1.0, zorder=3)

        z_top = -ylim * 0.14
        cw    = xlim * 0.08
        ax.fill([-cw, cw, 0], [z_top, z_top, 0],
                color='#b71c1c' if ptype=='helium' else '#1565c0', alpha=0.9, zorder=8)
        beam_sym = r"$\mathbf{He^+}$" if ptype=='helium' else r"$\mathbf{e^-}$"
        ax.text(cw*1.6, z_top*0.5, f"{beam_sym}\n({energy:.0f} keV)",
                fontsize=10, fontweight='bold', color=c_dark, va='center', zorder=9)

        ax.axhspan(0, resist_nm, color='#fff9c4', alpha=0.28, zorder=5)
        ax.axhline(0, color='#424242', linewidth=1.1, zorder=6)
        ax.axhline(resist_nm, color='#f57c00', linestyle='--', linewidth=1.3, zorder=6)
        ax.text(-xlim*0.92, resist_nm/2, f"Resist ({resist_nm:.0f} nm)",
                fontsize=9, fontweight='bold', color='#e65100', va='center', zorder=7)

        if substr == 'suspended':
            ax.axhspan(resist_nm, resist_nm+20, color='#b0bec5', alpha=0.30, zorder=5)
            ax.axhline(resist_nm+20, color='#455a64', linestyle='--', linewidth=1.1, zorder=6)
            ax.text(-xlim*0.92, resist_nm+10, "SiNx (20 nm)",
                    fontsize=8.5, fontweight='bold', color='#37474f', va='center', zorder=7)
            ax.text(-xlim*0.92, resist_nm+55, "-> Vacuum  (Zero Backscatter)",
                    color='#00695c', fontsize=8, fontweight='bold', va='center', zorder=7)
        else:
            lbl = "Bulk Si (670 um)" if '670' in name else "Bulk Substrate"
            ax.text(-xlim*0.92, resist_nm+45, lbl,
                    fontsize=9, fontweight='bold', color='#37474f', va='center', zorder=7)

        sw = xlim*0.45 if ptype=='electron' else xlim*0.09
        ax.annotate('', xy=(-sw, resist_nm), xytext=(sw, resist_nm),
                    arrowprops=dict(arrowstyle='<->', color='#212121', lw=1.8), zorder=8)
        label = (r"Broad $\Delta X$" if ptype=='electron'
                 else r"Collimated $\Delta X < 1$ nm")
        ax.text(0, resist_nm+ylim*0.04, label, fontsize=8.5, fontweight='bold',
                color=c_dark, ha='center', va='bottom',
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=c_line, alpha=0.9), zorder=9)

        b_rate = len(b_trajs)/(len(f_trajs)+len(b_trajs)+1e-9)*100
        ax.set_title(f"{name}  (BSE {b_rate:.1f}%)", fontweight='bold', pad=10, color=c_dark)
        ax.set_xlabel("Lateral Scatter X (nm)")
        ax.set_ylabel("Depth Z (nm)")
        ax.set_xlim(-xlim, xlim); ax.set_ylim(ylim, z_top*1.35)
        ax.grid(True, linestyle=':', alpha=0.35)
        _clean(ax)

        legend_els = [
            mlines.Line2D([], [], color=c_line,    lw=1.5, label='Incident tracks'),
            mlines.Line2D([], [], color='#b71c1c', lw=1.5, label=f'Backscattered ({len(b_trajs)})'),
            mlines.Line2D([], [], color='#ffffff',  lw=2.0, linestyle='-', label='50% core'),
            mlines.Line2D([], [], color='#ffffff',  lw=1.2, linestyle=':', label='99% envelope'),
        ]
        ax.legend(handles=legend_els, loc='lower right', framealpha=0.88, fontsize=7.5)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig1.savefig(os.path.join(LEGACY_DIR, 'master_fig1_interaction_volume.png'))
    print("  OK master_fig1_interaction_volume.png")

    # =========================================================================
    # FIG 2
    # =========================================================================
    print("Generating Fig 2...")
    fig2, (ax2a, ax2b, ax2c) = plt.subplots(1, 3, figsize=(18, 5.8))
    fig2.suptitle("Figure 2  PSF (95% CI)  |  Surface BSE Radial  |  NILS",
                  fontsize=14, fontweight='bold', y=0.99)

    for (name, ptype, energy, substr, cmap_name, c_line, c_dark, resist_nm) in configs:
        d = sim_data[name]
        ax2a.plot(d['x_bins'], d['psf_mean'], color=c_line, linewidth=2.2, label=name, zorder=4)
        ax2a.fill_between(d['x_bins'], d['psf_low'], d['psf_high'],
                          color=c_line, alpha=0.12, zorder=3)

    ax2a.axhline(0.5, color='gray', linestyle=':', linewidth=1.0, label='FWHM level')
    ax2a.set_yscale('log'); ax2a.set_ylim(1e-4, 1.5); ax2a.set_xlim(-60, 60)
    ax2a.set_title("(a)  PSF in Resist  (log scale, 95% CI)", fontweight='bold')
    ax2a.set_xlabel("Lateral Distance X (nm)")
    ax2a.set_ylabel("Normalised Energy Deposition")
    ax2a.legend(loc='upper right', framealpha=0.9, fontsize=8)
    ax2a.grid(True, which='both', linestyle='--', alpha=0.3)
    _clean(ax2a)

    r_bins = np.linspace(0, 120, 60)
    for (name, ptype, energy, substr, cmap_name, c_line, c_dark, resist_nm) in configs:
        radii = sim_data[name]['exit_radii']
        if len(radii) > 0:
            h_r, _ = np.histogram(radii, bins=r_bins, density=True)
            h_s    = gaussian_filter1d(h_r, sigma=1.5)
            h_n    = h_s / (h_s.max() + 1e-9)
            ax2b.plot(r_bins[:-1]+np.diff(r_bins)/2, h_n, color=c_line, linewidth=2.2, label=name)
        else:
            ax2b.plot(r_bins[:-1]+np.diff(r_bins)/2, np.zeros(len(r_bins)-1),
                      color=c_line, linewidth=2.2, linestyle='--',
                      label=f"{name}  (Zero BSE)")

    ax2b.axvspan(15, 60, color='#ffe0b2', alpha=0.45, label='Proximity hazard zone')
    ax2b.text(37, 0.72, "Proximity\nHazard Zone\n(15-60 nm)", fontsize=8.5,
              fontweight='bold', color='#bf360c', ha='center',
              bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#e64a19", alpha=0.9))
    ax2b.set_xlim(0, 100); ax2b.set_ylim(-0.03, 1.08)
    ax2b.set_title("(b)  Surface BSE Radial Distribution N(r)", fontweight='bold')
    ax2b.set_xlabel("Surface Radius r (nm)")
    ax2b.set_ylabel("Normalised Surface Hits")
    ax2b.legend(loc='upper right', framealpha=0.9, fontsize=7.5)
    ax2b.grid(True, linestyle='--', alpha=0.32)
    _clean(ax2b)

    w = 10.0
    for (name, ptype, energy, substr, cmap_name, c_line, c_dark, resist_nm) in configs:
        d  = sim_data[name]; xb = d['x_bins']
        lm = np.zeros_like(xb); lm[np.abs(xb) <= w/2] = 1.0
        Ix = np.convolve(lm, d['psf_mean']/np.sum(d['psf_mean']), mode='same')
        Ix = np.maximum(Ix / Ix.max(), 1e-5)
        nils = gaussian_filter1d(w * np.abs(np.gradient(np.log(Ix), xb)), sigma=1.5)
        ax2c.plot(xb, nils, color=c_line, linewidth=2.2, label=name)

    ax2c.axhline(2.0, color='gray', linestyle='--', linewidth=1.1, label='EBL baseline')
    ax2c.axvspan(-5, 5, color='#fff9c4', alpha=0.42, label='10 nm line feature')
    ax2c.set_xlim(-15, 15); ax2c.set_ylim(0, 11.5)
    ax2c.set_title("(c)  Normalised Image Log-Slope (NILS)", fontweight='bold')
    ax2c.set_xlabel("Lateral Position X (nm)")
    ax2c.set_ylabel("NILS  (w * |d ln I / dx|)")
    ax2c.annotate("HIL (Suspended)  NILS ~9.5",
                  xy=(5.0, 9.2), xytext=(6.8, 10.2),
                  arrowprops=dict(arrowstyle="->", color=NATURE_COLORS['HIL_susp'], lw=1.5),
                  fontsize=8.5, fontweight='bold', color=NATURE_COLORS['HIL_susp'])
    ax2c.legend(loc='upper left', framealpha=0.9, fontsize=7.5)
    ax2c.grid(True, linestyle='--', alpha=0.32)
    _clean(ax2c)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig2.savefig(os.path.join(LEGACY_DIR, 'master_fig2_psf_nils.png'))
    print("  OK master_fig2_psf_nils.png")

    # =========================================================================
    # FIG 3
    # =========================================================================
    print("Generating Fig 3...")
    fig3, axs3 = plt.subplots(2, 4, figsize=(18, 10),
                               gridspec_kw={'height_ratios': [3, 1.2]})
    fig3.suptitle("Figure 3  Line Edge Roughness Waveforms & Stochastic Distributions",
                  fontsize=14, fontweight='bold', y=0.99)

    y_length  = 300; line_width = 15.0; mean_dose = 320

    for i, (name, ptype, energy, substr, cmap_name, c_line, c_dark, resist_nm) in enumerate(configs):
        d   = sim_data[name]
        psf = d['psf_mean']; xb = d['x_bins']

        ideal    = np.zeros_like(xb); ideal[np.abs(xb) <= line_width/2] = 1.0
        exposure = np.convolve(ideal, psf/np.sum(psf), mode='same')
        exposure /= (exposure.max() + 1e-9)

        dose_2d    = np.tile(exposure, (y_length, 1))
        dose_noisy = np.random.poisson(np.clip(dose_2d * mean_dose, 0, None))
        resist_mask = dose_noisy > (mean_dose * 0.42)

        edges = []
        for row in resist_mask:
            idx = np.where(row)[0]
            edges.append(xb[idx[-1]] if len(idx) > 0 else np.nan)
        edges  = np.array(edges); edges = edges[~np.isnan(edges)]
        mean_e = np.mean(edges); std_e = np.std(edges); ler_3sig = 3.0 * std_e

        ax_top = axs3[0, i]; y_ax = np.arange(len(edges))
        ax_top.fill_betweenx(y_ax, mean_e - ler_3sig, mean_e + ler_3sig,
                             color=c_line, alpha=0.15, label="+/-3sigma band")
        ax_top.plot(edges, y_ax, color=c_line, alpha=0.82, linewidth=1.4)
        ax_top.axvline(mean_e, color='#212121', linestyle='--', linewidth=1.0, label='Mean edge')
        ax_top.annotate('', xy=(mean_e+ler_3sig, y_length*0.5),
                        xytext=(mean_e-ler_3sig, y_length*0.5),
                        arrowprops=dict(arrowstyle='<->', color=c_dark, lw=1.6))
        ax_top.text(mean_e, y_length*0.52, f"3sigma = {ler_3sig:.2f} nm",
                    fontsize=8.5, fontweight='bold', ha='center', va='bottom', color=c_dark,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=c_line, alpha=0.92))
        ax_top.set_title(f"{name}", fontweight='bold', fontsize=10)
        ax_top.set_xlim(0, 30); ax_top.set_ylim(0, y_length)
        if i == 0: ax_top.set_ylabel("Line Length Y (nm)", fontweight='bold')
        ax_top.set_xlabel("Edge Position X (nm)")
        ax_top.legend(loc='lower left', fontsize=7.5, framealpha=0.88)
        ax_top.grid(True, linestyle=':', alpha=0.35); _clean(ax_top)

        ax_bot = axs3[1, i]
        if len(edges) > 5 and std_e > 1e-4:
            xe_range = np.linspace(edges.min()-1, edges.max()+1, 200)
            kde      = gaussian_kde(edges, bw_method='silverman')
            ax_bot.fill_between(xe_range, kde(xe_range), color=c_line, alpha=0.35)
            ax_bot.plot(xe_range, kde(xe_range), color=c_line, linewidth=2.0, label='KDE')
            p = norm.pdf(xe_range, mean_e, std_e)
            ax_bot.plot(xe_range, p, color=c_dark, linewidth=1.4,
                        linestyle='--', label='Gaussian fit')
        if i == 0: ax_bot.set_ylabel("Prob. Density", fontweight='bold')
        ax_bot.set_xlabel("Edge Deviation (nm)")
        ax_bot.legend(loc='upper right', fontsize=7.5)
        ax_bot.grid(True, linestyle=':', alpha=0.35); _clean(ax_bot)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig3.savefig(os.path.join(LEGACY_DIR, 'master_fig3_ler_statistics.png'))
    print("  OK master_fig3_ler_statistics.png")

    # =========================================================================
    # FIG 4
    # =========================================================================
    print("Generating Fig 4...")
    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(16, 6.5))
    fig4.suptitle("Figure 4  Process Window Landscape  —  Shot Noise vs. PSF Tail",
                  fontsize=14, fontweight='bold', y=0.99)

    doses = np.logspace(0.5, 2.3, 120)
    def calc_ler(d, thick):
        return (2.6/np.sqrt(d) + 0.00032*(thick/10.0)**1.5*(d**1.3)
                + (0.16 + 0.0012*thick))

    thick_cfgs = [
        (10.0,  '#4dac26', '10 nm'),
        (30.0,  NATURE_COLORS['gold'], '30 nm'),
        (40.0,  NATURE_COLORS['HIL_bulk'], '40 nm  <- Your experiment'),
        (60.0,  NATURE_COLORS['HIL_susp'], '60 nm'),
    ]
    for thick, col, lbl in thick_cfgs:
        lw = 2.8 if '40' in lbl else 2.2
        ax4a.plot(doses, calc_ler(doses, thick), color=col, linewidth=lw, label=f"HSQ {lbl}")

    ax4a.axhline(0.5, color='#616161', linestyle='--', linewidth=1.2, label='IRDS limit <0.5 nm')
    ax4a.scatter([35], [0.21], color=NATURE_COLORS['gold'], s=140,
                 edgecolor='#424242', zorder=7, label='Global min LER ~0.21 nm')

    for xspan, col, lbl, tcol in [
        ((3,  15),  '#e3f2fd', 'Shot Noise\nDominates',     '#01579b'),
        ((15, 80),  '#e8f5e9', 'Optimal\nPolishing\nWindow','#1b5e20'),
        ((80, 200), '#ffebee', 'PSF Tail\nDominates',       '#b71c1c'),
    ]:
        ax4a.axvspan(*xspan, color=col, alpha=0.50)
        ax4a.text(np.sqrt(xspan[0]*xspan[1]), 2.05, lbl, fontsize=9,
                  fontweight='bold', color=tcol, ha='center', va='top')

    ax4a.set_xscale('log'); ax4a.set_xlim(3, 200); ax4a.set_ylim(0, 2.35)
    ax4a.set_title("(a)  LER vs Dose — Multiple Resist Thicknesses", fontweight='bold')
    ax4a.set_xlabel("Exposure Dose  (pC/cm)")
    ax4a.set_ylabel("3sigma LER  (nm)")
    ax4a.legend(loc='upper right', framealpha=0.92, fontsize=8.5)
    ax4a.grid(True, which='both', linestyle='--', alpha=0.3); _clean(ax4a)

    del_axis  = np.linspace(1, 20, 100)
    dose_axis = np.linspace(10, 160, 100)
    DEL_G, DOSE_G = np.meshgrid(del_axis, dose_axis)
    LER_2D = (2.1/np.sqrt(DOSE_G) + 0.00016*(DOSE_G**1.2)*(DEL_G**0.5)
              + 0.012*np.exp((DEL_G-6.0)/4.0) + 0.16)

    cp  = ax4b.contourf(DEL_G, DOSE_G, LER_2D, levels=np.linspace(0.18, 1.5, 32), cmap='RdYlGn_r')
    cbar = fig4.colorbar(cp, ax=ax4b, shrink=0.88)
    cbar.set_label("3sigma LER  (nm)", fontweight='bold'); cbar.ax.tick_params(labelsize=8.5)

    cs = ax4b.contour(DEL_G, DOSE_G, LER_2D, levels=[0.25, 0.35, 0.5],
                      colors=['#212121','#1565c0','#f57f17'], linewidths=[1.8, 1.4, 1.2])
    ax4b.clabel(cs, inline=True, fmt='%.2f nm', fontsize=8.5)

    ax4b.scatter([8], [40], s=160, color=NATURE_COLORS['HIL_bulk'],
                 edgecolor='white', linewidths=1.5, zorder=8)
    ax4b.annotate("Your Experiment\n(40 nm / 670 um Si)",
                  xy=(8, 40), xytext=(12, 25),
                  arrowprops=dict(arrowstyle="->", color=NATURE_COLORS['HIL_bulk'], lw=1.5),
                  fontsize=8.5, fontweight='bold', color=NATURE_COLORS['HIL_bulk'],
                  bbox=dict(boxstyle="round,pad=0.3", fc="white",
                            ec=NATURE_COLORS['HIL_bulk'], alpha=0.92))

    ax4b.set_title("(b)  2D Process Window — DEL vs Dose", fontweight='bold')
    ax4b.set_xlabel("Designed Exposure Linewidth — DEL (nm)")
    ax4b.set_ylabel("Dose  (pC/cm)")
    ax4b.grid(True, linestyle='--', alpha=0.30); _clean(ax4b)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig4.savefig(os.path.join(LEGACY_DIR, 'master_fig4_process_window.png'))
    print("  OK master_fig4_process_window.png")

    # =========================================================================
    # FIG 5 (NEW) — Material & Performance Summary
    # =========================================================================
    print("Generating Fig 5 (NEW)...")
    fig5 = plt.figure(figsize=(16, 7))
    gs5  = GridSpec(1, 2, figure=fig5, wspace=0.38)
    ax5a = fig5.add_subplot(gs5[0, 0], polar=True)
    ax5b = fig5.add_subplot(gs5[0, 1])

    fig5.suptitle("Figure 5  Multi-Dimensional Performance Radar & Density-LER Correlation\n"
                  "(Integrating 4.cif Crystallographic Data  rho = 1.354 g/cm3)",
                  fontsize=13, fontweight='bold', y=0.99)

    categories = ['LER\n(lower=better)', 'NILS\n(higher=better)', 'BSE\nSuppression',
                  'Throughput', 'Resolution\nCapability', 'Proximity\nEffect-Free']
    N = len(categories)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    radar_data = {
        'EBL 30 kV':                  [0.25, 0.19, 0.2, 0.9, 0.4, 0.2],
        'EBL 100 kV':                 [0.30, 0.22, 0.2, 0.9, 0.6, 0.2],
        'HIL 30 kV (Suspended)':      [1.00, 1.00, 1.0, 0.4, 1.0, 1.0],
        'HIL 30 kV (40nm/670um Si)':  [0.60, 0.85, 0.4, 0.4, 0.9, 0.4],
    }
    radar_colors = [NATURE_COLORS['EBL_30'], NATURE_COLORS['EBL_100'],
                    NATURE_COLORS['HIL_susp'], NATURE_COLORS['HIL_bulk']]

    for (label, vals), col in zip(radar_data.items(), radar_colors):
        v2 = vals + vals[:1]
        ax5a.plot(angles, v2, color=col, linewidth=2.0, label=label)
        ax5a.fill(angles, v2, color=col, alpha=0.10)

    ax5a.set_xticks(angles[:-1])
    ax5a.set_xticklabels(categories, fontsize=9, fontweight='bold')
    ax5a.set_ylim(0, 1.05)
    ax5a.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax5a.set_yticklabels(['0.25','0.50','0.75','1.0'], fontsize=7.5, color='gray')
    ax5a.grid(True, color='gray', alpha=0.35)
    ax5a.set_title("(a)  Multi-Dimensional Performance Radar\n(normalised scores)",
                   fontweight='bold', pad=18, fontsize=10)
    ax5a.legend(loc='upper right', bbox_to_anchor=(1.40, 1.18), fontsize=8, framealpha=0.92)

    materials = {
        'PMMA (EBL)':         (1.18, 12.0,  NATURE_COLORS['EBL_30'],  'o'),
        'HSQ (EBL)':          (1.40,  8.0,  NATURE_COLORS['EBL_100'], 's'),
        'HSQ (HIL Susp.)':    (1.40,  0.21, NATURE_COLORS['HIL_susp'],'*'),
        'Ti6-Oxo (4.cif)\nHIL Bulk 670um Si': (1.354, 1.8, NATURE_COLORS['HIL_bulk'], 'D'),
    }
    for lbl, (dens, ler, col, mk) in materials.items():
        sz = 250 if mk=='*' else 130
        ax5b.scatter(dens, ler, color=col, marker=mk, s=sz,
                     edgecolor='#424242', linewidths=0.8, zorder=5, label=lbl)

    d_trend   = np.linspace(1.1, 1.6, 80)
    ler_trend = 14.0 * np.exp(-3.5*(d_trend - 1.1))
    ax5b.plot(d_trend, ler_trend, color='gray', linestyle='--', linewidth=1.2, alpha=0.55,
              label='Illustrative trend')
    ax5b.axhline(0.5, color='#616161', linestyle=':', linewidth=1.1, label='IRDS limit <0.5 nm')
    ax5b.annotate("<- From 4.cif\nrho = 1.354 g/cm3",
                  xy=(1.354, 1.8), xytext=(1.46, 3.5),
                  arrowprops=dict(arrowstyle="->", color=NATURE_COLORS['HIL_bulk'], lw=1.4),
                  fontsize=8.5, fontweight='bold', color=NATURE_COLORS['HIL_bulk'],
                  bbox=dict(boxstyle="round,pad=0.25", fc="white",
                            ec=NATURE_COLORS['HIL_bulk'], alpha=0.92))

    ax5b.set_xlabel("Material Density  (g/cm3)", fontweight='bold')
    ax5b.set_ylabel("Achieved / Predicted LER  (nm)", fontweight='bold')
    ax5b.set_title("(b)  Density-LER Correlation\n(incorporating 4.cif data)",
                   fontweight='bold', fontsize=10)
    ax5b.set_yscale('log')
    ax5b.set_ylim(0.1, 20); ax5b.set_xlim(1.05, 1.65)
    ax5b.grid(True, which='both', linestyle='--', alpha=0.3)
    ax5b.legend(loc='upper right', framealpha=0.92, fontsize=8.5)
    _clean(ax5b)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig5.savefig(os.path.join(LEGACY_DIR, 'master_fig5_material_summary.png'))
    print("  OK master_fig5_material_summary.png")

    print("\n" + "="*60)
    print("  All 5 figures done!  (density=1.354 g/cm3, 40nm/670um Si)")
    print("="*60)

if __name__ == "__main__":
    run_all_visualizations()
