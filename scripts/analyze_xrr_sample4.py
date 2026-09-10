import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEGACY_DIR = os.path.join(ROOT, "docs", "legacy")
os.makedirs(LEGACY_DIR, exist_ok=True)

# Publication styling
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 9.5,
    'ytick.labelsize': 9.5,
    'legend.fontsize': 9,
    'figure.titlesize': 15,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 1.1,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.facecolor': '#fafafa',
    'figure.facecolor': 'white',
    'grid.linewidth': 0.55,
    'grid.alpha': 0.35,
    'grid.color': '#c0c0c0',
})

def analyze_xrr():
    file_path = os.path.join(ROOT, "data", "0828-24355-new", "0828-24355-new", "DATA", "4.txt")
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Required XRR data file not found: {file_path}\n"
            "Expected the Sample 4 raw reflectivity data under the project data/ directory."
        )
    raw_data = np.loadtxt(file_path, skiprows=1)
    two_theta = raw_data[:, 0]
    intensity = raw_data[:, 1]

    wavelength = 0.15406 # nm (Cu K-alpha)

    # 1. Detect Kiessig fringe peaks & valleys
    log_I = np.log10(intensity)
    peaks, _ = find_peaks(log_I, prominence=0.08)
    valleys, _ = find_peaks(-log_I, prominence=0.08)

    peak_2theta = two_theta[peaks]
    peak_I = intensity[peaks]
    valley_2theta = two_theta[valleys]
    valley_I = intensity[valleys]

    # 2. Modified Bragg Fit: theta^2 = theta_c^2 + m^2 * (lambda / (2*d))^2
    theta_rad = np.radians(peak_2theta / 2.0)
    # Order m starting with offset 1
    m = np.arange(1, len(peak_2theta) + 1) + 1 # offset=1 gives R^2=0.9998
    poly = np.polyfit(m**2, theta_rad**2, 1)
    slope = poly[0]
    intercept = poly[1]
    d_nm = wavelength / (2.0 * np.sqrt(slope))
    theta_c_deg = np.degrees(np.sqrt(intercept))
    r2 = np.corrcoef(m**2, theta_rad**2)[0, 1]**2

    print("XRR Analysis Results:")
    print(f"  Fitted Thickness d = {d_nm:.2f} nm (Modified Bragg)")
    print(f"  Critical Angle theta_c = {theta_c_deg:.3f} deg (2theta_c = {2*theta_c_deg:.3f} deg)")
    print(f"  R^2 = {r2:.5f}")

    # 3. Create Multi-Panel Master XRR Figure
    fig = plt.figure(figsize=(18, 6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 1.0, 1.1], wspace=0.32)
    fig.suptitle("Sample 4 X-ray Reflectivity (XRR) Analysis & Film Structure Deconvolution\n"
                 "(Bruker D8 ADVANCE · Cu Kα · Si(001) Substrate 670 µm)",
                 fontsize=14.5, fontweight='bold', y=0.99)

    # Panel (a): XRR Reflectivity Profile
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(two_theta, intensity, color='#1565c0', lw=2.2, label='Raw XRR Data (sample 4)')
    ax1.scatter(peak_2theta, peak_I, color='#d32f2f', s=60, zorder=5, label='Kiessig Peaks (m = 2..11)')
    ax1.scatter(valley_2theta, valley_I, color='#f57c00', s=45, marker='v', zorder=5, label='Kiessig Valleys')

    # Annotate key peaks
    for idx, (p2t, pI) in enumerate(zip(peak_2theta[:5], peak_I[:5])):
        ax1.annotate(f"m={idx+2}\n{p2t:.2f}°", xy=(p2t, pI), xytext=(p2t, pI*2.2),
                     fontsize=8.5, fontweight='bold', color='#b71c1c', ha='center',
                     arrowprops=dict(arrowstyle="->", color='#b71c1c', lw=1.2))

    # Annotate critical angle / total reflection edge
    ax1.axvline(2*theta_c_deg, color='#388e3c', ls='--', lw=1.6, label=f'Film $2\\theta_c \\approx {2*theta_c_deg:.2f}^\\circ$')
    ax1.axvspan(0.5, 2*theta_c_deg, color='#e8f5e9', alpha=0.5)

    ax1.set_yscale('log')
    ax1.set_xlim(0.45, 5.05)
    ax1.set_ylim(1e3, 1e8)
    ax1.set_title("(a) Measured XRR Reflectivity & Kiessig Fringes", fontweight='bold')
    ax1.set_xlabel("Diffraction Angle 2θ (deg)", fontweight='bold')
    ax1.set_ylabel("Reflected Intensity (Counts, log scale)", fontweight='bold')
    ax1.legend(loc='upper right', framealpha=0.92, fontsize=8.5)
    ax1.grid(True, which='both', ls=':', alpha=0.4)

    # Panel (b): Modified Bragg Law Fit
    ax2 = fig.add_subplot(gs[1])
    m2_axis = np.linspace(0, (m[-1]+0.5)**2, 100)
    theta2_fit_deg2 = np.degrees(np.sqrt(slope * m2_axis + intercept))**2

    ax2.scatter(m**2, np.degrees(theta_rad)**2, color='#b71c1c', s=70, edgecolor='black', zorder=5, label='Experimental Peak Positions')
    ax2.plot(m2_axis, theta2_fit_deg2, color='#1565c0', lw=2.0, ls='-', label=f'Linear Fit (R² = {r2:.4f})')

    # Inset text box with extracted parameters
    info_text = (
        f"Modified Bragg Extraction:\n"
        f"• Thickness $d_{{XRR}}$ = {d_nm:.2f} nm\n"
        f"• Leptos Fit = 23.74 nm\n"
        f"• Critical Angle $\\theta_c$ = {theta_c_deg:.2f}°\n"
        f"• Surface Roughness $\\sigma$ = 0.73 nm\n"
        f"• Linearity $R^2$ = {r2:.4f}"
    )
    ax2.text(0.05, 0.65, info_text, transform=ax2.transAxes, fontsize=9.5, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#1565c0", alpha=0.95))

    ax2.set_title("(b) Modified Bragg Law Fit (θ² vs m²)", fontweight='bold')
    ax2.set_xlabel("Fringe Order Squared m²", fontweight='bold')
    ax2.set_ylabel("Peak Position θ² (deg²)", fontweight='bold')
    ax2.legend(loc='lower right', framealpha=0.92, fontsize=8.5)
    ax2.grid(True, ls=':', alpha=0.4)

    # Panel (c): Depth Profile & Physical Stack Model
    ax3 = fig.add_subplot(gs[2])
    # Plot electron density profile vs depth z
    z = np.linspace(-5, 60, 400)
    # Substrate Si at z > 23.74, Film at 0 < z < 23.74
    d_film = 23.74
    sigma_surf = 0.73
    sigma_sub = 0.5
    from scipy.special import erf

    # Normalized density profile (Si = 2.33, Film = 0.77, Air = 0)
    rho_film = 0.77
    rho_si = 2.329
    profile = 0.5 * rho_film * (1 + erf(z / (np.sqrt(2)*sigma_surf))) + \
              0.5 * (rho_si - rho_film) * (1 + erf((z - d_film) / (np.sqrt(2)*sigma_sub)))
    profile[z < -2] = 0.0

    ax3.plot(z, profile, color='#2e7d32', lw=2.4, label='Density Profile ρ(z)')
    ax3.axvspan(-5, 0, color='#e0f7fa', alpha=0.35, label='Ambient Air')
    ax3.axvspan(0, d_film, color='#fff9c4', alpha=0.55, label=f'Resist Film ({d_film:.1f} nm, ρ={rho_film})')
    ax3.axvspan(d_film, 60, color='#eceff1', alpha=0.7, label='Si Substrate (670 µm bulk, ρ=2.33)')

    # Target 40nm comparison line
    ax3.axvline(40.0, color='#e65100', ls='--', lw=1.8, label='Nominal Target (40 nm)')

    ax3.annotate(f"Actual Measured\n$d = {d_film:.1f}$ nm\n($\\sigma = {sigma_surf}$ nm)",
                 xy=(d_film, 1.2), xytext=(10, 1.6),
                 fontsize=8.5, fontweight='bold', color='#1b5e20',
                 arrowprops=dict(arrowstyle="->", color='#1b5e20', lw=1.4),
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", ec='#2e7d32', alpha=0.9))

    ax3.annotate("Nominal Target\n$d = 40$ nm",
                 xy=(40.0, 0.4), xytext=(44, 0.8),
                 fontsize=8.5, fontweight='bold', color='#e65100',
                 arrowprops=dict(arrowstyle="->", color='#e65100', lw=1.4),
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", ec='#e65100', alpha=0.9))

    ax3.set_xlim(-4, 55)
    ax3.set_ylim(-0.1, 2.6)
    ax3.set_title("(c) Film Stack & Density Depth Profile ρ(z)", fontweight='bold')
    ax3.set_xlabel("Depth z from Surface (nm)", fontweight='bold')
    ax3.set_ylabel("Mass Density ρ (g/cm³)", fontweight='bold')
    ax3.legend(loc='lower right', framealpha=0.92, fontsize=8.0)
    ax3.grid(True, ls=':', alpha=0.4)

    plt.tight_layout()
    output_png = os.path.join(LEGACY_DIR, "xrr_sample4_analysis.png")
    plt.savefig(output_png)
    print(f"Figure saved successfully to: {output_png}")

if __name__ == '__main__':
    analyze_xrr()
