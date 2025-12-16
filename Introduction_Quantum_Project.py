import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib as mpl
#######################################################################################################################
# Here you will need to add the path where you dowloaded the software ffmpeg

mpl.rcParams["animation.ffmpeg_path"] = r"C:\Users\me4ad\Downloads\ffmpeg-2025-12-01-git-7043522fe0-essentials_build\ffmpeg-2025-12-01-git-7043522fe0-essentials_build\bin\ffmpeg.exe"
##################################################################################################################################

hbar = 1.0 
m = 1.0     # Particle mass


# Double-barrier potential geometry

a = 4.0     # Inner edge of barriers 
b = 5.2     # Outer edge of barriers
V0 = 0.6    # Barrier height


# Initial Gaussian wave packet parameters

k0 = 1.05      # Initial wave vector
sigma0 = 5.0   # Spatial width of the wave packet
x0 = -120.0    # Initial center position


# Numerical grid and time evolution parameters
x_min = -200.0  # Left boundary of spatial domain
x_max = 200.0   # Right boundary of spatial domain
N_grid = 2**12  # Number of grid points (4096)
dt = 0.05       # Time step
n_steps = 6000  # Total number of time steps


# Barrier region widths

d1 = b - a      # Width of left barrier
d2 = 2.0 * a    # Width of central well
d3 = b - a      # Width of right barrier

# Compute local wave vector k for a particle 
# with energy E in a region with potential V
def k_region(E, V):
    arg = 2.0 * m * (E - V) / (hbar ** 2)
    if np.real(arg) >= 0:
        return np.sqrt(arg + 0j)
    else:
        return 1j * np.sqrt(-arg)

# Compute the transfer matrix at an interface between 
# two regions with wave vectors kL and kR
def interface_matrix(kL, kR):
    eps = 1e-12
    kL_eff = kL if abs(kL) > eps else eps
    kR_eff = kR if abs(kR) > eps else eps
    alpha = kR_eff / kL_eff
    return 0.5 * np.array([[1 + alpha, 1 - alpha],
                           [1 - alpha, 1 + alpha]], dtype=complex)

# Compute the transfer matrix for propagation 
#through a region of length L with wave vector k
def propagation_matrix(k, L):
    phase = k * L
    return np.array([[np.exp(1j * phase), 0],
                     [0, np.exp(-1j * phase)]], dtype=complex)

# Build the total transfer matrix M for the double-barrier system
def M_matrix(E, VL, VR):
    k0r = k_region(E, 0.0)
    kL = k_region(E, VL)
    kW = k_region(E, 0.0)
    kR = k_region(E, VR)
    k4 = k_region(E, 0.0)
    M01 = interface_matrix(k0r, kL)
    M12 = interface_matrix(kL, kW)
    M23 = interface_matrix(kW, kR)
    M34 = interface_matrix(kR, k4)
    P1 = propagation_matrix(kL, d1)
    P2 = propagation_matrix(kW, d2)
    P3 = propagation_matrix(kR, d3)
    M = M34 @ P3 @ M23 @ P2 @ M12 @ P1 @ M01
    return M

# Convert the transfer matrix M to the scattering matrix S
def S_matrix_from_M(M):
    M11, M12 = M[0,0], M[0,1]
    M21, M22 = M[1,0], M[1,1]
    detM = M11*M22 - M12*M21

    eps = 1e-14
    if (not np.isfinite(M22)) or abs(M22) < eps:
        return np.full((2,2), np.nan + 1j*np.nan, dtype=complex)

    tLR = 1.0 / M22
    rL  = M21 / M22
    rR  = -M12 / M22
    tRL = detM / M22

    S = np.array([[rL,  tRL],
                  [tLR, rR ]], dtype=complex)
    return S

# Compute both transfer matrix M and scattering
# matrix S for a given energy and barrier heights
def M_and_S(E, VL, VR):
    M = M_matrix(E, VL, VR)
    S = S_matrix_from_M(M)
    return M, S

# Print matrix in a formatted way
def pretty_print_matrix(name, Mat):
    print(f"\n{name} =")
    for row in Mat:
        print("  ", "  ".join([f"{z.real:+.6e}{z.imag:+.6e}j" for z in row]))

# Extract transmission (t) and reflection (r)
# amplitudes from the transfer matrix
def scattering_amplitudes(E, VL, VR):
    M = M_matrix(E, VL, VR)
    if not np.isfinite(M).all():
        t = 0.0 + 0j
        r = 1.0 + 0j
        return t, r
    M21, M22 = M[1,0], M[1,1]
    if (not np.isfinite(M22)) or (np.abs(M22) < 1e-12):
        t = 0.0 + 0j
        r = np.exp(1j * np.angle(M21 + 1e-20))
    else:
        t = 1.0 / M22
        r = M21 / M22
    return t, r

# Compute the transmission probability T(E) = |t|²
def transmission_probability(E, VL, VR):
    t, r = scattering_amplitudes(E, VL, VR)
    return np.abs(t) ** 2

# Compute the Wigner phase time τ(E) = ℏ dφ/dE
def phase_time(E_grid, VL, VR):
    t_vals = np.array([scattering_amplitudes(E, VL, VR)[0] for E in E_grid])
    bad = ~np.isfinite(t_vals)
    t_vals[bad] = 1e-20 + 0j
    phases = np.unwrap(np.angle(t_vals))
    dphi_dE = np.gradient(phases, E_grid)
    tau = hbar * dphi_dE
    return tau, t_vals

# Define the double-barrier potential profile V(x)
def potential_profile(x, VL, VR):
    V = np.zeros_like(x)
    V += np.where((x > -b) & (x < -a), VL, 0.0)
    V += np.where((x > a) & (x < b), VR, 0.0)
    return V

# Create a normalized Gaussian wave packet
def initial_packet(x):
    norm = (1.0 / (2 * np.pi * sigma0 ** 2)) ** 0.25
    psi = norm * np.exp(-(x - x0) ** 2 / (4 * sigma0 ** 2) + 1j * k0 * x)
    dx = x[1] - x[0]
    prob0 = np.abs(psi) ** 2
    psi *= 1.0 / np.sqrt(prob0.sum() * dx)
    return psi

# Compute the mean energy <E> of the initial wave packet
def mean_energy_of_packet():
    x = np.linspace(x_min, x_max, N_grid)
    dx = x[1] - x[0]
    psi = initial_packet(x)
    k = np.fft.fftfreq(N_grid, d=dx) * 2 * np.pi
    psi_k = np.fft.fft(psi)
    dk = 2 * np.pi / (len(x) * dx)
    prob_k = np.abs(psi_k) ** 2
    prob_k /= prob_k.sum() * dk
    Ek = (hbar ** 2 * k ** 2) / (2 * m)
    E_mean = np.sum(Ek * prob_k) * dk
    return np.real(E_mean)

# Time evolution of the wave packet using the split-operator Fourier method
def simulate_wave_packet(VL, VR,
                         store_psi=False, frame_step=10,
                         steps=n_steps):
    x = np.linspace(x_min, x_max, N_grid)
    dx = x[1] - x[0]
    Vx = potential_profile(x, VL, VR)
    k = np.fft.fftfreq(N_grid, d=dx) * 2 * np.pi
    psi = initial_packet(x)
    Ek = (hbar ** 2 * k ** 2) / (2 * m)
    expV_half = np.exp(-1j * Vx * dt / (2 * hbar))
    expT = np.exp(-1j * Ek * dt / hbar)
    times = []
    x_expect = []
    p_expect = []
    psi_frames = []
    t_frames = []
    for n in range(steps):
        t = n * dt
        if store_psi and n % frame_step == 0:
            psi_frames.append(psi.copy())
            t_frames.append(t)
        if n % 10 == 0:
            prob = np.abs(psi) ** 2
            prob /= prob.sum() * dx
            x_exp = np.sum(x * prob) * dx
            psi_k = np.fft.fft(psi)
            dk = 2 * np.pi / (len(x) * dx)
            prob_k = np.abs(psi_k) ** 2
            prob_k /= prob_k.sum() * dk
            p_vals = hbar * k
            p_exp = np.sum(p_vals * prob_k) * dk
            times.append(t)
            x_expect.append(x_exp)
            p_expect.append(p_exp)
        psi = expV_half * psi
        psi_k = np.fft.fft(psi)
        psi_k *= expT
        psi = np.fft.ifft(psi_k)
        psi = expV_half * psi
    return x, np.array(times), np.array(x_expect), np.array(p_expect), np.array(t_frames), psi_frames

# evolve wave packet for a short time and plot
def short_time_plot(VL, VR):
    x = np.linspace(x_min, x_max, N_grid)
    dx = x[1] - x[0]
    Vx = potential_profile(x, VL, VR)
    psi = initial_packet(x)
    t_short = 2.0
    steps = int(t_short / dt)
    k = np.fft.fftfreq(N_grid, d=dx) * 2 * np.pi
    Ek = (hbar ** 2 * k ** 2) / (2 * m)
    expV_half = np.exp(-1j * Vx * dt / (2 * hbar))
    expT = np.exp(-1j * Ek * dt / hbar)
    for n in range(steps):
        psi = expV_half * psi
        psi_k = np.fft.fft(psi)
        psi_k *= expT
        psi = np.fft.ifft(psi_k)
        psi = expV_half * psi
    prob = np.abs(psi) ** 2
    prob /= prob.sum() * dx
    plt.figure()
    plt.plot(x, prob)
    plt.xlabel("x")
    plt.ylabel(r"$|\Psi(x,t)|^2$")
    plt.title(r"Normalized $|\Psi(x,t)|^2$ at short time (symmetric V0-V0)")
    plt.tight_layout()

# Compute T(E) and τ(E) for symmetric and asymmetric configurations
def compute_spectra_and_tunneling_time():
    E_min, E_max = 0.0, 3.0 * V0
    E_grid = np.linspace(E_min, E_max, 800)
    T_sym = np.array([transmission_probability(E, V0, V0) for E in E_grid])
    tau_sym, _ = phase_time(E_grid, V0, V0)
    T_asym = np.array([transmission_probability(E, V0, 2.0 * V0) for E in E_grid])
    tau_asym, _ = phase_time(E_grid, V0, 2.0 * V0)

    plt.figure()
    plt.plot(E_grid, T_sym, label="V0-V0")
    plt.plot(E_grid, T_asym, "--", label="V0-2V0")
    plt.xlabel("E")
    plt.ylabel("T(E)")
    plt.title("Transmission probability comparison")
    plt.legend()
    plt.tight_layout()

    plt.figure()
    plt.plot(E_grid, tau_sym, label="V0-V0")
    plt.plot(E_grid, tau_asym, "--", label="V0-2V0")
    plt.xlabel("E")
    plt.ylabel(r"$\tau(E)$")
    plt.title("Phase tunneling time comparison")
    plt.legend()
    plt.tight_layout()

    E0 = mean_energy_of_packet()
    idx = np.argmin(np.abs(E_grid - E0))
    tau_E0 = tau_sym[idx]
    print("<E> from packet =", E0, " tunneling time tau(<E>) =", tau_E0)

# Generate movie of wave packet evolution
def make_movie_mp4(frames, times, x, VL, VR, filename):
    fig, ax = plt.subplots()
    ax.set_xlim(-30, 30)
    ax.set_ylim(0, 0.05)
    ax.set_xlabel("x")
    ax.set_ylabel(r"$|\Psi(x,t)|^2$")

    ax.fill_between(x, 0, 1, where=((x > -b) & (x < -a)),
                    color="gray", alpha=0.35, step="mid")
    ax.fill_between(x, 0, 1, where=((x > a) & (x < b)),
                    color="gray", alpha=0.35, step="mid")

    line, = ax.plot([], [], lw=2)

    def init():
        line.set_data([], [])
        return line,

    def update(i):
        psi = frames[i]
        prob = np.abs(psi)**2
        prob /= prob.sum() * (x[1] - x[0])

        line.set_data(x, prob)

        mask = (x >= -30) & (x <= 30)
        ymax = prob[mask].max()
        ax.set_ylim(0, max(0.02, 1.2*ymax))

        ax.set_title(r"$|\Psi(x,t)|^2$, t = {:.2f}".format(times[i]))
        return line,

    ani = FuncAnimation(fig, update, frames=len(frames),
                        init_func=init, blit=True)

    writer = FFMpegWriter(fps=20, bitrate=1800)
    ani.save(filename, writer=writer)
    plt.close(fig)
    print("MP4 saved:", filename)

# Compute transmitted probability P(x > b) as a function of time
def transmission_vs_time(x, psi_frames):
    dx = x[1]-x[0]
    Tt = []
    for psi in psi_frames:
        prob = np.abs(psi)**2
        prob /= prob.sum() * dx
        Tt.append(np.sum(prob[x > b]) * dx)
    return np.array(Tt)

# Main function
def main():
    compute_spectra_and_tunneling_time()
    short_time_plot(V0, V0)

    x, t_sym, x_sym, p_sym, t_frames_sym, psi_frames_sym = simulate_wave_packet(
        V0, V0, store_psi=True, frame_step=10, steps=n_steps
    )
    x, t_as, x_as, p_as, t_frames_as, psi_frames_as = simulate_wave_packet(
        V0, 2.0 * V0, store_psi=True, frame_step=10, steps=n_steps
    )

    T_sym_t = transmission_vs_time(x, psi_frames_sym)
    T_as_t  = transmission_vs_time(x, psi_frames_as)

    E0 = mean_energy_of_packet()
    print("\nUsing E0 = <E> =", E0)

    M_sym, S_sym = M_and_S(E0, V0, V0)
    M_as,  S_as  = M_and_S(E0, V0, 2.0*V0)

    pretty_print_matrix("M (symmetric V0-V0)", M_sym)
    pretty_print_matrix("S (symmetric V0-V0)", S_sym)

    pretty_print_matrix("M (asymmetric V0-2V0)", M_as)
    pretty_print_matrix("S (asymmetric V0-2V0)", S_as)

    tLR_sym = S_sym[1,0]
    rL_sym  = S_sym[0,0]
    print("\nSymmetric check: |t|^2 =", abs(tLR_sym)**2, " |r|^2 =", abs(rL_sym)**2, " sum =", abs(tLR_sym)**2 + abs(rL_sym)**2)

    tLR_as = S_as[1,0]
    rL_as  = S_as[0,0]
    print("Asymmetric check: |t|^2 =", abs(tLR_as)**2, " |r|^2 =", abs(rL_as)**2, " sum =", abs(tLR_as)**2 + abs(rL_as)**2)

    plt.figure()
    plt.plot(t_frames_sym, T_sym_t, label="V0-V0")
    plt.plot(t_frames_as,  T_as_t,  "--", label="V0-2V0")
    plt.xlabel("t")
    plt.ylabel("P(x > b)")
    plt.title("Transmitted probability vs time")
    plt.legend()
    plt.tight_layout()

    plt.figure()
    plt.plot(t_sym, x_sym, label="V0-V0")
    plt.plot(t_as, x_as, "--", label="V0-2V0")
    plt.xlabel("t")
    plt.ylabel(r"$\langle x \rangle$")
    plt.title(r"$\langle x \rangle(t)$ comparison")
    plt.legend()
    plt.tight_layout()

    plt.figure()
    plt.plot(t_sym, p_sym, label="V0-V0")
    plt.plot(t_as, p_as, "--", label="V0-2V0")
    plt.xlabel("t")
    plt.ylabel(r"$\langle p \rangle$")
    plt.title(r"$\langle p \rangle(t)$ comparison")
    plt.legend()
    plt.tight_layout()

    make_movie_mp4(psi_frames_sym, t_frames_sym, x, V0, V0,
                   "wavepacket_symmetric.mp4")
    make_movie_mp4(psi_frames_as, t_frames_as, x, V0, 2.0 * V0,
                   "wavepacket_asymmetric.mp4")

    plt.show()

if __name__ == "__main__":
    main()