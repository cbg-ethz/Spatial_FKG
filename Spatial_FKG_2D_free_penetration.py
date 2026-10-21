import numpy as np
import time
import math
import matplotlib.pyplot as plt
from filterpy.monte_carlo import systematic_resample
from copy import deepcopy

# ============================================================
# Spatial-FKG reference implementation
# Tumor state: [x1, x2, x3, rx, ry]
# Immune state: [z, rx, ry] where z matches tumor x2
# Macro steps freeze neighborhood fields N_k(r), I_k(r), g_k(r)
# ============================================================

# -----------------------------
# Global settings
# -----------------------------

T = 30
delay_time = 6.0
num_points = 6
random_resampling_points = np.sort(np.random.uniform(delay_time, T, num_points - 2))
resampling_time = [0.0, delay_time/2.0] + list(np.linspace(delay_time, T, num_points-2))
if resampling_time[-1] != T:
    resampling_time.append(T)

intervals = np.diff(resampling_time)
print("interval variance",np.var(intervals))

dt_tumor = 0.5       # tumor micro step inside each immune macro interval

num = 5000           # tumor weighted trajectories
num_immune = 5000    # immune weighted trajectories
h = 0.002            # immune weight grid step (micro ODE grid for influx weighting)

# Spatial lattice
L = 70              # grid size LxL
R_neigh = 2         # neighborhood radius for density fields
R_move = 2          # movement radius
R_place = 2         # division placement radius (kernel K_R)
R_edge = 1          # edge neighborhood radius
theta_edge = 1      # threshold on local tumor density to define "tumor region"
K = 20.0            # base capacity
eps = 1.0           # capacity floor

# Tumor kinetics
k_death_1 = 0.1     # baseline death
k_div = 0.5         # base division
k_A = 0.5           # mutation acquisition rate
p_A = 1/3           # neg-bin parameter (scalar x2)
p_IE = 1e-3         # immune escape prob

eta_move_tumor = 0.5
beta_m_tumor = 0.08

# Immune kinetics
eta_move_immune = 0.2
beta_m_immune = -0.08   # negative = attracted to high tumor density
immune_death = 0.05
lambda_in_const = 0.001  # immune influx strength
free_penetration_min_tumor_density = 1e-12

# Delay: immune affects tumor after this many macro steps
delay_steps = 1

# State-space sizes (for x1,x2,x3 only; spatial is handled separately)
x1_size = 1001
x2_size = 1001
x3_size = 2

time_record = []
cell_count_record = []
immune_count_record = []

# -----------------------------
# Helpers: neighborhood operations
# -----------------------------
def make_disk_kernel(R: int) -> np.ndarray:
    y, x = np.ogrid[-R:R+1, -R:R+1]
    return ((x*x + y*y) <= R*R).astype(float)

def conv2_same(a: np.ndarray, k: np.ndarray) -> np.ndarray:
    H, W = a.shape
    kh, kw = k.shape
    ph, pw = kh//2, kw//2
    ap = np.pad(a, ((ph, ph), (pw, pw)), mode="edge")
    out = np.zeros_like(a, dtype=float)
    for i in range(H):
        for j in range(W):
            out[i, j] = np.sum(ap[i:i+kh, j:j+kw] * k)
    return out

def compute_fields(tumor_grid: np.ndarray, immune_grid: np.ndarray):
    kernel = make_disk_kernel(R_neigh)
    area = max(1.0, float(kernel.sum()))
    N_sum = conv2_same(tumor_grid.astype(float), kernel)
    I_sum = conv2_same(immune_grid.astype(float), kernel)

    N_t = N_sum / area
    I_t = I_sum / area # summed immune in neighborhood

    K_t = np.maximum(eps, K - I_t)
    g_t = np.maximum(0.0, 1.0 - (N_t / K_t))
    return N_t, I_t, K_t, g_t

def tumor_edge_mask(N_t: np.ndarray):
    core = (N_t >= theta_edge)
    kernel = make_disk_kernel(R_edge).astype(bool)
    kh, kw = kernel.shape
    ph, pw = kh//2, kw//2
    Np = np.pad(N_t, ((ph, ph), (pw, pw)), mode="edge")
    edge = np.zeros_like(core, dtype=bool)
    H, W = N_t.shape
    for i in range(H):
        for j in range(W):
            if not core[i, j]:
                continue
            window = Np[i:i+kh, j:j+kw]
            neigh = window[kernel]
            if np.any(neigh < theta_edge):
                edge[i, j] = True
    return edge

def neighbor_coords(i, j, Rm):
    coords = []
    for di in range(-Rm, Rm+1):
        for dj in range(-Rm, Rm+1):
            if di*di + dj*dj <= Rm*Rm:
                ii, jj = i + di, j + dj
                if 0 <= ii < L and 0 <= jj < L:
                    coords.append((ii, jj))
    return coords


def placement_coords(i, j, Rp):
    # Same geometry as movement neighborhood B_R(r)
    return neighbor_coords(i, j, Rp)


def K_R_sample(i, j, Rp):
    # Placement kernel K_R(r'->r) = 1_{r in B_R(r')}/|B_R(r')|
    # Returns a uniformly sampled destination r in B_R((i,j)).
    coords = placement_coords(i, j, Rp)
    if len(coords) == 0:
        return i, j
    ii, jj = coords[np.random.randint(len(coords))]
    return ii, jj

# -----------------------------
# Rate functions
# -----------------------------
def lambda_death(n):
    return 5 * (1-n[2])

def lambda_div(n):
    return k_div

def lambda_double_div(n):
    return 2 * lambda_div(n)

def lambda_in(n):
    return lambda_in_const

def div(p):
    return [0.5 for _ in p]

# -----------------------------
# Reaction / System classes (names preserved)
# -----------------------------
class Reaction:
    def __init__(self, propensity, num_lefts=None, num_rights=None):
        self.propensity = propensity
        self.num_lefts = np.array(num_lefts) if num_lefts is not None else None
        self.num_rights = np.array(num_rights) if num_rights is not None else None
        self.num_diff = None

class System:
    """
    Tumor SSA for single-particle state [x1,x2,x3,rx,ry]
    reactions[0] = division/mutation event
    reactions[1] = movement event
    """
    def __init__(self, num_elements):
        self.num_elements = num_elements
        self.reactions = []

    def add_reaction(self, propensity, num_lefts=None, num_rights=None):
        self.reactions.append(Reaction(propensity, num_lefts, num_rights))

    def evolute(self, final_time, inits=None, steps=200000, N_frozen=None, g_frozen=None):
        self.t = [0.0]
        if inits is None:
            self.n = [np.zeros(self.num_elements)]
        else:
            self.n = [np.array(inits, dtype=float)]

        for _ in range(steps):
            cur = self.n[-1]
            rx, ry = int(cur[3]), int(cur[4])

            a_div = lambda_double_div(cur) * (g_frozen[rx, ry] if g_frozen is not None else 1.0)
            a_mov = eta_move_tumor
            A0 = a_div + a_mov
            if A0 <= 0:
                self.t.append(final_time); self.n.append(cur); break

            dt = -np.log(np.random.random()) / A0
            if self.t[-1] + dt > final_time:
                self.t.append(final_time); self.n.append(cur); break
            self.t.append(self.t[-1] + dt)

            if np.random.random() < (a_div / A0):
                DX1 = np.random.poisson(lam=k_A)
                if DX1 > 0:
                    DX2 = np.random.negative_binomial(n=DX1, p=p_A)
                else:
                    DX2 = 0

                if cur[2] == 0:
                    DX3 = np.random.binomial(n=1, p=p_IE)
                else:
                    DX3 = 0

                new_state = cur.copy()
                new_state[0] += DX1
                new_state[1] += DX2
                new_state[2] += DX3

                # --- Option A: division-time placement for the tracked lineage ---
                # One daughter stays at (rx, ry), the other is placed uniformly in B_{R_place}(rx, ry).
                # Since we track one randomly chosen daughter, with prob 1/2 we stay, with prob 1/2 we jump via K_R.
                if np.random.random() < 0.5:
                    new_rx, new_ry = K_R_sample(rx, ry, R_place)
                    new_state[3] = new_rx
                    new_state[4] = new_ry

                self.n.append(new_state)
            else:
                nbrs = neighbor_coords(rx, ry, R_move)
                w = []
                for (ii, jj) in nbrs:
                    Nv = N_frozen[ii, jj] if N_frozen is not None else 0.0
                    gv = g_frozen[ii, jj] if g_frozen is not None else 1.0
                    w.append(np.exp(-beta_m_tumor * Nv) * gv)
                w = np.array(w, dtype=float)
                s = w.sum()
                new_state = cur.copy()
                if s > 0:
                    p = w / s
                    idx = np.random.choice(np.arange(len(nbrs)), p=p)
                    new_state[3] = nbrs[idx][0]
                    new_state[4] = nbrs[idx][1]
                self.n.append(new_state)

        return deepcopy(self.t), deepcopy(self.n)

class System_immune:
    """
    Immune SSA for single-particle state [z, rx, ry]
    Only movement events here (you can extend).
    """
    def __init__(self, num_elements):
        self.num_elements = num_elements
        self.reactions = []

    def evolute(self, final_time, inits=None, steps=200000, N_frozen=None):
        self.t = [0.0]
        if inits is None:
            self.n = [np.zeros(self.num_elements)]
        else:
            self.n = [np.array(inits, dtype=float)]

        for _ in range(steps):
            cur = self.n[-1]
            rx, ry = int(cur[1]), int(cur[2])

            A0 = eta_move_immune
            dt = -np.log(np.random.random()) / A0
            if self.t[-1] + dt > final_time:
                self.t.append(final_time); self.n.append(cur); break
            self.t.append(self.t[-1] + dt)

            nbrs = neighbor_coords(rx, ry, R_move)
            w = []
            for (ii, jj) in nbrs:
                Nv = N_frozen[ii, jj] if N_frozen is not None else 0.0
                w.append(np.exp(-beta_m_immune * Nv))
            w = np.array(w, dtype=float)
            s = w.sum()
            new_state = cur.copy()
            if s > 0:
                p = w / s
                idx = np.random.choice(np.arange(len(nbrs)), p=p)
                new_state[1] = nbrs[idx][0]
                new_state[2] = nbrs[idx][1]
            self.n.append(new_state)

        return deepcopy(self.t), deepcopy(self.n)

# -----------------------------
# FKG-SSA tumor step (spatial)
# -----------------------------
def _immune_multiplier_for_tumor_state(n_state, I_frozen):
    """
    Returns 1 if there exists at least one immune cell within radius R_neigh
    whose immune inner-state label z matches the tumor antigen label x2.
    Otherwise returns 0.
    I_frozen is expected to be a label-resolved immune grid of shape (x2_size, L, L).
    """
    if I_frozen is None:
        return 0.0

    x2 = int(n_state[1])
    rx, ry = int(n_state[3]), int(n_state[4])

    if x2 < 0 or x2 >= I_frozen.shape[0]:
        return 0.0

    immune_mass = 0.0
    for (ii, jj) in neighbor_coords(rx, ry, R_neigh):
        immune_mass += float(I_frozen[x2, ii, jj])
    return immune_mass

def FGK_SSA_one_period(sample_size, time_period, mu_total, init_particles=None, h=0.0001, I_frozen=None, N_frozen=None, g_frozen=None):
    # initial conditions
    initial_conditions = []
    if init_particles is None:
        initial_conditions = [np.array([0, 0, 0, L//2, L//2], dtype=float)] * sample_size
    else:
        total_weight = float(sum(init_particles['weight_list']))
        weights = [w / total_weight for w in init_particles['weight_list']]
        index = systematic_resample(weights)
        for i in index:
            initial_conditions.append(init_particles['state_list'][i])

    system = System(num_elements=5)
    system.add_reaction(lambda_double_div)
    system.add_reaction(lambda_double_div)

    weight_all = []
    state_all = []
    traj_n = []
    traj_t = []
    for i in range(sample_size):
        single_t, single_trj = system.evolute(final_time=time_period, inits=initial_conditions[i], N_frozen=N_frozen, g_frozen=g_frozen)
        traj_n.append(single_trj)
        traj_t.append(single_t)

    for i in range(len(traj_n)):
        t_grid = traj_t[i]
        n_at_tgrid = traj_n[i]
        w = 1.0
        for j in range(len(t_grid)-1):
            dt = t_grid[j+1] - t_grid[j]
            g_mult = float(g_frozen[int(n_at_tgrid[j][3]), int(n_at_tgrid[j][4])]) if g_frozen is not None else 1.0
            I_mult = _immune_multiplier_for_tumor_state(n_at_tgrid[j], I_frozen)
            integrand = lambda_div(n_at_tgrid[j]) * g_mult - lambda_death(n_at_tgrid[j]) * I_mult - k_death_1
            w = np.exp(integrand * dt) * w
        weight_all.append(w)
        state_all.append(n_at_tgrid[-1])

    ESS = sum(weight_all)**2 / sum([x**2 for x in weight_all])

    dist = {}
    for i in range(sample_size):
        s = state_all[i]
        x1, x2, x3, rx, ry = int(s[0]), int(s[1]), int(s[2]), int(s[3]), int(s[4])
        if 0 <= x1 < x1_size and 0 <= x2 < x2_size and 0 <= x3 < x3_size:
            key = (x1, x2, x3, rx, ry)
            dist[key] = dist.get(key, 0.0) + weight_all[i]
    for k in list(dist.keys()):
        dist[k] = dist[k] / sample_size * mu_total

    return dist, ESS, weight_all, state_all

# -----------------------------
# Immune FKG-SSA step (spatial)
# -----------------------------
def trajectory_on_the_grid(t, n, T, h):
    steps = int(T / h) + 1
    tgrid = np.linspace(0, T, steps)
    n_at_tgrid = []
    i_current = 0
    for time_point in tgrid:
        while i_current + 1 < len(t) and t[i_current + 1] < time_point:
            i_current += 1
        n_at_tgrid.append(n[i_current])
    return tgrid, n_at_tgrid

def tumor_marginal_x2(dist):
    probs_x2 = np.zeros(x2_size, dtype=float)
    for (_, x2, _, _, _), mass in dist.items():
        if 0 <= x2 < x2_size:
            probs_x2[x2] += mass
    return probs_x2

def get_edge_sites(edge_mask):
    if edge_mask is None:
        return []
    return list(zip(*np.where(edge_mask)))

def FGK_SSA_one_period_immune(sample_size, time_period, h, mu_total, dist_tumor=None, N_frozen=None, edge_mask=None, init_particles=None):
    initial_conditions = []

    # Keep half of the previous immune states.
    previous_states = []
    previous_weights = np.array([], dtype=float)

    if init_particles is not None:
        previous_states = init_particles['state_list']
        previous_weights = np.asarray(init_particles['weight_list'], dtype=float)

        if len(previous_states) > 0 and previous_weights.sum() > 0:
            previous_weights = previous_weights / previous_weights.sum()

            index = np.random.choice(
                len(previous_states),
                size=sample_size // 2,
                replace=True,
                p=previous_weights
            )

            initial_conditions.extend(
                np.array(previous_states[i], dtype=float).reshape(-1)
                for i in index
            )

    # Free-penetrating recruitment, matched to the ABM:
    # sample immune labels from the current positive-x2 tumor marginal,
    # and sample entry positions throughout the tumor-associated region.
    # Entry sites are weighted by exp(-beta_m_immune * N_frozen), so with
    # beta_m_immune < 0 immune influx is biased toward higher tumor density.
    recruited_from_free_penetration = False

    if dist_tumor is not None and N_frozen is not None:
        probs_x2 = tumor_marginal_x2(dist_tumor)
        probs_x2[0] = 0.0

        entry_sites = list(zip(*np.where(N_frozen > free_penetration_min_tumor_density)))
        n_missing = sample_size - len(initial_conditions)

        if n_missing > 0 and probs_x2.sum() > 0 and len(entry_sites) > 0:
            probs = probs_x2 / probs_x2.sum()

            z_picks = np.random.choice(
                np.arange(len(probs)),
                size=n_missing,
                replace=True,
                p=probs
            )

            entry_weights = np.array(
                [np.exp(-beta_m_immune * float(N_frozen[rx, ry])) for (rx, ry) in entry_sites],
                dtype=float
            )
            if entry_weights.sum() > 0:
                entry_weights /= entry_weights.sum()
            else:
                entry_weights = np.full(len(entry_sites), 1.0 / len(entry_sites), dtype=float)

            entry_idx = np.random.choice(
                np.arange(len(entry_sites)),
                size=n_missing,
                replace=True,
                p=entry_weights
            )

            for i in range(n_missing):
                rx, ry = entry_sites[int(entry_idx[i])]
                initial_conditions.append(
                    np.array([int(z_picks[i]), int(rx), int(ry)], dtype=float)
                )

            recruited_from_free_penetration = True

    # Empty-support fallback:
    # keep the existing immune population by resampling it.
    # This fills Monte Carlo trajectories without adding new immune influx.
    n_missing = sample_size - len(initial_conditions)

    if (
        n_missing > 0
        and len(previous_states) > 0
        and previous_weights.size > 0
        and previous_weights.sum() > 0
    ):
        fallback_idx = np.random.choice(
            len(previous_states),
            size=n_missing,
            replace=True,
            p=previous_weights
        )

        initial_conditions.extend(
            np.array(previous_states[i], dtype=float).reshape(-1)
            for i in fallback_idx
        )

        if not recruited_from_free_penetration:
            print(
                "Warning: no valid free-penetration tumor support or antigen distribution; "
                "existing immune particles were resampled without new recruitment."
            )

    # Last resort for the first immune step.
    n_missing = sample_size - len(initial_conditions)

    if n_missing > 0:
        initial_conditions.extend(
            np.array([0, L // 2, L // 2], dtype=float)
            for _ in range(n_missing)
        )

    if len(initial_conditions) != sample_size:
        raise RuntimeError(
            f"Expected {sample_size} immune initial conditions, "
            f"but obtained {len(initial_conditions)}."
        )

    system = System_immune(num_elements=3)
    traj_n = []
    traj_t = []
    for i in range(sample_size):
        single_t, single_trj = system.evolute(final_time=time_period, inits=initial_conditions[i], N_frozen=N_frozen)
        t_grid, n_at_tgrid = trajectory_on_the_grid(single_t, single_trj, time_period, h)
        traj_t.append(t_grid)
        traj_n.append(n_at_tgrid)

    steps = int(time_period / h) + 1

    p_hat = [dict() for _ in range(steps)]
    for i in range(sample_size):
        for j in range(steps):
            s = traj_n[i][j]
            key = (int(s[0]), int(s[1]), int(s[2]))
            p_hat[j][key] = p_hat[j].get(key, 0.0) + 1.0 / sample_size

    weight_all_immune = []
    pro_num = []
    pos_final = []
    for i in range(sample_size):
        n_at_tgrid = traj_n[i]
        w = np.zeros(steps, dtype=float)
        w[0] = 1.0
        for j in range(steps-1):
            #n_scalar = [int(s[0]) for s in n_at_tgrid]
            z = int(n_at_tgrid[j][0])
            rx = int(n_at_tgrid[j][1])
            ry = int(n_at_tgrid[j][2])
            key = (z, rx, ry)
            p_val = p_hat[j].get(key, 0.0)
            if p_val > 0:
                w[j + 1] = w[j] + h * ((0.1 - k_death_1) * w[j]) + h * (
                            lambda_in(z) / (max(mu_total, 1e-12) * p_val))
            else:
                w[j + 1] = w[j] + h * ((0.1 - k_death_1) * w[j])
        weight_all_immune.append(w[-1])
        pro_num.append(int(n_at_tgrid[-1][0]))
        pos_final.append((int(n_at_tgrid[-1][1]), int(n_at_tgrid[-1][2])))

    ESS_immune = sum(weight_all_immune)**2 / sum([x**2 for x in weight_all_immune])

    weight_pro_dist = np.zeros(x2_size, dtype=float)
    for i in range(sample_size):
        z = pro_num[i]
        if 0 <= z < x2_size:
            weight_pro_dist[z] += weight_all_immune[i]
    weight_pro_dist = weight_pro_dist / sample_size * mu_total

    immune_grid_label_est = np.zeros((x2_size, L, L), dtype=np.float32)
    for i in range(sample_size):
        z = pro_num[i]
        rx, ry = pos_final[i]
        if 0 <= z < x2_size:
            immune_grid_label_est[z, rx, ry] += weight_all_immune[i]
    immune_grid_label_est = immune_grid_label_est / sample_size * mu_total

    immune_grid_est = immune_grid_label_est.sum(axis=0)

    return weight_pro_dist, immune_grid_est, immune_grid_label_est, ESS_immune, weight_all_immune, pro_num, pos_final

# -----------------------------
# Macro iteration
# -----------------------------
def dist_to_grids(dist):
    tumor_grid = np.zeros((L, L), dtype=float)
    for (_, _, _, rx, ry), mass in dist.items():
        tumor_grid[rx, ry] += mass
    return tumor_grid

start_time = time.time()

dist = None
weight_all = None
state_all = None

weight_pro_dist = np.zeros(x2_size, dtype=float)
immune_grid = np.zeros((L, L), dtype=float)
immune_grid_label = np.zeros((x2_size, L, L), dtype=np.float32)

I_label_history = []
immune_init_particles = None
for i in range(len(resampling_time)-1):
    tumor_grid = dist_to_grids(dist) if dist is not None else np.zeros((L, L), dtype=float)
    N_field, I_field, K_field, g_field = compute_fields(tumor_grid, immune_grid)
    edge_mask = tumor_edge_mask(N_field)

    I_label_history.append(immune_grid_label.copy())
    if len(I_label_history) > (delay_steps + 1):
        I_label_history.pop(0)
    I_for_tumor = I_label_history[0] if len(I_label_history) > delay_steps else None

    tumor_time = [t for t in np.arange(resampling_time[i], resampling_time[i+1], dt_tumor)]
    if len(tumor_time) == 0 or tumor_time[0] != resampling_time[i]:
        tumor_time = [resampling_time[i]] + tumor_time
    if tumor_time[-1] < resampling_time[i+1]:
        tumor_time.append(resampling_time[i+1])

    for j in range(len(tumor_time)-1):
        tumor_grid = dist_to_grids(dist) if dist is not None else np.zeros((L, L), dtype=float)
        N_field, I_field_tmp, K_field, g_field = compute_fields(tumor_grid, immune_grid)

        if dist is None:
            dist, ESS, weight_all, state_all = FGK_SSA_one_period(
                sample_size=num,
                time_period=tumor_time[j+1]-tumor_time[j],
                mu_total=10.0,
                init_particles=None,
                I_frozen=None,
                N_frozen=N_field,
                g_frozen=g_field
            )
        else:
            init_particles = {'state_list': state_all, 'weight_list': weight_all}
            mu_total = float(sum(dist.values()))

            dist, ESS, weight_all, state_all = FGK_SSA_one_period(
                sample_size=num,
                time_period=tumor_time[j+1]-tumor_time[j],
                mu_total=mu_total,
                init_particles=init_particles,
                I_frozen=(I_for_tumor if i >= 2 else None),
                N_frozen=N_field,
                g_frozen=g_field
            )

    tumor_grid = dist_to_grids(dist)
    N_field, I_field, K_field, g_field = compute_fields(tumor_grid, immune_grid)
    edge_mask = tumor_edge_mask(N_field)
    current_edge_mask = edge_mask.copy()

    if i >= 1:
        mu_total_immune = float(weight_pro_dist.sum()) if weight_pro_dist.sum() > 0 else 1.0
        dt_immune = resampling_time[i+1]-resampling_time[i]

        weight_pro_dist, immune_grid, immune_grid_label, ESS_immune, weight_all_immune, pro_num, pos_final = FGK_SSA_one_period_immune(
            sample_size=num_immune,
            time_period=dt_immune,
            h=h,
            mu_total=mu_total_immune,
            dist_tumor=dist,
            N_frozen=N_field,
            edge_mask=current_edge_mask,  # retained for compatibility; free-penetration recruitment uses N_frozen
            init_particles=immune_init_particles
        )
        immune_init_particles = {
            'state_list': [np.array([int(pro_num[k]), int(pos_final[k][0]), int(pos_final[k][1])], dtype=float)
                           for k in range(len(pro_num))],
            'weight_list': weight_all_immune
        }
    current_time = resampling_time[i+1]
    current_cell_count = float(sum(dist.values()))
    time_record.append(current_time)
    cell_count_record.append(current_cell_count)
    immune_count_record.append(float(weight_pro_dist.sum()))

end_time = time.time()
print("Time used:", end_time - start_time)

# -----------------------------
# Visualizations
# -----------------------------
font = {'family':'Arial', 'size':17}

marginal_x1 = np.zeros(x1_size, dtype=float)
marginal_x2 = np.zeros(x2_size, dtype=float)
marginal_x3 = np.zeros(x3_size, dtype=float)
for (x1, x2, x3, rx, ry), mass in dist.items():
    if 0 <= x1 < x1_size:
        marginal_x1[x1] += mass
    if 0 <= x2 < x2_size:
        marginal_x2[x2] += mass
    if 0 <= x3 < x3_size:
        marginal_x3[x3] += mass

m1 = [marginal_x1[i] for i in range(len(marginal_x1)) if marginal_x1[i] > 0]
m2 = [marginal_x2[i] for i in range(len(marginal_x2)) if marginal_x2[i] > 0]
total_x3_mass = float(marginal_x3.sum())
escaped_fraction = float(marginal_x3[1] / total_x3_mass) if total_x3_mass > 0 and len(marginal_x3) > 1 else 0.0
print("Final marginal x3 mass [non-escaped, escaped]:", marginal_x3.tolist())
print("Final escaped fraction x3=1:", escaped_fraction)


fig, ax = plt.subplots(1, 3, figsize=(18, 5))
ax[0].tick_params(labelsize=15)
ax[0].tick_params(pad=0.03)
ax[0].plot(time_record, cell_count_record, linewidth=5, color='indianred')
ax[0].set_xlabel("Time", font=font, fontsize=20)
ax[0].set_ylabel("Total number of cells (mass)", font=font, fontsize=20)
ax[0].set_title("Tumor cell population over time", font=font, fontsize=22)

ax[1].tick_params(labelsize=15)
ax[1].tick_params(pad=0.03)
ax[1].bar(np.arange(len(m1)), m1, color='steelblue')
ax[1].set_xlabel(r'Number of antigenic mutations ($x_{1}$)', font=font, fontsize=20)
ax[1].set_title(r'Spatial model (dev)', font=font, fontsize=22)
ax[1].set_ylabel('Number of cells (mass)', font=font, fontsize=20)

ax[2].tick_params(labelsize=15)
ax[2].tick_params(pad=0.03)
ax[2].bar(np.arange(len(m2)), m2, color='sandybrown')
ax[2].set_xlabel(r'Antigenicity of cells ($x_{2}$)', font=font, fontsize=20)
ax[2].set_title(r'Spatial model (dev)', font=font, fontsize=22)
ax[2].set_ylabel('Number of cells (mass)', font=font, fontsize=20)
plt.show()

fig_x3, ax_x3 = plt.subplots(1, 1, figsize=(7, 5))
ax_x3.tick_params(labelsize=15)
ax_x3.tick_params(pad=0.03)
ax_x3.bar(np.arange(x3_size), marginal_x3)
ax_x3.set_xticks(np.arange(x3_size))
ax_x3.set_xticklabels([r"$x_3=0$\nnon-escaped", r"$x_3=1$\nescaped"])
ax_x3.set_ylabel('Number of cells (mass)', font=font, fontsize=20)
ax_x3.set_title(r'Marginal distribution of immune escape ($x_3$ / DX3)', font=font, fontsize=22)
plt.tight_layout()
plt.show()

fig2, ax2 = plt.subplots(1, 2, figsize=(16, 5))
ax2[0].tick_params(labelsize=15)
ax2[0].tick_params(pad=0.03)
ax2[0].plot(time_record, immune_count_record, linewidth=5, color='seagreen')
ax2[0].set_xlabel("Time", font=font, fontsize=20)
ax2[0].set_ylabel("Total number of immune cells (mass)", font=font, fontsize=20)
ax2[0].set_title("Immune cell population over time", font=font, fontsize=22)

ax2[1].tick_params(labelsize=15)
ax2[1].tick_params(pad=0.03)
m_final = np.array(weight_pro_dist, dtype=float)
z_support = np.where(m_final > 0)[0]
if len(z_support) > 0:
    ax2[1].bar(z_support, m_final[z_support], color='mediumseagreen')
    ax2[1].set_xlabel(r'Immune label ($z=x_{2}$)', font=font, fontsize=20)
    ax2[1].set_ylabel('Immune mass', font=font, fontsize=20)
    ax2[1].set_title("Final immune label distribution", font=font, fontsize=22)
else:
    ax2[1].text(0.5, 0.5, "No immune cells", ha='center', va='center', transform=ax2[1].transAxes, fontsize=18)
    ax2[1].set_axis_off()
plt.show()

tumor_grid_final = dist_to_grids(dist)
fig3, ax3 = plt.subplots(1, 2, figsize=(14, 6))
im0 = ax3[0].imshow(tumor_grid_final, origin='lower')
ax3[0].set_title("Final tumor (mass per site)")
plt.colorbar(im0, ax=ax3[0], fraction=0.046, pad=0.04)

im1 = ax3[1].imshow(immune_grid, origin='lower')
ax3[1].set_title("Final immune (mass per site)")
plt.colorbar(im1, ax=ax3[1], fraction=0.046, pad=0.04)
plt.show()
