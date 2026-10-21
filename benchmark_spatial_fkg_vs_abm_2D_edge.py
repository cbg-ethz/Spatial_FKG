import os
import csv
import time
from copy import deepcopy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

"""
2D agent-based test benchmark matched to the spatial-FKG tumor dynamics, using
edge-dependent immune influx.

Purpose
-------
This script is not meant to replace the spatial-FKG implementation. It is a
direct stochastic ABM used for qualitative benchmarking against the 2D FKG-SSA
model. The key choices are aligned with Script_test_FKG_SSA_spatial_2D.py:

    - tumor state: [x1, x2, x3, rx, ry]
    - immune state: [z, rx, ry]
    - local density fields N_t(r), I_t(r)
    - local capacity factor g_t(r) = max(0, 1 - N_t(r) / K_t(r))
    - K_t(r) = max(eps, K - I_t(r))
    - tumor division with mutation and immune escape state x3
    - tumor and immune movement biased by local density
    - free-penetrating immune influx attracted toward high tumor-density regions
    - local antigen-specific immune recognition

Important benchmarking note
---------------------------
The spatial comparison should be distributional, not cell-by-cell. Compare
averaged ABM maps and shared spatial summaries against FKG mass maps.
"""


# -----------------------------------------------------------------------------
# Parameters matched to the current 2D FKG script
# -----------------------------------------------------------------------------
RUN_SEED = int(os.environ.get("BENCHMARK_SEED", "1"))
RUN_ID = os.environ.get("BENCHMARK_RUN_ID", f"seed{RUN_SEED}")
OUTPUT_DIR = os.environ.get("BENCHMARK_OUTPUT_DIR", "benchmark_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
np.random.seed(RUN_SEED)

T = float(os.environ.get("BENCHMARK_T", "30.0"))
# Fixed-terminal-time benchmark: do not stop at a target tumor population.
target_population = None
record_every = 100
max_events = int(os.environ.get("BENCHMARK_MAX_EVENTS", "250000"))
# Emergency-only population ceiling; the scientific stopping rule is time T.
max_cell = int(os.environ.get("BENCHMARK_MAX_TUMOR_CELLS", "1000000000"))
initial_tumor_cells = 10
prefix = os.path.join(OUTPUT_DIR, f"edge_influx_linear_mass_abm_{RUN_ID}")

# Immune flare schedule: more frequent, linearly spaced update times.
# By default use 7 non-terminal flare times between delay_time and T.
# For T=20 this gives: 6, 8, 10, 12, 14, 16, 18.
# BENCHMARK_FLARE_TIMES still overrides the generated schedule exactly.
delay_time = 6.0
num_flare_points = int(os.environ.get("BENCHMARK_NUM_FLARES", "7"))
_flare_times_env = os.environ.get("BENCHMARK_FLARE_TIMES", "6,8,10,12,14,16,18,20,22,24,26,28").strip()
if _flare_times_env:
    benchmark_flare_times = [float(x) for x in _flare_times_env.split(",") if x.strip()]
else:
    if num_flare_points < 1:
        benchmark_flare_times = []
    elif T <= delay_time:
        benchmark_flare_times = []
    else:
        # Divide [delay_time, T] into num_flare_points+1 equal intervals and
        # use the left endpoints, excluding the terminal time T.
        benchmark_flare_times = list(
            np.linspace(delay_time, T, num_flare_points + 1, endpoint=True)[:-1]
        )

resampling_time = [0.0, delay_time / 2.0] + list(benchmark_flare_times)
if not resampling_time or resampling_time[-1] != T:
    resampling_time.append(T)
num_points = len(resampling_time)
use_discrete_immune_flares = True
print("Immune flare times:", [round(x, 6) for x in benchmark_flare_times])
# Deterministic flare mass is generated from the current tumor mass:
#   n_new = round(immune_flare_mass_scale * lambda_in_const * M_tumor * interval_length)
# This keeps influx linear in tumor burden, influx rate, and elapsed response interval.
immune_flare_mass_scale = float(os.environ.get("BENCHMARK_INFLUX_SCALE", "1.0"))

# Spatial lattice
L = int(os.environ.get("BENCHMARK_L", "220"))
R_neigh = 2
R_move = 2
R_place = 2
R_edge = 1
theta_edge = 1.0
min_edge_sites = 5
prefer_occupied_boundary_for_abm = False
print_immune_flare_info = True
K = 20.0
eps = 1.0

# Tumor kinetics
k_death_1 = 0.1
k_div = 0.5
k_A = 0.5
p_A = 1 / 3
p_IE = 1e-3

# Immune kinetics and recognition
eta_move_tumor = 0.5
beta_m_tumor = 0.08
eta_move_immune = 0
beta_m_immune = -0.08
# For this linear-influx test, do not impose artificial macro-step deletion.
# Old immune cells remain and move; later, larger edge flares dilute their relative contribution.
immune_death = 0.0
immune_retention_fraction = float(os.environ.get("BENCHMARK_IMMUNE_RETENTION", "1.0"))
if not (0.0 <= immune_retention_fraction <= 1.0):
    raise ValueError("BENCHMARK_IMMUNE_RETENTION must be between 0 and 1")
lambda_in_const = 0.03
immune_kill_coeff = 0.1


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def disk_coords(i, j, radius, L_value=L, include_self=True):
    coords = []
    for di in range(-radius, radius + 1):
        for dj in range(-radius, radius + 1):
            if not include_self and di == 0 and dj == 0:
                continue
            if di * di + dj * dj <= radius * radius:
                ii, jj = i + di, j + dj
                if 0 <= ii < L_value and 0 <= jj < L_value:
                    coords.append((ii, jj))
    return coords


def local_average(grid, pos, radius):
    coords = disk_coords(pos[0], pos[1], radius, L_value=grid.shape[0], include_self=True)
    if not coords:
        return 0.0
    return float(sum(grid[ii, jj] for ii, jj in coords)) / float(len(coords))


def local_sum(grid, pos, radius):
    coords = disk_coords(pos[0], pos[1], radius, L_value=grid.shape[0], include_self=True)
    return float(sum(grid[ii, jj] for ii, jj in coords))


def weighted_choice(items, weights):
    weights = np.asarray(weights, dtype=float)
    total = weights.sum()
    if total <= 0:
        return items[np.random.randint(len(items))]
    probs = weights / total
    return items[int(np.random.choice(np.arange(len(items)), p=probs))]


def entropy_from_grid(grid):
    total = float(grid.sum())
    if total <= 0:
        return 0.0
    p = grid.ravel() / total
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def immune_flare_schedule(final_time):
    # Deterministic recruitment times are supplied by the Euler benchmark.
    schedule = []
    previous_time = delay_time / 2.0
    for flare_time in benchmark_flare_times:
        if flare_time > final_time:
            continue
        interval_length = flare_time - previous_time
        schedule.append((flare_time, interval_length))
        previous_time = flare_time
    return schedule


# -----------------------------------------------------------------------------
# Cell objects
# -----------------------------------------------------------------------------
class TumorCell:
    def __init__(self, pos, x1=0, x2=0, x3=0):
        self.pos = tuple(pos)
        self.x1 = int(x1)
        self.x2 = int(x2)
        self.x3 = int(x3)

    @property
    def x2_state(self):
        return int(self.x2)


class ImmuneCell:
    def __init__(self, pos, z=0):
        self.pos = tuple(pos)
        self.z = int(z)


# -----------------------------------------------------------------------------
# Main system
# -----------------------------------------------------------------------------
class SpatialTumorImmuneABM:
    def __init__(self, L_value=L, max_cell_value=max_cell):
        self.L = int(L_value)
        self.max_cell = int(max_cell_value)

        self.tumor_grid = np.zeros((self.L, self.L), dtype=float)
        self.immune_grid = np.zeros((self.L, self.L), dtype=float)
        self.immune_entry_grid = np.zeros((self.L, self.L), dtype=float)
        self.immune_grid_by_state = {}
        # Tumor killing uses a frozen, delayed immune field.  The live immune
        # population continues to move/die between flare/update times.
        self.killing_immune_grid_by_state = {}

        self.cells = []
        self.immune_cells = []
        self.time = 0.0
        self.last_edge_source = "none"

        center = (self.L // 2, self.L // 2)
        for _ in range(initial_tumor_cells):
            founder = TumorCell(center, x1=0, x2=0, x3=0)
            self.cells.append(founder)
            self.tumor_grid[center] += 1.0

        self.log = {
            "time": [0.0],
            "tumor_cells": [initial_tumor_cells],
            "immune_cells": [0],
            "escaped_cells": [0],
            "tumor_entropy": [0.0],
            "immune_entropy": [0.0],
            "immune_in_tumor_fraction": [0.0],
            "immune_edge_fraction": [0.0],
            "mean_tumor_immune_neighborhood": [0.0],
        }

    # -------------------------------------------------------------------------
    # Local FKG-like fields
    # -------------------------------------------------------------------------
    def local_N(self, pos):
        return local_average(self.tumor_grid, pos, R_neigh)

    def local_I(self, pos):
        return local_average(self.immune_grid, pos, R_neigh)

    def local_K(self, pos):
        return max(eps, K - self.local_I(pos))

    def local_g(self, pos):
        return max(0.0, 1.0 - self.local_N(pos) / self.local_K(pos))

    def local_matching_immune_mass(self, pos, z_state):
        grid = self.killing_immune_grid_by_state.get(int(z_state))
        if grid is None:
            return 0.0
        return local_sum(grid, pos, R_neigh)

    def tumor_death_rate(self, cell):
        matching_immune = self.local_matching_immune_mass(cell.pos, cell.x2_state)
        immune_death_rate = immune_kill_coeff * (1 - cell.x3) * matching_immune
        return k_death_1 + immune_death_rate

    def tumor_division_rate(self, cell):
        return k_div * self.local_g(cell.pos)

    def tumor_move_rate(self, cell):
        return eta_move_tumor

    # -------------------------------------------------------------------------
    # Tumor events
    # -------------------------------------------------------------------------
    def mutate_daughter(self, parent):
        daughter = deepcopy(parent)
        dx1 = np.random.poisson(lam=k_A)
        dx2 = np.random.negative_binomial(n=dx1, p=p_A) if dx1 > 0 else 0
        dx3 = np.random.binomial(n=1, p=p_IE) if parent.x3 == 0 else 0

        daughter.x1 += int(dx1)
        daughter.x2 += int(dx2)
        daughter.x3 = max(int(daughter.x3), int(dx3))
        return daughter

    def add_tumor_cell(self, cell, pos):
        cell.pos = tuple(pos)
        self.cells.append(cell)
        self.tumor_grid[cell.pos] += 1.0

    def remove_tumor_cell(self, cell):
        self.tumor_grid[cell.pos] -= 1.0
        self.cells.remove(cell)

    def birth_event(self, rates):
        parent = weighted_choice(self.cells, rates)
        parent_idx = self.cells.index(parent)
        self.tumor_grid[parent.pos] -= 1.0

        daughter_stay = self.mutate_daughter(parent)
        daughter_move = self.mutate_daughter(parent)

        daughter_stay.pos = parent.pos
        self.cells[parent_idx] = daughter_stay
        self.tumor_grid[daughter_stay.pos] += 1.0

        # Match the FKG placement kernel: uniform placement in B_R(parent),
        # including the parent site.
        place_options = disk_coords(parent.pos[0], parent.pos[1], R_place, self.L, include_self=True)
        new_pos = place_options[np.random.randint(len(place_options))]
        self.add_tumor_cell(daughter_move, new_pos)

    def tumor_death_event(self, rates):
        dead_cell = weighted_choice(self.cells, rates)
        self.remove_tumor_cell(dead_cell)

    def tumor_move_event(self, rates):
        cell = weighted_choice(self.cells, rates)
        # Match FKG neighbor_coords, where B_R(r) includes r itself.
        options = disk_coords(cell.pos[0], cell.pos[1], R_move, self.L, include_self=True)
        if not options:
            return
        weights = []
        for pos in options:
            n_val = self.local_N(pos)
            g_val = self.local_g(pos)
            weights.append(np.exp(-beta_m_tumor * n_val) * max(g_val, 0.0))
        new_pos = weighted_choice(options, weights)
        self.tumor_grid[cell.pos] -= 1.0
        cell.pos = tuple(new_pos)
        self.tumor_grid[cell.pos] += 1.0

    # -------------------------------------------------------------------------
    # Immune events
    # -------------------------------------------------------------------------
    def add_immune_cell(self, cell, pos):
        cell.pos = tuple(pos)
        self.immune_cells.append(cell)
        self.immune_grid[cell.pos] += 1.0
        if cell.z not in self.immune_grid_by_state:
            self.immune_grid_by_state[cell.z] = np.zeros((self.L, self.L), dtype=float)
        self.immune_grid_by_state[cell.z][cell.pos] += 1.0

    def remove_immune_cell(self, cell):
        self.immune_grid[cell.pos] -= 1.0
        self.immune_grid_by_state[cell.z][cell.pos] -= 1.0
        self.immune_cells.remove(cell)

    def immune_move_event(self):
        if not self.immune_cells:
            return
        cell = self.immune_cells[np.random.randint(len(self.immune_cells))]
        # Match FKG neighbor_coords, where B_R(r) includes r itself.
        options = disk_coords(cell.pos[0], cell.pos[1], R_move, self.L, include_self=True)
        if not options:
            return
        weights = [np.exp(-beta_m_immune * self.local_N(pos)) for pos in options]
        new_pos = weighted_choice(options, weights)
        self.immune_grid[cell.pos] -= 1.0
        self.immune_grid_by_state[cell.z][cell.pos] -= 1.0
        cell.pos = tuple(new_pos)
        self.immune_grid[cell.pos] += 1.0
        self.immune_grid_by_state[cell.z][cell.pos] += 1.0

    def immune_death_event(self):
        if not self.immune_cells:
            return
        cell = self.immune_cells[np.random.randint(len(self.immune_cells))]
        self.remove_immune_cell(cell)

    def tumor_x2_state_distribution(self):
        counts = {}
        for cell in self.cells:
            z = cell.x2_state
            counts[z] = counts.get(z, 0) + 1
        if not counts:
            return np.array([0], dtype=int), np.array([1.0], dtype=float)
        states = np.array(sorted(counts.keys()), dtype=int)
        weights = np.array([counts[s] for s in states], dtype=float)
        return states, weights / weights.sum()

    def occupied_boundary_sites(self):
        fallback = []
        occupied = set(zip(*np.where(self.tumor_grid > 0)))
        for pos in occupied:
            nbrs = disk_coords(pos[0], pos[1], R_edge, self.L, include_self=True)
            if any(nbr not in occupied for nbr in nbrs):
                fallback.append(pos)
        return fallback

    def edge_sites(self):
        """Return the exact FKG density-level-set tumor edge.

        Match tumor_edge_mask(N_t) in the edge-based FKG script:
        1) N_t(r) is the radius-R_neigh local average of tumor mass;
        2) a site belongs to the tumor core when N_t(r) >= theta_edge;
        3) a core site is an edge site when at least one radius-R_edge
           Euclidean-disk neighbor has N_t < theta_edge.

        No occupied-support fallback is used in this diagnostic version.
        """
        edge = []
        for i in range(self.L):
            for j in range(self.L):
                pos = (i, j)
                if self.local_N(pos) < theta_edge:
                    continue
                nbrs = disk_coords(i, j, R_edge, self.L, include_self=True)
                if any(self.local_N(nbr) < theta_edge for nbr in nbrs):
                    edge.append(pos)

        self.last_edge_source = "FKG_density_level_set_edge" if edge else "none"
        return edge

    def edge_influx_sites_and_weights(self):
        """Return immune entry sites on the exact FKG tumor edge.

        The edge-based FKG implementation samples entry positions uniformly
        from its edge mask, so all detected edge sites receive equal weight.
        """
        sites = self.edge_sites()
        if not sites:
            return [], []
        return sites, [1.0] * len(sites)

    def freeze_immune_field_for_tumor_killing(self):
        """Freeze the pre-influx live immune field for tumor killing.

        At each flare/update time, the immune population that has already
        migrated through the preceding delay interval becomes the killing
        field for the next interval.  The newly recruited flare is added only
        after this snapshot, so it can move during the next interval but does
        not kill immediately.
        """
        self.killing_immune_grid_by_state = {
            int(z): grid.copy() for z, grid in self.immune_grid_by_state.items()
        }

    def discrete_immune_turnover_event(self):
        """Retain a random fraction of the current live immune cells.

        This is a direct-ABM test analogue of the FKG fixed-budget immune
        refresh, where only half of the previous immune particle states are
        carried into the next immune update.  Because ABM immune cells have
        equal unit mass, retained cells are sampled uniformly without
        replacement.  This operation changes state/spatial persistence; it is
        intentionally separated from the frozen delayed killing field.
        """
        n_before = len(self.immune_cells)
        if n_before == 0 or immune_retention_fraction >= 1.0:
            return n_before, n_before

        n_keep = int(round(immune_retention_fraction * n_before))
        n_keep = max(0, min(n_before, n_keep))

        if n_keep == n_before:
            return n_before, n_before

        if n_keep > 0:
            keep_idx = set(np.random.choice(n_before, size=n_keep, replace=False).tolist())
        else:
            keep_idx = set()

        old_cells = list(self.immune_cells)
        for idx, cell in enumerate(old_cells):
            if idx not in keep_idx:
                self.remove_immune_cell(cell)

        return n_before, len(self.immune_cells)

    def immune_influx_event(self):
        entry_sites, entry_weights = self.edge_influx_sites_and_weights()
        if not entry_sites:
            return

        states, weights = self.tumor_x2_state_distribution()
        if len(states) > 1 or states[0] != 0:
            weights = weights.copy()
            weights[states == 0] = 0.0
            if weights.sum() > 0:
                weights = weights / weights.sum()

        z = int(np.random.choice(states, p=weights))
        pos = weighted_choice(entry_sites, entry_weights)
        self.immune_entry_grid[pos] += 1.0
        self.add_immune_cell(ImmuneCell(pos=pos, z=z), pos)

    def immune_flare_event(self, interval_length):
        entry_sites, entry_weights = self.edge_influx_sites_and_weights()
        if not entry_sites:
            return

        tumor_mass_now = float(self.tumor_grid.sum())
        expected_mass = (
            immune_flare_mass_scale
            * lambda_in_const
            * tumor_mass_now
            * float(interval_length)
        )
        n_new = int(round(max(expected_mass, 0.0)))
        if n_new <= 0:
            return
        if print_immune_flare_info:
            print(
                f"Immune flare at t={self.time:.3f}: "
                f"{n_new} cells from M_tumor={tumor_mass_now:.3f}, "
                f"dt={interval_length:.3f}, lambda_in={lambda_in_const:.6g}, "
                f"scale={immune_flare_mass_scale:.6g}, "
                f"{len(entry_sites)} edge sites, source={self.last_edge_source}"
            )

        states, weights = self.tumor_x2_state_distribution()
        if len(states) > 1 or states[0] != 0:
            weights = weights.copy()
            weights[states == 0] = 0.0
            if weights.sum() > 0:
                weights = weights / weights.sum()

        z_picks = np.random.choice(states, size=n_new, replace=True, p=weights)
        for k in range(n_new):
            pos = weighted_choice(entry_sites, entry_weights)
            self.immune_entry_grid[pos] += 1.0
            self.add_immune_cell(ImmuneCell(pos=pos, z=int(z_picks[k])), pos)

    # -------------------------------------------------------------------------
    # Summaries and simulation
    # -------------------------------------------------------------------------
    def spatial_summaries(self):
        tumor_region = np.zeros((self.L, self.L), dtype=bool)
        edge_region = np.zeros((self.L, self.L), dtype=bool)

        for i in range(self.L):
            for j in range(self.L):
                pos = (i, j)
                tumor_region[i, j] = self.local_N(pos) >= theta_edge

        for pos in self.edge_sites():
            edge_region[pos] = True

        immune_total = float(self.immune_grid.sum())
        tumor_total = float(self.tumor_grid.sum())
        if immune_total > 0:
            immune_in_tumor = float(self.immune_grid[tumor_region].sum()) / immune_total
            immune_edge = float(self.immune_grid[edge_region].sum()) / immune_total
        else:
            immune_in_tumor = 0.0
            immune_edge = 0.0

        neighborhood_values = []
        weights = []
        occupied = list(zip(*np.where(self.tumor_grid > 0)))
        for pos in occupied:
            n_val = self.local_N(pos)
            i_val = self.local_I(pos)
            if n_val + i_val > 0:
                neighborhood_values.append(i_val / (n_val + i_val))
                weights.append(self.tumor_grid[pos])
        if weights:
            mean_neighborhood = float(np.average(neighborhood_values, weights=weights))
        else:
            mean_neighborhood = 0.0

        return {
            "tumor_entropy": entropy_from_grid(self.tumor_grid),
            "immune_entropy": entropy_from_grid(self.immune_grid),
            "immune_in_tumor_fraction": immune_in_tumor,
            "immune_edge_fraction": immune_edge,
            "mean_tumor_immune_neighborhood": mean_neighborhood,
            "tumor_total": tumor_total,
            "immune_total": immune_total,
        }

    def record_state(self):
        escaped = sum(1 for cell in self.cells if cell.x3 == 1)
        summaries = self.spatial_summaries()
        self.log["time"].append(self.time)
        self.log["tumor_cells"].append(len(self.cells))
        self.log["immune_cells"].append(len(self.immune_cells))
        self.log["escaped_cells"].append(escaped)
        for key in [
            "tumor_entropy",
            "immune_entropy",
            "immune_in_tumor_fraction",
            "immune_edge_fraction",
            "mean_tumor_immune_neighborhood",
        ]:
            self.log[key].append(summaries[key])

    def immune_escaped_tumor_grid(self):
        grid = np.zeros((self.L, self.L), dtype=float)
        for cell in self.cells:
            if cell.x3 == 1:
                grid[cell.pos] += 1.0
        return grid

    def evolute(self, final_time=T, max_events_value=max_events):
        flare_schedule = immune_flare_schedule(final_time) if use_discrete_immune_flares else []
        flare_idx = 0
        step = 0
        while step < max_events_value:
            if self.time >= final_time or len(self.cells) < 1 or len(self.cells) >= self.max_cell:
                break

            next_flare = flare_schedule[flare_idx] if flare_idx < len(flare_schedule) else None
            next_flare_time = next_flare[0] if next_flare is not None else None

            birth_rates = np.array([self.tumor_division_rate(cell) for cell in self.cells], dtype=float)
            tumor_death_rates = np.array([self.tumor_death_rate(cell) for cell in self.cells], dtype=float)
            tumor_move_rates = np.array([self.tumor_move_rate(cell) for cell in self.cells], dtype=float)

            total_birth = float(birth_rates.sum())
            total_tdeath = float(tumor_death_rates.sum())
            total_tmove = float(tumor_move_rates.sum())
            total_imove = eta_move_immune * len(self.immune_cells)
            total_ideath = immune_death * len(self.immune_cells)
            if use_discrete_immune_flares:
                total_iinflux = 0.0
            else:
                total_iinflux = lambda_in_const * max(1, len(self.edge_influx_sites_and_weights()[0]))

            a0 = total_birth + total_tdeath + total_tmove + total_imove + total_ideath + total_iinflux
            if a0 <= 0:
                if next_flare_time is not None and next_flare_time <= final_time:
                    self.time = next_flare_time
                    self.freeze_immune_field_for_tumor_killing()
                    n_before, n_after = self.discrete_immune_turnover_event()
                    if print_immune_flare_info:
                        print(f"Immune turnover at t={self.time:.3f}: retained {n_after}/{n_before}")
                    self.immune_flare_event(next_flare[1])
                    flare_idx += 1
                    self.record_state()
                    continue
                break

            dt = -np.log(np.random.random()) / a0
            if next_flare_time is not None and self.time < next_flare_time <= self.time + dt:
                self.time = next_flare_time
                self.freeze_immune_field_for_tumor_killing()
                n_before, n_after = self.discrete_immune_turnover_event()
                if print_immune_flare_info:
                    print(f"Immune turnover at t={self.time:.3f}: retained {n_after}/{n_before}")
                self.immune_flare_event(next_flare[1])
                flare_idx += 1
                self.record_state()
                continue
            if self.time + dt > final_time:
                self.time = final_time
                break
            self.time += dt

            u = np.random.random() * a0
            c1 = total_birth
            c2 = c1 + total_tdeath
            c3 = c2 + total_tmove
            c4 = c3 + total_imove
            c5 = c4 + total_ideath

            if u < c1:
                self.birth_event(birth_rates)
            elif u < c2:
                self.tumor_death_event(tumor_death_rates)
            elif u < c3:
                self.tumor_move_event(tumor_move_rates)
            elif u < c4:
                self.immune_move_event()
            elif u < c5:
                self.immune_death_event()
            else:
                self.immune_influx_event()

            step += 1
            if step % record_every == 0:
                self.record_state()

        if self.log["time"][-1] != self.time:
            self.record_state()

    def save_outputs(self, cpu_seconds=None):
        def write_count_distribution(filename, value_label, values):
            counts = {}
            for value in values:
                counts[value] = counts.get(value, 0) + 1
            with open(filename, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([value_label, "mass"])
                for value in sorted(counts):
                    writer.writerow([value, counts[value]])

        with open(f"{prefix}_growth_and_spatial_summaries.csv", "w", newline="") as f:
            writer = csv.writer(f)
            keys = list(self.log.keys())
            writer.writerow(keys)
            for i in range(len(self.log["time"])):
                writer.writerow([self.log[key][i] for key in keys])

        # Dedicated immune population recording for direct FKG comparison.
        with open(f"{prefix}_immune_count_record.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time", "immune_mass"])
            for t_value, immune_value in zip(self.log["time"], self.log["immune_cells"]):
                writer.writerow([t_value, immune_value])

        np.savetxt(f"{prefix}_tumor_grid.csv", self.tumor_grid, delimiter=",")
        np.savetxt(f"{prefix}_immune_grid.csv", self.immune_grid, delimiter=",")
        np.savetxt(f"{prefix}_immune_entry_grid.csv", self.immune_entry_grid, delimiter=",")
        escaped_grid = self.immune_escaped_tumor_grid()
        np.savetxt(f"{prefix}_immune_escaped_tumor_grid.csv", escaped_grid, delimiter=",")

        spatial_metadata = np.array(
            [
                ["method", "abm_fkg_matched_edge_influx"],
                ["T", str(T)],
                ["L", str(self.L)],
                ["R_neigh", str(R_neigh)],
                ["R_move", str(R_move)],
                ["R_place", str(R_place)],
                ["K", str(K)],
                ["stopping_rule", "fixed_terminal_time"],
                ["num_points", str(num_points)],
                ["immune_flare_mass_scale", str(immune_flare_mass_scale)],
                ["immune_flare_mass_scale", str(immune_flare_mass_scale)],
                ["immune_retention_fraction", str(immune_retention_fraction)],
                ["continuous_immune_death", str(immune_death)],
                ["lambda_in_const", str(lambda_in_const)],
                ["p_IE", str(p_IE)],
                ["k_div", str(k_div)],
                ["final_time", str(self.time)],
                ["final_tumor_cells", str(len(self.cells))],
                ["final_immune_cells", str(len(self.immune_cells))],
                ["final_escaped_tumor_cells", str(int(escaped_grid.sum()))],
                ["cpu_seconds", str(cpu_seconds) if cpu_seconds is not None else "NA"],
            ],
            dtype=str,
        )
        np.savez_compressed(
            f"{prefix}_spatial_mass.npz",
            tumor_mass=self.tumor_grid.astype(float),
            immune_mass=self.immune_grid.astype(float),
            immune_escaped_tumor_mass=escaped_grid.astype(float),
            immune_entry_mass=self.immune_entry_grid.astype(float),
            metadata=spatial_metadata,
        )
        print(f"Saved benchmark spatial mass arrays to {prefix}_spatial_mass.npz")

        cell_states = np.array(
            [[cell.x1, cell.x2, cell.x3, cell.pos[0], cell.pos[1]] for cell in self.cells],
            dtype=float,
        )
        immune_states = np.array(
            [[cell.z, cell.pos[0], cell.pos[1]] for cell in self.immune_cells],
            dtype=float,
        )
        np.savetxt(f"{prefix}_final_tumor_cells.csv", cell_states, delimiter=",", header="x1,x2,x3,rx,ry", comments="")
        np.savetxt(f"{prefix}_final_immune_cells.csv", immune_states, delimiter=",", header="z,rx,ry", comments="")

        tumor_coordinates = np.array(
            [[cell.pos[0], cell.pos[1], cell.x1, cell.x2, cell.x3] for cell in self.cells],
            dtype=float,
        )
        np.savetxt(
            f"{prefix}_tumor_spatial_coordinates.csv",
            tumor_coordinates,
            delimiter=",",
            header="rx,ry,x1,x2,x3",
            comments="",
        )

        write_count_distribution(f"{prefix}_marginal_x1.csv", "x1", [cell.x1 for cell in self.cells])
        write_count_distribution(f"{prefix}_marginal_x2.csv", "x2", [cell.x2 for cell in self.cells])
        write_count_distribution(f"{prefix}_marginal_x3.csv", "x3", [cell.x3 for cell in self.cells])
        write_count_distribution(f"{prefix}_immune_mass_distribution.csv", "z", [cell.z for cell in self.immune_cells])

        with open(f"{prefix}_final_immune_mass.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["final_time", "final_immune_mass"])
            writer.writerow([self.time, len(self.immune_cells)])

        x1_vals = np.array([cell.x1 for cell in self.cells], dtype=int)
        x2_vals = np.array([cell.x2 for cell in self.cells], dtype=int)
        x3_vals = np.array([cell.x3 for cell in self.cells], dtype=int)
        marginal_x1 = np.bincount(
            x1_vals, minlength=(int(x1_vals.max()) + 1 if x1_vals.size else 1)
        ).astype(float)
        marginal_x2 = np.bincount(
            x2_vals, minlength=(int(x2_vals.max()) + 1 if x2_vals.size else 1)
        ).astype(float)
        marginal_x3 = np.bincount(x3_vals, minlength=2).astype(float)

        immune_labels = np.array([cell.z for cell in self.immune_cells], dtype=int)
        immune_label_mass = np.bincount(
            immune_labels,
            minlength=(int(immune_labels.max()) + 1 if immune_labels.size else 1)
        ).astype(float)

        np.savez_compressed(
            f"{prefix}_benchmark_bundle.npz",
            time=np.asarray(self.log["time"], dtype=float),
            tumor_count=np.asarray(self.log["tumor_cells"], dtype=float),
            immune_count=np.asarray(self.log["immune_cells"], dtype=float),
            marginal_x1=marginal_x1,
            marginal_x2=marginal_x2,
            marginal_x3=marginal_x3,
            immune_label_mass=immune_label_mass,
            tumor_grid=self.tumor_grid.astype(float),
            immune_grid=self.immune_grid.astype(float),
            immune_entry_grid=self.immune_entry_grid.astype(float),
        )


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def plot_results(system):
    font = {"family": "Arial", "size": 17}

    # Match the FKG plotting convention exactly:
    # construct integer-support marginals, keep only positive-mass bins,
    # and plot those masses against consecutive bar indices.
    x1_vals = np.array([cell.x1 for cell in system.cells], dtype=int)
    x2_vals = np.array([cell.x2 for cell in system.cells], dtype=int)

    if len(x1_vals) > 0:
        marginal_x1 = np.bincount(x1_vals, minlength=int(x1_vals.max()) + 1).astype(float)
        m1 = [marginal_x1[i] for i in range(len(marginal_x1)) if marginal_x1[i] > 0]
    else:
        m1 = []

    if len(x2_vals) > 0:
        marginal_x2 = np.bincount(x2_vals, minlength=int(x2_vals.max()) + 1).astype(float)
        m2 = [marginal_x2[i] for i in range(len(marginal_x2)) if marginal_x2[i] > 0]
    else:
        m2 = []

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))

    ax[0].tick_params(labelsize=15)
    ax[0].tick_params(pad=0.03)
    ax[0].plot(system.log["time"], system.log["tumor_cells"], linewidth=5, color="indianred")
    ax[0].set_xlabel("Time", fontdict=font, fontsize=20)
    ax[0].set_ylabel("Total number of cells", fontdict=font, fontsize=20)
    ax[0].set_title("Tumor cell population over time", fontdict=font, fontsize=22)

    ax[1].tick_params(labelsize=15)
    ax[1].tick_params(pad=0.03)
    ax[1].bar(np.arange(len(m1)), m1, color="steelblue")
    ax[1].set_xlabel(r"Number of antigenic mutations ($x_{1}$)", fontdict=font, fontsize=20)
    ax[1].set_title(r"Spatial model", fontdict=font, fontsize=22)
    ax[1].set_ylabel("Number of cells", fontdict=font, fontsize=20)

    ax[2].tick_params(labelsize=15)
    ax[2].tick_params(pad=0.03)
    ax[2].bar(np.arange(len(m2)), m2, color="sandybrown")
    ax[2].set_xlabel(r"Antigenicity of cells ($x_{2}$)", fontdict=font, fontsize=20)
    ax[2].set_title(r"Spatial model", fontdict=font, fontsize=22)
    ax[2].set_ylabel("Number of cells", fontdict=font, fontsize=20)

    plt.tight_layout()
    fig.savefig(f"{prefix}_tumor_marginals.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Explicit immune recording, analogous to the FKG immune-output figure.
    fig_immune, ax_immune = plt.subplots(1, 2, figsize=(16, 5))

    ax_immune[0].tick_params(labelsize=15)
    ax_immune[0].plot(
        system.log["time"],
        system.log["immune_cells"],
        linewidth=5,
        color="seagreen"
    )
    ax_immune[0].set_xlabel("Time", fontdict=font, fontsize=20)
    ax_immune[0].set_ylabel("Total number of immune cells", fontdict=font, fontsize=20)
    ax_immune[0].set_title("Immune cell population over time", fontdict=font, fontsize=22)

    immune_counts = {}
    for cell in system.immune_cells:
        immune_counts[int(cell.z)] = immune_counts.get(int(cell.z), 0) + 1

    if immune_counts:
        z_support = np.array(sorted(immune_counts.keys()), dtype=int)
        z_mass = np.array([immune_counts[z] for z in z_support], dtype=float)
        ax_immune[1].bar(z_support, z_mass, color="mediumseagreen")
        ax_immune[1].set_xlabel(r"Immune label ($z=x_{2}$)", fontdict=font, fontsize=20)
        ax_immune[1].set_ylabel("Immune mass", fontdict=font, fontsize=20)
        ax_immune[1].set_title("Final immune label distribution", fontdict=font, fontsize=22)
        ax_immune[1].tick_params(labelsize=15)
    else:
        ax_immune[1].text(
            0.5, 0.5, "No immune cells",
            ha="center", va="center",
            transform=ax_immune[1].transAxes,
            fontsize=18
        )
        ax_immune[1].set_axis_off()

    plt.tight_layout()
    fig_immune.savefig(f"{prefix}_immune_summary.png", dpi=200, bbox_inches="tight")
    plt.close()

    fig2, ax2 = plt.subplots(1, 4, figsize=(22, 5))
    im0 = ax2[0].imshow(system.tumor_grid, origin="lower")
    ax2[0].set_title("Final tumor occupancy")
    plt.colorbar(im0, ax=ax2[0], fraction=0.046, pad=0.04)

    im1 = ax2[1].imshow(system.immune_grid, origin="lower")
    ax2[1].set_title("Final immune occupancy")
    if np.any(system.tumor_grid > 0):
        ax2[1].contour((system.tumor_grid > 0).astype(float), levels=[0.5],
                       colors="black", linestyles="--", linewidths=1.0)
    plt.colorbar(im1, ax=ax2[1], fraction=0.046, pad=0.04)

    im_entry = ax2[2].imshow(system.immune_entry_grid, origin="lower")
    ax2[2].set_title("Immune entry sites")
    if np.any(system.tumor_grid > 0):
        ax2[2].contour((system.tumor_grid > 0).astype(float), levels=[0.5],
                       colors="black", linestyles="--", linewidths=1.0)
    plt.colorbar(im_entry, ax=ax2[2], fraction=0.046, pad=0.04)

    x3_grid = system.immune_escaped_tumor_grid()
    im2 = ax2[3].imshow(x3_grid, origin="lower")
    ax2[3].set_title(r"Final immune-escaped tumor cells ($x_3=1$)")
    if np.any(system.tumor_grid > 0):
        ax2[3].contour((system.tumor_grid > 0).astype(float), levels=[0.5],
                       colors="black", linestyles="--", linewidths=1.0)
    plt.colorbar(im2, ax=ax2[3], fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig2.savefig(f"{prefix}_heatmaps.png", dpi=200, bbox_inches="tight")
    plt.close()

    fig3, ax3 = plt.subplots(1, 3, figsize=(18, 5))
    ax3[0].plot(system.log["time"], system.log["tumor_entropy"], linewidth=3)
    ax3[0].set_title("Tumor spatial entropy")
    ax3[0].set_xlabel("Time")

    ax3[1].plot(system.log["time"], system.log["immune_in_tumor_fraction"], linewidth=3)
    ax3[1].set_title("Immune fraction inside tumor region")
    ax3[1].set_xlabel("Time")

    ax3[2].plot(system.log["time"], system.log["mean_tumor_immune_neighborhood"], linewidth=3)
    ax3[2].set_title("Tumor-weighted immune neighborhood")
    ax3[2].set_xlabel("Time")

    plt.tight_layout()
    fig3.savefig(f"{prefix}_spatial_summaries.png", dpi=200, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    cpu_start = time.perf_counter()
    system = SpatialTumorImmuneABM(L_value=L, max_cell_value=max_cell)
    system.evolute(final_time=T, max_events_value=max_events)
    if system.time < T - 1e-10:
        raise RuntimeError(
            f"ABM terminated before fixed terminal time T={T}: final_time={system.time}. "
            f"Increase BENCHMARK_MAX_EVENTS if the event cap was reached."
        )
    cpu_seconds = time.perf_counter() - cpu_start
    system.save_outputs(cpu_seconds=cpu_seconds)
    with open(f"{prefix}_runtime_summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "run_id", "seed", "L", "stopping_rule",
            "final_tumor_cells", "final_immune_cells", "final_time",
            "terminal_time", "cpu_seconds", "immune_flare_mass_scale", "n_flares"
        ])
        writer.writerow([
            "edge_influx_abm", RUN_ID, RUN_SEED, L, "fixed_terminal_time",
            len(system.cells), len(system.immune_cells), system.time,
            T, cpu_seconds, immune_flare_mass_scale,
            len(immune_flare_schedule(T))
        ])
    plot_results(system)

    print(f"Final time: {system.time:.3f}")
    print(f"Final tumor cells: {len(system.cells)}")
    print(f"Final immune cells: {len(system.immune_cells)}")
    print(f"Final immune-escaped tumor cells: {sum(cell.x3 == 1 for cell in system.cells)}")
    print(f"CPU seconds: {cpu_seconds:.3f}")
