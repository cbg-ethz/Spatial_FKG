# Spatial-FKG

Spatial-FKG is a spatial tumor--immune modeling framework for studying population-level tumor evolution with intracellular cell states, local spatial interactions, immune recruitment, movement, and immune escape.

This repository provides directly runnable reference implementations of the main Spatial-FKG model variants together with a matched agent-based benchmark.

## Overview

Tumor cells are represented by intracellular states and spatial coordinates.

In 2D:

```text
[x1, x2, x3, rx, ry]
```

In 3D:

```text
[x1, x2, x3, rx, ry, rz]
```

where:

- `x1` is the number of accumulated antigenic mutations;
- `x2` is the antigenicity state;
- `x3` is a binary immune-escape state;
- `rx`, `ry`, and `rz` are spatial lattice coordinates.

Immune trajectories carry a recognition label `z` together with their spatial coordinates. In the baseline implementation, immune recognition is antigen-specific through matching between `z` and the tumor antigenicity state `x2`.

Spatial-FKG approximates mean spatial population dynamics using weighted auxiliary trajectories rather than simulating the complete lineage history of every individual cell.

## Scripts

### 2D edge-dependent recruitment

`Spatial_FKG_2D_edge_dependent.py`

Immune trajectories are recruited at detected tumor-edge sites.

### 2D free penetration

`Spatial_FKG_2D_free_penetration.py`

Immune trajectories can enter throughout the tumor-associated region, with entry probabilities weighted by local tumor density.

### 2D immunotherapy example

`Spatial_FKG_2D_immunotherapy.py`

Illustrates how a treatment-dependent change in immune-mediated tumor killing can be incorporated into the model.

### 3D edge-dependent recruitment

`Spatial_FKG_3D_edge_dependent.py`

Three-dimensional version of the edge-dependent recruitment model.

### 3D free penetration

`Spatial_FKG_3D_free_penetration.py`

Three-dimensional version of the free-penetration model.

### Agent-based benchmark

`benchmark_spatial_fkg_vs_abm_2D_edge.py`

A matched 2D stochastic agent-based model used to compare Spatial-FKG with direct cell-based simulation.

## Requirements

The reference implementations are provided as standalone Python scripts rather than as a Python package.

Python 3.10 or later is recommended.

Required dependencies:

- NumPy
- Matplotlib
- FilterPy

The dependencies can be installed with:

```bash
python -m pip install -r requirements.txt
```

## Running the scripts

Each Spatial-FKG script contains its own default parameter settings and can be run directly.

For example:

```bash
python Spatial_FKG_2D_edge_dependent.py
```

Other examples:

```bash
python Spatial_FKG_2D_free_penetration.py
python Spatial_FKG_3D_edge_dependent.py
python Spatial_FKG_3D_free_penetration.py
python Spatial_FKG_2D_immunotherapy.py
```

The scripts are stochastic, so repeated runs may produce slightly different realizations.

## Parameters

Parameters are kept directly in each script so that every example is self-contained and can be run without a separate configuration file.

### Simulation and numerical parameters

| Parameter | Description |
|---|---|
| `T` | Terminal simulation time. In the immunotherapy example, `T` is the treatment start time. |
| `T_final` | Final simulation time in the immunotherapy example. |
| `delay_time` | Initial delay before immune-response updates. |
| `num_points` | Number of macro time points controlling the immune-response schedule. |
| `dt_tumor` | Tumor micro-interval used within each immune macro-interval. |
| `num` | Number of weighted tumor trajectories. |
| `num_immune` | Number of weighted immune trajectories. |
| `h` | Time-grid step used for immune weight integration. |
| `x1_size` | Allocated size of the mutation-state dimension. |
| `x2_size` | Allocated size of the antigenicity-state dimension. |
| `x3_size` | Allocated size of the immune-escape dimension. |

Increasing `num` or `num_immune` generally reduces Monte Carlo variability at increased computational cost. Decreasing `h` or `dt_tumor` increases numerical resolution but also increases runtime.

### Spatial parameters

| Parameter | Description |
|---|---|
| `L` | Linear lattice size. The domain is `L x L` in 2D and `L x L x L` in 3D. |
| `R_neigh` | Radius used to define local tumor and immune neighborhoods. |
| `R_move` | Radius of the movement neighborhood. |
| `R_place` | Radius used for daughter placement after division. |
| `R_edge` | Radius used in tumor-edge detection. |
| `theta_edge` | Local tumor-density threshold used to define the tumor region for edge detection. |
| `K` | Baseline local carrying-capacity parameter. |
| `eps` | Lower bound used for the effective local carrying capacity. |
| `free_penetration_min_tumor_density` | Minimum tumor density used to define free-penetration entry sites. |

### Tumor kinetic parameters

| Parameter | Description |
|---|---|
| `k_div` | Baseline tumor division rate. |
| `k_death_1` | Baseline tumor-death contribution. |
| `k_A` | Mutation-acquisition parameter. |
| `p_A` | Negative-binomial parameter controlling antigenicity changes conditional on mutation acquisition. |
| `p_IE` | Probability of acquiring immune escape during a division event. |
| `eta_move_tumor` | Tumor movement-event rate. |
| `beta_m_tumor` | Strength of tumor-density-dependent movement bias. |

### Immune parameters

| Parameter | Description |
|---|---|
| `eta_move_immune` | Immune movement-event rate. |
| `beta_m_immune` | Strength and direction of tumor-density-dependent immune movement or entry bias. |
| `lambda_in_const` | Immune recruitment/influx parameter. |
| `immune_death` | Immune-death parameter defined in the current scripts. |
| `delay_steps` | Number of macro steps used for delayed immune influence on tumor dynamics. |

The default values differ across the example scripts because the scripts illustrate different spatial and biological regimes. For reproducible analyses, the exact script and parameter values used should be reported.

## Benchmark

The agent-based benchmark is designed for distributional comparison with Spatial-FKG.

A Spatial-FKG run approximates mean spatial population behavior using weighted trajectories, whereas an individual agent-based simulation is one stochastic realization.

The benchmark script contains its own parameter settings and output options.

## Citation

If you use Spatial-FKG, please cite the Spatial-FKG manuscript associated with this repository.

**Spatial-FKG:** citation to be added after the final manuscript or preprint information is available.

Spatial-FKG builds on the weighted-trajectory FKG-SSA methodology developed in our earlier work. Please also cite the corresponding FKG-SSA publication when using this implementation.
