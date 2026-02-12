# FVDiagnosticTests.jl

Learning quasistationarity diagnostics for accelerated molecular dynamics.

Sampling QSDs is a key part of accelerated MD algorithms. The goal is to explore alternatives to classical MCMC convergence diagnostics which tend to be overly conservative and reduce sampling efficiency, which in turn has a significant impact on parallel efficiency.

In this experiment, a Fleming-Viot system of interacting particles, which are killed at the boundary of a domain, and branched uniformly at random from surviving particles, is used to generate approximate samples from the QSD.

A recurrent neural network learns to predict, from a sequence of features of the Fleming-Viot process, whether the underlying non-linear Fokker--Planck equation has converged to its stationary state (which is none other than the QSD). 

The corresponding quasistationarity diagnostic is trained on low-dimensional systems where ground truth data is available, with the aim of deploying it on low-dimensional traces of Fleming-Viot trajectories from MD simulations, projected to a low-dimensional space through a collective variable $\xi$. Under suitable assumptions on this collective variable, this procedure defines another approach to the estimation of decorrelation times.

## Context

In molecular dynamics (MD), estimating dynamical quantities (reaction rates, transition paths) is hampered by **metastability**: the system remains trapped in local free-energy wells for timescales many orders of magnitude longer than the simulation timestep. The **parallel replica** method (ParRep, [V98](#refs)) exploits this separation of timescales by:

1. **Decorrelation**: waiting for the process to reach a local equilibrium (the QSD) inside a metastable state $\Omega$
2. **Dephasing**: preparing $N$ i.i.d. replicas sampled from the QSD inside $\Omega$ (for instance using a Fleming-Viot process)
3. **Parallel exit**: running all replicas in parallel until the first one exits $\Omega$, yielding an $N$-fold acceleration in the simulation of the longest portion of the trajectory

The algorithm's parallel efficiency depends critically on an accurate estimate of the **decorrelation time** $t_{\mathrm{corr}}(\Omega)$. In condensed matter systems, mathematically principled ways to estimate these decorrelation time are available (see e.g. [BLS25](#refs)).
In biophysical systems, there are currently no equivalents for these heuristics. The Gelman-Rubin diagnostic from MCMC has been proposed as a practical way to assess quasistationary convergence. This gives the so-called generalized parallel replica (GenParRep [BLS15](#refs)). Unfortunately, the Gelman Rubin diagnostic can be shown to be systematically conservative, in a way which restricts the practicality of GenParRep to highly metastable systems only.

It is therefore of practical importance to explore alternatives to the Gelman-Rubin diagnostic, since the ParRep method should apply in principle to a large range of systems. This package explores a data-driven strategy, by training a recurrent neural network to learn a diagnostic on a synthetic ensemble of Fleming-Viot systems, sampled from a distribution of one-dimensional dynamics representative of typical effective dynamics encountered in target molecular systems.

## File structure

### Core functionality

| File | Description |
|------|-------------|
| `FVDiagnosticTests.jl` | Module definition |
| `generate_data.jl` | Defines and samples the synthetic data ensemble, and generates ground-truth labels |
| `lstm.jl` | Defines the `RNNDiagnostic` model architecture and various convenience functions |
| `train_lstm.jl` | Defines training-related methods |
| `tournament.jl` | Jointly train and select model hyperparameters by a tournament-style elimination process |
| `main.jl` | Training run |

### Experiments
TODO

## Dependencies

```julia
Flux, ParameterSchedulers, MLUtils, JLD2, ProgressMeter
Random, Distributions, StatsBase, Statistics, LinearAlgebra
SparseArrays, Arpack
```

Postprocessing scripts additionally use: `Plots`, `LaTeXStrings`, `JSON`, `DelimitedFiles`, `ColorSchemes`, `DataInterpolations`.

## Usage

### 1. Train

```bash
julia main.jl
```

Searches over CNN/LSTM/MLP depth and width combinations, eliminating half the candidates each round based on validation loss. Outputs `best_hope.jld2` with the winning model state.

Architecture search grid (from `main.jl`):
- CNN depth: 3-5, width exponent: 3-4
- LSTM depth: 1-2, width exponent: 5-6
- MLP depth: 1-2, width exponent: 5-6

### 2. Online inference

```julia
using .FVDiagnosticTests

model = load_rnn_from_state(64, model_state)
online = RNNDiagnosticOnline(model)
testmode!(online)
diag_tol = 0.95

for frame in frames # MD Fleming-Viot frames (featurized as time-averaged histograms for now)
    logit = online(frame)
    if sigmoid(logit[1]) > diag_tol
        println("Converged")
        break
    end
end

reset_rnn_state!(online)  # reset state for next trajectory
```

## Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `input_dim` | 64 | Feature vector dimensionality (currently histogram bins) |
| `Ngrid` | 500 | Lattice discretization parameter for killed generator |
| `dt` | 1e-3 | Simulation time step |
| `stride` | 50 | Steps per lag time tau (so tau = stride * dt) |
| `Nreplicas` | 50 | Particles in FV ensemble |
| `βlims` | (1.0, 3.0) | Inverse temperature sampling range |
| `tol` | 0.05 | TV distance threshold defining t_corr |
| `npot` | 5 | Random potentials per training batch |
| `ntrace` | 5 | FV traces per potential |
| `ncut` | 1 | Numbe of random subsequences extracted per trace |

## References

<a id="refs"></a>
- V98: [A. F. Voter, *Parallel Replica method for dynamics of infrequent events*, Physical Review B, 1998](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.57.R13985)
- BLS15: [A. Binder, T. Lelièvre, G. Simpson, *A generalized parallel replica dynamics*, Journal of Computational Physics, 2015](https://www.sciencedirect.com/science/article/pii/S0021999115000030)
- BLS25: [N. Blassel, T. Lelièvre, G. Stoltz, *Quantitative spectral asymptotics for reversible diffusions in temperature-dependent domains*, arXiv preprint ,2025](https://arxiv.org/abs/2501.16082)
