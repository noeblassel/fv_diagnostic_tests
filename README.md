# FVDiagnosticTests.jl

Learning quasistationarity diagnostics for accelerated molecular dynamics.

Sampling QSDs is a key part of accelerated MD algorithms. The goal is to explore alternatives to classical MCMC convergence diagnostics which tend to be overly conservative and reduce sampling efficiency, which in turn has a significant impact on parallel efficiency.

In this experiment, a Fleming-Viot system of interacting particles, which are killed at the boundary of a domain, and branched uniformly at random from surviving particles, is used to generate approximate samples from the QSD.

A recurrent neural network learns to predict, from a sequence of features of the Fleming-Viot process, whether the underlying non-linear Fokker-Planck equation has converged to its stationary state (which is none other than the QSD).

The corresponding quasistationarity diagnostic is trained on low-dimensional systems where ground truth data is available, with the aim of deploying it on low-dimensional traces of Fleming-Viot trajectories from MD simulations, projected to a low-dimensional space through a collective variable $\xi$. Under suitable assumptions on $\xi$, this procedure gives a way to estimate valid decorrelation times.

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
| `FVDiagnosticTests.jl` | Module definition and exports |
| `generate_data.jl` | Synthetic data generation, ground-truth labels, and feature functions |
| `lstm.jl` | Model architecture: `CNNFeaturizer`, `RNNDiagnostic`, `RNNDiagnosticOnline`, hyperparameter structs |
| `train_lstm.jl` | `TrainingRun` struct and training/evaluation loops |
| `retrain.jl` | Retrain a model from a checkpoint |

### Tests

`tests/runtests.jl` — smoke tests for both featurizer paths, variable τ/N, online inference, and checkpoint round-trips.

### Experiments

See `experiments/`.

## Dependencies

```julia
Flux, ParameterSchedulers, MLUtils, JLD2, ProgressMeter
Random, Distributions, StatsBase, Statistics, LinearAlgebra
SparseArrays, Arpack
```

Postprocessing scripts additionally use: `Plots`, `LaTeXStrings`, `JSON`, `DelimitedFiles`, `ColorSchemes`, `DataInterpolations`.

## Architecture

The model is `RNNDiagnostic`, a sequence-to-sequence classifier composed of three stages:

```
frame sequence  →  featurizer  →  LSTM stack  →  MLP head  →  logit sequence
```

Each frame (a snapshot of the FV ensemble) is embedded independently by the featurizer before being fed into the recurrent stack.

### Featurizers

The featurizer is selected via the `featurizer` field of `RNNDiagnosticHyperParams`.

#### `CNNFeaturizer`

Encodes a fixed-size statistical summary of the particle distribution (histogram or ECDF) using a 1D CNN with max-pooling.

```julia
CNNFeaturizer(; input_dim, kernel_dims, nchannels, rng)
```

- **Input**: `(input_dim, 1, batch)` — one scalar feature vector per frame
- **Output**: `(cnn_output_dim, batch)` — flattened after the last conv+pool block
- **Feature functions**: `hist_feature`, `ecdf_feature`, `tecdf_feature` (all map the particle set to a fixed-length vector)

Controlled by `CNNFeaturizerHyperParams(depth, width_exponent)`:
- `depth` — number of Conv+MaxPool blocks
- `width_exponent` — first channel count is `2^width_exponent`, doubling each block

### Constructing a model

```julia
# CNN featurizer (explicit)
feat = CNNFeaturizer(; input_dim=64, kernel_dims=[5,5,5], nchannels=[16,32,64], rng)
model = RNNDiagnostic(feat; dims_rnn=[128], dims_mlp=[64,32], rng)

# Via hyperparameter struct
hp = RNNDiagnosticHyperParams(
    CNNFeaturizerHyperParams(depth=3, width_exponent=4),
    rnn_depth=2, rnn_width_exponent=6,
    mlp_depth=1, mlp_width_exponent=6)
model = RNNDiagnostic(hp; input_dim=64, rng)
```

## Usage

### 1. Train

```bash
julia --project=. hyperparameter_tuning/tune_neps.jl
```

Runs NePS IfBO (In-context Freeze-Thaw Bayesian Optimization) over CNN architecture and training hyperparameters. Outputs results to `hyperparameter_tuning/neps_results_cnn/`.

### 2. Resume from checkpoint

```bash
julia --project=. retrain.jl <checkpoint>.jld2
```

`load_rnn_from_state` detects the architecture automatically from the saved state (supports both new-format checkpoints with a `featurizer` field, and pre-refactor checkpoints with the legacy `cnn_encoder` field).

### 3. Online inference

```julia
include("FVDiagnosticTests.jl")
using .FVDiagnosticTests, Flux, JLD2

model_state = JLD2.load("best_hope_trained.jld2", "model_state")
model = load_rnn_from_state(input_dim, model_state)
online = RNNDiagnosticOnline(model)
testmode!(online)

diag_tol = 0.95

for frame in frames   # featurized FV frames, length = input_dim
    logit = online(frame)
    if Flux.σ(logit[1]) > diag_tol
        println("Converged")
        break
    end
end

reset_rnn_state!(online)  # reset hidden state for next trajectory
```

### 4. Run tests

```bash
julia --project=. tests/runtests.jl
```

### 5. Hyperparameter optimization

Hyperparameter search uses **NePS** (Neural Pipeline Search) with IfBO (In-context Freeze-Thaw Bayesian Optimization). It operates on a pre-generated **offline dataset** of FV trajectories, avoiding repeated simulation cost during the search.

#### Step 0 — Generate the offline dataset

```bash
julia --project=. hyperparameter_tuning/make_dataset.jl
```

Writes `hyperparameter_tuning/hp_dataset.jld2` containing `train` and `test` splits.

#### Run the search

```bash
julia --project=. hyperparameter_tuning/tune_neps.jl
```

Searches over 8 hyperparameters (learning rate, CNN/RNN/MLP depth and width, input dimension) with epoch as the fidelity parameter. Results are saved to `hyperparameter_tuning/neps_results_cnn/`.

Inspect results with:
```bash
python -c "import neps; neps.status('hyperparameter_tuning/neps_results_cnn')"
```

## Key parameters

### Data generation (`get_batch`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `input_dim` | 64 | Feature vector length (histogram bins for CNN) |
| `feature` | `hist_feature` | Feature function: `hist_feature`, `ecdf_feature`, or `tecdf_feature` |
| `stride_lims` | `(50, 50)` | Range `(min, max)` for the lag time stride τ; one value sampled uniformly per potential |
| `Nreplicas_lims` | `(50, 50)` | Range `(min, max)` for the FV ensemble size N; one value sampled uniformly per trace |
| `Ngrid` | 500 | Lattice discretization for the killed generator |
| `dt` | 1e-3 | Simulation time step |
| `βlims` | `(1.0, 1.0)` | Inverse temperature sampling range |
| `tol` | 0.05 | TV distance threshold defining $t_\mathrm{corr}$ |
| `npot` | 5 | Random potentials per batch |
| `ntrace` | 5 | FV traces per potential |
| `ncut` | 1 | Random subsequences extracted per trace |

Setting `stride_lims=(a,b)` with `a < b` trains the model to be robust to variable lag times. Similarly for `Nreplicas_lims`. Both default to fixed values `(50,50)` for backward compatibility.

### Training (`TrainingRun`)

| Field | Description |
|-------|-------------|
| `model` | An `RNNDiagnostic` instance |
| `feature` | Feature function passed to `get_batch` |
| `input_dim` | Must match the featurizer's expected input size |
| `stride_lims` | Forwarded to `get_batch`; default `(50,50)` |
| `Nreplicas_lims` | Forwarded to `get_batch`; default `(50,50)` |
| `βlims` | Inverse temperature range for training potentials |
| `pot_per_batch` | Potentials per gradient step |
| `trace_per_pot` | Traces per potential |
| `cut_per_trace` | Subsequences per trace |

## References

<a id="refs"></a>
- V98: [A. F. Voter, *Parallel Replica method for dynamics of infrequent events*, Physical Review B, 1998](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.57.R13985)
- BLS15: [A. Binder, T. Lelièvre, G. Simpson, *A generalized parallel replica dynamics*, Journal of Computational Physics, 2015](https://www.sciencedirect.com/science/article/pii/S0021999115000030)
- BLS25: [N. Blassel, T. Lelièvre, G. Stoltz, *Quantitative spectral asymptotics for reversible diffusions in temperature-dependent domains*, arXiv preprint, 2025](https://arxiv.org/abs/2501.16082)
