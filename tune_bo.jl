using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

include("FVDiagnosticTests.jl")

using .FVDiagnosticTests
using GaussianProcesses
using Flux, JLD2, Statistics, Random

# ============================================================
# Minimal Bayesian Optimisation loop
#
# Strategy:
#   - GP surrogate with fixed kernel hyperparameters (SE ARD,
#     skip MLE fitting due to PDMats ambiguity in GP.optimize!).
#   - Upper-Confidence-Bound acquisition, optimised by random
#     search over the box (cheap and robust for ≤ 9 dims).
#   - All discrete dims handled by continuous relaxation:
#     round(Int, x[i]) inside the objective.
# ============================================================

"""
    bo_search(objective, lb, ub; n_init, n_iter, β, rng)

Maximise `objective` over the box [lb, ub] using BO with UCB.
Returns a NamedTuple with:
  - `observed_optimizer` — x vector at the best observed value
  - `observed_optimum`   — corresponding objective value
  - `X`, `y`            — full history of evaluations
"""
function bo_search(objective, lb, ub;
        n_init     = 5,
        n_iter     = 45,
        β          = 2.0,
        n_cand     = 2000,
        rng        = Xoshiro(42))

    d = length(lb)

    X = Vector{Vector{Float64}}()
    y = Vector{Float64}()

    # ── Initial random exploration ───────────────────────────
    println("  [BO] Random initialisation ($n_init points)...")
    for i in 1:n_init
        x = lb .+ rand(rng, d) .* (ub .- lb)
        push!(X, x)
        val = objective(x)
        push!(y, val)
        println("  init $i/$n_init  loss=$(-val)")
        flush(stdout)
    end

    best_idx = argmax(y)
    best_x, best_y = X[best_idx], y[best_idx]

    # ── Sequential BO iterations ─────────────────────────────
    for iter in 1:n_iter
        # Fit GP (fixed kernel hyperparameters; skip optimize! due to
        # PDMats ldiv! ambiguity in GaussianProcesses v0.12)
        X_mat = hcat(X...)         # d × n
        y_vec = Float64.(y)

        kern = SEArd(zeros(d), 0.0)   # ARD squared-exponential
        gp   = GP(X_mat, y_vec, MeanConst(mean(y_vec)), kern, -2.0)

        # Random UCB search over the box
        cands = [lb .+ rand(rng, d) .* (ub .- lb) for _ in 1:n_cand]
        X_cand = hcat(cands...)
        mu, sigma2 = predict_y(gp, X_cand)
        ucb = mu .+ β .* sqrt.(max.(sigma2, 1e-10))
        x_next = cands[argmax(ucb)]

        # Evaluate objective
        y_next = objective(x_next)
        push!(X, x_next)
        push!(y, y_next)

        if y_next > best_y
            best_y = y_next
            best_x = x_next
        end

        println("  [BO] iter $iter/$n_iter  loss=$(-y_next)  best_loss=$(-best_y)")
        flush(stdout)
    end

    return (observed_optimizer=best_x, observed_optimum=best_y, X=X, y=y)
end

# ============================================================
# Objective factory
# ============================================================

"""
    make_objective(; base_seed, input_dim, βlims, ...)

Returns a closure `x -> -loss` that wraps one training epoch.
The first 5 dims of `x` are shared (lr, rnn/mlp depths & widths);
`featurizer_builder(x)` extracts the remaining dims.
"""
function make_objective(;
        base_seed,
        input_dim,
        βlims,
        train_batches,
        test_batches,
        pot_per_batch,
        trace_per_pot,
        cut_per_trace,
        feature,
        featurizer_builder)

    call_count = Ref(0)

    function objective(x)
        call_count[] += 1

        lr            = exp(x[1])
        rnn_depth     = round(Int, x[2])
        rnn_width_exp = round(Int, x[3])
        mlp_depth     = round(Int, x[4])
        mlp_width_exp = round(Int, x[5])
        feat_hp, _    = featurizer_builder(x)

        h = RNNDiagnosticHyperParams(feat_hp, rnn_depth, rnn_width_exp,
                                      mlp_depth, mlp_width_exp)

        run = build_candidate_run((lr, h);
            base_seed     = base_seed + call_count[],
            input_dim     = input_dim,
            βlims         = βlims,
            pot_per_batch = pot_per_batch,
            trace_per_pot = trace_per_pot,
            cut_per_trace = cut_per_trace,
            feature       = feature)

        run_epoch!(run, train_batches, false)   # false = skip per-trial checkpoint
        _, loss = test_accuracy!(run, test_batches)
        return -loss    # BO maximises; we minimise loss
    end

    return objective
end

# ============================================================
# Shared search settings
# ============================================================

const BASE_SEED     = 2022
const INPUT_DIM_CNN = 64
const INPUT_DIM_DS  = 50    # max particles for DeepSet
const BETA_LIMS     = (1.0, 3.0)
const TRAIN_BATCHES = 30
const TEST_BATCHES  = 15
const POT_PER_BATCH = 5
const TRACE_PER_POT = 5
const CUT_PER_TRACE = 2

# ============================================================
# CNN BO search (7-dimensional)
# ============================================================
# x[1]: log(lr)         [log(1e-4), log(1e-2)]
# x[2]: rnn_depth       [0.5, 2.5]  → 1 or 2
# x[3]: rnn_width_exp   [4.5, 6.5]  → 5 or 6
# x[4]: mlp_depth       [0.5, 2.5]  → 1 or 2
# x[5]: mlp_width_exp   [4.5, 6.5]  → 5 or 6
# x[6]: cnn_depth       [2.5, 5.5]  → 3, 4, or 5
# x[7]: cnn_width_exp   [2.5, 4.5]  → 3 or 4

cnn_builder(x) = (CNNFeaturizerHyperParams(round(Int, x[6]), round(Int, x[7])), 2)

obj_cnn = make_objective(;
    base_seed          = BASE_SEED,
    input_dim          = INPUT_DIM_CNN,
    βlims              = BETA_LIMS,
    train_batches      = TRAIN_BATCHES,
    test_batches       = TEST_BATCHES,
    pot_per_batch      = POT_PER_BATCH,
    trace_per_pot      = TRACE_PER_POT,
    cut_per_trace      = CUT_PER_TRACE,
    feature            = hist_feature,
    featurizer_builder = cnn_builder)

println("\n=== CNN Bayesian Optimisation (7 dims, 50 iterations) ===")

result_cnn = bo_search(obj_cnn,
    [log(1e-4), 0.5, 4.5, 0.5, 4.5, 2.5, 2.5],
    [log(1e-2), 2.5, 6.5, 2.5, 6.5, 5.5, 4.5];
    n_init = 5,
    n_iter = 45,
    rng    = Xoshiro(BASE_SEED))

# ============================================================
# DeepSet BO search (9-dimensional)
# ============================================================
# x[1–5]: same as CNN
# x[6]: phi_depth       [0.5, 3.5]  → 1, 2, or 3
# x[7]: phi_width_exp   [4.5, 7.5]  → 5, 6, or 7
# x[8]: rho_depth       [0.5, 3.5]  → 1, 2, or 3
# x[9]: rho_width_exp   [4.5, 7.5]  → 5, 6, or 7

ds_builder(x) = (DeepSetFeaturizerHyperParams(
    round(Int, x[6]), round(Int, x[7]),
    round(Int, x[8]), round(Int, x[9])), 4)

obj_ds = make_objective(;
    base_seed          = BASE_SEED,
    input_dim          = INPUT_DIM_DS,
    βlims              = BETA_LIMS,
    train_batches      = TRAIN_BATCHES,
    test_batches       = TEST_BATCHES,
    pot_per_batch      = POT_PER_BATCH,
    trace_per_pot      = TRACE_PER_POT,
    cut_per_trace      = CUT_PER_TRACE,
    feature            = deep_set_feature,
    featurizer_builder = ds_builder)

println("\n=== DeepSet Bayesian Optimisation (9 dims, 60 iterations) ===")

result_ds = bo_search(obj_ds,
    [log(1e-4), 0.5, 4.5, 0.5, 4.5, 0.5, 4.5, 0.5, 4.5],
    [log(1e-2), 2.5, 6.5, 2.5, 6.5, 3.5, 7.5, 3.5, 7.5];
    n_init = 5,
    n_iter = 55,
    rng    = Xoshiro(BASE_SEED + 1))

# ============================================================
# Decode helpers
# ============================================================

function decode_cnn(x)
    lr            = exp(x[1])
    rnn_depth     = round(Int, x[2])
    rnn_width_exp = round(Int, x[3])
    mlp_depth     = round(Int, x[4])
    mlp_width_exp = round(Int, x[5])
    cnn_depth     = round(Int, x[6])
    cnn_width_exp = round(Int, x[7])
    feat_hp = CNNFeaturizerHyperParams(cnn_depth, cnn_width_exp)
    h = RNNDiagnosticHyperParams(feat_hp, rnn_depth, rnn_width_exp,
                                  mlp_depth, mlp_width_exp)
    return lr, h
end

function decode_ds(x)
    lr            = exp(x[1])
    rnn_depth     = round(Int, x[2])
    rnn_width_exp = round(Int, x[3])
    mlp_depth     = round(Int, x[4])
    mlp_width_exp = round(Int, x[5])
    phi_depth     = round(Int, x[6])
    phi_width_exp = round(Int, x[7])
    rho_depth     = round(Int, x[8])
    rho_width_exp = round(Int, x[9])
    feat_hp = DeepSetFeaturizerHyperParams(phi_depth, phi_width_exp,
                                            rho_depth, rho_width_exp)
    h = RNNDiagnosticHyperParams(feat_hp, rnn_depth, rnn_width_exp,
                                  mlp_depth, mlp_width_exp)
    return lr, h
end

# ============================================================
# Retrain best configs from scratch and save
# ============================================================

println("\n=== Retraining best CNN config ===")
lr_cnn, h_cnn = decode_cnn(result_cnn.observed_optimizer)
println("lr=$(lr_cnn)  hyperparams=$(h_cnn)")

best_run_cnn = build_candidate_run((lr_cnn, h_cnn);
    base_seed     = BASE_SEED,
    input_dim     = INPUT_DIM_CNN,
    βlims         = BETA_LIMS,
    pot_per_batch = POT_PER_BATCH,
    trace_per_pot = TRACE_PER_POT,
    cut_per_trace = CUT_PER_TRACE,
    feature       = hist_feature)

run_epoch!(best_run_cnn, TRAIN_BATCHES, false)
acc_cnn, loss_cnn = test_accuracy!(best_run_cnn, TEST_BATCHES)
println("CNN retrain: val_loss=$(loss_cnn)  acc=$(acc_cnn)")
JLD2.jldsave("best_hope_bo_cnn.jld2", model_state=Flux.state(best_run_cnn.model))
println("Saved → best_hope_bo_cnn.jld2")

println("\n=== Retraining best DeepSet config ===")
lr_ds, h_ds = decode_ds(result_ds.observed_optimizer)
println("lr=$(lr_ds)  hyperparams=$(h_ds)")

best_run_ds = build_candidate_run((lr_ds, h_ds);
    base_seed     = BASE_SEED,
    input_dim     = INPUT_DIM_DS,
    βlims         = BETA_LIMS,
    pot_per_batch = POT_PER_BATCH,
    trace_per_pot = TRACE_PER_POT,
    cut_per_trace = CUT_PER_TRACE,
    feature       = deep_set_feature)

run_epoch!(best_run_ds, TRAIN_BATCHES, false)
acc_ds, loss_ds = test_accuracy!(best_run_ds, TEST_BATCHES)
println("DeepSet retrain: val_loss=$(loss_ds)  acc=$(acc_ds)")
JLD2.jldsave("best_hope_bo_ds.jld2", model_state=Flux.state(best_run_ds.model))
println("Saved → best_hope_bo_ds.jld2")

# ── Overall winner ───────────────────────────────────────────
if loss_cnn <= loss_ds
    println("\nOverall winner: CNN (loss=$(loss_cnn))")
    JLD2.jldsave("best_hope_bo.jld2", model_state=Flux.state(best_run_cnn.model))
else
    println("\nOverall winner: DeepSet (loss=$(loss_ds))")
    JLD2.jldsave("best_hope_bo.jld2", model_state=Flux.state(best_run_ds.model))
end
println("Saved → best_hope_bo.jld2")
