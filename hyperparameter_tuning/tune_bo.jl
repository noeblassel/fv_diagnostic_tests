using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

include(joinpath(@__DIR__, "..", "FVDiagnosticTests.jl"))

using .FVDiagnosticTests
using GaussianProcesses
using Flux, JLD2, Statistics, Random, LinearAlgebra, MLUtils, ProgressMeter

include(joinpath(@__DIR__, "offline_training.jl"))

# ── Load offline dataset ──────────────────────────────────────────────────────

const DATASET_PATH = joinpath(@__DIR__, "hp_dataset.jld2")
isfile(DATASET_PATH) || error("Dataset not found. Run make_dataset.jl first.")
const HP_TRAIN = JLD2.load(DATASET_PATH, "train")
const HP_TEST  = JLD2.load(DATASET_PATH, "test")
@info "train=$(length(HP_TRAIN.sequences))  test=$(length(HP_TEST.sequences))"

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

include(joinpath(@__DIR__, "gp_utils.jl"))

# ============================================================
# MAP GP hyperparameter fitting
# ============================================================

function fit_gp_hyperparams(X_mat, y_vec, lb, ub; n_restarts=3)
    d, n = size(X_mat)
    y_c  = y_vec .- mean(y_vec)
    log_ℓ0  = log.((ub .- lb) ./ 2)
    log_σf0 = log(max(std(y_vec), 1e-6))
    log_σn0 = log(0.1)
    θ0 = [log_ℓ0; log_σf0; log_σn0]

    function neg_map(θ)
        ℓ2 = exp.(2θ[1:d]); σf = exp(θ[d+1]); σn = exp(θ[d+2])
        K  = [σf^2 * exp(-0.5*sum((X_mat[:,i].-X_mat[:,j]).^2 ./ ℓ2))
              for i in 1:n, j in 1:n]
        C  = cholesky(Symmetric(K + (σn^2+1e-9)*I(n)); check=false)
        issuccess(C) || return Inf
        α  = C.L' \ (C.L \ y_c)
        lml = -0.5*dot(y_c,α) - sum(log.(diag(C.L))) - 0.5n*log(2π)
        # log prior (Normal on each log-param)
        lp  = -0.5*sum(((θ[1:d].-log_ℓ0)/1.0).^2) -
               0.5*((θ[d+1]-log_σf0)/1.0)^2 -
               0.5*((θ[d+2]-log_σn0)/1.5)^2
        return -(lml + lp)
    end

    best_θ, best_val = θ0, Inf
    for r in 1:n_restarts
        θ_init = r==1 ? θ0 : θ0 .+ 0.5.*randn(d+2)
        try
            θr, vr = _nelder_mead(neg_map, θ_init)
            vr < best_val && (best_θ = θr; best_val = vr)
        catch; end
    end
    return best_θ[1:d], best_θ[d+1], best_θ[d+2]  # log_ℓ, log_σf, log_σn
end

"""
    bo_search(objective, lb, ub; n_init, n_iter, β, rng)

Maximise `objective` over the box [lb, ub] using BO with UCB.
Returns a NamedTuple with:
  - `observed_optimizer` — x vector at the best observed value
  - `observed_optimum`   — corresponding objective value
  - `X`, `y`            — full history of evaluations
"""
function bo_search(objective, lb, ub;
        n_init        = 5,
        n_iter        = 45,
        β             = 2.0,
        n_cand        = 2000,
        hp_fit_every  = 5,
        config_key    = nothing,   # x -> comparable key; nothing = no dedup
        rng           = Xoshiro(42))

    d = length(lb)

    X = Vector{Vector{Float64}}()
    y = Vector{Float64}()
    seen_keys = Set{Any}()

    # ── Initial random exploration ───────────────────────────
    println("  [BO] Random initialisation ($n_init points)...")
    for i in 1:n_init
        x = lb .+ rand(rng, d) .* (ub .- lb)
        if config_key !== nothing
            key = config_key(x)
            # Redraw up to 20 times if already seen
            attempts = 0
            while key in seen_keys && attempts < 20
                x   = lb .+ rand(rng, d) .* (ub .- lb)
                key = config_key(x)
                attempts += 1
            end
            push!(seen_keys, key)
        end
        push!(X, x)
        val = objective(x)
        push!(y, val)
        println("  init $i/$n_init  loss=$(-val)")
        flush(stdout)
    end

    best_idx = argmax(y)
    best_x, best_y = X[best_idx], y[best_idx]

    hp_log_ℓ, hp_log_σf, hp_log_σn = zeros(d), 0.0, -2.0
    hp_min_obs = 2*(d+2)

    # ── Sequential BO iterations ─────────────────────────────
    for iter in 1:n_iter
        # Build GP; refit MAP hyperparameters periodically
        X_mat = hcat(X...)         # d × n
        y_vec = Float64.(y)
        n_obs = length(y)

        if n_obs >= hp_min_obs && mod(iter-1, hp_fit_every) == 0
            hp_log_ℓ, hp_log_σf, hp_log_σn =
                fit_gp_hyperparams(X_mat, y_vec, lb, ub)
            println("  [BO] fitted GP hyperparams: ℓ=$(round.(exp.(hp_log_ℓ); digits=3))" *
                    "  σf=$(round(exp(hp_log_σf); digits=3))" *
                    "  σn=$(round(exp(hp_log_σn); digits=3))")
        end

        kern = SEArd(hp_log_ℓ, hp_log_σf)
        gp   = GP(X_mat, y_vec, MeanConst(mean(y_vec)), kern, hp_log_σn)

        # Random UCB search over the box
        cands = [lb .+ rand(rng, d) .* (ub .- lb) for _ in 1:n_cand]
        if config_key !== nothing
            cands_new = filter(xc -> config_key(xc) ∉ seen_keys, cands)
            cands = isempty(cands_new) ? cands : cands_new
        end
        X_cand = hcat(cands...)
        mu, sigma2 = predict_y(gp, X_cand)
        ucb = mu .+ β .* sqrt.(max.(sigma2, 1e-10))
        x_next = cands[argmax(ucb)]

        # Evaluate objective
        y_next = objective(x_next)
        push!(X, x_next)
        push!(y, y_next)
        if config_key !== nothing
            push!(seen_keys, config_key(x_next))
        end

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

Returns a closure `x -> -loss` that wraps one training epoch using offline data.
The first 5 dims of `x` are shared (lr, rnn/mlp depths & widths);
`featurizer_builder(x)` extracts the remaining dims.
"""
function make_objective(;
        base_seed,
        input_dim,
        βlims,
        train_epochs,
        pot_per_batch,
        trace_per_pot,
        cut_per_trace,
        feature,
        featurizer_builder,
        stride_lims    = STRIDE_LIMS,
        Nreplicas_lims = NREPLICAS_LIMS)

    history    = NamedTuple[]
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
            base_seed      = base_seed + call_count[],
            input_dim      = input_dim,
            βlims          = βlims,
            pot_per_batch  = pot_per_batch,
            trace_per_pot  = trace_per_pot,
            cut_per_trace  = cut_per_trace,
            feature        = feature,
            stride_lims    = stride_lims,
            Nreplicas_lims = Nreplicas_lims)

        println("  model: $(repr("text/plain",run.model))")

        train_losses = Float64[]
        test_losses  = Float64[]
        test_accs    = Float64[]
        for epoch in 1:train_epochs
            train_loss  = run_epoch_offline!(run, HP_TRAIN)
            test_result = test_loss_offline!(run, HP_TEST)
            push!(train_losses, train_loss)
            push!(test_losses,  test_result.loss)
            push!(test_accs,    test_result.acc)
            println("    epoch $epoch/$train_epochs  train=$(round(train_loss; digits=4))" *
                    "  test=$(round(test_result.loss; digits=4))" *
                    "  acc=$(round(test_result.acc * 100; digits=1))%")
            flush(stdout)
        end

        push!(history, (
            call         = call_count[],
            x            = copy(x),
            train_losses = train_losses,
            test_losses  = test_losses,
            test_accs    = test_accs,
            model_state  = Flux.state(run.model),
        ))

        return -test_losses[end]    # BO maximises; we minimise loss
    end

    return objective, history
end

# ============================================================
# Shared search settings
# ============================================================

const BASE_SEED       = 2022
const INPUT_DIM_CNN   = 64
const INPUT_DIM_DS    = 200    # max particles for DeepSet
const BETA_LIMS       = (1.0, 3.0)
const TRAIN_EPOCHS    = 10     # epochs per BO candidate evaluation
const POT_PER_BATCH   = 5
const TRACE_PER_POT   = 5
const CUT_PER_TRACE   = 2
const STRIDE_LIMS     = (10, 200)
const NREPLICAS_LIMS  = (10, 200)

# ============================================================
# DeepSet BO search (9-dimensional)
# ============================================================
# x[1–5]: same as CNN
# x[6]: phi_depth       [0.5, 3.5]  → 1, 2, or 3
# x[7]: phi_width_exp   [2.5, 6.5]  → 3, 4, 5, or 6  (narrowed for 1-D input)
# x[8]: rho_depth       [0.5, 3.5]  → 1, 2, or 3
# x[9]: rho_width_exp   [2.5, 6.5]  → 3, 4, 5, or 6  (narrowed for 1-D input)

ds_builder(x) = (DeepSetFeaturizerHyperParams(
    round(Int, x[6]), round(Int, x[7]),
    round(Int, x[8]), round(Int, x[9])), 4)

obj_ds, hist_ds = make_objective(;
    base_seed          = BASE_SEED,
    input_dim          = INPUT_DIM_DS,
    βlims              = BETA_LIMS,
    train_epochs       = TRAIN_EPOCHS,
    pot_per_batch      = POT_PER_BATCH,
    trace_per_pot      = TRACE_PER_POT,
    cut_per_trace      = CUT_PER_TRACE,
    feature            = deep_set_feature,
    featurizer_builder = ds_builder,
    stride_lims        = STRIDE_LIMS,
    Nreplicas_lims     = NREPLICAS_LIMS)

cnn_config_key(x) = (round(x[1]; digits=1),
                     round(Int,x[2]), round(Int,x[3]),
                     round(Int,x[4]), round(Int,x[5]),
                     round(Int,x[6]), round(Int,x[7]))
ds_config_key(x)  = (round(x[1]; digits=1),
                     round(Int,x[2]), round(Int,x[3]),
                     round(Int,x[4]), round(Int,x[5]),
                     round(Int,x[6]), round(Int,x[7]),
                     round(Int,x[8]), round(Int,x[9]))

println("\n=== DeepSet Bayesian Optimisation (9 dims, 60 iterations) ===")

result_ds = bo_search(obj_ds,
    [log(1e-4), 0.5, 4.5, 0.5, 4.5, 0.5, 2.5, 0.5, 2.5],
    [log(1e-2), 2.5, 6.5, 2.5, 6.5, 3.5, 6.5, 3.5, 6.5];
    n_init     = 5,
    n_iter     = 55,
    config_key = ds_config_key,
    rng        = Xoshiro(BASE_SEED + 1))

JLD2.jldsave(joinpath(@__DIR__, "bo_results_ds.jld2");
    X         = result_ds.X,
    y         = result_ds.y,
    best_x    = result_ds.observed_optimizer,
    best_loss = -result_ds.observed_optimum,
    history   = hist_ds,
)
println("Saved → $(joinpath(@__DIR__, "bo_results_ds.jld2"))")

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

obj_cnn, hist_cnn = make_objective(;
    base_seed          = BASE_SEED,
    input_dim          = INPUT_DIM_CNN,
    βlims              = BETA_LIMS,
    train_epochs       = TRAIN_EPOCHS,
    pot_per_batch      = POT_PER_BATCH,
    trace_per_pot      = TRACE_PER_POT,
    cut_per_trace      = CUT_PER_TRACE,
    feature            = hist_feature,
    featurizer_builder = cnn_builder,
    stride_lims        = STRIDE_LIMS,
    Nreplicas_lims     = NREPLICAS_LIMS)

println("\n=== CNN Bayesian Optimisation (7 dims, 50 iterations) ===")

result_cnn = bo_search(obj_cnn,
    [log(1e-4), 0.5, 4.5, 0.5, 4.5, 2.5, 2.5],
    [log(1e-2), 2.5, 6.5, 2.5, 6.5, 5.5, 4.5];
    n_init     = 5,
    n_iter     = 45,
    config_key = cnn_config_key,
    rng        = Xoshiro(BASE_SEED))

JLD2.jldsave(joinpath(@__DIR__, "bo_results_cnn.jld2");
    X         = result_cnn.X,
    y         = result_cnn.y,
    best_x    = result_cnn.observed_optimizer,
    best_loss = -result_cnn.observed_optimum,
    history   = hist_cnn,
)
println("Saved → $(joinpath(@__DIR__, "bo_results_cnn.jld2"))")

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
# Retrain best configs from scratch using online data and save
# ============================================================

println("\n=== Retraining best CNN config ===")
lr_cnn, h_cnn = decode_cnn(result_cnn.observed_optimizer)
println("lr=$(lr_cnn)  hyperparams=$(h_cnn)")

best_run_cnn = build_candidate_run((lr_cnn, h_cnn);
    base_seed      = BASE_SEED,
    input_dim      = INPUT_DIM_CNN,
    βlims          = BETA_LIMS,
    pot_per_batch  = POT_PER_BATCH,
    trace_per_pot  = TRACE_PER_POT,
    cut_per_trace  = CUT_PER_TRACE,
    feature        = hist_feature,
    stride_lims    = STRIDE_LIMS,
    Nreplicas_lims = NREPLICAS_LIMS)

run_epoch!(best_run_cnn, TRAIN_BATCHES, false)
acc_cnn, loss_cnn = test_accuracy!(best_run_cnn, TEST_BATCHES)
println("CNN retrain: val_loss=$(loss_cnn)  acc=$(acc_cnn)")
JLD2.jldsave(joinpath(@__DIR__, "best_hope_bo_cnn.jld2"), model_state=Flux.state(best_run_cnn.model))
println("Saved → $(joinpath(@__DIR__, "best_hope_bo_cnn.jld2"))")

println("\n=== Retraining best DeepSet config ===")
lr_ds, h_ds = decode_ds(result_ds.observed_optimizer)
println("lr=$(lr_ds)  hyperparams=$(h_ds)")

best_run_ds = build_candidate_run((lr_ds, h_ds);
    base_seed      = BASE_SEED,
    input_dim      = INPUT_DIM_DS,
    βlims          = BETA_LIMS,
    pot_per_batch  = POT_PER_BATCH,
    trace_per_pot  = TRACE_PER_POT,
    cut_per_trace  = CUT_PER_TRACE,
    feature        = deep_set_feature,
    stride_lims    = STRIDE_LIMS,
    Nreplicas_lims = NREPLICAS_LIMS)

run_epoch!(best_run_ds, TRAIN_BATCHES, false)
acc_ds, loss_ds = test_accuracy!(best_run_ds, TEST_BATCHES)
println("DeepSet retrain: val_loss=$(loss_ds)  acc=$(acc_ds)")
JLD2.jldsave(joinpath(@__DIR__, "best_hope_bo_ds.jld2"), model_state=Flux.state(best_run_ds.model))
println("Saved → $(joinpath(@__DIR__, "best_hope_bo_ds.jld2"))")

# ── Overall winner ───────────────────────────────────────────
if loss_cnn <= loss_ds
    println("\nOverall winner: CNN (loss=$(loss_cnn))")
    JLD2.jldsave(joinpath(@__DIR__, "best_hope_bo.jld2"), model_state=Flux.state(best_run_cnn.model))
else
    println("\nOverall winner: DeepSet (loss=$(loss_ds))")
    JLD2.jldsave(joinpath(@__DIR__, "best_hope_bo.jld2"), model_state=Flux.state(best_run_ds.model))
end
println("Saved → $(joinpath(@__DIR__, "best_hope_bo.jld2"))")
