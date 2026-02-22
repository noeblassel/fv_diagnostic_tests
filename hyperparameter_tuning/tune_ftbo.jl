using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

include(joinpath(@__DIR__, "..", "FVDiagnosticTests.jl"))

using .FVDiagnosticTests
using Flux, JLD2, Statistics, Random, LinearAlgebra, MLUtils, ProgressMeter

include(joinpath(@__DIR__, "offline_training.jl"))

# ── Load offline dataset ──────────────────────────────────────────────────────

const DATASET_PATH = joinpath(@__DIR__, "hp_dataset.jld2")
isfile(DATASET_PATH) || error("Dataset not found. Run make_dataset.jl first.")
const HP_TRAIN = JLD2.load(DATASET_PATH, "train")
const HP_TEST  = JLD2.load(DATASET_PATH, "test")
@info "train=$(length(HP_TRAIN.sequences))  test=$(length(HP_TEST.sequences))"

# ============================================================
# Freeze-Thaw Bayesian Optimisation  (Swersky, Snoek & Adams 2014)
#
# Key idea: maintain a pool of partially-trained configs.
# At each step, fit a 2D GP over (hyperparams × training_time)
# and use LCB on the GP-extrapolated final loss to decide:
#   (a) continue ("thaw") an existing config, or
#   (b) start a brand-new one.
#
# Unlike tune_bo.jl (fixed 30-batch budget per config), configs
# here are trained in small "chunks" (CHUNK_BATCHES steps each).
# Clearly bad configs are abandoned early; promising ones get more
# budget.
#
# GP model:
#   f(x, t) ~ GP(m, k_x(x,x') * k_t(t,t'))
# where
#   k_x  = SE-ARD over hyperparameter space
#   k_t(t,t') = β_t^α_t / (t + t' + β_t)^α_t   (freeze-thaw kernel)
#
# Acquisition: LCB at (x, T_final) — lower = better final loss.
# ============================================================

# ── Pool entry ───────────────────────────────────────────────────────────────

mutable struct FrozenConfig
    x::Vector{Float64}       # hyperparameter vector
    run::TrainingRun         # live model + optimizer (in memory)
    ts::Vector{Int}          # chunk indices at which loss was measured
    losses::Vector{Float64}  # corresponding validation losses
    config_key::Any          # decoded discrete key; `nothing` if dedup disabled
end

# ── Product kernel: SE-ARD(x) ⊗ Freeze-Thaw time kernel(t) ─────────────────

function _kx(x1, x2, ℓ2)
    s = 0.0
    @inbounds for i in eachindex(x1)
        s += (x1[i] - x2[i])^2 / ℓ2[i]
    end
    return exp(-0.5 * s)
end

# Freeze-thaw kernel: k_t(t,t') = β_t^α_t / (t + t' + β_t)^α_t
# Captures smooth decay toward a finite asymptote (Swersky et al. 2014 eq. 12).
function _kt(t1, t2; α_t::Float64 = 1.0, β_t::Float64 = 1.0)
    return (β_t^α_t) / (t1 + t2 + β_t)^α_t
end

function _kprod(x1, t1, x2, t2, ℓ2; α_t = 1.0, β_t = 1.0)
    return _kx(x1, x2, ℓ2) * _kt(t1, t2; α_t = α_t, β_t = β_t)
end

# ── GP cache: pre-factorised Cholesky for efficient multi-point prediction ───

struct GPCache
    L::Matrix{Float64}             # lower Cholesky factor of (K + noise·I)
    α::Vector{Float64}             # (K + noise·I)⁻¹ (y − m)
    xs::Vector{Vector{Float64}}    # observed hyperparameter vectors
    ts::Vector{Int}                # observed time steps
    m::Float64                     # constant prior mean
    ℓ2::Vector{Float64}
    α_t::Float64
    β_t::Float64
end

function build_gp_cache(xs_obs, ts_obs, ys_obs, ℓ2;
                         α_t = 1.0, β_t = 1.0, noise = 1e-3)
    n   = length(ys_obs)
    m   = mean(ys_obs)
    y_c = ys_obs .- m

    K = [_kprod(xs_obs[i], ts_obs[i], xs_obs[j], ts_obs[j], ℓ2;
                α_t = α_t, β_t = β_t) for i in 1:n, j in 1:n]

    K_reg = Symmetric(K + (noise + 1e-6) * I(n))
    L     = Matrix(cholesky(K_reg).L)
    α_vec = L' \ (L \ y_c)

    return GPCache(L, α_vec, collect(xs_obs), collect(ts_obs), m, ℓ2, α_t, β_t)
end

function gp_predict(cache::GPCache, x_pred, t_pred)
    (; L, α, xs, ts, m, ℓ2, α_t, β_t) = cache
    n = length(α)

    k_star = [_kprod(x_pred, t_pred, xs[i], ts[i], ℓ2; α_t = α_t, β_t = β_t)
              for i in 1:n]
    k_ss   = _kprod(x_pred, t_pred, x_pred, t_pred, ℓ2; α_t = α_t, β_t = β_t)

    v   = L \ k_star
    μ   = m + dot(k_star, α)
    σ2  = max(k_ss - dot(v, v), 1e-10)
    return μ, σ2
end

include(joinpath(@__DIR__, "gp_utils.jl"))

# ============================================================
# MAP GP hyperparameter fitting for the 2D (x,t) product kernel
# ============================================================

function fit_gp_hyperparams_2d(xs_obs, ts_obs, ys_obs, lb, ub; n_restarts=3)
    d   = length(lb)
    n   = length(ys_obs)
    y_c = ys_obs .- mean(ys_obs)
    log_ℓ0 = log.((ub .- lb) ./ 2)
    θ0 = [log_ℓ0; 0.0; 0.0; log(0.03)]   # log_α_t=0, log_β_t=0, log_noise

    function neg_map(θ)
        ℓ2    = exp.(2θ[1:d])
        α_t_  = exp(θ[d+1]); β_t_ = exp(θ[d+2]); noise_ = exp(θ[d+3])
        K     = [_kprod(xs_obs[i], ts_obs[i], xs_obs[j], ts_obs[j], ℓ2;
                        α_t=α_t_, β_t=β_t_) for i in 1:n, j in 1:n]
        C     = cholesky(Symmetric(K + (noise_+1e-9)*I(n)); check=false)
        issuccess(C) || return Inf
        α_vec = C.L' \ (C.L \ y_c)
        lml   = -0.5*dot(y_c,α_vec) - sum(log.(diag(C.L))) - 0.5n*log(2π)
        lp    = -0.5*sum(((θ[1:d].-log_ℓ0)/1.0).^2) -
                 0.5*(θ[d+1]/0.75)^2 - 0.5*(θ[d+2]/0.75)^2 -
                 0.5*((θ[d+3]-log(0.03))/1.0)^2
        return -(lml + lp)
    end

    best_θ, best_val = θ0, Inf
    for r in 1:n_restarts
        θi = r==1 ? θ0 : θ0 .+ 0.5.*randn(d+3)
        try
            θr, vr = _nelder_mead(neg_map, θi; max_iter=600)
            vr < best_val && (best_θ = θr; best_val = vr)
        catch; end
    end
    return exp.(2best_θ[1:d]), exp(best_θ[d+1]), exp(best_θ[d+2]), exp(best_θ[d+3])
    #        ℓ2_fit,               α_t_fit,         β_t_fit,          noise_fit
end

# ── Main freeze-thaw BO loop ─────────────────────────────────────────────────

"""
    freeze_thaw_bo_search(build_run, lb, ub; kwargs...) -> NamedTuple

Freeze-Thaw Bayesian Optimisation over the box [lb, ub].

`build_run(x, seed_offset)` must return a freshly-initialised (untrained)
`TrainingRun` for hyperparameter vector `x`.

Keyword arguments
─────────────────
- `n_init`        : random configs in the initial pool (default 5)
- `n_iter`        : sequential acquisition steps (default 45)
- `T_final`       : max training epochs per config; used for GP extrapolation
- `β_ucb`         : LCB exploration bonus  (lcb = μ − β·σ)
- `n_cand`        : random candidates drawn for the "new config" option
- `α_t`, `β_t`   : freeze-thaw time kernel parameters (initial values; fitted by MAP)
- `noise`         : GP observation noise (initial value; fitted by MAP)
- `hp_fit_every`  : refit GP hyperparameters every this many iterations (default 3)
- `rng`           : random number generator

Returns a NamedTuple with:
- `pool`        — all FrozenConfig objects explored
- `best_config` — the FrozenConfig with the lowest observed validation loss
- `best_x`      — its hyperparameter vector
- `best_loss`   — its lowest observed validation loss
"""
function freeze_thaw_bo_search(build_run, lb, ub;
        n_init          = 5,
        n_iter          = 45,
        T_final         = 6,
        β_ucb           = 2.0,
        n_cand          = 2000,
        α_t             = 1.0,
        β_t             = 1.0,
        noise           = 1.0,
        hp_fit_every    = 3,
        rng             = Xoshiro(42),
        config_key      = nothing,   # x -> comparable key; nothing = no dedup
        prefix          = "",
        checkpoint_path = nothing)

    d = length(lb)

    # SE-ARD length scales: set to half the range of each dimension,
    # so configs within ~half-range are considered correlated.
    range_w = ub .- lb
    ℓ2 = (range_w ./ 2.0) .^ 2

    current_ℓ2, current_α_t, current_β_t, current_noise = ℓ2, α_t, β_t, noise
    hp_min_obs = 2*(d+3)

    pool       = FrozenConfig[]
    call_count = Ref(0)

    # Run one full training epoch then measure test loss.
    # Each FTBO acquisition step corresponds to exactly one training epoch.
    function eval_chunk!(run)
        run_epoch_offline!(run, HP_TRAIN)
        result = test_loss_offline!(run, HP_TEST)
        println("      acc=$(round(result.acc * 100; digits=1))%")
        return result.loss
    end

    # Build a fresh config, run its first chunk, add it to the pool
    function launch_config!(x)
        call_count[] += 1
        run  = build_run(x, call_count[])
        loss = eval_chunk!(run)
        key  = config_key !== nothing ? config_key(x) : nothing
        push!(pool, FrozenConfig(copy(x), run, [1], [loss], key))
        return loss
    end

    # ── Initialisation: n_init random configs ────────────────────────────────
    println("  [FTBO] Initialising ($n_init random configs, $T_final epochs/config)...")
    for i in 1:n_init
        x = lb .+ rand(rng, d) .* (ub .- lb)
        # --- dedup check ---
        if config_key !== nothing && !isempty(pool)
            key = config_key(x)
            match_idx = findfirst(j -> pool[j].config_key == key &&
                                       maximum(pool[j].ts) < T_final, 1:length(pool))
            if match_idx !== nothing
                cfg    = pool[match_idx]
                t_next = maximum(cfg.ts) + 1
                loss   = eval_chunk!(cfg.run)
                push!(cfg.ts, t_next); push!(cfg.losses, loss)
                println("  init $i/$n_init  DUPLICATE→THAW pool[$match_idx]  loss=$(round(loss; digits=4))")
                flush(stdout)
                continue
            end
        end
        # --- end dedup check ---
        loss = launch_config!(x)
        println("  init $i/$n_init  loss=$(round(loss; digits=4))")
        flush(stdout)
    end

    # ── Sequential acquisition loop ──────────────────────────────────────────
    for iter in 1:n_iter

        # Collect all (x, t, loss) observations into flat vectors
        xs_obs = Vector{Float64}[]
        ts_obs = Int[]
        ys_obs = Float64[]
        for cfg in pool, (t, l) in zip(cfg.ts, cfg.losses)
            push!(xs_obs, cfg.x)
            push!(ts_obs, t)
            push!(ys_obs, l)
        end

        # Refit MAP hyperparameters periodically, then build GP cache
        n_obs = length(ys_obs)
        if n_obs >= hp_min_obs && mod(iter-1, hp_fit_every) == 0
            try
                current_ℓ2, current_α_t, current_β_t, current_noise =
                    fit_gp_hyperparams_2d(xs_obs, ts_obs, ys_obs, lb, ub)
                println("  [FTBO] fitted GP hyperparams:" *
                        "  α_t=$(round(current_α_t; digits=3))" *
                        "  β_t=$(round(current_β_t; digits=3))" *
                        "  noise=$(round(current_noise; digits=5))")
            catch e
                @warn "FTBO hyperparam fit failed: $e"
            end
        end
        cache = build_gp_cache(xs_obs, ts_obs, ys_obs, current_ℓ2;
                                α_t = current_α_t, β_t = current_β_t,
                                noise = current_noise)

        best_lcb        = Inf
        chosen_action   = :new
        chosen_pool_idx = 0
        chosen_x_new    = similar(lb)

        # Option A: thaw an existing config (if not yet exhausted)
        for (i, cfg) in enumerate(pool)
            maximum(cfg.ts) >= T_final && continue   # budget exhausted

            μ, σ2 = gp_predict(cache, cfg.x, T_final)
            lcb   = μ - β_ucb * sqrt(σ2)
            if lcb < best_lcb
                best_lcb        = lcb
                chosen_action   = :thaw
                chosen_pool_idx = i
            end
        end

        # Option B: start a new config from n_cand random draws
        for _ in 1:n_cand
            xc        = lb .+ rand(rng, d) .* (ub .- lb)
            μ, σ2     = gp_predict(cache, xc, T_final)
            lcb       = μ - β_ucb * sqrt(σ2)
            if lcb < best_lcb
                best_lcb      = lcb
                chosen_action = :new
                chosen_x_new  = xc
            end
        end

        # Redirect :new → :thaw if candidate decodes to an existing non-exhausted config
        if chosen_action == :new && config_key !== nothing
            key_new   = config_key(chosen_x_new)
            match_idx = findfirst(i -> pool[i].config_key == key_new &&
                                       maximum(pool[i].ts) < T_final, 1:length(pool))
            if match_idx !== nothing
                println("  [FTBO] iter $iter/$n_iter  NEW→THAW: candidate has same config as pool[$match_idx]")
                chosen_action   = :thaw
                chosen_pool_idx = match_idx
            end
        end

        # Execute the chosen action
        if chosen_action == :thaw
            cfg    = pool[chosen_pool_idx]
            t_next = maximum(cfg.ts) + 1
            loss   = eval_chunk!(cfg.run)
            push!(cfg.ts, t_next)
            push!(cfg.losses, loss)
            println("  [FTBO] iter $iter/$n_iter  THAW config $chosen_pool_idx" *
                    " (chunk $t_next/$T_final)" *
                    "  loss=$(round(loss; digits=4))" *
                    "  best=$(round(minimum(minimum(cfg.losses) for cfg in pool); digits=4))" *
                     "  model: $(repr("text/plain",cfg.run.model))")
        else
            loss = launch_config!(chosen_x_new)
            println("  [FTBO] iter $iter/$n_iter  NEW config #$(length(pool))" *
                    "  loss=$(round(loss; digits=4))" *
                    "  best=$(round(minimum(minimum(cfg.losses) for cfg in pool); digits=4))")
        end
        flush(stdout)

        if checkpoint_path !== nothing
            pool_ckpt = [(x=cfg.x, ts=cfg.ts, losses=cfg.losses,
                          model_state=Flux.state(cfg.run.model)) for cfg in pool]
            JLD2.jldsave(checkpoint_path; pool=pool_ckpt, iter=iter)
        end
    end

    best_cfg = argmin(cfg -> minimum(cfg.losses), pool)

    return (pool        = pool,
            best_config = best_cfg,
            best_x      = best_cfg.x,
            best_loss   = minimum(best_cfg.losses))
end

# ============================================================
# Shared settings (match tune_bo.jl where applicable)
# ============================================================

const BASE_SEED      = 2026
const INPUT_DIM_CNN  = 64
const INPUT_DIM_DS   = 200
const INPUT_DIM_ATTN = 200
const BETA_LIMS      = (1.0, 3.0)
const T_FINAL        = 30      # max training epochs per config (GP extrapolation horizon)
const POT_PER_BATCH  = 5
const TRACE_PER_POT  = 5
const CUT_PER_TRACE  = 2
const STRIDE_LIMS    = (10, 200)
const NREPLICAS_LIMS = (10, 200)

# ============================================================
# Decode helpers  (x -> lr, hyperparams)
# ============================================================

# x[1]: log(lr)       [log(1e-4), log(1e-2)]
# x[2]: rnn_depth     [0.5, 2.5]  → 1 or 2
# x[3]: rnn_width_exp [4.5, 6.5]  → 5 or 6
# x[4]: mlp_depth     [0.5, 2.5]  → 1 or 2
# x[5]: mlp_width_exp [4.5, 6.5]  → 5 or 6
# CNN x[6]: cnn_depth [2.5, 5.5]  → 3, 4, or 5
# CNN x[7]: cnn_width [2.5, 4.5]  → 3 or 4

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

# DeepSet x[6]: phi_depth [0.5, 3.5]  → 1, 2, or 3
# DeepSet x[7]: phi_width [2.5, 6.5]  → 3, 4, 5, or 6  (narrowed for 1-D input)
# DeepSet x[8]: rho_depth [0.5, 3.5]  → 1, 2, or 3
# DeepSet x[9]: rho_width [2.5, 6.5]  → 3, 4, 5, or 6  (narrowed for 1-D input)

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
# build_run factories: (x, seed_offset) -> TrainingRun
# ============================================================

function make_build_run_cnn()
    function build_run(x, seed_offset)
        lr, h = decode_cnn(x)
        return build_candidate_run((lr, h);
            base_seed      = BASE_SEED + seed_offset,
            input_dim      = INPUT_DIM_CNN,
            βlims          = BETA_LIMS,
            pot_per_batch  = POT_PER_BATCH,
            trace_per_pot  = TRACE_PER_POT,
            cut_per_trace  = CUT_PER_TRACE,
            feature        = hist_feature,
            stride_lims    = STRIDE_LIMS,
            Nreplicas_lims = NREPLICAS_LIMS)
    end
    return build_run
end

function make_build_run_ds()
    function build_run(x, seed_offset)
        lr, h = decode_ds(x)
        return build_candidate_run((lr, h);
            base_seed      = BASE_SEED + seed_offset,
            input_dim      = INPUT_DIM_DS,
            βlims          = BETA_LIMS,
            pot_per_batch  = POT_PER_BATCH,
            trace_per_pot  = TRACE_PER_POT,
            cut_per_trace  = CUT_PER_TRACE,
            feature        = deep_set_feature,
            stride_lims    = STRIDE_LIMS,
            Nreplicas_lims = NREPLICAS_LIMS,
            n_meta         = 2)
    end
    return build_run
end

# x[1]: log(lr)          [log(1e-4), log(1e-2)]
# x[2]: rnn_depth        [0.5, 2.5]
# x[3]: rnn_width_exp    [4.5, 6.5]
# x[4]: mlp_depth        [0.5, 2.5]
# x[5]: mlp_width_exp    [4.5, 6.5]
# x[6]: log2(n_subsample) [5.5, 10.5]  → 64 to 1024
# x[7]: n_heads_exp      [0.5, 3.5]  → 1, 2, or 3
# x[8]: width_exp        [2.5, 5.5]  → 3, 4, or 5 (clamped ≥ x[7])
# x[9]: depth            [0.5, 3.5]  → 1, 2, or 3

function decode_attn(x)
    lr            = exp(x[1])
    rnn_depth     = round(Int, x[2]);  rnn_width_exp = round(Int, x[3])
    mlp_depth     = round(Int, x[4]);  mlp_width_exp = round(Int, x[5])
    n_subsample   = 2^round(Int, x[6])
    n_heads_exp   = round(Int, x[7])
    width_exp     = max(round(Int, x[8]), n_heads_exp)
    depth         = round(Int, x[9])
    feat_hp = AttentionFeaturizerHyperParams(n_subsample, n_heads_exp, width_exp, depth)
    h = RNNDiagnosticHyperParams(feat_hp, rnn_depth, rnn_width_exp, mlp_depth, mlp_width_exp)
    return lr, h
end

function make_build_run_attn()
    function build_run(x, seed_offset)
        lr, h = decode_attn(x)
        return build_candidate_run((lr, h);
            base_seed      = BASE_SEED + seed_offset,
            input_dim      = INPUT_DIM_ATTN,
            βlims          = BETA_LIMS,
            pot_per_batch  = POT_PER_BATCH,
            trace_per_pot  = TRACE_PER_POT,
            cut_per_trace  = CUT_PER_TRACE,
            feature        = deep_set_feature,
            stride_lims    = STRIDE_LIMS,
            Nreplicas_lims = NREPLICAS_LIMS,
            n_meta         = 2)
    end
    return build_run
end

attn_config_key(x) = (round(x[1]; digits=1),
                       round(Int,x[2]), round(Int,x[3]),
                       round(Int,x[4]), round(Int,x[5]),
                       round(Int,x[6]),
                       round(Int,x[7]),
                       max(round(Int,x[8]), round(Int,x[7])),
                       round(Int,x[9]))

# ============================================================
# CNN Freeze-Thaw BO  (7-dimensional, 50 total steps)
# ============================================================

cnn_config_key(x) = (round(x[1]; digits=1),
                     round(Int,x[2]), round(Int,x[3]),
                     round(Int,x[4]), round(Int,x[5]),
                     round(Int,x[6]), round(Int,x[7]))

println("\n=== CNN Freeze-Thaw BO (7 dims, n_init=5, n_iter=45) ===")

result_cnn = freeze_thaw_bo_search(
    make_build_run_cnn(),
    [log(1e-4), 0.5, 4.5, 0.5, 4.5, 2.5, 2.5],
    [log(1e-2), 2.5, 6.5, 2.5, 6.5, 5.5, 4.5];
    n_init          = 10,
    n_iter          = 500,
    T_final         = T_FINAL,
    rng             = Xoshiro(BASE_SEED),
    config_key      = cnn_config_key,
    prefix          = "cnn_",
    checkpoint_path = joinpath(@__DIR__, "ftbo_checkpoint_cnn.jld2"))

println("\nCNN FTBO best loss: $(result_cnn.best_loss)")

pool_data_cnn = [(x=cfg.x, ts=cfg.ts, losses=cfg.losses,
                  model_state=Flux.state(cfg.run.model)) for cfg in result_cnn.pool]
JLD2.jldsave(joinpath(@__DIR__, "ftbo_results_cnn.jld2");
    pool             = pool_data_cnn,
    best_x           = result_cnn.best_x,
    best_loss        = result_cnn.best_loss,
    best_model_state = Flux.state(result_cnn.best_config.run.model),
)
println("Saved → $(joinpath(@__DIR__, "ftbo_results_cnn.jld2"))")

# ============================================================
# DeepSet Freeze-Thaw BO  (9-dimensional, 60 total steps)
# ============================================================

ds_config_key(x)  = (round(x[1]; digits=1),
                     round(Int,x[2]), round(Int,x[3]),
                     round(Int,x[4]), round(Int,x[5]),
                     round(Int,x[6]), round(Int,x[7]),
                     round(Int,x[8]), round(Int,x[9]))

println("\n=== DeepSet Freeze-Thaw BO (9 dims, n_init=5, n_iter=55) ===")

result_ds = freeze_thaw_bo_search(
    make_build_run_ds(),
    [log(1e-4), 0.5, 4.5, 0.5, 4.5, 0.5, 4.5, 0.5, 4.5],
    [log(1e-2), 2.5, 6.5, 2.5, 6.5, 3.5, 8.5, 3.5, 8.5];
    n_init          = 10,
    n_iter          = 500,
    T_final         = T_FINAL,
    rng             = Xoshiro(BASE_SEED + 1),
    config_key      = ds_config_key,
    prefix          = "deepset_",
    checkpoint_path = joinpath(@__DIR__, "ftbo_checkpoint_ds.jld2"))

println("\nDeepSet FTBO best loss: $(result_ds.best_loss)")

pool_data_ds = [(x=cfg.x, ts=cfg.ts, losses=cfg.losses,
                 model_state=Flux.state(cfg.run.model)) for cfg in result_ds.pool]
JLD2.jldsave(joinpath(@__DIR__, "ftbo_results_ds.jld2");
    pool             = pool_data_ds,
    best_x           = result_ds.best_x,
    best_loss        = result_ds.best_loss,
    best_model_state = Flux.state(result_ds.best_config.run.model),
)
println("Saved → $(joinpath(@__DIR__, "ftbo_results_ds.jld2"))")

# ============================================================
# Attention Freeze-Thaw BO  (9-dimensional)
# ============================================================

println("\n=== Attention Freeze-Thaw BO (9 dims, n_init=10, n_iter=200) ===")

result_attn = freeze_thaw_bo_search(
    make_build_run_attn(),
    [log(1e-4), 0.5, 4.5, 0.5, 4.5, 5.5, 0.5, 2.5, 0.5],
    [log(1e-2), 2.5, 6.5, 2.5, 6.5, 10.5, 3.5, 5.5, 3.5];
    n_init          = 10,
    n_iter          = 500,
    T_final         = T_FINAL,
    rng             = Xoshiro(BASE_SEED + 2),
    config_key      = attn_config_key,
    prefix          = "attn_",
    checkpoint_path = joinpath(@__DIR__, "ftbo_checkpoint_attn.jld2"))

println("\nAttention FTBO best loss: $(result_attn.best_loss)")

pool_data_attn = [(x=cfg.x, ts=cfg.ts, losses=cfg.losses,
                   model_state=Flux.state(cfg.run.model)) for cfg in result_attn.pool]
JLD2.jldsave(joinpath(@__DIR__, "ftbo_results_attn.jld2");
    pool             = pool_data_attn,
    best_x           = result_attn.best_x,
    best_loss        = result_attn.best_loss,
    best_model_state = Flux.state(result_attn.best_config.run.model),
)
println("Saved → $(joinpath(@__DIR__, "ftbo_results_attn.jld2"))")