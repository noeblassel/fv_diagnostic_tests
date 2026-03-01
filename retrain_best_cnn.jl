#!/usr/bin/env julia
#
# retrain_best_cnn.jl
#
# Loads the best hyperparameter vector from the CNN FTBO run,
# retrains the model online (on-the-fly data generation) using
# multiple seeds, and saves the best model.

using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

include("FVDiagnosticTests.jl")

using .FVDiagnosticTests
using Flux, JLD2, Statistics, Random, Dates, ProgressMeter

# ── Settings ─────────────────────────────────────────────────────────────────

const N_SEEDS          = 5        # number of independent seeds to try
const N_TRAIN_BATCHES  = 50      # training batches per epoch
const N_TEST_BATCHES   = 50       # test batches for validation
const N_EPOCHS         = 200       # total training epochs per seed
const BETA_LIMS        = (1.0, 3.0)
const POT_PER_BATCH    = 5
const TRACE_PER_POT    = 5
const CUT_PER_TRACE    = 2
const STRIDE_LIMS      = (10, 200)
const NREPLICAS_LIMS   = (10, 200)

const RESULTS_PATH     = joinpath(@__DIR__, "ftbo_results_cnn.jld2")
const OUTPUT_DIR       = joinpath(@__DIR__, "retrain_results")

# ── Load best hyperparameters from FTBO ──────────────────────────────────────

isfile(RESULTS_PATH) || error("FTBO results not found at $RESULTS_PATH")
best_x    = JLD2.load(RESULTS_PATH, "best_x")
best_loss = JLD2.load(RESULTS_PATH, "best_loss")

# Decode the hyperparameter vector (same logic as tune_ftbo.jl)
function decode_cnn(x)
    lr            = exp(x[1])
    rnn_depth     = round(Int, x[2])
    rnn_width_exp = round(Int, x[3])
    mlp_depth     = round(Int, x[4])
    mlp_width_exp = round(Int, x[5])
    input_dim_exp = round(Int, x[8])
    cnn_depth     = min(round(Int, x[6]), input_dim_exp)
    cnn_width_exp = round(Int, x[7])
    feat_hp = CNNFeaturizerHyperParams(input_dim_exp, cnn_depth, cnn_width_exp)
    h = RNNDiagnosticHyperParams(feat_hp, rnn_depth, rnn_width_exp,
                                  mlp_depth, mlp_width_exp)
    return lr, h
end

lr, hp = decode_cnn(best_x)

println("=" ^ 70)
println("Retraining best CNN from FTBO (online data generation)")
println("=" ^ 70)
println("  FTBO best loss:    $best_loss")
println("  Learning rate:     $lr")
println("  RNN depth:         $(hp.rnn_depth)")
println("  RNN width:         2^$(hp.rnn_width_exponent) = $(2^hp.rnn_width_exponent)")
println("  MLP depth:         $(hp.mlp_depth)")
println("  MLP width:         2^$(hp.mlp_width_exponent) = $(2^hp.mlp_width_exponent)")
println("  CNN depth:         $(hp.featurizer.depth)")
println("  CNN width:         2^$(hp.featurizer.width_exponent) = $(2^hp.featurizer.width_exponent)")
println("  Input dim:         2^$(hp.featurizer.input_dim_exponent) = $(2^hp.featurizer.input_dim_exponent)")
println("  Seeds:             $N_SEEDS")
println("  Epochs per seed:   $N_EPOCHS")
println("  Train batches:     $N_TRAIN_BATCHES")
println("  Test batches:      $N_TEST_BATCHES")
println("=" ^ 70)

# ── Create output directory ──────────────────────────────────────────────────

mkpath(OUTPUT_DIR)

# ── Train each seed ──────────────────────────────────────────────────────────

seed_results = []

for seed_idx in 1:N_SEEDS
    seed = 1000 * seed_idx
    rng  = Xoshiro(seed)

    println("\n", "─" ^ 70)
    println("Seed $seed_idx/$N_SEEDS  (seed=$seed)")
    println("─" ^ 70)

    input_dim = 2^hp.featurizer.input_dim_exponent
    model     = RNNDiagnostic(hp; n_meta=1, rng=Xoshiro(seed))
    opt_state = Flux.setup(Adam(lr), model)

    run = TrainingRun(
        rng            = rng,
        βlims          = BETA_LIMS,
        opt_state      = opt_state,
        model          = model,
        feature        = hist_feature,
        input_dim      = input_dim,
        pot_per_batch  = POT_PER_BATCH,
        trace_per_pot  = TRACE_PER_POT,
        cut_per_trace  = CUT_PER_TRACE,
        id             = "retrain_seed$(seed)",
        stride_lims    = STRIDE_LIMS,
        Nreplicas_lims = NREPLICAS_LIMS,
        n_meta         = 1,
    )

    n_params = sum(length, Flux.trainables(model))
    println("  Model parameters: $n_params")

    best_val_loss = Inf
    best_epoch    = 0
    best_state    = nothing
    train_losses  = Float64[]
    val_losses    = Float64[]
    val_accs      = Float64[]

    for epoch in 1:N_EPOCHS
        # Train
        epoch_losses = run_epoch!(run, N_TRAIN_BATCHES, false)
        mean_train   = mean(epoch_losses)
        push!(train_losses, mean_train)

        # Validate
        acc, val_loss = test_accuracy!(run, N_TEST_BATCHES)
        push!(val_losses, val_loss)
        push!(val_accs, acc[1])

        improved = val_loss < best_val_loss
        if improved
            best_val_loss = val_loss
            best_epoch    = epoch
            best_state    = Flux.state(model)
        end

        println("  Epoch $epoch/$N_EPOCHS" *
                "  train=$(round(mean_train; digits=4))" *
                "  val=$(round(val_loss; digits=4))" *
                "  acc=$(round(acc[1]*100; digits=1))%" *
                (improved ? "  ★ new best" : ""))
        flush(stdout)

        # Save checkpoint for this seed
        ckpt_path = joinpath(OUTPUT_DIR, "seed$(seed_idx)_epoch$(epoch).jld2")
        JLD2.jldsave(ckpt_path; model_state=Flux.state(model))
    end

    push!(seed_results, (;
        seed_idx, seed,
        best_val_loss, best_epoch, best_state,
        train_losses, val_losses, val_accs,
    ))

    # Save this seed's best model
    seed_best_path = joinpath(OUTPUT_DIR, "best_seed$(seed_idx).jld2")
    JLD2.jldsave(seed_best_path; model_state=best_state)
    println("  → Best epoch $best_epoch, val loss $(round(best_val_loss; digits=4))")
    println("  → Saved to $seed_best_path")
end

# ── Pick the overall best seed ───────────────────────────────────────────────

println("\n", "=" ^ 70)
println("Summary")
println("=" ^ 70)

for r in seed_results
    println("  Seed $(r.seed_idx) (seed=$(r.seed)): " *
            "best_val_loss=$(round(r.best_val_loss; digits=4)) " *
            "at epoch $(r.best_epoch), " *
            "final_acc=$(round(r.val_accs[end]*100; digits=1))%")
end

best_run = argmin(r -> r.best_val_loss, seed_results)

println("\nOverall best: Seed $(best_run.seed_idx)" *
        " (val_loss=$(round(best_run.best_val_loss; digits=4))" *
        ", epoch $(best_run.best_epoch))")

# Save the overall best
overall_path = joinpath(@__DIR__, "best_retrained_cnn.jld2")
JLD2.jldsave(overall_path;
    model_state   = best_run.best_state,
    best_x        = best_x,
    lr            = lr,
    hp            = hp,
    seed          = best_run.seed,
    best_epoch    = best_run.best_epoch,
    best_val_loss = best_run.best_val_loss,
    all_results   = [(seed_idx=r.seed_idx, seed=r.seed,
                      best_val_loss=r.best_val_loss, best_epoch=r.best_epoch,
                      train_losses=r.train_losses, val_losses=r.val_losses,
                      val_accs=r.val_accs) for r in seed_results],
)

println("Saved overall best → $overall_path")
println("Done.")
