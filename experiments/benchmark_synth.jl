using ProgressMeter
using Statistics, Random
using Flux, JLD2
using JSON, Dates

include("../FVDiagnosticTests.jl")

prefix = ARGS[1]

using .FVDiagnosticTests

# --- Helpers ---
@inline trapz(x, y) = sum((y[i] + y[i+1]) / 2 * (x[i+1] - x[i]) for i in 1:length(x)-1)
@inline get_bin(val, minval, maxval, nbins) = 1 + clamp(floor(Int, nbins * (val - minval) / (maxval - minval)), 0, nbins - 1)

function auc_func(fpr, tpr)
    fpr_full = [1.0; fpr; 0.0]
    tpr_full = [1.0; tpr; 0.0]
    p = sortperm(fpr_full)
    return trapz(fpr_full[p], tpr_full[p])
end

function compute_roc(scores, labels; thresholds=collect(0.01f0:0.01f0:0.99f0))
    n = length(scores)
    n == 0 && return (fpr=Float64[], tpr=Float64[], auc=NaN)
    n_pos = sum(labels .== 1.0f0)
    n_neg = n - n_pos
    tpr_vec = Float64[]
    fpr_vec = Float64[]
    for th in thresholds
        pred_pos = scores .> th
        tp = sum(@. pred_pos && (labels == 1.0f0))
        fp = sum(@. pred_pos && (labels == 0.0f0))
        push!(tpr_vec, n_pos > 0 ? tp / n_pos : 0.0)
        push!(fpr_vec, n_neg > 0 ? fp / n_neg : 0.0)
    end
    auc_val = auc_func(fpr_vec, tpr_vec)
    return (fpr=fpr_vec, tpr=tpr_vec, auc=auc_val)
end

# --- Load Model ---
rseed = 2027
rng = Xoshiro(rseed)
βlims = (1.0, 3.0)
ckpt = JLD2.load("$(prefix).jld2")
state = ckpt["model_state"]

if haskey(ckpt, "hp")
    hp = ckpt["hp"]
    input_dim = 2^hp.featurizer.input_dim_exponent
    model = RNNDiagnostic(hp; n_meta=1, rng=Xoshiro(rseed))
    Flux.loadmodel!(model, state)
else
    input_dim = 64
    model = load_rnn_from_state(input_dim, state)
end

testmode!(model)
println("Loaded model from $(prefix).jld2  (input_dim=$input_dim, params=$(sum(length, Flux.trainables(model))))")

# --- Configuration ---
stride_values    = [10, 25, 50, 100, 150, 200]
Nreplicas_values = [10, 25, 50, 100, 150, 200]
num_batches      = 20
npot_per_batch   = 5
ntrace_per_pot   = 5
alpha_values     = Float32[0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
ncorr            = 5
time_bins        = 0.0:0.1:Float64(ncorr)   # t/t_corr ∈ [0, ncorr]
nbins_time       = length(time_bins) - 1
detect_bins      = 0.0:0.05:Float64(ncorr)  # finer bins for detection time histogram
nbins_detect     = length(detect_bins) - 1

function evaluate_grid(model, rng;
                       stride_values, Nreplicas_values,
                       num_batches, npot_per_batch, ntrace_per_pot,
                       alpha_values, time_bins, nbins_time, ncorr,
                       detect_bins, nbins_detect,
                       input_dim=64, βlims=(1.0, 3.0))

    ns = length(stride_values)
    nn = length(Nreplicas_values)
    na = length(alpha_values)

    # Parameter grid accumulators: (stride, Nreplicas, alpha)
    tp_grid = zeros(Int, ns, nn, na)
    fp_grid = zeros(Int, ns, nn, na)
    tn_grid = zeros(Int, ns, nn, na)
    fn_grid = zeros(Int, ns, nn, na)

    # Time-resolved accumulators
    # Denominators are alpha-independent (frame counts by class)
    acc_denom = zeros(nbins_time)
    fpr_denom = zeros(nbins_time)
    fnr_denom = zeros(nbins_time)
    # Numerators depend on alpha: (time_bin, alpha)
    acc_numer = zeros(nbins_time, na)
    fpr_numer = zeros(nbins_time, na)
    fnr_numer = zeros(nbins_time, na)

    # ROC sample storage: one independent sample per trajectory
    roc_samples = [Tuple{Float32,Float32}[] for _ in 1:ns, _ in 1:nn]
    roc_samples_all = Tuple{Float32,Float32}[]

    # Detection time histogram: (nbins_detect, na)
    detect_hist = zeros(Int, nbins_detect, na)

    total_cells = ns * nn * num_batches
    p = Progress(total_cells; desc="Evaluating grid: ")

    for (si, s) in enumerate(stride_values)
        for (ni, N) in enumerate(Nreplicas_values)
            for b in 1:num_batches
                X, Y, mask = get_batch(rng;
                    input_dim=input_dim,
                    stride_lims=(s, s),
                    Nreplicas_lims=(N, N),
                    ncut=0,
                    ncorr=ncorr,
                    npot=npot_per_batch,
                    ntrace=ntrace_per_pot,
                    feature=hist_feature,
                    βlims=βlims)

                Flux.reset!(model)
                Yhat_prob = Flux.σ(model(X))

                ntraj = size(Y, 2)

                for j in 1:ntraj
                    traj_len = sum(mask[:, j])
                    traj_len == 0 && continue

                    decorr_step = findfirst(==(1.0f0), Y[:, j])
                    decorr_step === nothing && continue

                    for k in 1:traj_len
                        prob = Yhat_prob[k, j]
                        true_pos = Y[k, j] == 1.0f0
                        tbin = get_bin(k / decorr_step, first(time_bins), last(time_bins), nbins_time)

                        # Alpha-independent time denominators
                        acc_denom[tbin] += 1
                        if !true_pos
                            fpr_denom[tbin] += 1
                        else
                            fnr_denom[tbin] += 1
                        end

                        # Accumulate for each alpha threshold
                        for (ai, α) in enumerate(alpha_values)
                            pred_pos = prob > α

                            if pred_pos && true_pos
                                tp_grid[si, ni, ai] += 1
                            elseif pred_pos && !true_pos
                                fp_grid[si, ni, ai] += 1
                            elseif !pred_pos && true_pos
                                fn_grid[si, ni, ai] += 1
                            else
                                tn_grid[si, ni, ai] += 1
                            end

                            acc_numer[tbin, ai] += (pred_pos == true_pos) ? 1 : 0
                            if !true_pos
                                fpr_numer[tbin, ai] += pred_pos ? 1 : 0
                            else
                                fnr_numer[tbin, ai] += !pred_pos ? 1 : 0
                            end
                        end
                    end

                    # ROC: one independent sample per trajectory (alpha-independent)
                    k_roc = rand(rng, 1:traj_len)
                    sample = (Yhat_prob[k_roc, j], Y[k_roc, j])
                    push!(roc_samples[si, ni], sample)
                    push!(roc_samples_all, sample)

                    # Detection time: first frame where model predicts positive
                    for (ai, α) in enumerate(alpha_values)
                        detect_frame = findfirst(>(α), @view Yhat_prob[1:traj_len, j])
                        if detect_frame !== nothing
                            ratio = detect_frame / decorr_step
                            tbin = get_bin(ratio, first(detect_bins), last(detect_bins), nbins_detect)
                            detect_hist[tbin, ai] += 1
                        end
                    end
                end

                next!(p)
            end
        end
    end

    # --- Derived grid metrics: (ns, nn, na) ---
    total_grid = tp_grid .+ fp_grid .+ tn_grid .+ fn_grid
    acc_grid = (tp_grid .+ tn_grid) ./ max.(total_grid, 1)
    fpr_grid = fp_grid ./ max.(fp_grid .+ tn_grid, 1)
    fnr_grid = fn_grid ./ max.(fn_grid .+ tp_grid, 1)

    # --- Per-cell AUC (alpha-independent): (ns, nn) ---
    auc_grid = zeros(ns, nn)
    roc_per_cell = Matrix{Any}(undef, ns, nn)
    for si in 1:ns, ni in 1:nn
        samples = roc_samples[si, ni]
        if length(samples) > 0
            scores = [s[1] for s in samples]
            labels = [s[2] for s in samples]
            roc_result = compute_roc(scores, labels)
            auc_grid[si, ni] = roc_result.auc
            roc_per_cell[si, ni] = roc_result
        else
            auc_grid[si, ni] = NaN
            roc_per_cell[si, ni] = (fpr=Float64[], tpr=Float64[], auc=NaN)
        end
    end

    # --- Global ROC ---
    global_scores = [s[1] for s in roc_samples_all]
    global_labels = [s[2] for s in roc_samples_all]
    global_roc = compute_roc(global_scores, global_labels)

    # --- Time-resolved rates: (nbins_time, na) ---
    # Broadcasting: (nbins_time, na) ./ (nbins_time,) works via reshape
    acc_time = acc_numer ./ max.(acc_denom, 1)
    fpr_time = fpr_numer ./ max.(fpr_denom, 1)
    fnr_time = fnr_numer ./ max.(fnr_denom, 1)

    # AUC statistics across cells
    valid_aucs = filter(!isnan, vec(auc_grid))
    mean_auc = isempty(valid_aucs) ? NaN : mean(valid_aucs)
    std_auc  = isempty(valid_aucs) ? NaN : std(valid_aucs)

    return (
        # Grid metrics: (ns, nn, na)
        acc_grid   = acc_grid,
        fpr_grid   = fpr_grid,
        fnr_grid   = fnr_grid,
        auc_grid   = auc_grid,       # (ns, nn)
        total_grid = total_grid,
        # Time-resolved: (nbins_time, na)
        acc_time  = acc_time,
        fpr_time  = fpr_time,
        fnr_time  = fnr_time,
        acc_denom = acc_denom,        # (nbins_time,)
        fpr_denom = fpr_denom,
        fnr_denom = fnr_denom,
        # Detection time histogram: (nbins_detect, na)
        detect_hist = detect_hist,
        detect_bins = collect(detect_bins),
        # ROC (alpha-independent)
        roc_fpr        = global_roc.fpr,
        roc_tpr        = global_roc.tpr,
        roc_thresholds = collect(0.01f0:0.01f0:0.99f0),
        mean_auc       = mean_auc,
        std_auc        = std_auc,
        global_auc     = global_roc.auc,
        roc_per_cell   = roc_per_cell,
        # Config
        stride_values    = stride_values,
        Nreplicas_values = Nreplicas_values,
        time_bins        = collect(time_bins),
        alpha_values     = alpha_values,
    )
end

# --- Run evaluation ---
println("Starting benchmark evaluation...")
t0 = now()
results = evaluate_grid(model, rng;
    stride_values    = stride_values,
    Nreplicas_values = Nreplicas_values,
    num_batches      = num_batches,
    npot_per_batch   = npot_per_batch,
    ntrace_per_pot   = ntrace_per_pot,
    alpha_values     = alpha_values,
    time_bins        = time_bins,
    nbins_time       = nbins_time,
    ncorr            = ncorr,
    detect_bins      = detect_bins,
    nbins_detect     = nbins_detect,
    input_dim        = input_dim,
    βlims            = βlims)
t1 = now()
println("Evaluation completed in $(t1 - t0)")

# --- Save JSON results ---
json_data = Dict(
    "acc_grid"          => results.acc_grid,
    "fpr_grid"          => results.fpr_grid,
    "fnr_grid"          => results.fnr_grid,
    "auc_grid"          => results.auc_grid,
    "total_grid"        => results.total_grid,
    "acc_time"          => results.acc_time,
    "fpr_time"          => results.fpr_time,
    "fnr_time"          => results.fnr_time,
    "detect_hist"       => results.detect_hist,
    "detect_bins"       => results.detect_bins,
    "roc_fpr"           => results.roc_fpr,
    "roc_tpr"           => results.roc_tpr,
    "roc_thresholds"    => results.roc_thresholds,
    "mean_auc"          => results.mean_auc,
    "std_auc"           => results.std_auc,
    "global_auc"        => results.global_auc,
    "stride_values"     => results.stride_values,
    "Nreplicas_values"  => results.Nreplicas_values,
    "time_bins"         => results.time_bins,
    "alpha_values"      => results.alpha_values,
)

open("$(prefix)_benchmark_results.json", "w") do f
    write(f, JSON.json(json_data; allownan=true))
end
println("Results saved to $(prefix)_benchmark_results.json")
println("Benchmark complete. Run plot_benchmark.jl to generate plots.")
