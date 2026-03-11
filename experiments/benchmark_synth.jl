using ProgressMeter
using Statistics, Random
using Flux, JLD2
using JSON, Dates

include("../FVDiagnosticTests.jl")

prefix = ARGS[1]

using .FVDiagnosticTests

# --- Parse optional OOD shift arguments ---
# Usage: julia benchmark_synth.jl <prefix> [--βlims=lo,hi] [--tol=0.05]
#        [--logσlims=lo,hi] [--κlims=lo,hi] [--δlims=lo,hi] [--clims=lo,hi]
#        [--mmax=6] [--Ngrid=100] [--dt=1e-3]
function parse_kwarg(args, key; type=Float64, default=nothing)
    for a in args
        if startswith(a, "--$(key)=")
            val = split(a, "="; limit=2)[2]
            if type <: Tuple
                parts = split(val, ",")
                return (parse(Float64, parts[1]), parse(Float64, parts[2]))
            elseif type == Int
                return parse(Int, val)
            else
                return parse(Float64, val)
            end
        end
    end
    return default
end

cli_βlims    = parse_kwarg(ARGS, "βlims";    type=Tuple)
cli_tol      = parse_kwarg(ARGS, "tol";      type=Float64)
cli_Ngrid    = parse_kwarg(ARGS, "Ngrid";     type=Int)
cli_dt       = parse_kwarg(ARGS, "dt";       type=Float64)
cli_logσlims = parse_kwarg(ARGS, "logσlims"; type=Tuple)
cli_κlims    = parse_kwarg(ARGS, "κlims";    type=Tuple)
cli_δlims    = parse_kwarg(ARGS, "δlims";    type=Tuple)
cli_clims    = parse_kwarg(ARGS, "clims";    type=Tuple)
cli_mmax     = parse_kwarg(ARGS, "mmax";     type=Int)

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
βlims = something(cli_βlims, (1.0, 3.0))
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
num_batches      = 200
npot_per_batch   = 5
ntrace_per_pot   = 5
alpha_values     = Float32[0.5, 0.6, 0.7, 0.8, 0.85, 0.9]
ncorr            = 10
time_bins        = 0.0:0.1:Float64(ncorr)   # t/t_corr ∈ [0, ncorr]
nbins_time       = length(time_bins) - 1
detect_bins      = 0.0:0.05:Float64(ncorr)  # finer bins for detection time histogram
nbins_detect     = length(detect_bins) - 1
tol              = something(cli_tol, 0.05)
Ngrid            = something(cli_Ngrid, 100)
dt               = something(cli_dt, 1e-3)

# Naive diagnostic thresholds
gr_alpha_values = Float64[0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
w1_alpha_values = Float64[0.01, 0.05, 0.1, 0.2, 0.3, 0.5]

# Build potential_kwargs from CLI overrides (only include what was specified)
potential_kwargs_dict = Dict{Symbol,Any}()
cli_logσlims !== nothing && (potential_kwargs_dict[:logσlims] = cli_logσlims)
cli_κlims    !== nothing && (potential_kwargs_dict[:κlims]    = cli_κlims)
cli_δlims    !== nothing && (potential_kwargs_dict[:δlims]    = cli_δlims)
cli_clims    !== nothing && (potential_kwargs_dict[:clims]    = cli_clims)
cli_mmax     !== nothing && (potential_kwargs_dict[:mmax]     = cli_mmax)
potential_kwargs = NamedTuple(potential_kwargs_dict)

# Collect all distribution shift overrides into a single dict for JSON recording
ood_config = Dict{String,Any}()
cli_βlims    !== nothing && (ood_config["βlims"]    = collect(cli_βlims))
cli_tol      !== nothing && (ood_config["tol"]      = cli_tol)
cli_Ngrid    !== nothing && (ood_config["Ngrid"]     = cli_Ngrid)
cli_dt       !== nothing && (ood_config["dt"]       = cli_dt)
cli_logσlims !== nothing && (ood_config["logσlims"] = collect(cli_logσlims))
cli_κlims    !== nothing && (ood_config["κlims"]    = collect(cli_κlims))
cli_δlims    !== nothing && (ood_config["δlims"]    = collect(cli_δlims))
cli_clims    !== nothing && (ood_config["clims"]    = collect(cli_clims))
cli_mmax     !== nothing && (ood_config["mmax"]     = cli_mmax)

# Build filename suffix from overrides: e.g. "_ood_βlims=3.0,6.0_tol=0.1"
if isempty(ood_config)
    ood_suffix = ""
else
    parts = sort(collect(keys(ood_config)))
    fmt(v) = v isa AbstractVector ? join(v, ',') : string(v)
    ood_suffix = "_ood_" * join(["$(k)=$(fmt(ood_config[k]))" for k in parts], "_")
    println("OOD overrides: $ood_config")
end
outprefix = "$(prefix)$(ood_suffix)"
println("βlims=$βlims  tol=$tol  Ngrid=$Ngrid  dt=$dt")
println("Output prefix: $(outprefix)")

function evaluate_grid(model, rng;
                       stride_values, Nreplicas_values,
                       num_batches, npot_per_batch, ntrace_per_pot,
                       alpha_values, gr_alpha_values, w1_alpha_values,
                       time_bins, nbins_time, ncorr,
                       detect_bins, nbins_detect,
                       input_dim=64, βlims=(1.0, 3.0),
                       tol=0.05, Ngrid=100, dt=1e-3,
                       potential_kwargs::NamedTuple=NamedTuple())

    ns = length(stride_values)
    nn = length(Nreplicas_values)
    na = length(alpha_values)
    na_gr = length(gr_alpha_values)
    na_w1 = length(w1_alpha_values)

    # --- NN accumulators ---
    tp_grid = zeros(Int, ns, nn, na)
    fp_grid = zeros(Int, ns, nn, na)
    tn_grid = zeros(Int, ns, nn, na)
    fn_grid = zeros(Int, ns, nn, na)

    acc_numer = zeros(nbins_time, na)
    fpr_numer = zeros(nbins_time, na)
    fnr_numer = zeros(nbins_time, na)

    roc_samples = [Tuple{Float32,Float32}[] for _ in 1:ns, _ in 1:nn]
    roc_samples_all = Tuple{Float32,Float32}[]

    detect_hist = zeros(Int, nbins_detect, na)

    # --- GR accumulators ---
    gr_tp_grid = zeros(Int, ns, nn, na_gr)
    gr_fp_grid = zeros(Int, ns, nn, na_gr)
    gr_tn_grid = zeros(Int, ns, nn, na_gr)
    gr_fn_grid = zeros(Int, ns, nn, na_gr)

    gr_acc_numer = zeros(nbins_time, na_gr)
    gr_fpr_numer = zeros(nbins_time, na_gr)
    gr_fnr_numer = zeros(nbins_time, na_gr)

    gr_roc_samples = [Tuple{Float32,Float32}[] for _ in 1:ns, _ in 1:nn]
    gr_roc_samples_all = Tuple{Float32,Float32}[]

    gr_detect_hist = zeros(Int, nbins_detect, na_gr)

    # --- W1 accumulators ---
    w1_tp_grid = zeros(Int, ns, nn, na_w1)
    w1_fp_grid = zeros(Int, ns, nn, na_w1)
    w1_tn_grid = zeros(Int, ns, nn, na_w1)
    w1_fn_grid = zeros(Int, ns, nn, na_w1)

    w1_acc_numer = zeros(nbins_time, na_w1)
    w1_fpr_numer = zeros(nbins_time, na_w1)
    w1_fnr_numer = zeros(nbins_time, na_w1)

    w1_roc_samples = [Tuple{Float32,Float32}[] for _ in 1:ns, _ in 1:nn]
    w1_roc_samples_all = Tuple{Float32,Float32}[]

    w1_detect_hist = zeros(Int, nbins_detect, na_w1)

    # --- Shared time-resolved denominators ---
    acc_denom = zeros(nbins_time)
    fpr_denom = zeros(nbins_time)
    fnr_denom = zeros(nbins_time)

    total_cells = ns * nn * num_batches
    p = Progress(total_cells; desc="Evaluating grid: ")

    for (si, s) in enumerate(stride_values)
        for (ni, N) in enumerate(Nreplicas_values)
            for b in 1:num_batches
                X, Y, mask, X_naive = get_batch(rng;
                    input_dim=input_dim,
                    stride_lims=(s, s),
                    Nreplicas_lims=(N, N),
                    ncut=0,
                    ncorr=ncorr,
                    npot=npot_per_batch,
                    ntrace=ntrace_per_pot,
                    feature=hist_feature,
                    βlims=βlims,
                    tol=tol, Ngrid=Ngrid, dt=dt,
                    potential_kwargs=potential_kwargs)

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
                        gr_val = X_naive[1, k, j]
                        w1_val = X_naive[2, k, j]
                        true_pos = Y[k, j] == 1.0f0
                        tbin = get_bin(k / decorr_step, first(time_bins), last(time_bins), nbins_time)

                        # Alpha-independent time denominators (shared)
                        acc_denom[tbin] += 1
                        if !true_pos
                            fpr_denom[tbin] += 1
                        else
                            fnr_denom[tbin] += 1
                        end

                        # NN: prob > alpha → converged
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

                        # GR: small → converged
                        for (ai, th) in enumerate(gr_alpha_values)
                            pred_pos = gr_val < th

                            if pred_pos && true_pos
                                gr_tp_grid[si, ni, ai] += 1
                            elseif pred_pos && !true_pos
                                gr_fp_grid[si, ni, ai] += 1
                            elseif !pred_pos && true_pos
                                gr_fn_grid[si, ni, ai] += 1
                            else
                                gr_tn_grid[si, ni, ai] += 1
                            end

                            gr_acc_numer[tbin, ai] += (pred_pos == true_pos) ? 1 : 0
                            if !true_pos
                                gr_fpr_numer[tbin, ai] += pred_pos ? 1 : 0
                            else
                                gr_fnr_numer[tbin, ai] += !pred_pos ? 1 : 0
                            end
                        end

                        # W1: small → converged
                        for (ai, th) in enumerate(w1_alpha_values)
                            pred_pos = w1_val < th

                            if pred_pos && true_pos
                                w1_tp_grid[si, ni, ai] += 1
                            elseif pred_pos && !true_pos
                                w1_fp_grid[si, ni, ai] += 1
                            elseif !pred_pos && true_pos
                                w1_fn_grid[si, ni, ai] += 1
                            else
                                w1_tn_grid[si, ni, ai] += 1
                            end

                            w1_acc_numer[tbin, ai] += (pred_pos == true_pos) ? 1 : 0
                            if !true_pos
                                w1_fpr_numer[tbin, ai] += pred_pos ? 1 : 0
                            else
                                w1_fnr_numer[tbin, ai] += !pred_pos ? 1 : 0
                            end
                        end
                    end

                    # ROC: one independent sample per trajectory (between 0 and 2*tcorr)
                    n_roc = min(traj_len, 2 * decorr_step)
                    k_roc = rand(rng, 1:n_roc)
                    true_label = Y[k_roc, j]

                    # NN ROC
                    push!(roc_samples[si, ni], (Yhat_prob[k_roc, j], true_label))
                    push!(roc_samples_all, (Yhat_prob[k_roc, j], true_label))

                    # Naive ROC: negate scores so compute_roc (uses >) works
                    gr_score = -Float32(X_naive[1, k_roc, j])
                    w1_score = -Float32(X_naive[2, k_roc, j])
                    push!(gr_roc_samples[si, ni], (gr_score, true_label))
                    push!(gr_roc_samples_all, (gr_score, true_label))
                    push!(w1_roc_samples[si, ni], (w1_score, true_label))
                    push!(w1_roc_samples_all, (w1_score, true_label))

                    # NN detection time
                    for (ai, α) in enumerate(alpha_values)
                        detect_frame = findfirst(>(α), @view Yhat_prob[1:traj_len, j])
                        if detect_frame !== nothing
                            ratio = detect_frame / decorr_step
                            tbin = get_bin(ratio, first(detect_bins), last(detect_bins), nbins_detect)
                            detect_hist[tbin, ai] += 1
                        end
                    end

                    # GR detection time
                    for (ai, th) in enumerate(gr_alpha_values)
                        detect_frame = findfirst(<(th), @view X_naive[1, 1:traj_len, j])
                        if detect_frame !== nothing
                            ratio = detect_frame / decorr_step
                            tbin = get_bin(ratio, first(detect_bins), last(detect_bins), nbins_detect)
                            gr_detect_hist[tbin, ai] += 1
                        end
                    end

                    # W1 detection time
                    for (ai, th) in enumerate(w1_alpha_values)
                        detect_frame = findfirst(<(th), @view X_naive[2, 1:traj_len, j])
                        if detect_frame !== nothing
                            ratio = detect_frame / decorr_step
                            tbin = get_bin(ratio, first(detect_bins), last(detect_bins), nbins_detect)
                            w1_detect_hist[tbin, ai] += 1
                        end
                    end
                end

                next!(p)
            end
        end
    end

    # --- NN derived grid metrics ---
    total_grid = tp_grid .+ fp_grid .+ tn_grid .+ fn_grid
    acc_grid = (tp_grid .+ tn_grid) ./ max.(total_grid, 1)
    fpr_grid = fp_grid ./ max.(fp_grid .+ tn_grid, 1)
    fnr_grid = fn_grid ./ max.(fn_grid .+ tp_grid, 1)

    # --- GR derived grid metrics ---
    gr_total = gr_tp_grid .+ gr_fp_grid .+ gr_tn_grid .+ gr_fn_grid
    gr_acc_grid = (gr_tp_grid .+ gr_tn_grid) ./ max.(gr_total, 1)
    gr_fpr_grid = gr_fp_grid ./ max.(gr_fp_grid .+ gr_tn_grid, 1)
    gr_fnr_grid = gr_fn_grid ./ max.(gr_fn_grid .+ gr_tp_grid, 1)

    # --- W1 derived grid metrics ---
    w1_total = w1_tp_grid .+ w1_fp_grid .+ w1_tn_grid .+ w1_fn_grid
    w1_acc_grid = (w1_tp_grid .+ w1_tn_grid) ./ max.(w1_total, 1)
    w1_fpr_grid = w1_fp_grid ./ max.(w1_fp_grid .+ w1_tn_grid, 1)
    w1_fnr_grid = w1_fn_grid ./ max.(w1_fn_grid .+ w1_tp_grid, 1)

    # --- NN per-cell AUC ---
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

    # --- GR/W1 per-cell AUC ---
    gr_auc_grid = zeros(ns, nn)
    w1_auc_grid = zeros(ns, nn)
    gr_roc_thresholds = collect(-0.2f0:0.002f0:0.0f0)
    w1_roc_thresholds = collect(-1.0f0:0.005f0:0.0f0)

    for si in 1:ns, ni in 1:nn
        gr_s = gr_roc_samples[si, ni]
        if length(gr_s) > 0
            gr_auc_grid[si, ni] = compute_roc([s[1] for s in gr_s], [s[2] for s in gr_s]; thresholds=gr_roc_thresholds).auc
        else
            gr_auc_grid[si, ni] = NaN
        end

        w1_s = w1_roc_samples[si, ni]
        if length(w1_s) > 0
            w1_auc_grid[si, ni] = compute_roc([s[1] for s in w1_s], [s[2] for s in w1_s]; thresholds=w1_roc_thresholds).auc
        else
            w1_auc_grid[si, ni] = NaN
        end
    end

    # --- Global ROC ---
    global_roc = compute_roc([s[1] for s in roc_samples_all], [s[2] for s in roc_samples_all])
    gr_global_roc = compute_roc([s[1] for s in gr_roc_samples_all], [s[2] for s in gr_roc_samples_all]; thresholds=gr_roc_thresholds)
    w1_global_roc = compute_roc([s[1] for s in w1_roc_samples_all], [s[2] for s in w1_roc_samples_all]; thresholds=w1_roc_thresholds)

    # --- Time-resolved rates ---
    acc_time = acc_numer ./ max.(acc_denom, 1)
    fpr_time = fpr_numer ./ max.(fpr_denom, 1)
    fnr_time = fnr_numer ./ max.(fnr_denom, 1)

    gr_acc_time = gr_acc_numer ./ max.(acc_denom, 1)
    gr_fpr_time = gr_fpr_numer ./ max.(fpr_denom, 1)
    gr_fnr_time = gr_fnr_numer ./ max.(fnr_denom, 1)

    w1_acc_time = w1_acc_numer ./ max.(acc_denom, 1)
    w1_fpr_time = w1_fpr_numer ./ max.(fpr_denom, 1)
    w1_fnr_time = w1_fnr_numer ./ max.(fnr_denom, 1)

    # AUC statistics
    valid_aucs = filter(!isnan, vec(auc_grid))
    mean_auc = isempty(valid_aucs) ? NaN : mean(valid_aucs)
    std_auc  = isempty(valid_aucs) ? NaN : std(valid_aucs)

    gr_valid_aucs = filter(!isnan, vec(gr_auc_grid))
    gr_mean_auc = isempty(gr_valid_aucs) ? NaN : mean(gr_valid_aucs)
    gr_std_auc  = isempty(gr_valid_aucs) ? NaN : std(gr_valid_aucs)

    w1_valid_aucs = filter(!isnan, vec(w1_auc_grid))
    w1_mean_auc = isempty(w1_valid_aucs) ? NaN : mean(w1_valid_aucs)
    w1_std_auc  = isempty(w1_valid_aucs) ? NaN : std(w1_valid_aucs)

    return (
        # NN grid metrics
        acc_grid   = acc_grid,
        fpr_grid   = fpr_grid,
        fnr_grid   = fnr_grid,
        auc_grid   = auc_grid,
        total_grid = total_grid,
        # NN time-resolved
        acc_time  = acc_time,
        fpr_time  = fpr_time,
        fnr_time  = fnr_time,
        # NN detection time
        detect_hist = detect_hist,
        detect_bins = collect(detect_bins),
        # NN ROC
        roc_fpr        = global_roc.fpr,
        roc_tpr        = global_roc.tpr,
        roc_thresholds = collect(0.01f0:0.01f0:0.99f0),
        mean_auc       = mean_auc,
        std_auc        = std_auc,
        global_auc     = global_roc.auc,
        roc_per_cell   = roc_per_cell,

        # GR grid metrics
        gr_acc_grid = gr_acc_grid,
        gr_fpr_grid = gr_fpr_grid,
        gr_fnr_grid = gr_fnr_grid,
        gr_auc_grid = gr_auc_grid,
        # GR time-resolved
        gr_acc_time = gr_acc_time,
        gr_fpr_time = gr_fpr_time,
        gr_fnr_time = gr_fnr_time,
        # GR detection time
        gr_detect_hist = gr_detect_hist,
        # GR ROC
        gr_roc_fpr = gr_global_roc.fpr,
        gr_roc_tpr = gr_global_roc.tpr,
        gr_global_auc = gr_global_roc.auc,
        gr_mean_auc = gr_mean_auc,
        gr_std_auc = gr_std_auc,

        # W1 grid metrics
        w1_acc_grid = w1_acc_grid,
        w1_fpr_grid = w1_fpr_grid,
        w1_fnr_grid = w1_fnr_grid,
        w1_auc_grid = w1_auc_grid,
        # W1 time-resolved
        w1_acc_time = w1_acc_time,
        w1_fpr_time = w1_fpr_time,
        w1_fnr_time = w1_fnr_time,
        # W1 detection time
        w1_detect_hist = w1_detect_hist,
        # W1 ROC
        w1_roc_fpr = w1_global_roc.fpr,
        w1_roc_tpr = w1_global_roc.tpr,
        w1_global_auc = w1_global_roc.auc,
        w1_mean_auc = w1_mean_auc,
        w1_std_auc = w1_std_auc,

        # Shared
        acc_denom = acc_denom,
        fpr_denom = fpr_denom,
        fnr_denom = fnr_denom,
        stride_values    = stride_values,
        Nreplicas_values = Nreplicas_values,
        time_bins        = collect(time_bins),
        alpha_values     = alpha_values,
        gr_alpha_values  = gr_alpha_values,
        w1_alpha_values  = w1_alpha_values,
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
    gr_alpha_values  = gr_alpha_values,
    w1_alpha_values  = w1_alpha_values,
    time_bins        = time_bins,
    nbins_time       = nbins_time,
    ncorr            = ncorr,
    detect_bins      = detect_bins,
    nbins_detect     = nbins_detect,
    input_dim        = input_dim,
    βlims            = βlims,
    tol              = tol,
    Ngrid            = Ngrid,
    dt               = dt,
    potential_kwargs = potential_kwargs)
t1 = now()
println("Evaluation completed in $(t1 - t0)")

# --- Save JSON results ---
json_data = Dict(
    # NN
    "acc_grid"          => results.acc_grid,
    "fpr_grid"          => results.fpr_grid,
    "fnr_grid"          => results.fnr_grid,
    "auc_grid"          => results.auc_grid,
    "total_grid"        => results.total_grid,
    "acc_time"          => results.acc_time,
    "fpr_time"          => results.fpr_time,
    "fnr_time"          => results.fnr_time,
    "acc_denom"         => results.acc_denom,
    "fpr_denom"         => results.fpr_denom,
    "fnr_denom"         => results.fnr_denom,
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
    "ood_config"        => ood_config,
    # GR
    "gr_acc_grid"       => results.gr_acc_grid,
    "gr_fpr_grid"       => results.gr_fpr_grid,
    "gr_fnr_grid"       => results.gr_fnr_grid,
    "gr_auc_grid"       => results.gr_auc_grid,
    "gr_acc_time"       => results.gr_acc_time,
    "gr_fpr_time"       => results.gr_fpr_time,
    "gr_fnr_time"       => results.gr_fnr_time,
    "gr_detect_hist"    => results.gr_detect_hist,
    "gr_roc_fpr"        => results.gr_roc_fpr,
    "gr_roc_tpr"        => results.gr_roc_tpr,
    "gr_global_auc"     => results.gr_global_auc,
    "gr_mean_auc"       => results.gr_mean_auc,
    "gr_std_auc"        => results.gr_std_auc,
    "gr_alpha_values"   => results.gr_alpha_values,
    # W1
    "w1_acc_grid"       => results.w1_acc_grid,
    "w1_fpr_grid"       => results.w1_fpr_grid,
    "w1_fnr_grid"       => results.w1_fnr_grid,
    "w1_auc_grid"       => results.w1_auc_grid,
    "w1_acc_time"       => results.w1_acc_time,
    "w1_fpr_time"       => results.w1_fpr_time,
    "w1_fnr_time"       => results.w1_fnr_time,
    "w1_detect_hist"    => results.w1_detect_hist,
    "w1_roc_fpr"        => results.w1_roc_fpr,
    "w1_roc_tpr"        => results.w1_roc_tpr,
    "w1_global_auc"     => results.w1_global_auc,
    "w1_mean_auc"       => results.w1_mean_auc,
    "w1_std_auc"        => results.w1_std_auc,
    "w1_alpha_values"   => results.w1_alpha_values,
)

open("$(outprefix)_benchmark_results.json", "w") do f
    write(f, JSON.json(json_data; allownan=true))
end
println("Results saved to $(outprefix)_benchmark_results.json")
println("Benchmark complete. Run: julia --project=. experiments/plot_benchmark.jl $(outprefix)")
