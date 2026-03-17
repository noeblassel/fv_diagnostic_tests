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
    model, input_dim = load_rnn_from_state(64, state)
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
tv_alpha_values = Float64[0.01, 0.02, 0.05, 0.1, 0.2, 0.5]

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

# --- Method abstraction ---
struct MethodConfig
    json_prefix::String        # "" for NN, "gr_" for GR, etc.
    tv_detect_prefix::String   # "nn_" for NN, same as json_prefix for others
    alpha_values::Vector{Float64}
    roc_thresholds::Vector{Float32}
    naive_channel::Int         # 0 = NN (uses Yhat_prob), 1-3 = X_naive channel index
    higher_is_positive::Bool   # true: score > α → converged; false: score < α → converged
end

function init_accum(ns, nn, na, nbins_time, nbins_detect)
    (tp = zeros(Int, ns, nn, na), fp = zeros(Int, ns, nn, na),
     tn = zeros(Int, ns, nn, na), fn = zeros(Int, ns, nn, na),
     acc_numer = zeros(nbins_time, na), fpr_numer = zeros(nbins_time, na),
     fnr_numer = zeros(nbins_time, na),
     roc_samples = [Tuple{Float32,Float32}[] for _ in 1:ns, _ in 1:nn],
     roc_samples_all = Tuple{Float32,Float32}[],
     detect_hist = zeros(Int, nbins_detect, na),
     detect_count = zeros(Int, ns, nn, na),
     tv_detect_sum = zeros(ns, nn, na),
     tv_detect_count = zeros(Int, ns, nn, na))
end

@inline function get_method_score(cfg::MethodConfig, k, j, Yhat_prob, X_naive)
    cfg.naive_channel == 0 ? Yhat_prob[k, j] : X_naive[cfg.naive_channel, k, j]
end

@inline function is_positive(cfg::MethodConfig, score, threshold)
    cfg.higher_is_positive ? (score > threshold) : (score < threshold)
end

function find_detection(cfg::MethodConfig, α, traj_len, j, Yhat_prob, X_naive)
    if cfg.naive_channel == 0
        findfirst(>(α), @view Yhat_prob[1:traj_len, j])
    else
        findfirst(<(α), @view X_naive[cfg.naive_channel, 1:traj_len, j])
    end
end

function evaluate_grid(model, rng;
                       stride_values, Nreplicas_values,
                       num_batches, npot_per_batch, ntrace_per_pot,
                       methods::Vector{MethodConfig},
                       time_bins, nbins_time, ncorr,
                       detect_bins, nbins_detect,
                       input_dim=64, βlims=(1.0, 3.0),
                       tol=0.05, Ngrid=100, dt=1e-3,
                       potential_kwargs::NamedTuple=NamedTuple())

    ns = length(stride_values)
    nn = length(Nreplicas_values)
    accs = [init_accum(ns, nn, length(m.alpha_values), nbins_time, nbins_detect) for m in methods]

    traj_count_grid = zeros(Int, ns, nn)
    acc_denom = zeros(nbins_time)
    fpr_denom = zeros(nbins_time)
    fnr_denom = zeros(nbins_time)

    prog = Progress(ns * nn * num_batches; desc="Evaluating grid: ")

    for (si, s) in enumerate(stride_values)
        for (ni, N) in enumerate(Nreplicas_values)
            for b in 1:num_batches
                X, Y, mask, X_naive = get_batch(rng;
                    input_dim=input_dim, stride_lims=(s, s), Nreplicas_lims=(N, N),
                    ncut=0, ncorr=ncorr, npot=npot_per_batch, ntrace=ntrace_per_pot,
                    feature=hist_feature, βlims=βlims,
                    tol=tol, Ngrid=Ngrid, dt=dt, true_tv=true,
                    potential_kwargs=potential_kwargs)

                Flux.reset!(model)
                Yhat_prob = Flux.σ(model(X))
                ntraj = size(Y, 2)

                for j in 1:ntraj
                    traj_len = sum(mask[:, j])
                    traj_len == 0 && continue
                    decorr_step = findfirst(==(1.0f0), Y[:, j])
                    decorr_step === nothing && continue
                    traj_count_grid[si, ni] += 1

                    # --- Frame-level confusion matrix + time-resolved rates ---
                    for k in 1:traj_len
                        true_pos = Y[k, j] == 1.0f0
                        tbin = get_bin(k / decorr_step, first(time_bins), last(time_bins), nbins_time)

                        acc_denom[tbin] += 1
                        if !true_pos; fpr_denom[tbin] += 1; else; fnr_denom[tbin] += 1; end

                        for (cfg, acc) in zip(methods, accs)
                            score = get_method_score(cfg, k, j, Yhat_prob, X_naive)
                            for (ai, α) in enumerate(cfg.alpha_values)
                                pred_pos = is_positive(cfg, score, α)
                                if pred_pos && true_pos;       acc.tp[si, ni, ai] += 1
                                elseif pred_pos && !true_pos;  acc.fp[si, ni, ai] += 1
                                elseif !pred_pos && true_pos;  acc.fn[si, ni, ai] += 1
                                else;                          acc.tn[si, ni, ai] += 1; end
                                acc.acc_numer[tbin, ai] += (pred_pos == true_pos) ? 1 : 0
                                if !true_pos
                                    acc.fpr_numer[tbin, ai] += pred_pos ? 1 : 0
                                else
                                    acc.fnr_numer[tbin, ai] += !pred_pos ? 1 : 0
                                end
                            end
                        end
                    end

                    # --- ROC: one independent sample per trajectory ---
                    n_roc = min(traj_len, 2 * decorr_step)
                    k_roc = rand(rng, 1:n_roc)
                    true_label = Y[k_roc, j]
                    for (cfg, acc) in zip(methods, accs)
                        raw = get_method_score(cfg, k_roc, j, Yhat_prob, X_naive)
                        s = cfg.higher_is_positive ? Float32(raw) : -Float32(raw)
                        push!(acc.roc_samples[si, ni], (s, true_label))
                        push!(acc.roc_samples_all, (s, true_label))
                    end

                    # --- Detection time + TV at detection ---
                    for (cfg, acc) in zip(methods, accs)
                        for (ai, α) in enumerate(cfg.alpha_values)
                            detect_frame = find_detection(cfg, α, traj_len, j, Yhat_prob, X_naive)
                            if detect_frame !== nothing
                                acc.detect_count[si, ni, ai] += 1
                                ratio = detect_frame / decorr_step
                                tbin = get_bin(ratio, first(detect_bins), last(detect_bins), nbins_detect)
                                acc.detect_hist[tbin, ai] += 1
                                acc.tv_detect_sum[si, ni, ai] += Float64(X_naive[4, detect_frame, j])
                                acc.tv_detect_count[si, ni, ai] += 1
                            end
                        end
                    end
                end

                next!(prog)
            end
        end
    end

    # --- Derive metrics and build JSON dict ---
    json_data = Dict{String,Any}(
        "stride_values"    => stride_values,
        "Nreplicas_values" => Nreplicas_values,
        "time_bins"        => collect(time_bins),
        "detect_bins"      => collect(detect_bins),
        "acc_denom"        => acc_denom,
        "fpr_denom"        => fpr_denom,
        "fnr_denom"        => fnr_denom,
        "traj_count_grid"  => traj_count_grid,
    )

    for (cfg, acc) in zip(methods, accs)
        p  = cfg.json_prefix
        tp = cfg.tv_detect_prefix

        # Grid metrics
        total    = acc.tp .+ acc.fp .+ acc.tn .+ acc.fn
        acc_grid = (acc.tp .+ acc.tn) ./ max.(total, 1)
        fpr_grid = acc.fp ./ max.(acc.fp .+ acc.tn, 1)
        fnr_grid = acc.fn ./ max.(acc.fn .+ acc.tp, 1)

        # Time-resolved rates
        acc_time = acc.acc_numer ./ max.(acc_denom, 1)
        fpr_time = acc.fpr_numer ./ max.(fpr_denom, 1)
        fnr_time = acc.fnr_numer ./ max.(fnr_denom, 1)

        # Per-cell AUC
        auc_grid = zeros(ns, nn)
        for si in 1:ns, ni in 1:nn
            samples = acc.roc_samples[si, ni]
            if !isempty(samples)
                auc_grid[si, ni] = compute_roc(
                    [s[1] for s in samples], [s[2] for s in samples];
                    thresholds=cfg.roc_thresholds).auc
            else
                auc_grid[si, ni] = NaN
            end
        end

        # Global ROC
        global_roc = compute_roc(
            [s[1] for s in acc.roc_samples_all],
            [s[2] for s in acc.roc_samples_all];
            thresholds=cfg.roc_thresholds)

        # AUC statistics
        valid_aucs = filter(!isnan, vec(auc_grid))
        mean_auc_val = isempty(valid_aucs) ? NaN : mean(valid_aucs)
        std_auc_val  = isempty(valid_aucs) ? NaN : std(valid_aucs)

        # TV at detection
        tv_at_detect = acc.tv_detect_sum ./ max.(acc.tv_detect_count, 1)

        # Write to JSON dict
        json_data["$(p)acc_grid"]          = acc_grid
        json_data["$(p)fpr_grid"]          = fpr_grid
        json_data["$(p)fnr_grid"]          = fnr_grid
        json_data["$(p)auc_grid"]          = auc_grid
        json_data["$(p)acc_time"]          = acc_time
        json_data["$(p)fpr_time"]          = fpr_time
        json_data["$(p)fnr_time"]          = fnr_time
        json_data["$(p)detect_hist"]       = acc.detect_hist
        json_data["$(p)roc_fpr"]           = global_roc.fpr
        json_data["$(p)roc_tpr"]           = global_roc.tpr
        json_data["$(p)global_auc"]        = global_roc.auc
        json_data["$(p)mean_auc"]          = mean_auc_val
        json_data["$(p)std_auc"]           = std_auc_val
        json_data["$(tp)tv_at_detect"]     = tv_at_detect
        json_data["$(p)detect_count_grid"] = acc.detect_count

        # NN-specific extra keys (no prefix)
        if p == ""
            json_data["total_grid"]     = total
            json_data["roc_thresholds"] = collect(cfg.roc_thresholds)
            json_data["alpha_values"]   = cfg.alpha_values
        else
            json_data["$(p)alpha_values"] = cfg.alpha_values
        end
    end

    return json_data
end

# --- Define methods ---
methods = MethodConfig[
    MethodConfig("",    "nn_", Float64.(alpha_values),    collect(0.01f0:0.01f0:0.99f0), 0, true),
    MethodConfig("gr_", "gr_", Float64.(gr_alpha_values), collect(-0.2f0:0.002f0:0.0f0), 1, false),
    MethodConfig("w1_", "w1_", Float64.(w1_alpha_values), collect(-1.0f0:0.005f0:0.0f0), 2, false),
    MethodConfig("tv_", "tv_", Float64.(tv_alpha_values), collect(-2.0f0:0.01f0:0.0f0),  3, false),
]

# --- Run evaluation ---
println("Starting benchmark evaluation...")
t0 = now()
json_data = evaluate_grid(model, rng;
    stride_values    = stride_values,
    Nreplicas_values = Nreplicas_values,
    num_batches      = num_batches,
    npot_per_batch   = npot_per_batch,
    ntrace_per_pot   = ntrace_per_pot,
    methods          = methods,
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

json_data["ood_config"] = ood_config

open("$(outprefix)_benchmark_results.json", "w") do f
    write(f, JSON.json(json_data; allownan=true))
end
println("Results saved to $(outprefix)_benchmark_results.json")
println("Benchmark complete. Run: julia --project=. experiments/plot_benchmark.jl $(outprefix)")
