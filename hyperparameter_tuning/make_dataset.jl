using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

include(joinpath(@__DIR__, "..", "FVDiagnosticTests.jl"))
using .FVDiagnosticTests
using JLD2, Random, Statistics

"""
    generate_offline_dataset(rng; kwargs...)

Generate and store a fixed offline HP-tuning dataset.

Mirrors the outer loop of `get_batch` in `generate_data.jl` with two changes:
1. No cutting — stores the full trajectory (all nframes_target frames).
2. Stores raw particle positions, not features.

Returns a NamedTuple:
- `sequences`  — `[traj][frame]` = Vector{Float32} of Nrep*stride positions
- `labels`     — `[traj][frame]` = 0f0 or 1f0
- `Nreplicas`  — particle count for each trajectory
- `strides`    — stride for each trajectory
- `T_convs`    — T_conv for each trajectory
- `dt`         — simulation time step
"""
function generate_offline_dataset(rng;
        tol          = 0.05,
        Ngrid        = 100,
        βlims        = (1.0, 3.0),
        dt           = 1e-3,
        tau_gt       = 0.1,
        stride_lims  :: Tuple{Int,Int} = (10, 200),
        Nreplicas_lims :: Tuple{Int,Int} = (10, 200),
        ntrace       = 5,
        npot         = 100,
        min_length   = 5,
        ncorr        = 2,
        max_attempts = 10)

    sequences  = Vector{Vector{Float32}}[]  # [traj][frame] = positions
    labels     = Vector{Float32}[]
    Nreplicas_out = Int[]
    strides_out   = Int[]
    T_convs_out   = Float64[]

    βmin, βmax = βlims
    i = 0

    while i < npot
        W, D, W′, D′ = generate_potential(rng=rng)

        β = βmin + (βmax - βmin) * rand(rng)

        L    = comp_generator(W, D, β, Ngrid)
        ν, _ = comp_qsd(W, β, L)

        P_gt = exp(-tau_gt * Matrix(L))

        # opt 1: pre-sample all per-trace random values from master RNG before parallelizing
        trace_seeds     = rand(rng, UInt64, ntrace)
        trace_strides   = rand(rng, stride_lims[1]:stride_lims[2], ntrace)
        trace_Nreplicas = rand(rng, Nreplicas_lims[1]:Nreplicas_lims[2], ntrace)
        trace_ixs       = rand(rng, 1:Ngrid, ntrace)

        trace_results = Vector{Any}(undef, ntrace)

        Threads.@threads for j = 1:ntrace
            local_rng   = Random.Xoshiro(trace_seeds[j])
            stride_j    = trace_strides[j]
            Nreplicas_j = trace_Nreplicas[j]
            ix_j        = trace_ixs[j]

            local_failed   = 0
            success        = false
            decorr_step_gt = -1
            T_conv         = 0.0
            local_fv       = nothing

            while !success
                try
                    x0             = (ix_j + 1) / (Ngrid + 2)
                    decorr_step_gt = max(2, conv_tv(P_gt, ν, ix_j, tol))
                    T_conv         = decorr_step_gt * tau_gt
                    nframes_target = max(min_length, round(Int, ncorr * T_conv / (stride_j * dt)))
                    nsteps_total   = nframes_target * stride_j
                    local_fv       = sim_fv(W′, D, D′, dt, β, Nreplicas_j, nsteps_total, stride_j, x0, local_rng)

                    if length(local_fv.fv_frames) >= min_length
                        success = true
                    else
                        local_failed += 1
                    end

                catch e
                    local_failed += 1
                    @warn "Fleming-Viot run failed..." exception=(e, catch_backtrace())
                end

                if local_failed >= max_attempts
                    @info "Exceeded maximum attempts ($max_attempts) for trace $j — skipping"
                    break
                end

                ix_j = rand(local_rng, 1:Ngrid)
            end

            trace_results[j] = success ? (
                fv_frames  = local_fv.fv_frames,
                Nreplicas  = Nreplicas_j,
                stride     = stride_j,
                T_conv     = T_conv,
                success    = true,
            ) : (success = false,)
        end

        potential_failed = any(r -> !r.success, trace_results)

        if !potential_failed
            for j = 1:ntrace
                r = trace_results[j]
                !r.success && continue

                fv_frames_j = r.fv_frames
                l           = length(fv_frames_j)
                T_conv_j    = r.T_conv
                stride_j    = r.stride
                Nreplicas_j = r.Nreplicas

                full_labels = Float32.((1:l) .* (stride_j * dt) .> T_conv_j)

                push!(sequences, [Float32.(f) for f in fv_frames_j])
                push!(labels, full_labels)
                push!(Nreplicas_out, Nreplicas_j)
                push!(strides_out, stride_j)
                push!(T_convs_out, T_conv_j)
            end
        end

        if potential_failed
            continue
        end

        i += 1
        println("Generated $i/$npot potentials")
    end

    return (sequences  = sequences,
            labels     = labels,
            Nreplicas  = Nreplicas_out,
            strides    = strides_out,
            T_convs    = T_convs_out,
            dt         = dt)
end

# ── Generate and save ─────────────────────────────────────────────────────────

dataset = generate_offline_dataset(Xoshiro(2024);
    npot           = 100,
    ntrace         = 5,
    βlims          = (1.0, 3.0),
    stride_lims    = (10, 200),
    Nreplicas_lims = (10, 200),
    ncorr          = 2,
    tol            = 0.05,
    min_length     = 5,
    max_attempts   = 10)

n_traj  = length(dataset.sequences)
lengths = length.(dataset.sequences)

println("Trajectories generated : $n_traj")
println("Length range           : $(minimum(lengths)) – $(maximum(lengths))  (median $(median(lengths)))")
println("T_conv range           : $(round(minimum(dataset.T_convs); digits=2)) – $(round(maximum(dataset.T_convs); digits=2))")

# ── 80/20 train/test split by trajectory index ────────────────────────────────

perm    = randperm(Xoshiro(2025), n_traj)
n_train = round(Int, 0.8 * n_traj)
train_idx = perm[1:n_train]
test_idx  = perm[n_train+1:end]

function _subset(ds, idx)
    (sequences = ds.sequences[idx],
     labels    = ds.labels[idx],
     Nreplicas = ds.Nreplicas[idx],
     strides   = ds.strides[idx],
     T_convs   = ds.T_convs[idx],
     dt        = ds.dt)
end

train_ds = _subset(dataset, train_idx)
test_ds  = _subset(dataset, test_idx)

for (name, ds) in (("train", train_ds), ("test", test_ds))
    ls = length.(ds.sequences)
    println("$name: $(length(ds.sequences)) trajectories  " *
            "length $(minimum(ls))–$(maximum(ls))  (median $(median(ls)))  " *
            "T_conv $(round(minimum(ds.T_convs); digits=2))–$(round(maximum(ds.T_convs); digits=2))")
end

out_path = joinpath(@__DIR__, "hp_dataset.jld2")
JLD2.jldsave(out_path, train=train_ds, test=test_ds)
println("Saved → $out_path")
