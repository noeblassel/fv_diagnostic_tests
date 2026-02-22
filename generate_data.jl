const dummy_val = -1.0f0 # dummy variable for masking in sequence batching

"""
Computes the Wassestein-1 distance between two equally-sized samples
"""
function w1(X,Y)
    @assert size(X) == size(Y)
    Xs = sort(X)
    Ys = sort(Y)
    return sum(abs,Xs-Ys)/length(X)
end

"""
    generate_potential(; kwargs...)

Generate a random 1D potential `V(x)` on (0,1) together with a diffusion profile `D(x)`.

Returns `(V, D)` where:
- `V` is the potential (shifted so min(V)=0), a univariate function
- `D` is the diffusion coefficient profile, a univariate positive function
"""
function generate_potential(; n_points::Int=1000,
    clims=(0.15, 0.85),
    δlims=(0.0, 0.2),
    κlims=(0.2, 0.6),
    logσlims=(-4, -1),
    mmax=6,
    poneWell=0.0,
    pconstD=0.5,
    rng::AbstractRNG=Random.default_rng())

    x = range(0, 1, length=n_points)

    δmin, δmax = δlims
    cmin, cmax = clims
    c_mw = cmin + (cmax - cmin) * rand(rng)
    κmin, κmax = κlims
    κ = κmin + rand(rng) * (κmax - κmin)

    main_well(x) = 1.0 - exp(-(x - c_mw)^2 / 2κ^2) # main potential well
    main_well′(x) = (x - c_mw) * exp(-(x - c_mw)^2 / 2κ^2) / κ^2 # derivative

    # add perturbations
    m_V = ((rand(rng) < poneWell) ? 0 : rand(rng, 1:mmax))
    c_V = rand(rng, m_V)
    h_V = (δmin .+ (δmax - δmin) * rand(rng, m_V))
    κ_V = (κmin .+ rand(rng, m_V) * (κmax - κmin)) / m_V # division by zero when m_V=0 is fine

    perturbation(x) = sum((h_V[i] * exp(-(x - c_V[i])^2 / 2κ_V[i]^2) for i = 1:m_V), init=0.0)
    perturbation′(x) = -sum((h_V[i] * ((x - c_V[i])) * exp(-(x - c_V[i])^2 / 2κ_V[i]^2) / κ_V[i]^2 for i = 1:m_V), init=0.0)

    δE = δmin + (δmax - δmin) * rand(rng) # potential difference between boundary points

    # Normalize V
    V0, V1 = main_well(0) + perturbation(0), main_well(1) + perturbation(1)
    b = (V0 < V1) ? (V0 - V1 + δE) : (V0 - V1 - δE)
    X = range(0, 1, n_points)
    a = -minimum(b * X + main_well.(X) + perturbation.(X))
    tmp_max = maximum(@. a + b * X + main_well(X) + perturbation(X))

    V(x) = (a + b * x + main_well(x) + perturbation(x)) / tmp_max
    V′(x) = (b + main_well′(x) + perturbation′(x)) / tmp_max

    # Fix diffusion profile

    logσmin, logσmax = logσlims
    logσscale = logσmin + rand(rng) * (logσmax - logσmin) # "degree of metastability"
    σ = exp(logσscale)

    m_D = ((rand(rng) < pconstD) ? 0 : rand(rng, 1:mmax))
    c_D = rand(rng, m_D)
    h_D = σ * randn(rng, m_D)
    κ_D = (κmin .+ rand(rng, m_D) * (κmax - κmin)) / m_D

    D_perturbation(x) = sum((h_D[i] * exp(-(x - c_D[i])^2 / 2κ_D[i]^2) for i = 1:m_D), init=0.0)

    d = minimum(D_perturbation.(X))
    D(x) = σ + D_perturbation(x) - d
    D′(x) = -sum((h_D[i] * (x - c_D[i]) * exp(-(x - c_D[i])^2 / 2κ_D[i]^2) / κ_D[i]^2 for i = 1:m_D), init=0.0)

    return V, D, V′, D′
end

"""
    Computes the generator and jump rates for a Gibbs-stationary jump process associated to an energy function V and a diffusion coefficient D

    Arguments

    F : energy
    a : diffusion coefficient
    β : inverse of kT, with T the temperature
    N : number of grid points in jump process discretization

    Returns

    L: a sparse matrix, giving the negative of the generator for the killed jump process on a N×N regular lattice inside (0,1), excluding cemetery boundary points {0,1}
"""
function comp_generator(F, a, β, N)
    X = range(0, 1, N + 2) # grid on [0,1]
    dx = inv(N + 1)

    V = F.(X)
    D = a.(X)

    @inline prev(i) = (i > 1) ? i - 1 : i # reflection at the boundary for the non-killed process
    @inline next(i) = (i < N + 2) ? i + 1 : i

    rate(i, j) = exp(-β * (V[j] - V[i]) / 2) * (D[i] + D[j]) / 2 # for |i-j|=1
    rows = vcat(([i, i, i] for i = 1:N+2)...)
    cols = vcat(([i, prev(i), next(i)] for i = 1:N+2)...)

    factor = inv(β * dx^2) # for consistent approximation of the generator
    vals = vcat((factor * [rate(i, prev(i)) + rate(i, next(i)), -rate(i, prev(i)), -rate(i, next(i))] for i = 1:N+2)...)

    L = sparse(rows, cols, vals)

    return L[2:N+1, 2:N+1] # return negative killed generator
end


"""
    Computes the QSD associated to the previous jump process, with hard killing at the boundary {0,1}.

    Arguments:
"""
function comp_qsd(V, β, L)
    λs, us = eigs(L', sigma=0.0, nev=2, which=:LR, tol=eps()) # returns eigenvalues of least magnitude
    λs = real.(λs)
    us = real.(us)
    ν = abs.(us[:, 1])
    ν /= sum(ν)

    @assert (0<λs[1]<λs[2])

    return ν, λs[2] - λs[1]
end

"""
    Computes the sequence of total variation distances between the QSD ν and the non-linear Fokker-Planck solution

    Argument: 
    P = a matrix, the killed semigroup for an (unspecified) lag time
    ν = the associated QSD
    ix = the index of the initial Dirac mass on the discretization grid

    Returns:
    errs = the sequence of total variation distances at multiples of the lag time
"""
function tv_trace(P, ν, ix; tv_tol=0.02, maxiter=1000)
    N = size(P, 1)
    l = zeros(N)
    l[ix] = 1

    errs = Float64[]

    for k = 1:maxiter
        push!(errs, sum(abs, l - ν))

        if last(errs) < tv_tol
            return errs
        end

        l = P'l
        l /= sum(l) # normalize
    end

    return errs
end

"""
    Computes the first step at which the total variation distances between the QSD ν and the non-linear Fokker-Planck solution drops below some threshold

    Argument: 
    P = a matrix, the killed semigroup for an (unspecified) lag time
    ν = the associated QSD
    ix = the index of the initial Dirac mass on the discretization grid
    tv_tol = the tolerance threshold

    Returns:
    k = the step at which convergence to the threshold occurs
"""
function conv_tv(P, ν, ix, tv_tol=0.1; max_iter=1000)
    N = size(P, 1)
    l = zeros(N)
    l[ix] = 1

    k = 0
    while true
        if sum(abs, l - ν) < tv_tol
            return k
        end

        l = P'l
        l /= sum(l) # normalize
        k += 1

        (k > max_iter) && return -1
    end
end

function sim_fv(V′, D, D′, dt, β, Nrep, nsteps, stride, x0, rng; naive::Bool=false)
    fv_frames = Vector{Float32}[]
    fv = fill(x0, Nrep)

    invβ = inv(β)
    σ = sqrt(2invβ * dt)

    # opt 4: pre-allocate buffers — avoids stride*nframes small heap allocations
    fv_trace_buf = Vector{Float64}(undef, Nrep * stride)
    survived_buf = Vector{Bool}(undef, Nrep)
    Dv           = Vector{Float64}(undef, Nrep)

    gr_hist  = Float64[]
    w1_hist  = Float64[Inf]

    sum_lin_gr = zeros(Nrep)
    sum_sq_gr  = zeros(Nrep)

    for k = 1:nsteps
        @. Dv = D(fv)                                                          # opt 2: cache D(fv), avoids double evaluation
        fv .+= (-Dv .* V′.(fv) + invβ * D′.(fv)) * dt + σ * sqrt.(Dv) .* randn(rng, Nrep)

        @. survived_buf = (0 < fv < 1)                                         # opt 4: reuse pre-allocated Bool buffer
        n_survived = count(survived_buf)

        (n_survived == 0) && error("Extinction Event !")

        fv[.!(survived_buf)] .= rand(rng, fv[survived_buf], Nrep - n_survived)

        # opt 4: write into pre-allocated window buffer instead of append!
        step_in_window = mod1(k, stride)
        fv_trace_buf[(step_in_window - 1)*Nrep + 1 : step_in_window*Nrep] .= fv

        if naive                                                                # opt 3: skip GR/W1 accumulation when not needed
            sum_lin_gr += fv
            sum_sq_gr  += fv .^ 2
        end

        if k % stride == 0
            if naive
                push!(gr_hist, (sum(sum_sq_gr) - sum(sum_lin_gr)^2/(Nrep*k)) / (sum(sum_sq_gr - (sum_lin_gr .^ 2)/k)) - 1)

                if !isempty(fv_frames)
                    push!(w1_hist, w1(last(fv_frames), fv_trace_buf))
                end
            end

            push!(fv_frames, copy(fv_trace_buf))                               # opt 4: one allocation per frame instead of stride allocations
        end
    end

    if naive
        w1_hist /= w1_hist[2]                                                  # normalize W1-distance by first decrement
    end

    return (fv_frames=fv_frames, gr_history=gr_hist, w1_history=w1_hist)
end

@inline get_bin(val,minval,maxval,nbins) = 1+clamp(floor(Int,nbins*(val-minval)/(maxval-minval)),0,nbins-1)

function raw_feature(pts,dim_feature)
    return Float32.(pts)
end

"""
histogram feature representation
"""
function hist_feature(pts,dim_feature)
        X = zeros(Float32,dim_feature)
        dx = 1.0f0 / dim_feature
        for p in pts
            bin = get_bin(p,0.0,1.0,dim_feature)
            X[bin] += 1.0f0
        end
        X /= (sum(X)*dx)
    return X
end

"""
empirical CDF feature representation
"""
function ecdf_feature(pts,dim_feature)
    X = zeros(Float32,dim_feature)
    G = ecdf(pts)
    dx = 1.0f0 / dim_feature

    for i = 1:dim_feature
        x = (i - 0.5f0) * dx
        X[i] = Float32(G(x))
    end
    return X 
end

"""
"tilted" empirical CDF feature representation
"""
function tecdf_feature(pts,dim_feature)
    X = zeros(Float32,dim_feature)
    G = ecdf(pts)
    dx = 1.0f0 / dim_feature

    for i = 1:dim_feature
        x = (i - 0.5f0) * dx
        X[i] = Float32(G(x))-x # avoid division by zero
    end
    return X
end

"""
Deep Sets feature representation — returns particle positions zero-padded to exactly Nmax.

Fills the first min(N, Nmax) entries with the actual positions; remaining slots are zero.
No resampling is performed; the caller is responsible for setting input_dim = Nreplicas_max × stride_max.
"""
function deep_set_feature(pts, Nmax)
    N = length(pts)
    out = zeros(Float32, Nmax)
    out[1:min(N, Nmax)] .= Float32.(pts[1:min(N, Nmax)])
    return out
end

"""
    get_batch(rng; kwargs...)

Generate a dataset of Fleming–Viot simulations over random one-dimensional potentials.

This function samples random energy landscapes, runs a Fleming–Viot particle process for
several trajectories, extracts statistical features from each frame, and packages them
into a batch suitable for machine learning or statistical modeling.

# Arguments
- `rng::AbstractRNG`: a pseudorandom number generator.
# Keyword Arguments
- `tol::Float64=0.05`: tolerance for convergence in total variation distance.
- `Ngrid::Int=100`: number of grid points for the discretized generator.
- `βlims=(1.0,1.0)`: range of possible inverse temperatures β to sample from.
- `dt::Float64=1e-3`: time step for Fleming–Viot simulation.
- `tau_gt::Float64=0.1`: fixed lag time for the ground-truth killed semigroup `P_gt = exp(-tau_gt * L)`;
  computed once per potential, decoupled from simulation stride.
- `stride_lims::Tuple{Int,Int}=(50,50)`: range of simulation sampling strides per trace;
  controls ONLY sampling frequency, not ground-truth accuracy.
- `Nreplicas_lims::Tuple{Int,Int}=(50,50)`: range of replica counts to sample per trace.
- `input_dim::Int=64`: dimensionality of feature vectors (e.g., histogram bins).
  For `deep_set_feature`, set `input_dim = Nreplicas_lims[2] × stride_lims[2]` (maximum
  particle slots × stride) so the padded raw-position vector fits without truncation.
- `feature::Function=hist_feature`: feature extraction function; one of
  `hist_feature`, `ecdf_feature`, `tecdf_feature`, or `deep_set_feature`.
- `n_meta::Int=1`: number of metadata scalars appended to each frame vector.
  - `n_meta=1` (default): appends `√(N·τ)` — legacy behaviour for CNN/histogram models.
  - `n_meta=3`: appends `[N_actual, Nreplicas, τ]` — use with `deep_set_feature` and the
    new `DeepSetFeaturizer(n_meta=3)` that performs masked mean-pooling instead of resampling.
- `ntrace::Int=5`: number of independent Fleming–Viot traces per potential.
- `ncut::Int=1`: number of random subsequences extracted from each trace.
- `npot::Int=5`: number of distinct random potentials to generate.
- `min_length::Int=2`: minimum sequence length per trace.
- `max_attempts::Int=10`: maximum number of failed simulation attempts allowed per potential
  before discarding it and moving to the next.

# Returns
A tuple `(X, Y, mask)` where:
- `X` is a 3D tensor `(input_dim + n_meta, max_length, batch_size)` of feature sequences.
- `Y` is a 2D tensor `(max_length, batch_size)` of binary labels indicating decorrelation.
- `mask` is a boolean matrix of the same shape as `Y`, where `true` marks valid entries.

# Notes
- The function automatically skips problematic potentials after `max_attempts` failed
  attempts to prevent infinite loops.
- Each batch element corresponds to one random trajectory segment.
- Randomness is controlled via the `rng` argument.
"""

function get_batch(rng;
    tol=0.05,
    Ngrid=100,
    βlims=(1.0,1.0),
    dt=1e-3,
    tau_gt::Float64=0.1,
    stride_lims::Tuple{Int,Int}=(50,50),
    Nreplicas_lims::Tuple{Int,Int}=(50,50),
    input_dim=64,
    feature::Function=hist_feature,
    n_meta::Int=1,
    ntrace=5,
    ncut=1,
    npot=5,
    min_length=5,
    ncorr=2,
    max_attempts::Int=10,
    naive = false)

    batch = Vector{Vector{Float32}}[]
    labels = Vector{Float32}[]
    βmin,βmax = βlims

    i = 0

    naive && (input_dim = 2)

    while i < npot
        W, D, W′, D′ = generate_potential(rng=rng)

        β = βmin+(βmax-βmin)*rand(rng) # sample random temperature

        L = comp_generator(W, D, β, Ngrid) # killed generator, Ngrid×Ngrid
        ν, gap = comp_qsd(W, β, L)

        P_gt = exp(-tau_gt * Matrix(L)) # killed semigroup at lag time tau_gt -- minus sign because comp_generator returns the negative generator

        potential_failed = false   # flag to skip to next potential if too many failures

        # opt 1: pre-sample all per-trace random values from master RNG before parallelizing
        trace_seeds     = rand(rng, UInt64, ntrace)
        trace_strides   = rand(rng, stride_lims[1]:stride_lims[2], ntrace)
        trace_Nreplicas = rand(rng, Nreplicas_lims[1]:Nreplicas_lims[2], ntrace)
        trace_ixs       = rand(rng, 1:Ngrid, ntrace)

        trace_results = Vector{Any}(undef, ntrace)

        # opt 1: ntrace FV simulations for this potential are embarrassingly parallel
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
                    local_fv       = sim_fv(W′, D, D′, dt, β, Nreplicas_j, nsteps_total, stride_j, x0, local_rng; naive=naive)

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
                fv_frames      = local_fv.fv_frames,
                gr_history     = local_fv.gr_history,
                w1_history     = local_fv.w1_history,
                Nreplicas      = Nreplicas_j,
                stride         = stride_j,
                decorr_step_gt = decorr_step_gt,
                T_conv         = T_conv,
                success        = true,
            ) : (success = false,)
        end

        # Skip potential if any trace failed to produce valid data
        potential_failed = any(r -> !r.success, trace_results)

        if !potential_failed
            # Collate successful trace results serially
            for j = 1:ntrace
                r = trace_results[j]
                !r.success && continue

                fv_frames = r.fv_frames
                gr_hist   = r.gr_history
                w1_hist   = r.w1_history
                Nreplicas = r.Nreplicas
                stride    = r.stride
                T_conv    = r.T_conv
                l         = length(fv_frames)

                features = Vector{Float32}[]

                if !naive
                    for f = fv_frames
                        push!(features, feature(f, input_dim))
                    end
                else
                    for k = 1:size(gr_hist, 1)
                        push!(features, Float32.([gr_hist[k], w1_hist[k]]))
                    end
                end

                # Append metadata scalars to every frame
                if n_meta == 2
                    N_actual  = Nreplicas * stride
                    meta_vals = Float32.([N_actual, sqrt(N_actual * dt)])
                    features  = [vcat(f, meta_vals) for f in features]
                else   # n_meta == 1 — legacy single scalar, backward-compatible
                    meta_val = Float32(sqrt(Nreplicas * stride * dt))
                    features = [vcat(f, [meta_val]) for f in features]
                end

                full_labels = (1:l) .* (stride * dt) .> T_conv

                for k = 1:ncut
                    # Start-anchored, uniformly-sampled length: keeps positive-label
                    # fraction distribution constant across batches
                    α_min = min_length / l
                    α     = α_min + rand(rng) * (1.0 - α_min)
                    len   = clamp(round(Int, α * l), min_length, l)
                    push!(batch, features[1:len])
                    push!(labels, Float32.(full_labels[1:len]))
                end

                if ncut == 0
                    push!(batch, features)
                    push!(labels, Float32.(full_labels))
                end
            end
        end

        if potential_failed
            continue
        end

        i += 1
    end

    nsamples = length(batch)
    if nsamples == 0
        error("No samples collected — consider increasing npot, ntrace, or max_attempts")
    end
    p = randperm(rng, nsamples)

    batch_X = batchseq(view(batch, p), zeros(Float32, input_dim + n_meta))
    batch_Y = batchseq(view(labels, p), dummy_val)

    Y = stack(batch_Y,dims=1)
    mask = (Y .!= dummy_val)

    return stack(batch_X, dims=2), Y, mask
end