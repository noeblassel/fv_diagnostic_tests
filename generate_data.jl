const dummy_val = -1.0f0 # dummy variable for masking in sequence batching

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

function sim_fv(V′, D, D′, dt, β, Nrep, nsteps ,stride, x0, rng)
    fv_frames = Vector{Float32}[]
    fv = fill(x0, Nrep)

    invβ = inv(β)
    σ = sqrt(2invβ * dt)

    fv_trace = Float64[]

    for k = 1:nsteps
        fv .+= (-D.(fv) .* V′.(fv) + invβ * D′.(fv)) * dt + σ * sqrt.(D.(fv)) .* randn(rng, Nrep)
        survived = (0 .< fv .< 1)
        n_survived = sum(survived)

        (n_survived == 0) && error("Extinction Event !")

        fv[.!(survived)] .= rand(rng, fv[survived], Nrep - sum(survived))
        append!(fv_trace, copy(fv))

        if k % stride == 0
            push!(fv_frames, copy(fv_trace))
            empty!(fv_trace)
        end
    end

    return fv_frames
end

@inline get_bin(val,minval,maxval,nbins) = 1+clamp(floor(Int,nbins*(val-minval)/(maxval-minval)),0,nbins-1)


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
    get_batch(rng; kwargs...)

Generate a dataset of Fleming–Viot simulations over random one-dimensional potentials.

This function samples random energy landscapes, runs a Fleming–Viot particle process for
several trajectories, extracts statistical features from each frame, and packages them
into a batch suitable for machine learning or statistical modeling.

# Arguments
- `rng::AbstractRNG`: a pseudorandom number generator.
# Keyword Arguments
- `tol::Float64=0.05`: tolerance for convergence in total variation distance.
- `Ngrid::Int=500`: number of grid points for the discretized generator.
- `βlims=(1.0,1.0)`: range of possible inverse temperatures β to sample from.
- `dt::Float64=1e-3`: time step for Fleming–Viot simulation.
- `lagtime=50dt`: lag time for killed semigroup propagation.
- `Nreplicas::Int=50`: number of replicas used in the Fleming–Viot ensemble.
- `input_dim::Int=64`: dimensionality of feature vectors (e.g., histogram bins).
- `feature::Function=hist_feature`: feature extraction function; one of
  `hist_feature`, `ecdf_feature`, or `tecdf_feature`.
- `ntrace::Int=5`: number of independent Fleming–Viot traces per potential.
- `ncut::Int=1`: number of random subsequences extracted from each trace.
- `npot::Int=5`: number of distinct random potentials to generate.
- `min_length::Int=2`: minimum sequence length per trace.
- `max_attempts::Int=10`: maximum number of failed simulation attempts allowed per potential
  before discarding it and moving to the next.

# Returns
A tuple `(X, Y, mask)` where:
- `X` is a 3D tensor `(input_dim, max_length, batch_size)` of feature sequences.
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
    Ngrid=500,
    βlims=(1.0,1.0),
    dt=1e-3,
    stride=50,
    Nreplicas=50,
    input_dim=64,
    feature::Function=hist_feature,
    ntrace=5,
    ncut=1,
    npot=5,
    min_length=5,
    max_attempts::Int=10) # <-- new parameter

    batch = Vector{Vector{Float32}}[]
    labels = Vector{Float32}[]
    βmin,βmax = βlims

    i = 0

    while i < npot
        W, D, W′, D′ = generate_potential(rng=rng)

        β = βmin+(βmax-βmin)*rand(rng) # sample random temperature

        L = comp_generator(W, D, β, Ngrid) # killed generator, Ngrid×Ngrid
        ν, gap = comp_qsd(W, β, L)
        P = exp(-(stride * dt) * Matrix(L)) # killed semigroup -- minus sign because comp_generator returns the negative generator

        attempts = 0                # total failed attempts for this potential
        potential_failed = false   # flag to skip to next potential if too many failures

        for j = 1:ntrace
            success = false
            fv_frames = Vector{Float32}[]
            decorr_step = -1

            while !success
                attempts += 1
                try
                    ix = rand(rng, 1:Ngrid)
                    x0 = (ix + 1) / (Ngrid + 2)
                    decorr_step = max(2,conv_tv(P, ν, ix, tol))

                    fv_frames = sim_fv(W′, D, D′, dt, β, Nreplicas, 2*decorr_step * stride,stride, x0, rng)
                    (length(fv_frames) >= min_length) && (success = true)
                catch e
                    @warn "Extinction event during Fleming-Viot simulation, retrying..." exception=(e,catch_backtrace())
                    # If we've retried too many times for this potential, give up and move on
                end

                if attempts >= max_attempts
                    @info "Exceeded maximum attempts ($max_attempts) for this potential — skipping to next potential"
                    potential_failed = true
                    break
                end
            end

            if potential_failed
                break
            end

            # process successful trace
            features = Vector{Float32}[]
            l = length(fv_frames)
            s_max = l - min_length + 1

            for f = fv_frames
                push!(features, feature(f,input_dim))
            end

            full_labels = (1:l .> decorr_step)

            for k=1:ncut
                s = rand(rng, 1:s_max)
                e_min = s + min_length - 1
                e = rand(rng, e_min:l)
                push!(batch, features[s:e])
                push!(labels, Float32.(full_labels[s:e]))
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

    batch_X = batchseq(view(batch, p), zeros(Float32, input_dim))
    batch_Y = batchseq(view(labels, p), dummy_val)

    Y = stack(batch_Y,dims=1)
    mask = (Y .!= dummy_val)

    return stack(batch_X, dims=2), Y, mask
end
