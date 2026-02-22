# ========================
# Featurizer abstraction
# ========================

abstract type AbstractFeaturizer end

# ========================
# CNN Featurizer
# ========================

struct CNNFeaturizer{S<:Chain} <: AbstractFeaturizer
    encoder::S
    output_dim::Int
end

Flux.@layer CNNFeaturizer

function CNNFeaturizer(; input_dim::Int, kernel_dims::Vector{Int}, nchannels::Vector{Int}, rng=Random.GLOBAL_RNG)
    initializer = Flux.glorot_uniform(rng)
    cnn_layers = []
    in_channels = 1

    for (kernel_size, out_channels) in zip(kernel_dims, nchannels)
        pad_size = div(kernel_size - 1, 2)
        push!(cnn_layers, Conv(tuple(kernel_size), in_channels => out_channels, pad=pad_size, leakyrelu, init=initializer))
        push!(cnn_layers, MaxPool((2,)))
        in_channels = out_channels
    end

    encoder = Chain(cnn_layers...)
    output_dim = prod(Flux.outputsize(encoder, (input_dim, 1, 1)))
    return CNNFeaturizer(encoder, output_dim)
end

function (f::CNNFeaturizer)(x)
    # x: (input_dim, 1, n_samples)
    z = f.encoder(x)
    return reshape(z, f.output_dim, size(x, 3))
end

# ========================
# Deep Sets Featurizer
# ========================

struct DeepSetFeaturizer{S<:Chain, T<:Chain} <: AbstractFeaturizer
    phi::S        # element-wise MLP: R^n_meta → R^d  (applied to each particle)
    rho::T        # aggregation MLP: R^d → R^e  (applied after mean pooling)
    output_dim::Int
    n_meta::Int   # 0 → original 1-D input; ≥1 → meta[1] is mask, meta[2:end] condition φ
end

Flux.@layer DeepSetFeaturizer

function DeepSetFeaturizer(; input_dim=nothing, dims_phi::Vector{Int}, dims_rho::Vector{Int}, n_meta::Int=0, rng=Random.GLOBAL_RNG)
    initializer = Flux.glorot_uniform(rng)

    # phi: Dense(max(1,n_meta) → dims_phi[1], leakyrelu) → ... → Dense(→ dims_phi[end], leakyrelu)
    # When n_meta==0: φ takes 1 particle coordinate (original behaviour)
    # When n_meta≥1: φ takes [xᵢ; meta[2:end]] = n_meta inputs
    phi_layers = []
    in_dim = max(1, n_meta)
    for out_dim in dims_phi
        push!(phi_layers, Dense(in_dim => out_dim, leakyrelu, init=initializer))
        in_dim = out_dim
    end
    phi = Chain(phi_layers...)

    # rho: Dense(dims_phi[end] → dims_rho[1], leakyrelu) → ... → Dense(→ dims_rho[end])
    # All layers have leakyrelu except the last (linear output)
    rho_layers = []
    in_dim = last(dims_phi)
    for (i, out_dim) in enumerate(dims_rho)
        if i < length(dims_rho)
            push!(rho_layers, Dense(in_dim => out_dim, leakyrelu, init=initializer))
        else
            push!(rho_layers, Dense(in_dim => out_dim, init=initializer))
        end
        in_dim = out_dim
    end
    rho = Chain(rho_layers...)

    return DeepSetFeaturizer(phi, rho, last(dims_rho), n_meta)
end

function (f::DeepSetFeaturizer)(x)
    if f.n_meta == 0
        # Original path: x is (Nmax, 1, n_samples) — Nmax particle positions per sample
        Nmax, _, n_samples = size(x)
        x_flat = reshape(x, 1, Nmax * n_samples)
        phi_out = f.phi(x_flat)                                    # (d, Nmax * n_samples)
        d = size(phi_out, 1)
        phi_out = reshape(phi_out, d, Nmax, n_samples)
        aggregated = dropdims(mean(phi_out; dims=2); dims=2)        # (d, n_samples)
        return f.rho(aggregated)                                    # (output_dim, n_samples)
    else
        # Metadata path: x is (Nmax + n_meta, 1, n_samples)
        # meta[1] = N_actual (mask), meta[2:end] = conditioning scalars broadcast to φ
        total_dim, _, n_samples = size(x)
        Nmax = total_dim - f.n_meta

        x_p = x[1:Nmax, 1, :]                                     # (Nmax, n_samples) — padded positions
        x_m = x[(Nmax+1):end, 1, :]                               # (n_meta, n_samples) — [N_actual; cond...]

        # Valid-particle mask: slot i is valid iff i ≤ N_actual for that sample
        N_vals = x_m[1:1, :]                                       # (1, n_samples)
        valid  = Float32.(reshape(Float32.(1:Nmax), Nmax, 1) .<= N_vals)  # (Nmax, n_samples)

        # Broadcast conditioning scalars (meta[2:end]) to every particle slot
        x_cond       = x_m[2:end, :]                              # (n_meta-1, n_samples)
        x_cond_broad = repeat(reshape(x_cond, f.n_meta-1, 1, n_samples), 1, Nmax, 1)  # (n_meta-1, Nmax, n_samples)

        # φ input: [xᵢ; cond...] for each slot → (n_meta, Nmax * n_samples)
        phi_input = vcat(reshape(x_p, 1, Nmax, n_samples), x_cond_broad)  # (n_meta, Nmax, n_samples)
        phi_input = reshape(phi_input, f.n_meta, Nmax * n_samples)

        phi_out = f.phi(phi_input)                                 # (d, Nmax * n_samples)
        d = size(phi_out, 1)
        phi_out = reshape(phi_out, d, Nmax, n_samples)

        # Masked mean: zero out padded slots, divide by N_actual (clamp to 1 to avoid NaN on zero-padded batch frames)
        phi_out    = phi_out .* reshape(valid, 1, Nmax, n_samples)
        N_vals_safe = max.(N_vals, 1f0)
        aggregated = dropdims(sum(phi_out; dims=2); dims=2) ./ N_vals_safe  # (d, n_samples)

        return f.rho(aggregated)                                   # (output_dim, n_samples)
    end
end

# ========================
# Attention Featurizer
# ========================

struct AttentionBlock{A, N1, N2, F1, F2}
    attn::A     # Flux.MultiHeadAttention(d_model; nheads=n_heads)
    norm1::N1   # Flux.LayerNorm(d_model)  — pre-LN before attention
    norm2::N2   # Flux.LayerNorm(d_model)  — pre-LN before FF
    ff1::F1     # Dense(d_model => 4*d_model, gelu)
    ff2::F2     # Dense(4*d_model => d_model)
end

Flux.@layer AttentionBlock

function AttentionBlock(d_model::Int, n_heads::Int; rng=Random.GLOBAL_RNG)
    init = Flux.glorot_uniform(rng)
    AttentionBlock(
        Flux.MultiHeadAttention(d_model; nheads=n_heads, init=init),
        Flux.LayerNorm(d_model),
        Flux.LayerNorm(d_model),
        Dense(d_model => 4*d_model, gelu; init=init),
        Dense(4*d_model => d_model; init=init),
    )
end

function (b::AttentionBlock)(x)
    # x: (d_model, n_particles, n_samples)
    x_n = b.norm1(x)
    attn_out, _ = b.attn(x_n, x_n, x_n)   # self-attention; returns (out, weights)
    x = x + attn_out
    d, np, ns = size(x)
    x_n = b.norm2(x)
    ff_out = b.ff2(b.ff1(reshape(x_n, d, np * ns)))
    x + reshape(ff_out, d, np, ns)
end

struct AttentionFeaturizer{E<:Dense, T<:Chain, N} <: AbstractFeaturizer
    embedding::E     # Dense(n_meta => d_model)   — projects [pos; cond] → token
    blocks::T        # Chain of AttentionBlocks
    output_norm::N   # Flux.LayerNorm(d_model)    — final norm before pooling
    output_dim::Int  # = d_model
    n_meta::Int      # must be ≥ 2: meta[1]=N_actual mask, meta[2:end]=conditioning
    n_subsample::Int
end

Flux.@layer AttentionFeaturizer

function AttentionFeaturizer(; n_meta::Int=2, d_model::Int, n_heads::Int,
                               depth::Int, n_subsample::Int, rng=Random.GLOBAL_RNG)
    @assert d_model % n_heads == 0 "d_model must be divisible by n_heads"
    @assert n_meta >= 2
    init = Flux.glorot_uniform(rng)
    # Each token input: [particle_pos (1); conditioning_scalars (n_meta-1)] = n_meta dims
    embedding   = Dense(n_meta => d_model; init=init)
    blocks      = Chain([AttentionBlock(d_model, n_heads; rng=rng) for _ in 1:depth]...)
    output_norm = Flux.LayerNorm(d_model)
    AttentionFeaturizer(embedding, blocks, output_norm, d_model, n_meta, n_subsample)
end

function (f::AttentionFeaturizer)(x)
    # x: (Nmax + n_meta, 1, n_samples)
    total_dim, _, n_samples = size(x)
    Nmax  = total_dim - f.n_meta
    x_p   = x[1:Nmax, 1, :]          # (Nmax, n_samples) — zero-padded positions
    x_m   = x[(Nmax+1):end, 1, :]    # (n_meta, n_samples) — [N_actual; cond...]
    N_vals = x_m[1, :]               # (n_samples,) — actual particle counts
    x_cond = x_m[2:end, :]           # (n_meta-1, n_samples)

    n_sub = f.n_subsample

    # Subsample particles — wrap in Zygote.ignore so discrete sampling is stop-gradient
    x_sub, sub_N = Zygote.ignore() do
        x_sub_inner = zeros(Float32, n_sub, n_samples)
        sub_N_inner = zeros(Float32, n_samples)
        x_p_cpu = Array(x_p)
        for s in 1:n_samples
            N_s = round(Int, N_vals[s])
            k   = min(N_s, n_sub)
            idx = k < N_s ? randperm(N_s)[1:k] : collect(1:k)
            x_sub_inner[1:k, s] = x_p_cpu[idx, s]
            sub_N_inner[s] = Float32(k)
        end
        x_sub_inner, sub_N_inner
    end

    # Build token inputs: [particle_pos; cond_scalars] for each slot
    x_cond_broad = repeat(reshape(x_cond, f.n_meta - 1, 1, n_samples), 1, n_sub, 1)
    phi_input = vcat(
        reshape(x_sub, 1, n_sub, n_samples),
        x_cond_broad,
    )  # (n_meta, n_sub, n_samples)

    d = f.output_dim
    embedded = reshape(f.embedding(reshape(phi_input, f.n_meta, n_sub * n_samples)),
                       d, n_sub, n_samples)

    h = f.blocks(embedded)          # (d_model, n_sub, n_samples)
    h = f.output_norm(h)

    # Masked mean-pool over valid particle slots
    valid  = Float32.(reshape(1:n_sub, n_sub, 1) .<= reshape(sub_N, 1, n_samples))
    h      = h .* reshape(valid, 1, n_sub, n_samples)
    pooled = dropdims(sum(h; dims=2); dims=2) ./ max.(sub_N', 1f0)
    pooled  # (output_dim, n_samples)
end

# ========================
# RNNDiagnostic
# ========================

struct RNNDiagnostic{F<:AbstractFeaturizer, T<:Chain, U<:Chain}
    featurizer::F
    rnn::T
    mlp_head::U
    n_meta::Int      # number of metadata scalars appended to each frame (bypass featurizer)
end

Flux.@layer RNNDiagnostic

function RNNDiagnostic(featurizer::AbstractFeaturizer;
    dims_rnn::Vector{Int}=[64],
    dims_mlp::Vector{Int}=[64, 32],
    n_meta::Int=0,
    rng=Random.GLOBAL_RNG)

    initializer = Flux.glorot_uniform(rng)

    rnn_layers = []
    input_dim_rnn = featurizer.output_dim + n_meta   # LSTM sees featurizer output + metadata
    for output_dim_rnn in dims_rnn
        push!(rnn_layers, LSTM(input_dim_rnn => output_dim_rnn, init_kernel=initializer, init_recurrent_kernel=initializer))
        input_dim_rnn = output_dim_rnn
    end
    rnn = Chain(rnn_layers...)

    mlp_layers = []
    input_dim_mlp = last(dims_rnn)
    for output_dim_mlp in dims_mlp
        push!(mlp_layers, Dense(input_dim_mlp => output_dim_mlp, leakyrelu, init=initializer))
        input_dim_mlp = output_dim_mlp
    end
    mlp = Chain(mlp_layers..., Dense(last(dims_mlp) => 1, init=initializer))

    return RNNDiagnostic(featurizer, rnn, mlp, n_meta)
end

# Backward-compatible constructor using a CNN featurizer
function RNNDiagnostic(; input_dim::Int=64,
    cnn_kernel_dims::Vector{Int}=[5,5,5],
    cnn_nchannels::Vector{Int}=[16,32,64],
    dims_rnn::Vector{Int}=[64],
    dims_mlp::Vector{Int}=[64, 32],
    n_meta::Int=0,
    rng=Random.GLOBAL_RNG)

    featurizer = CNNFeaturizer(; input_dim=input_dim, kernel_dims=cnn_kernel_dims, nchannels=cnn_nchannels, rng=rng)
    return RNNDiagnostic(featurizer; dims_rnn=dims_rnn, dims_mlp=dims_mlp, n_meta=n_meta, rng=rng)
end

function (m::RNNDiagnostic)(x)
    @assert ndims(x) == 3 "Input size error: expected data in format (input_dim,sequence_length,batch_size)"

    total_dim, seq_len, batch_size = size(x)
    feat_n_meta = hasproperty(m.featurizer, :n_meta) ? m.featurizer.n_meta : 0

    if feat_n_meta > 0
        # Featurizer handles the full input (particles + metadata); no bypass needed
        x_flat = reshape(x, total_dim, 1, seq_len * batch_size)
        z = m.featurizer(x_flat)                       # (feat_dim, seq_len*batch)
        z = reshape(z, size(z, 1), seq_len, batch_size)
        # m.n_meta == 0 in this branch; no additional bypass
    else
        feature_dim = total_dim - m.n_meta
        x_feat = x[1:feature_dim, :, :]
        x_meta = x[(feature_dim+1):end, :, :]          # (n_meta, seq_len, batch_size)

        x_flat = reshape(x_feat, feature_dim, 1, seq_len * batch_size)
        z = m.featurizer(x_flat)                       # (feat_dim, seq_len*batch)
        z = reshape(z, size(z, 1), seq_len, batch_size)

        m.n_meta > 0 && (z = vcat(z, x_meta))          # (feat_dim + n_meta, seq_len, batch_size)
    end

    h = m.rnn(z)

    yhat = m.mlp_head(h)                               # (1, seq_len, batch_size)
    return reshape(yhat, seq_len, batch_size)
end

# ========================
# Hyperparameter structs
# ========================

struct CNNFeaturizerHyperParams
    depth::Int
    width_exponent::Int   # channels = [2^w, 2^(w+1), ..., 2^(w+depth-1)]
end

struct DeepSetFeaturizerHyperParams
    phi_depth::Int
    phi_width_exponent::Int
    rho_depth::Int
    rho_width_exponent::Int
end

struct AttentionFeaturizerHyperParams
    n_subsample::Int        # particles to subsample per frame (e.g. 128–512)
    n_heads_exponent::Int   # n_heads = 2^n_heads_exponent
    width_exponent::Int     # d_model = 2^width_exponent  (must be ≥ n_heads_exponent)
    depth::Int              # number of transformer blocks
end

mutable struct RNNDiagnosticHyperParams
    featurizer::Union{CNNFeaturizerHyperParams, DeepSetFeaturizerHyperParams,
                      AttentionFeaturizerHyperParams}
    rnn_depth::Int
    rnn_width_exponent::Int
    mlp_depth::Int
    mlp_width_exponent::Int
end

# Backward-compatible positional constructor (assumes CNN featurizer)
function RNNDiagnosticHyperParams(cnn_depth::Int, cnn_width_exponent::Int,
    rnn_depth::Int, rnn_width_exponent::Int,
    mlp_depth::Int, mlp_width_exponent::Int)
    return RNNDiagnosticHyperParams(
        CNNFeaturizerHyperParams(cnn_depth, cnn_width_exponent),
        rnn_depth, rnn_width_exponent, mlp_depth, mlp_width_exponent
    )
end

function build_featurizer(hp::CNNFeaturizerHyperParams; input_dim::Int, n_meta::Int=0, rng=Random.GLOBAL_RNG)
    nchannels = [2^i for i in hp.width_exponent:(hp.width_exponent + hp.depth - 1)]
    kernel_dims = fill(5, hp.depth)
    return CNNFeaturizer(; input_dim=input_dim, kernel_dims=kernel_dims, nchannels=nchannels, rng=rng)
end

function build_featurizer(hp::DeepSetFeaturizerHyperParams; input_dim::Int, n_meta::Int=0, rng=Random.GLOBAL_RNG)
    dims_phi = fill(2^hp.phi_width_exponent, hp.phi_depth)
    dims_rho = fill(2^hp.rho_width_exponent, hp.rho_depth)
    return DeepSetFeaturizer(; dims_phi=dims_phi, dims_rho=dims_rho, n_meta=n_meta, rng=rng)
end

function build_featurizer(hp::AttentionFeaturizerHyperParams; input_dim::Int, n_meta::Int=2, rng=Random.GLOBAL_RNG)
    width_exp = max(hp.width_exponent, hp.n_heads_exponent)  # enforce d_model ≥ n_heads
    AttentionFeaturizer(;
        n_meta      = n_meta,
        d_model     = 2^width_exp,
        n_heads     = 2^hp.n_heads_exponent,
        depth       = hp.depth,
        n_subsample = hp.n_subsample,
        rng         = rng)
end

function RNNDiagnostic(hp::RNNDiagnosticHyperParams; input_dim::Int=64, n_meta::Int=1, rng=Random.GLOBAL_RNG)
    featurizer  = build_featurizer(hp.featurizer; input_dim=input_dim, n_meta=n_meta, rng=rng)
    handles_meta_internally = (featurizer isa DeepSetFeaturizer && featurizer.n_meta > 0) ||
                              (featurizer isa AttentionFeaturizer && featurizer.n_meta > 0)
    lstm_n_meta = handles_meta_internally ? 0 : n_meta
    dims_rnn    = fill(2^hp.rnn_width_exponent, hp.rnn_depth)
    dims_mlp    = fill(2^hp.mlp_width_exponent, hp.mlp_depth)
    return RNNDiagnostic(featurizer; dims_rnn=dims_rnn, dims_mlp=dims_mlp, n_meta=lstm_n_meta, rng=rng)
end

# ========================
# RNNDiagnosticOnline
# ========================

mutable struct RNNDiagnosticOnline{F<:AbstractFeaturizer, T<:AbstractVector, U<:Chain, V}
    featurizer::F
    rnn_cells::T
    mlp_head::U
    rnn_state::V
    n_meta::Int
end

function RNNDiagnosticOnline(model::T) where {T}
    rnn_cells = [layer.cell for layer in model.rnn.layers]
    dims_rnn = [size(cell.bias, 1) ÷ 4 for cell in rnn_cells]
    rnn_state = [(zeros(Float32, d), zeros(Float32, d)) for d in dims_rnn]
    return RNNDiagnosticOnline(model.featurizer, rnn_cells, model.mlp_head, rnn_state, model.n_meta)
end

function reset_rnn_state!(model::RNNDiagnosticOnline)
    model.rnn_state = [(zero(h), zero(c)) for (c, h) in model.rnn_state]
end

function (m::RNNDiagnosticOnline)(x)
    @assert ndims(x) == 1 "RNNDiagnosticOnline only accepts 1D input vectors"
    feat_n_meta = hasproperty(m.featurizer, :n_meta) ? m.featurizer.n_meta : 0

    if feat_n_meta > 0
        # Full vector (particles + meta); featurizer handles the split
        # For online inference all particle slots are real (N_actual = total particles), no padding
        z = vec(m.featurizer(reshape(x, length(x), 1, 1)))
    else
        feature_dim = length(x) - m.n_meta
        z = vec(m.featurizer(reshape(x[1:feature_dim], feature_dim, 1, 1)))
        m.n_meta > 0 && (z = vcat(z, x[(feature_dim+1):end]))  # append metadata before first LSTM cell
    end

    for (i, cell) in enumerate(m.rnn_cells)
        (z, m.rnn_state[i]) = cell(z, m.rnn_state[i])
    end

    return m.mlp_head(z)
end

# ========================
# Checkpoint loading
# ========================

"""
Convenience function reconstructing a RNNDiagnostic model from a saved state (NamedTuple).

Supports both:
- New format: state has a `featurizer` field (CNNFeaturizer or DeepSetFeaturizer)
- Old format: state has a `cnn_encoder` field (pre-refactor checkpoints)
"""
function load_rnn_from_state(input_dim, state)
    if haskey(state, :featurizer)
        feat_state = state.featurizer
        if haskey(feat_state, :encoder)
            # CNNFeaturizer
            kernel_widths = [size(l.weight, 1) for l in feat_state.encoder.layers[1:2:end]]
            nchannels = [size(l.bias, 1) for l in feat_state.encoder.layers[1:2:end]]
            featurizer = CNNFeaturizer(; input_dim=input_dim, kernel_dims=kernel_widths, nchannels=nchannels)
        elseif haskey(feat_state, :phi)
            # DeepSetFeaturizer
            dims_phi = [size(l.bias, 1) for l in feat_state.phi.layers]
            dims_rho = [size(l.bias, 1) for l in feat_state.rho.layers]
            # Detect n_meta from φ input dimension:
            # n_meta==0 (old) → phi_input_dim==1; n_meta==k≥2 → phi_input_dim==k
            phi_input_dim = size(feat_state.phi.layers[1].weight, 2)
            feat_n_meta   = (phi_input_dim == 1) ? 0 : phi_input_dim
            featurizer = DeepSetFeaturizer(; dims_phi=dims_phi, dims_rho=dims_rho, n_meta=feat_n_meta)
        elseif haskey(feat_state, :embedding)
            # AttentionFeaturizer — recover architecture from saved state
            d_model     = size(feat_state.embedding.weight, 1)
            n_meta_in   = size(feat_state.embedding.weight, 2)
            depth       = length(feat_state.blocks.layers)
            n_heads     = feat_state.blocks.layers[1].attn.nheads
            n_subsample = feat_state.n_subsample
            featurizer  = AttentionFeaturizer(;
                n_meta=n_meta_in, d_model=d_model, n_heads=n_heads,
                depth=depth, n_subsample=n_subsample)
        else
            error("Unknown featurizer type in saved state")
        end

        rnn_widths = [size(l.cell.Wh, 2) for l in state.rnn.layers]
        mlp_widths = [size(l.bias, 1) for l in state.mlp_head.layers]
        pop!(mlp_widths)

        lstm_input_size = size(state.rnn.layers[1].cell.Wi, 2)
        n_meta = lstm_input_size - featurizer.output_dim

        model = RNNDiagnostic(featurizer; dims_rnn=rnn_widths, dims_mlp=mlp_widths, n_meta=n_meta)
        Flux.loadmodel!(model.featurizer, state.featurizer)
        Flux.loadmodel!(model.rnn, state.rnn)
        Flux.loadmodel!(model.mlp_head, state.mlp_head)
    else
        # Old format: cnn_encoder field (backward compatibility)
        kernel_widths = [size(l.weight, 1) for l in state.cnn_encoder.layers[1:2:end]]
        nchannels = [size(l.bias, 1) for l in state.cnn_encoder.layers[1:2:end]]
        featurizer = CNNFeaturizer(; input_dim=input_dim, kernel_dims=kernel_widths, nchannels=nchannels)

        rnn_widths = [size(l.cell.Wh, 2) for l in state.rnn.layers]
        mlp_widths = [size(l.bias, 1) for l in state.mlp_head.layers]
        pop!(mlp_widths)

        model = RNNDiagnostic(featurizer; dims_rnn=rnn_widths, dims_mlp=mlp_widths)
        # Load field-by-field since old state has different field names
        Flux.loadmodel!(model.featurizer.encoder, state.cnn_encoder)
        Flux.loadmodel!(model.rnn, state.rnn)
        Flux.loadmodel!(model.mlp_head, state.mlp_head)
    end

    return model
end
