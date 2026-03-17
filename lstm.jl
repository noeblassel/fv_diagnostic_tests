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
    feature_dim = total_dim - m.n_meta
    x_feat = x[1:feature_dim, :, :]
    x_meta = x[(feature_dim+1):end, :, :]          # (n_meta, seq_len, batch_size)

    x_flat = reshape(x_feat, feature_dim, 1, seq_len * batch_size)
    z = m.featurizer(x_flat)                       # (feat_dim, seq_len*batch)
    z = reshape(z, size(z, 1), seq_len, batch_size)

    m.n_meta > 0 && (z = vcat(z, x_meta))          # (feat_dim + n_meta, seq_len, batch_size)

    h = m.rnn(z)

    yhat = m.mlp_head(h)                               # (1, seq_len, batch_size)
    return reshape(yhat, seq_len, batch_size)
end

# ========================
# Hyperparameter structs
# ========================

struct CNNFeaturizerHyperParams
    input_dim_exponent::Int  # input_dim = 2^e, e.g. 5→32, 6→64, 7→128, 8→256
    depth::Int
    width_exponent::Int      # channels = [2^w, 2^(w+1), ..., 2^(w+depth-1)]
end

mutable struct RNNDiagnosticHyperParams
    featurizer::CNNFeaturizerHyperParams
    rnn_depth::Int
    rnn_width_exponent::Int
    mlp_depth::Int
    mlp_width_exponent::Int
end

# Backward-compatible positional constructor (assumes CNN featurizer, default input_dim=64)
function RNNDiagnosticHyperParams(cnn_depth::Int, cnn_width_exponent::Int,
    rnn_depth::Int, rnn_width_exponent::Int,
    mlp_depth::Int, mlp_width_exponent::Int;
    input_dim_exponent::Int = 6)
    return RNNDiagnosticHyperParams(
        CNNFeaturizerHyperParams(input_dim_exponent, cnn_depth, cnn_width_exponent),
        rnn_depth, rnn_width_exponent, mlp_depth, mlp_width_exponent
    )
end

function build_featurizer(hp::CNNFeaturizerHyperParams; n_meta::Int=0, rng=Random.GLOBAL_RNG)
    input_dim   = 2^hp.input_dim_exponent
    nchannels   = [2^i for i in hp.width_exponent:(hp.width_exponent + hp.depth - 1)]
    kernel_dims = fill(5, hp.depth)
    return CNNFeaturizer(; input_dim=input_dim, kernel_dims=kernel_dims, nchannels=nchannels, rng=rng)
end

function RNNDiagnostic(hp::RNNDiagnosticHyperParams; n_meta::Int=1, rng=Random.GLOBAL_RNG)
    featurizer = build_featurizer(hp.featurizer; n_meta=n_meta, rng=rng)
    dims_rnn   = fill(2^hp.rnn_width_exponent, hp.rnn_depth)
    dims_mlp   = fill(2^hp.mlp_width_exponent, hp.mlp_depth)
    return RNNDiagnostic(featurizer; dims_rnn=dims_rnn, dims_mlp=dims_mlp, n_meta=n_meta, rng=rng)
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

Flux.@layer RNNDiagnosticOnline

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
    feature_dim = length(x) - m.n_meta
    z = vec(m.featurizer(reshape(x[1:feature_dim], feature_dim, 1, 1)))
    m.n_meta > 0 && (z = vcat(z, x[(feature_dim+1):end]))  # append metadata before first LSTM cell

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
- New format: state has a `featurizer` field (CNNFeaturizer)
- Old format: state has a `cnn_encoder` field (pre-refactor checkpoints)
"""
function load_rnn_from_state(input_dim, state)
    if haskey(state, :featurizer)
        feat_state = state.featurizer
        kernel_widths = [size(l.weight, 1) for l in feat_state.encoder.layers[1:2:end]]
        nchannels = [size(l.bias, 1) for l in feat_state.encoder.layers[1:2:end]]

        rnn_widths = [size(l.cell.Wh, 2) for l in state.rnn.layers]
        mlp_widths = [size(l.bias, 1) for l in state.mlp_head.layers]
        pop!(mlp_widths)

        # Infer correct input_dim from the LSTM input weight dimensions.
        # Each Conv(same-pad) + MaxPool(2) layer halves the spatial dim,
        # so output_dim = (input_dim ÷ 2^depth) * nchannels[end].
        # lstm_input_size = output_dim + n_meta.
        lstm_input_size = size(state.rnn.layers[1].cell.Wi, 2)
        depth = length(nchannels)
        n_meta = -1
        for n_meta_try in (1, 0)
            target = lstm_input_size - n_meta_try
            inferred = target * 2^depth ÷ nchannels[end]
            if inferred > 0 && ispow2(inferred) &&
               (inferred ÷ 2^depth) * nchannels[end] == target
                input_dim = inferred
                n_meta = n_meta_try
                break
            end
        end
        n_meta >= 0 || error("Could not infer input_dim / n_meta from saved state " *
                             "(lstm_input=$lstm_input_size, depth=$depth, last_ch=$(nchannels[end]))")

        featurizer = CNNFeaturizer(; input_dim=input_dim, kernel_dims=kernel_widths, nchannels=nchannels)
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

    return model, input_dim
end
