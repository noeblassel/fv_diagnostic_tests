struct RNNDiagnostic{S<:Chain,T<:Chain,U<:Chain}
    cnn_encoder::S
    cnn_output_dim::Int
    rnn::T
    mlp_head::U
end

Flux.@layer RNNDiagnostic

function RNNDiagnostic(; input_dim::Int=64,
    cnn_kernel_dims::Vector{Int}=[5,5,5],
    cnn_nchannels::Vector{Int}=[16,32,64],
    dims_rnn::Vector{Int}=[64],
    dims_mlp::Vector{Int}=[64, 32],
    rng=Random.GLOBAL_RNG)

    initializer = Flux.glorot_uniform(rng)

    cnn_layers = []
    in_channels = 1

    for (kernel_size, out_channels) = zip(cnn_kernel_dims, cnn_nchannels)
        pad_size = div(kernel_size-1,2)
        push!(cnn_layers, Conv(tuple(kernel_size), in_channels => out_channels,pad=pad_size, leakyrelu, init=initializer))
        push!(cnn_layers, MaxPool((2,)))
        in_channels = out_channels
    end

    cnn_encoder = Chain(cnn_layers...)

    cnn_output_dim = prod(Flux.outputsize(cnn_encoder, (input_dim, 1, 1)))

    rnn_layers = []
    input_dim_rnn = cnn_output_dim

    for output_dim_rnn = dims_rnn
        push!(rnn_layers, LSTM(input_dim_rnn => output_dim_rnn, init_kernel=initializer, init_recurrent_kernel=initializer))
        input_dim_rnn = output_dim_rnn
    end

    rnn = Chain(rnn_layers...)

    mlp_layers = []
    input_dim_mlp = last(dims_rnn)

    for output_dim_mlp = dims_mlp
        push!(mlp_layers, Dense(input_dim_mlp => output_dim_mlp, leakyrelu, init=initializer))
        input_dim_mlp = output_dim_mlp
    end

    mlp = Chain(mlp_layers..., Dense(last(dims_mlp) => 1, init=initializer))

    return RNNDiagnostic(cnn_encoder, cnn_output_dim, rnn, mlp)
end

function (m::RNNDiagnostic)(x)
    @assert ndims(x) == 3 "Input size error: expected data in format (input_dim,sequence_length,batch_size)"

    input_dim, sequence_length, batch_size = size(x)
    x = reshape(x, input_dim, 1, sequence_length * batch_size)

    z = m.cnn_encoder(x)
    z = reshape(z, m.cnn_output_dim, sequence_length, batch_size)

    h = m.rnn(z)

    yhat_logits = m.mlp_head(h) # (1,sequence_length,batch_size) -- ouput logits
    return reshape(yhat_logits, sequence_length, batch_size)
end


mutable struct RNNDiagnosticHyperParams
    cnn_depth::Int
    cnn_width_exponent::Int
    
    rnn_depth::Int
    rnn_width_exponent::Int

    mlp_depth::Int
    mlp_width_exponent::Int
end

function RNNDiagnostic(hyperparams::RNNDiagnosticHyperParams; input_dim::Int=64, rng=Random.GLOBAL_RNG)
    
    cnn_nchannels = [2^i for i=hyperparams.cnn_width_exponent:(hyperparams.cnn_width_exponent+hyperparams.cnn_depth-1)]
    cnn_kernel_dims = fill(5, hyperparams.cnn_depth)

    dims_rnn = fill(2^(hyperparams.rnn_width_exponent), hyperparams.rnn_depth)

    dims_mlp = fill(2^(hyperparams.mlp_width_exponent), hyperparams.mlp_depth)

    return RNNDiagnostic(input_dim=input_dim,
        cnn_nchannels=cnn_nchannels,
        cnn_kernel_dims=cnn_kernel_dims,
        dims_rnn=dims_rnn,
        dims_mlp=dims_mlp,
        rng=rng)
end


"""
Similar to 
"""
mutable struct RNNDiagnosticOnline{S<:Chain,T<:Chain,U<:Chain,V}
    cnn_encoder::S
    cnn_output_dim::Int
    rnn_cells::T
    mlp_head::U
    rnn_state::V
end

function RNNDiagnosticOnline(model::RNNDiagnostic)

    rnn_cells = [layer.cell for layer=model.rnn.layers]
    dims_rnn = [size(cell.bias,1) ÷ 4 for cell=rnn_cells]
    rnn_state = [(zeros(Float32,d),zeros(Float32,d)) for d=dims_rnn] # hidden and cell states

    RNNDiagnosticOnline(model.cnn_encoder,model.cnn_output_dim,rnn_cells,model.mlp_head,rnn_state)
end

function reset_state!(model::RNNDiagnosticOnline)
    model.rnn_state = [(zero(h),zero(c)) for (c,h)=model.rnn_state]
end

function (m::RNNDiagnosticOnline)(x)
    @assert ndims(x) == 1 "RNNDiagnosticOnline only"
    input_dim = size(x,1)
    
end

function (m::RNNDiagnostic)(x)
    @assert ndims(x) == 3 "Input size error: expected data in format (input_dim,sequence_length,batch_size)"

    input_dim, sequence_length, batch_size = size(x)
    x = reshape(x, input_dim, 1, sequence_length * batch_size)

    z = m.cnn_encoder(x)
    z = reshape(z, m.cnn_output_dim, sequence_length, batch_size)

    h = m.rnn(z)

    yhat_logits = m.mlp_head(h) # (1,sequence_length,batch_size) -- ouput logits
    return reshape(yhat_logits, sequence_length, batch_size)
end

"""
Convenience function reconstructing a RNNDiagnostic model from a saved state (NamedTuple)
"""
function load_rnn_from_state(input_dim,state)
    kernel_widths = [size(l.weight,1) for l=state.cnn_encoder.layers[1:2:end]] # skip max pool layers
    nchannels = [size(l.bias,1) for l=state.cnn_encoder.layers[1:2:end]]
    rnn_widths = [size(l.cell.Wh,2) for l=state.rnn.layers]
    mlp_widths = [size(l.bias,1) for l=state.mlp_head.layers]

    pop!(mlp_widths) # constructor expect hidden dimensions

    model = RNNDiagnostic(input_dim=input_dim,cnn_kernel_dims=kernel_widths,cnn_nchannels=nchannels,dims_rnn=rnn_widths,dims_mlp = mlp_widths)
    Flux.loadmodel!(model,state)

    return model
end