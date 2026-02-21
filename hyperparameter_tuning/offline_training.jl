# offline_training.jl
#
# Included (not run directly) by the BO scripts.
# Requires the parent FVDiagnosticTests module to already be loaded,
# as well as Flux, MLUtils, JLD2, Statistics, Random.

const _OFFLINE_DUMMY      = -1.0f0   # padding sentinel — same value as dummy_val in generate_data.jl
const OFFLINE_MINIBATCH_SIZE = 32    # trajectories per minibatch

# ── Private helper ────────────────────────────────────────────────────────────

# Assemble one (X, Y, mask) from the trajectories at `indices`.
# Returns `nothing` if all trajectories are too short.
function _assemble_batch(dataset, indices, rng; feature, input_dim, min_length=5)
    dt           = dataset.dt
    batch        = Vector{Vector{Float32}}[]
    labels_batch = Vector{Float32}[]

    for k in indices
        fv_frames_k   = dataset.sequences[k]
        full_labels_k = dataset.labels[k]
        l = length(fv_frames_k)
        l < min_length && continue

        meta_val = Float32(sqrt(dataset.Nreplicas[k] * dataset.strides[k] * dt))
        features = [vcat(feature(f, input_dim), [meta_val]) for f in fv_frames_k]

        α   = (min_length / l) + rand(rng) * (1.0 - min_length / l)
        len = clamp(round(Int, α * l), min_length, l)
        push!(batch,        features[1:len])
        push!(labels_batch, Float32.(full_labels_k[1:len]))
    end

    isempty(batch) && return nothing

    batch_X = MLUtils.batchseq(batch,        zeros(Float32, input_dim + 1))
    batch_Y = MLUtils.batchseq(labels_batch, _OFFLINE_DUMMY)
    Y       = stack(batch_Y, dims=1)
    return stack(batch_X, dims=2), Y, (Y .!= _OFFLINE_DUMMY)
end

# ── Public API ────────────────────────────────────────────────────────────────

"""
    epoch_batch(dataset, rng; feature, input_dim, min_length) -> (X, Y, mask)

Assembles **all** trajectories in `dataset` into a single batch (useful for
inspection / smoke-tests).  One random start-anchored cut per trajectory;
appends `sqrt(N·stride·dt)` metadata scalar to every frame.
"""
function epoch_batch(dataset, rng; feature=hist_feature, input_dim=64, min_length=5)
    perm   = randperm(rng, length(dataset.sequences))
    result = _assemble_batch(dataset, perm, rng;
                             feature=feature, input_dim=input_dim, min_length=min_length)
    result === nothing && error("epoch_batch: no valid trajectories")
    return result
end

"""
    run_epoch_offline!(params::TrainingRun, dataset; minibatch_size) -> Float32

One training epoch: shuffles all trajectories, splits into minibatches of
`minibatch_size` trajectories each, performs one gradient step per minibatch.
Returns mean training loss across minibatches.
"""
function run_epoch_offline!(params::TrainingRun, dataset;
                            minibatch_size::Int = OFFLINE_MINIBATCH_SIZE)
    trainmode!(params.model)
    perm   = randperm(params.rng, length(dataset.sequences))
    losses = Float32[]

    @showprogress for chunk in Iterators.partition(perm, minibatch_size)
        mb = _assemble_batch(dataset, chunk, params.rng;
                             feature=params.feature, input_dim=params.input_dim)
        mb === nothing && continue
        X, Y, mask = mb

        loss, grads = Flux.withgradient(params.model) do m
            Flux.logitbinarycrossentropy(m(X), Y, agg=x -> mean(x[mask]))
        end
        Flux.update!(params.opt_state, params.model, grads[1])
        push!(losses, Float32(loss))
    end

    return isempty(losses) ? 0.0f0 : mean(losses)
end

"""
    test_loss_offline!(params::TrainingRun, dataset; minibatch_size) -> (loss, acc)

Forward pass on all trajectories in `dataset`, minibatch by minibatch.
Returns a NamedTuple `(loss, acc)` where `loss` is the mean cross-entropy loss
(the BO objective) and `acc` is the fraction of correctly classified snapshots.
"""
function test_loss_offline!(params::TrainingRun, dataset;
                            minibatch_size::Int = OFFLINE_MINIBATCH_SIZE)
    testmode!(params.model)
    perm      = randperm(params.rng, length(dataset.sequences))
    losses    = Float64[]
    n_correct = 0
    n_total   = 0

    @showprogress for chunk in Iterators.partition(perm, minibatch_size)
        mb = _assemble_batch(dataset, chunk, params.rng;
                             feature=params.feature, input_dim=params.input_dim)
        mb === nothing && continue
        X, Y, mask = mb
        logits = params.model(X)
        push!(losses, Float64(Flux.logitbinarycrossentropy(logits, Y, agg=x -> mean(x[mask]))))
        probs      = Flux.σ.(logits)[mask]
        n_correct += sum((probs .> 0.5f0) .== Y[mask])
        n_total   += sum(mask)
    end

    loss = isempty(losses) ? 0.0 : mean(losses)
    acc  = n_total > 0 ? n_correct / n_total : 0.0
    return (; loss, acc)
end
