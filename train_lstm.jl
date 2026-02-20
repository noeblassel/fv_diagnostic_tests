@kwdef mutable struct TrainingRun{R,S,T,U,V}
    rng::R

    βlims::S

    opt_state::T
    model::U

    feature::V
    input_dim::Int

    pot_per_batch::Int
    trace_per_pot::Int
    cut_per_trace::Int

    id::String

    stride_lims::Tuple{Int,Int} = (50, 50)
    Nreplicas_lims::Tuple{Int,Int} = (50, 50)
end

function test_accuracy!(params::TrainingRun, n)

    acc = zeros(5)
    loss = 0.0
    testmode!(params.model)

    @showprogress for b = 1:n
        X, Y, mask = get_batch(params.rng;
        input_dim=params.input_dim,
        ntrace=params.trace_per_pot,
        npot=params.pot_per_batch,
        ncut=params.cut_per_trace,
        feature=params.feature,
        βlims=params.βlims,
        stride_lims=params.stride_lims,
        Nreplicas_lims=params.Nreplicas_lims)

        Yhat_logits = params.model(X)
        Yhat_prob = Flux.σ(Yhat_logits)[mask]
        Y_true = Y[mask]

        acc[1] += mean(((Yhat_prob .> 0.5).== Y_true)) # 0.5 threshold

        for i=1:4
            thr_mask_pos = (Yhat_prob .> (0.5+0.1*i)) # higher confidence thresholds
            thr_mask_neg = (Yhat_prob .< (0.5-0.1*i))

            n_pos = sum(thr_mask_pos)
            n_neg = sum(thr_mask_neg)

            acc[i+1] += (n_pos*mean(Y_true[thr_mask_pos]) + n_neg*(1-mean(Y_true[thr_mask_neg])))/(n_pos+n_neg) # looser thresholds
        end

        loss += Flux.logitbinarycrossentropy(Yhat_logits, Y, agg=x -> mean(x[mask]))
    end

    return (acc / n, loss/ n)

end

function run_epoch!(params::TrainingRun,n,save_checkpoint=true)
    trainmode!(params.model)
    losses = Float32[]

    @showprogress for b = 1:n
                X, Y, mask = get_batch(params.rng;
                            input_dim=params.input_dim,
                            ntrace=params.trace_per_pot,
                            npot=params.pot_per_batch,
                            ncut=params.cut_per_trace,
                            feature=params.feature,
                            βlims=params.βlims,
                            stride_lims=params.stride_lims,
                            Nreplicas_lims=params.Nreplicas_lims)

        loss, grads = Flux.withgradient(params.model) do m
            Yhat_logits = m(X)

            Flux.logitbinarycrossentropy(Yhat_logits, Y, agg=x -> mean(x[mask]))
        end

        Flux.update!(params.opt_state, params.model, grads[1])
        push!(losses, loss)
    end

    if save_checkpoint
        JLD2.jldsave("history/run_$(params.id).jld2", model_state=Flux.state(params.model)) # checkpoint model state
    end

    return losses
end