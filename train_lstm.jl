include("./generate_data.jl")
include("./lstm.jl")


@kwdef mutable struct TrainingParams{R,S,T,U,V,W}
    rng::R

    βlims::S

    lr_schedule::T
    opt::U
    model::V

    feature::W

    pot_per_batch::Int
    trace_per_pot::Int
    cut_per_trace::Int

    batch_per_epoch::Int
    test_per_epoch::Int
    n_epochs::Int
end

io = open("training_log_$(rseed).out", "w")

params= RNNDiagnosticTrainingParams(rseed=rseed,
        rng=rng,
        βlims=βlims,
        lr_schedule=lr_schedule,
        opt_state=opt_state,
        batch_per_epoch=batch_per_epoch,
        test_per_epoch=test_per_epoch,
        n_epochs=n_epochs,
        pot_per_batch=pot_per_batch,
        trace_per_pot=trace_per_pot,
        cut_per_trace=ncut,
        feature=feature,
        tstamp=tstamp,
        model=model)

println(io,"Training parameters: ",params)
flush(io)


function test_accuracy!(params::TrainingParams, n)
    acc = 0.0
    loss = 0.0
    testmode!(params.model)

    @showprogress for b = 1:n
        X, Y, mask = get_batch(params.rng;
        input_dim=params.input_dim,
        ntrace=params.trace_per_pot,
        npot=params.pot_per_batch,
        ncut=params.cut_per_trace,
        feature=params.feature,
        βlims=params.βlims)

        Yhat_logits = params.model(X)
        Yhat_prob = Flux.σ(Yhat_logits)

        acc += mean(((Yhat_prob .> 0.5).==Y)[mask])
        loss += Flux.logitbinarycrossentropy(Yhat_logits, Y, agg=x -> mean(x .* mask))
    end

    return (acc / n, loss/ n)

end

function run_epoch!(params::RNNDiagnosticTrainingParams,epoch_number,save_checkpoint=true)
    η = params.lr_schedule[epoch_number]
    Flux.adjust!(params.opt_state,η)
    train_mode!(params.model)


    batch_losses = Float32[]

        @showprogress for b = 1:params.batch_per_epoch
                    X, Y, mask = get_batch(params.rng;
                                input_dim=params.input_dim,
                                ntrace=params.trace_per_pot,
                                npot=params.pot_per_batch,
                                ncut=params.cut_per_trace,
                                feature=params.feature,
                                βlims=params.βlims)

            loss, grads = Flux.withgradient(params.model) do m
                Yhat_logits = m(X)

                Flux.logitbinarycrossentropy(Yhat_logits, Y, agg=x -> mean(x .* mask))
            end

            Flux.update!(params.opt_state, params.model, grads[1])
            push!(batch_losses, loss)
        end

        if save_checkpoint
            JLD2.jldsave("history/epoch_$(epoch_number)_$(params.rseed)_$(params.tstamp).jld2", model_state=Flux.state(params.model)) # checkpoint model state
        end

        return batch_losses
end

#         println(io,"Epoch $epoch completed. Average training loss: $(mean(batch_losses))")
#         append!(loss_trace, copy(batch_losses))
#         push!(epoch_loss, mean(batch_losses))

#         plot(loss_trace, xlabel="batch", ylabel="training loss", label="")
#         for (j, loss_v) = enumerate(epoch_loss)
#             plot!(t -> loss_v, (j - 1) * params.batch_per_epoch + 1, j * params.batch_per_epoch, color=:red, label="")
#         end

#         savefig("training_loss.pdf")

#         JLD2.jldsave("history/epoch_$(epoch)_$(params.rseed)_$(params.tstamp).jld2", model_state=Flux.state(params.model)) # checkpoint model state



#     opt_state = Flux.setup(opt, model)

#     loss_trace = Float32[]
#     epoch_loss = Float32[]
#     test_acc = Float32[]

#     println(io,"Starting training...")

# for (η,epoch) in zip(lr_schedule,1:n_epochs)
#     Flux.adjust!(opt_state,η) # set learning rate to scheduled value

#     batch_losses = Float32[]
#     trainmode!(model)

#     @showprogress for b = 1:batch_per_epoch
#         X, Y, mask = get_batch(rng; input_dim=input_dim, ntrace=trace_per_pot, npot=pot_per_batch,ncut = ncut, feature=feature,βlims=βlims)

#         loss, grads = Flux.withgradient(model) do m
#             Yhat_logits = m(X)

#             Flux.logitbinarycrossentropy(Yhat_logits, Y, agg=x -> mean(x .* mask))
#         end
#         Flux.update!(opt_state, model, grads[1])
#         push!(batch_losses, loss)
#     end

#     println(io,"Epoch $epoch completed. Average training loss: $(mean(batch_losses))")
#     append!(loss_trace, copy(batch_losses))
#     push!(epoch_loss, mean(batch_losses))

#     plot(loss_trace, xlabel="batch", ylabel="training loss", label="")
#     for (j, loss_v) = enumerate(epoch_loss)
#         plot!(t -> loss_v, (j - 1) * batch_per_epoch + 1, j * batch_per_epoch, color=:red, label="")
#     end

#     savefig("training_loss.pdf")

#     JLD2.jldsave("history/epoch_$(epoch)_$(rseed)_$(tstamp).jld2", model_state=Flux.state(model)) # checkpoint model state

    
#     push!(test_acc, test_accuracy(model,test_per_epoch,rng))

#     plot(test_acc, xlabel="epoch", ylabel="test accuracy", label="")
#     savefig("test_accuracy.pdf")

#     println(io,"Test classification accuracy : $(100*last(test_acc)) %")
#     flush(io)
# end