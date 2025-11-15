function build_candidate_run(hyp;
        base_seed = 2025,
        input_dim = 64,
        βlims = (1.0,3.0),
        pot_per_batch = 5,
        trace_per_pot = 5,
        cut_per_trace = 1,
        feature = hist_feature)

    lr, cd, cw, rd, rw, md, mw = hyp

    seed = base_seed + hash(hyp)
    rng = Xoshiro(seed)

    h = RNNDiagnosticHyperParams(cd, cw, rd, rw, md, mw)

    model = RNNDiagnostic(h; input_dim=input_dim, rng=rng)

    opt = Adam(lr)
    opt_state = Flux.setup(opt, model)

    tstamp = string(Dates.now())

    return TrainingRun(
        rng=rng,
        βlims=βlims,
        opt_state=opt_state,
        model=model,
        feature=feature,
        input_dim=input_dim,
        pot_per_batch=pot_per_batch,
        trace_per_pot=trace_per_pot,
        cut_per_trace=cut_per_trace,
        id="$(tstamp)_$(hash(hyp))_$(base_seed)"
    )
end

function run_tournament!(candidates;
        max_epochs = 100,
        reduction_factor = 2,
        train_batches = 20,
        test_batches = 20,
        io=stdout)

    n = length(candidates)
    alive = collect(1:n)
    @assert n > 0 "No candidates supplied"
    round = 0

    while length(alive) > 1
        round += 1
        println(io,"\n=== Round $round: # candidate configs=$(length(alive))")

        val_scores = Float64[]
        val_ixs = Int[]

        for (i,ix)=enumerate(alive)
                println(io," Training candidate $ix  ($i/$(length(alive))) ... (id=$(candidates[ix].id)) ")
                run_epoch!(candidates[ix], train_batches)

                acc, loss = test_accuracy!(candidates[ix], test_batches)

                println(io,"Validation accuracies (50% → 90% decision thresholds):",acc)
                println(io,"Validation loss : ",loss)
                push!(val_scores,loss) # primary metric: test loss
                push!(val_ixs, ix)

                flush(io)
        end

        p = sortperm(val_scores)

        println(io,"Min loss: $(minimum(val_scores)). Max loss: $(maximum(val_scores)). Mean loss $(mean(val_scores))")

        n_survivors = max(1, fld(length(alive), reduction_factor))
        alive = sort(val_ixs[p[1:n_survivors]])
    end

    return candidates[first(alive)]
end