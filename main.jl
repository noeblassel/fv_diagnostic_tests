
using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

include("FVDiagnosticTests.jl")

using .FVDiagnosticTests
using Flux,JLD2

lrs = [1e-3]
input_dim_exponent_range = 5:7   # 32, 64, or 128 histogram bins
cnn_depth_range = 3:5
cnn_width_exponent_range = 3:4
rnn_depth_range = 1:2
rnn_width_exponent_range = 5:6
mlp_depth_range = 1:2
mlp_width_exponent_range = 5:6


candidates = [build_candidate_run((lr, RNNDiagnosticHyperParams(CNNFeaturizerHyperParams(ide, cd, cw), rd, rw, md, mw)); base_seed=2022)
 for lr in lrs for ide in input_dim_exponent_range for cd in cnn_depth_range for cw in cnn_width_exponent_range
               for rd in rnn_depth_range for rw in rnn_width_exponent_range for md in mlp_depth_range
               for mw in mlp_width_exponent_range]

candidates = reshape(candidates,:)

open("model_summaries.out","w") do f
    for m in candidates
        println(f,"-------------",m.id,"----------------")
        println(f,repr("text/plain",m.model))
    end
end

io = open("tournament_log.out","w")
best_hope = run_tournament!(candidates,train_batches=100,test_batches=50,io=io)
JLD2.jldsave("best_hope.jld2", model_state=Flux.state(best_hope.model)) # checkpoint model state