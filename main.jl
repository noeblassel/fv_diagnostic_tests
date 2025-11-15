include("FVDiagnosticTests.jl")

using .FVDiagnosticTests

lrs = [1e-3]
cnn_depth_range = 3:5
cnn_width_exponent_range = 4:6
rnn_depth_range = 1:1
rnn_width_exponent_range = 5:7
mlp_depth_range = 1:2
mlp_width_exponent_range = 5:7


candidates = [build_candidate_run((lr, cd, cw, rd, rw, md, mw))
 for lr in lrs for cd in cnn_depth_range for cw in cnn_width_exponent_range
               for rd in rnn_depth_range for rw in rnn_width_exponent_range for md in mlp_depth_range
               for mw in mlp_width_exponent_range]

candidates = reshape(candidates,:)

open("model_summaries.out","w") do f
    for m in candidates
        println(f,"-------------",m.id,"----------------")
        println(f,repr("text/plain",m.model))
    end
end

best_hope = run_tournament!(candidates)
JLD2.jldsave("best_hope.jld2", model_state=Flux.state(best_hope.model)) # checkpoint model state