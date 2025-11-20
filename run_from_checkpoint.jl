prefix = ARGS[1]
n_epochs = parse(Int,ARGS[2])

include("FVDiagnosticTests.jl")

using .FVDiagnosticTests
using Flux, JLD2, Random, Statistics

n_train_per_epoch = 100
n_test_per_epoch = 50

rng = Random.Xoshiro(2025)

state = JLD2.load("$(prefix).jld2","model_state")
model = load_rnn_from_state(64,state)

opt_state = Flux.setup(Adam(),model)

training_params = TrainingRun(rng = rng,
βlims = (1.0,3.0),
opt_state=opt_state,
model=model,
feature=hist_feature,
input_dim=64,
pot_per_batch=5,
trace_per_pot=5,
cut_per_trace=2,
id="$(prefix)_0")

io = open("$(prefix)_training_log.out","w")

println(io,"Training $prefix for $n_epochs episodes.")
println(io,"Pot per batch: $(training_params.pot_per_batch)")
println(io,"Trace per pot: $(training_params.trace_per_pot)")
println(io,"Cut per trace: $(training_params.cut_per_trace)")

flush(io)

for k=1:n_epochs
    training_params.id = "$(prefix)_$(k)"
    println(io,"Episode $k")

    losses = run_epoch!(training_params,n_train_per_epoch)

    println(io,"Training loss (min/max/mean/std) : $(minimum(losses))/$(maximum(losses))/$(mean(losses))/$(std(losses))")
    flush(io)
    acc,loss = test_accuracy!(training_params,n_test_per_epoch)
    println(io,"Validation metrics (50% → 90% decision threshold accuracies /mean testing loss) : $(acc)/$(loss)")
    flush(io)
end
