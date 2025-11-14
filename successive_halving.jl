using Flux, ParameterSchedulers, MLUtils, JLD2
using Plots, ProgressMeter

include("./generate_data.jl")
include("./lstm.jl")

rseed = 2023
rng = Xoshiro(rseed)

input_dim = 64 # dimension of input feature
βlims = (1.0,1.0) # temperature range
model = RNNDiagnostic(input_dim=input_dim, rng=rng)

display(model)

lr_schedule = Step(1e-3, 0.2, 10)
opt = Adam(1e-3)
opt_state = Flux.setup(opt, model)

batch_per_epoch = 50
test_per_epoch = 20
n_epochs = 30

pot_per_batch = 5
trace_per_pot = 5 
ncut = 1 # number of cuts per trace in data generation

train_params = (; rseed=rseed,
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
        input_dim=input_dim,
        model=model)

feature = hist_feature # feature extraction function

loss_trace = Float32[]
epoch_loss = Float32[]
test_acc = Float32[]