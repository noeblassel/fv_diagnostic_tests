module FVDiagnosticTests

    using Random, Dates
    using Flux, Zygote, ParameterSchedulers, MLUtils, JLD2, ProgressMeter
    using StatsBase

    using Random, Distributions, StatsBase, Statistics
    using SparseArrays, Arpack, MLUtils

    include("./generate_data.jl")
    include("./lstm.jl")
    include("./train_lstm.jl")
    include("./tournament.jl")

    export generate_potential, comp_generator, comp_qsd, tv_trace, conv_tv, sim_fv, hist_feature, ecdf_feature, tecdf_feature, deep_set_feature, get_batch
    export AbstractFeaturizer, CNNFeaturizer, DeepSetFeaturizer, AttentionFeaturizer
    export CNNFeaturizerHyperParams, DeepSetFeaturizerHyperParams, AttentionFeaturizerHyperParams
    export RNNDiagnostic, RNNDiagnosticHyperParams, RNNDiagnosticOnline, load_rnn_from_state, reset_rnn_state!
    export TrainingRun, test_accuracy!, run_epoch!
    export build_candidate_run, run_tournament!


end