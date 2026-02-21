using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

include(joinpath(@__DIR__, "..", "FVDiagnosticTests.jl"))
using .FVDiagnosticTests
using Flux, Random, Statistics, Test, JLD2

# ── Test 1: CNN featurizer forward pass ──────────────────────────────────────
@testset "CNN featurizer forward pass" begin
    rng = Xoshiro(1)
    feat = CNNFeaturizer(; input_dim=64, kernel_dims=[5,5,5], nchannels=[16,32,64], rng=rng)
    model = RNNDiagnostic(feat; dims_rnn=[64], dims_mlp=[64,32], n_meta=1, rng=rng)
    x = randn(Float32, 65, 10, 4)   # 64 feature dims + 1 metadata
    y = model(x)
    @test size(y) == (10, 4)
    @test all(isfinite, y)
end

# ── Test 2: Deep Sets featurizer forward pass ─────────────────────────────────
@testset "DeepSet featurizer forward pass" begin
    rng = Xoshiro(2)
    feat = DeepSetFeaturizer(; dims_phi=[32,64], dims_rho=[64,32], rng=rng)
    model = RNNDiagnostic(feat; dims_rnn=[64], dims_mlp=[64,32], n_meta=1, rng=rng)
    x = randn(Float32, 51, 10, 4)   # 50 particle dims + 1 metadata
    y = model(x)
    @test size(y) == (10, 4)
    @test all(isfinite, y)
end

# ── Test 3: Variable stride ───────────────────────────────────────────────────
@testset "Variable stride (τ)" begin
    strides = [rand(Xoshiro(k), 10:100) for k in 1:50]
    @test length(unique(strides)) > 1
    @test all(10 .<= strides .<= 100)
    # Constant stride_lims=(50,50) should always give 50
    @test all(rand(Xoshiro(k), 50:50) == 50 for k in 1:10)
end

# ── Test 4: Variable N + deep_set_feature ────────────────────────────────────
@testset "Variable N + deep_set_feature" begin
    X, Y, mask = get_batch(Xoshiro(1);
        stride_lims=(50,50), Nreplicas_lims=(20,80),
        npot=1, ntrace=2, ncut=1,
        input_dim=50, feature=deep_set_feature)
    @test size(X, 1) == 51          # 50 particle dims + 1 metadata scalar
    @test ndims(X) == 3
    @test size(Y) == size(mask)
    @test all(isfinite, X)
end

# ── Test 5: Online inference ──────────────────────────────────────────────────
@testset "Online inference" begin
    rng = Xoshiro(5)

    @testset "CNN online" begin
        feat = CNNFeaturizer(; input_dim=64, kernel_dims=[5,5,5], nchannels=[16,32,64], rng=rng)
        model = RNNDiagnostic(feat; dims_rnn=[64], dims_mlp=[32], n_meta=1, rng=rng)
        online = RNNDiagnosticOnline(model)
        out = online(randn(Float32, 65))   # 64 feature dims + 1 metadata
        @test length(out) == 1
        @test all(isfinite, out)
    end

    @testset "DeepSet online" begin
        feat = DeepSetFeaturizer(; dims_phi=[32,64], dims_rho=[64,32], rng=rng)
        model = RNNDiagnostic(feat; dims_rnn=[64], dims_mlp=[32], n_meta=1, rng=rng)
        online = RNNDiagnosticOnline(model)
        frame = randn(Float32, 51)   # 50 particle dims + 1 metadata
        out1 = online(frame)
        @test length(out1) == 1
        @test all(isfinite, out1)

        reset_rnn_state!(online)
        out2 = online(frame)
        @test all(isfinite, out2)
        # After reset, same input should give the same output as the first call
        @test out1 ≈ out2
    end
end

# ── Test 6: Training step (Deep Sets, variable τ and N) ──────────────────────
@testset "Training step with Deep Sets, variable τ/N" begin
    feat = DeepSetFeaturizer(; dims_phi=[32], dims_rho=[32], rng=Xoshiro(7))
    model = RNNDiagnostic(feat; dims_rnn=[32], dims_mlp=[32], n_meta=1, rng=Xoshiro(7))
    opt_state = Flux.setup(Adam(1e-3), model)

    losses = Float32[]
    trainmode!(model)
    for k in 1:3
        X, Y, mask = get_batch(Xoshiro(k);
            stride_lims=(30,70), Nreplicas_lims=(20,60),
            npot=2, ntrace=2, ncut=1,
            input_dim=32, feature=deep_set_feature)
        loss, grads = Flux.withgradient(model) do m
            yhat = m(X)
            Flux.logitbinarycrossentropy(yhat, Y, agg=x -> mean(x[mask]))
        end
        Flux.update!(opt_state, model, grads[1])
        push!(losses, loss)
    end
    @test length(losses) == 3
    @test all(isfinite, losses)
end

# ── Test 7: Checkpoint round-trip ─────────────────────────────────────────────
@testset "Checkpoint round-trip" begin
    x_ds  = randn(Float32, 51, 5, 2)   # 50 particle dims + 1 metadata
    x_cnn = randn(Float32, 65, 5, 2)   # 64 feature dims + 1 metadata

    @testset "DeepSet" begin
        feat = DeepSetFeaturizer(; dims_phi=[32,64], dims_rho=[64,32], rng=Xoshiro(99))
        model = RNNDiagnostic(feat; dims_rnn=[64], dims_mlp=[32], n_meta=1, rng=Xoshiro(99))
        y_before = model(x_ds)
        mktempdir() do dir
            path = joinpath(dir, "ckpt.jld2")
            JLD2.jldsave(path; model_state=Flux.state(model))
            model2 = load_rnn_from_state(50, JLD2.load(path, "model_state"))
            @test model2(x_ds) ≈ y_before
        end
    end

    @testset "CNN" begin
        feat = CNNFeaturizer(; input_dim=64, kernel_dims=[5,5], nchannels=[16,32], rng=Xoshiro(5))
        model = RNNDiagnostic(feat; dims_rnn=[32], dims_mlp=[32], n_meta=1, rng=Xoshiro(5))
        y_before = model(x_cnn)
        mktempdir() do dir
            path = joinpath(dir, "ckpt.jld2")
            JLD2.jldsave(path; model_state=Flux.state(model))
            model2 = load_rnn_from_state(64, JLD2.load(path, "model_state"))
            @test model2(x_cnn) ≈ y_before
        end
    end
end
