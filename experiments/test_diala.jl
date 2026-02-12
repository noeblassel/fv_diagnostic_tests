    using Plots, ProgressMeter, LaTeXStrings
    using Statistics, Random, StatsBase
    using Flux,JLD2,MLUtils, DataInterpolations
    using DelimitedFiles

    include("FVDiagnosticTests.jl")

    using .FVDiagnosticTests

    ### CV parameters

    const saddles = Dict(1=>(-3,89), 2=>(44,-92), 3=>(61,122), 4=>(124,-19),5=>(53,28)) # saddle 5 = free-energy minimum

    const ϕmin,ϕmax = (-91.5,188.9)
    const ψmin,ψmax = (-113,165.1)

    const ϕm,ψm = saddles[5]

    reperiodize_phi(ϕ) = ϕm-180 + mod(ϕ-ϕm+180,360.0)
    reperiodize_psi(ψ) = ψm-180 + mod(ψ-ψm+180,360.0)


    @inline affine_shift(X,m,M) = (X .- m) ./ (M-m)
    unit_shift(X) = affine_shift(X,minimum(X),maximum(X)) # affine transform onto (0,1)

    ###

    function read_trace_diala(datadir,file_id;
        input_dim::Int = 64,
        stride = 50
    )

        filename = joinpath(datadir,"decorr_fv_$(file_id).out")
        M = readdlm(filename)

        
        ϕdat = reperiodize_phi.(M[:,1:2:end]) # to avoid jumps when crossing periodic boundaries
        ψdat = reperiodize_psi.(M[:,2:2:end])

        ϕ = unit_shift(ϕdat)
        ψ = unit_shift(ψdat)
        
        nchunks = size(M,1)÷stride
        
        ϕchunks = [@view ϕ[1+stride*(k-1):stride*k,:] for k=1:nchunks]
        ψchunks = [@view ψ[1+stride*(k-1):stride*k,:] for k=1:nchunks]

        Xϕ = [hist_feature(chunk,input_dim) for chunk=ϕchunks]
        Xψ = [hist_feature(chunk,input_dim) for chunk=ψchunks]
        
        gr_hist_ϕ = Float64[]
        gr_hist_ψ = Float64[]

        w1_hist_ϕ = Float64[Inf]
        w1_hist_ψ = Float64[Inf]

        Nrep = size(M,2)÷2
        
        sum_lin_gr_ϕ = zeros(Nrep)
        sum_sq_gr_ϕ = zeros(Nrep)

        sum_lin_gr_ψ = zeros(Nrep)
        sum_sq_gr_ψ = zeros(Nrep)

        for i=1:nchunks-1
            push!(w1_hist_ϕ,FVDiagnosticTests.w1(vec(ϕchunks[i]),vec(ϕchunks[i+1])))
            push!(w1_hist_ψ,FVDiagnosticTests.w1(vec(ψchunks[i]),vec(ψchunks[i+1])))

        end
        
        for i=1:nchunks
            ϕchunk = ϕchunks[i]
            ψchunk = ψchunks[i]

            sum_lin_gr_ϕ += sum(ϕchunk,dims=1) |> vec
            sum_lin_gr_ψ += sum(ψchunk,dims=1) |> vec

            sum_sq_gr_ϕ += sum(abs2,ϕchunk,dims=1) |> vec
            sum_sq_gr_ψ += sum(abs2,ψchunk,dims=1) |> vec

            push!(gr_hist_ϕ,(sum(sum_sq_gr_ϕ) -sum(sum_lin_gr_ϕ)^2/(Nrep*i*stride)) / (sum(sum_sq_gr_ϕ - (sum_lin_gr_ϕ .^ 2)/(i*stride))) - 1)
            push!(gr_hist_ψ,(sum(sum_sq_gr_ψ) -sum(sum_lin_gr_ψ)^2/(Nrep*i*stride)) / (sum(sum_sq_gr_ψ - (sum_lin_gr_ψ .^ 2)/(i*stride))) - 1)
        end

        times = collect(1:nchunks) .* (stride*dt*log_stride)

        return stack(batchseq([Xϕ,Xψ],zeros(Float32, input_dim)),dims=2),[w1_hist_ϕ w1_hist_ψ],[gr_hist_ϕ gr_hist_ψ],times
    end


### load RNN Diagnostic

rseed = 2027
rng = Xoshiro(rseed)
model_state = JLD2.load("best_hope_trained.jld2", "model_state")

input_dim = 64
model = load_rnn_from_state(input_dim,model_state)
testmode!(model)


### MD data parameters

const dt = 2e-3 # time step in ps
const log_stride = 2 # CV output frequency
const basedir = "/home/nblassel/Documents/ArticlesQSD/numerical/experiments/fv"

for gamma=[1,2,5,10,20]
    for domain = ["opt","basin"]
        for saddle_ix=1:5
            println("gamma: $(gamma), domain: $(domain), saddle: $(saddles[saddle_ix])")
            ϕz,ψz = saddles[saddle_ix]
            datadir = joinpath(basedir,"trajectories_$(domain)_$(gamma)","saddle$(ϕz)_$(ψz)")

            times = zeros(5)

            w1_ϕ = Vector{Float64}[]
            w1_ψ = Vector{Float64}[]
            gr_ϕ = Vector{Float64}[]
            gr_ψ = Vector{Float64}[]
            rnn_ϕ = Vector{Float64}[]
            rnn_ψ = Vector{Float64}[]

            ## "Ground truth" total variation error data
            err_ϕ = readdlm(joinpath(basedir,"series/errs_phi_$(saddle_ix)_trajectories_$(domain)_$(gamma).out")) |> vec
            err_ψ = readdlm(joinpath(basedir,"series/errs_psi_$(saddle_ix)_trajectories_$(domain)_$(gamma).out")) |> vec

            times_gt = collect(range(0,30,1+size(err_ϕ,1))) # 30 ps total time
            popfirst!(times_gt) # remove t=0 entry

            lerp_gt_ϕ = LinearInterpolation(err_ϕ,times_gt,extrapolation_left=ExtrapolationType.Linear)
            lerp_gt_ψ = LinearInterpolation(err_ψ,times_gt,extrapolation_left=ExtrapolationType.Linear)

            @showprogress for traj_ix=1:50
                X,w1,gr,times = read_trace_diala(datadir,traj_ix,input_dim=input_dim,stride=50)
                Yhat = model(X) |> Flux.σ

                push!(w1_ϕ,w1[:,1])
                push!(w1_ψ,w1[:,2])
                push!(gr_ϕ,gr[:,1])
                push!(gr_ψ,gr[:,2])
                push!(rnn_ϕ,Yhat[:,1])
                push!(rnn_ψ,Yhat[:,2])
                
            end

            outdir = "results_diala"
            isdir(outdir) || mkpath(outdir)
            
            outfile = joinpath(outdir,"diagnostics_$(saddle_ix)_$(domain)_$(gamma).jld2")

            w1_ϕ_mat  = reduce(hcat, w1_ϕ)
            w1_ψ_mat  = reduce(hcat, w1_ψ)
            gr_ϕ_mat  = reduce(hcat, gr_ϕ)
            gr_ψ_mat  = reduce(hcat, gr_ψ)
            rnn_ϕ_mat = reduce(hcat, rnn_ϕ)
            rnn_ψ_mat = reduce(hcat, rnn_ψ)

            jldsave(outfile;
                times  = times,
                w1_ϕ   = w1_ϕ_mat,
                w1_ψ   = w1_ψ_mat,
                gr_ϕ   = gr_ϕ_mat,
                gr_ψ   = gr_ψ_mat,
                rnn_ϕ  = rnn_ϕ_mat,
                rnn_ψ  = rnn_ψ_mat,
                tv_ϕ   = lerp_gt_ϕ.(times),
                tv_ψ   = lerp_gt_ψ.(times),
            )

        end
    end
end
