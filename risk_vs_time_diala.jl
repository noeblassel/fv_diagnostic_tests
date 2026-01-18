using JLD2, Plots, Statistics, Measurements, ColorSchemes

# --- Configuration ---
const DATADIR = "results_diala"
const PLOTDIR = "figures_diala"
mkpath(PLOTDIR)

const GAMMAS = [1, 2, 5, 10, 20]
const SADDLES = 1:5
const TV_THRESHOLD = 0.05

# Tolerance Sweeps
const ALPHAS_RNN = [0.5, 0.6, 0.7, 0.8, 0.9] # From loose to strict
const ALPHAS_GR  = [0.2,0.1, 0.05, 0.02, 0.01]   # From loose to strict

# --- Data Loading ---

struct PerfPoint
    time_ratio_mean :: Float64
    time_ratio_std  :: Float64
    err_mean        :: Float64
    err_std         :: Float64
end

function get_performance_curve(method, alphas, gammas)
    # Store results: curve_data[gamma] = Vector{PerfPoint} (one per alpha)
    curve_data = Dict{Int, Vector{PerfPoint}}()

    for γ in gammas
        points = PerfPoint[]
        
        for α in alphas
            # Aggregate across all saddles for this gamma/alpha combo
            all_ratios = Float64[]
            all_errors = Float64[]

            for s in SADDLES
                fpath = joinpath(DATADIR, "diagnostics_$(s)_basin_$(γ).jld2")
                if !isfile(fpath) continue end
                d = load(fpath)

                times = d["times"]
                tv_ϕ, tv_ψ = d["tv_ϕ"], d["tv_ψ"]

                # Ground Truth
                ix_true_ϕ = findfirst(x -> x < TV_THRESHOLD, tv_ϕ)
                ix_true_ψ = findfirst(x -> x < TV_THRESHOLD, tv_ψ)
                t_true = (isnothing(ix_true_ϕ) || isnothing(ix_true_ψ)) ? times[end] : times[max(ix_true_ϕ, ix_true_ψ)]

                # Method Data
                if method == :RNN
                    data_ϕ, data_ψ = d["rnn_ϕ"], d["rnn_ψ"]
                    comp = (vals) -> vals .>= α
                else
                    data_ϕ, data_ψ = d["gr_ϕ"], d["gr_ψ"]
                    comp = (vals) -> vals .<= α
                end

                # Trajectories
                for i in 1:size(data_ϕ, 2)
                    col_ϕ, col_ψ = data_ϕ[:, i], data_ψ[:, i]
                    ix_ϕ = findfirst(comp(col_ϕ))
                    ix_ψ = findfirst(comp(col_ψ))

                    if isnothing(ix_ϕ) || isnothing(ix_ψ)
                        # Penalty for non-convergence: Count as Max Time + Max Error
                        push!(all_ratios, times[end] / t_true)
                        push!(all_errors, max(tv_ϕ[end], tv_ψ[end]))
                    else
                        ix = max(ix_ϕ, ix_ψ)
                        push!(all_ratios, times[ix] / t_true)
                        push!(all_errors, max(tv_ϕ[ix], tv_ψ[ix]))
                    end
                end
            end
            
            # Compute stats for this point on the curve
            if isempty(all_ratios)
                push!(points, PerfPoint(NaN, NaN, NaN, NaN))
            else
                push!(points, PerfPoint(mean(all_ratios), std(all_ratios), mean(all_errors), std(all_errors)))
            end
        end
        curve_data[γ] = points
    end
    return curve_data
end

# --- Plotting ---

function plot_risk_vs_time()
    println("Processing data...")
    rnn_curves = get_performance_curve(:RNN, ALPHAS_RNN, GAMMAS)
    gr_curves  = get_performance_curve(:GR,  ALPHAS_GR,  GAMMAS)

    # Setup Plot
    pl = plot(
        size=(800, 600),
        title="Bias vs. Time\n(averaged over saddles & realizations)",
        xlabel="Diagnostic time /  \"honest\" convergence time",
        ylabel="Total variation error at diagnostic time",
        legend=:topright,
        grid=true,
        minorgrid=true,
        framestyle=:box
    )

    # Reference Zones
    scatter!(pl, [1.0],[TV_THRESHOLD], color=:green, alpha=0.3, label="Goal", linestyle=:solid,ms=30)
    
    colors = cgrad(:plasma, length(GAMMAS), categorical=true)

    for (i, γ) in enumerate(GAMMAS)
        c = colors[i]
        
        # --- RNN (Solid) ---
        pts = rnn_curves[γ]
        X = [p.time_ratio_mean for p in pts]
        Y = [p.err_mean for p in pts]
        
        # Filter NaNs
        valid = .!isnan.(X)
        X, Y = X[valid], Y[valid]

        if !isempty(X)
            plot!(pl, X, Y, 
                label=(i==1 ? "RNN" : ""),
                color=c, lw=2.5, linestyle=:solid,
                marker=:circle, markersize=5
            )
         end

        pts = gr_curves[γ]
        X = [p.time_ratio_mean for p in pts]
        Y = [p.err_mean for p in pts]
        
        valid = .!isnan.(X)
        X, Y = X[valid], Y[valid]

        if !isempty(X)
            plot!(pl, X, Y, 
                label=(i==1 ? "Gelman-Rubin" : ""), 
                color=c, lw=1.5, linestyle=:dash,
                marker=:square, markersize=4, markeralpha=0.7
            )
        end
    end

    # Dummy Legend for Gamma Colors
    # (Plot invisible points just to populate the legend correctly)
    for (i, γ) in enumerate(GAMMAS)
        plot!(pl, [NaN], [NaN], color=colors[i], label="γ = $γ")
    end

    # Formatting limits
    ylims!(pl, 0.0, 0.5) # Focus on the relevant error range
    xlims!(pl, 0.2, 15.0)  # Focus on the transition around 1.0

    savefig(pl, joinpath(PLOTDIR, "risk_vs_time_pareto.pdf"))
    println("Plot saved to $(joinpath(PLOTDIR, "risk_vs_time_pareto.pdf"))")
end

plot_risk_vs_time()