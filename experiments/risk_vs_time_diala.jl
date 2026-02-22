using JLD2, Plots, Statistics, ColorSchemes

# --- Configuration ---
const DATADIR = "results_diala"
const PLOTDIR = "figures_diala"
mkpath(PLOTDIR)

const GAMMAS = [1, 2, 5, 10,20]
const SADDLES = 1:5
const TV_THRESHOLD = 0.05
const MIN_BURNIN = 1.0 # ps

const ALPHAS_RNN = [0.5, 0.6, 0.7, 0.8, 0.9]
const ALPHAS_GR  = [0.2, 0.1, 0.05, 0.02, 0.01] 

# --- Data Collection ---

"""
Gathers all (time_ratio, error) pairs for a specific method and threshold alpha,
aggregating across ALL Gammas and Saddles.
"""
function gather_points(method, α)
    ratios = Float64[]
    errors = Float64[]

    comp = if method == :RNN
        (vals) -> vals .>= α
    else
        (vals) -> vals .<= α
    end

    for γ in GAMMAS
        for s in SADDLES
            fpath = joinpath(DATADIR, "diagnostics_$(s)_basin_$(γ).jld2")
            if !isfile(fpath) continue end
            d = load(fpath)

            times = d["times"]
            tv_ϕ, tv_ψ = d["tv_ϕ"], d["tv_ψ"]

            # --- Ground Truth Calculation (Robust) ---
            start_idx = findfirst(t -> t >= MIN_BURNIN, times)
            if isnothing(start_idx); start_idx = 1; end

            ix_true_ϕ = findnext(x -> x < TV_THRESHOLD, tv_ϕ, start_idx)
            ix_true_ψ = findnext(x -> x < TV_THRESHOLD, tv_ψ, start_idx)
            
            if isnothing(ix_true_ϕ) || isnothing(ix_true_ψ)
                t_true = times[end]
            else
                t_true = times[max(ix_true_ϕ, ix_true_ψ)]
            end

            # --- Method Trajectories ---
            if method == :RNN
                data_ϕ, data_ψ = d["rnn_ϕ"], d["rnn_ψ"]
            else
                data_ϕ, data_ψ = d["gr_ϕ"], d["gr_ψ"]
            end

            # Check every trajectory in the batch
            for i in 1:size(data_ϕ, 2)
                col_ϕ, col_ψ = data_ϕ[:, i], data_ψ[:, i]
                
                # Apply burn-in skip to method detection as well
                ix_ϕ = findnext(comp(col_ϕ), start_idx)
                ix_ψ = findnext(comp(col_ψ), start_idx)

                if isnothing(ix_ϕ) || isnothing(ix_ψ)
                    # Penalty: Max Time, Max Error
                    push!(ratios, times[end] / t_true)
                    push!(errors, max(tv_ϕ[end], tv_ψ[end]))
                else
                    ix = max(ix_ϕ, ix_ψ)
                    push!(ratios, times[ix] / t_true)
                    push!(errors, max(tv_ϕ[ix], tv_ψ[ix]))
                end
            end
        end
    end
    return ratios, errors
end

# --- Plotting ---

function isoluminant_gradient(n_steps::Int; L=60, C=30)
    hues = range(0, 360, length=n_steps)

    lch_colors = [LCHab(L, C, h) for h in hues]

    return RGB.(lch_colors)
end

function jitter(data, amount) # for visusalization purposes
    return data .+ (rand(length(data)) .- 0.5) .* amount
end

function plot_single_method(method_sym, alphas, method_name, filename)
    println("Generating plot for $method_name...")
    
    pl = plot(
        size=(800, 600),
        title="$method_name: Efficiency vs Bias",
        xlabel="Diagnostic time / Decorrelation time",
        ylabel="TV error at diagnostic time",
        grid=true,
        minorgrid=true,
        framestyle=:box,
        legend=:topright
    )

    n_alphas = length(alphas)

    color_palette = isoluminant_gradient(2n_alphas; L=60, C=30)

    all_points = [gather_points(method_sym, α) for α in alphas]
    tot_points = sum(length(p[1]) for p in all_points)
    println(" Total points to plot: $tot_points")

    for (i, α) in enumerate(alphas)
        X, Y = all_points[i]
        
        c = color_palette[i]

        X_jitt = jitter(X, 0.2) 
        Y_jitt = jitter(Y, 0.01)

        scatter!(pl, X_jitt, Y_jitt, 
            label="", 
            color=c, alpha=50/tot_points, markershape=:circle, markersize=5, msw=0
        )
    end

    # --- Goal Region ---
    plot!(pl, t->1+0.5cos(t), t->0.05 + 0.04sin(t), 0, 2π, 
        color=:blue, linewidth=2, label="Ideal")

    # --- Legend ---
    # Manually add entries for the Loose (Light) and Strict (Dark) ends
    
    scatter!(pl, [NaN], [NaN], color=color_palette[1],  label="α=$(alphas[1])",  markershape=:circle, msw=0)
    scatter!(pl, [NaN], [NaN], color=color_palette[n_alphas], label="α=$(alphas[end])", markershape=:circle, msw=0)

    # --- Formatting ---
    # Keeping limits identical for easy comparison between the two files
    ylims!(pl, 0.0, 1.0)
    xlims!(pl, 0.0, 20.0)
    
    outpath = joinpath(PLOTDIR, filename)
    savefig(pl, outpath)
    println("Saved: $outpath")
end

# --- Execution ---

# 1. RNN Plot (Red Scheme)
plot_single_method(:RNN, ALPHAS_RNN, "RNN", "risk_vs_time_gradient_rnn.pdf")

# 2. GR Plot (Blue Scheme)
plot_single_method(:GR, ALPHAS_GR, "Gelman-Rubin", "risk_vs_time_gradient_gr.pdf")