using JLD2,Plots,Statistics,Printf

datadir = "results_diala"

mkpath("figures_diala")

const domain = "basin"
const saddles = Dict(1=>(-3,89), 2=>(44,-92), 3=>(61,122), 4=>(124,-19),5=>(53,28)) # saddle 5 = free-energy minimum

for gamma=[1,2,5,10,20]
        for saddle_ix=1:5
            d = load(joinpath(datadir,"diagnostics_$(saddle_ix)_$(domain)_$(gamma).jld2"))
            pl = plot(title="γ=$(gamma), (ϕ₀,ψ₀)=$(saddles[saddle_ix])",xlabel="time (ps)",ylabel="Diagnostic metric",legend=:topright)

            times = d["times"]

            w1_ϕ = d["w1_ϕ"]
            w1_ψ = d["w1_ψ"]

            gr_ϕ = d["gr_ϕ"]
            gr_ψ = d["gr_ψ"]

            rnn_ϕ = d["rnn_ϕ"]
            rnn_ψ = d["rnn_ψ"]
            

            tv_ϕ = d["tv_ϕ"]
            tv_ψ = d["tv_ψ"]

            J = size(w1_ϕ,2)

            plot!(pl,times,tv_ϕ,label="TV (ϕ)",color=:red,lw=2)
            plot!(pl,times,tv_ψ,label="TV (ψ)",color=:blue,lw=2)

            plot!(pl,[0.0],[NaN],label="GR",color=:black,linestyle=:dash)
            plot!(pl,[0.0],[NaN],label="RNN",color=:black,linestyle=:dot)

            plot!(pl,times,mean(gr_ϕ,dims=2),ribbon=std(gr_ϕ,dims=2)/sqrt(J),color=:red,label="",linestyle=:dash)
            plot!(pl,times,mean(gr_ψ,dims=2),ribbon=std(gr_ψ,dims=2)/sqrt(J),color=:blue,label="",linestyle=:dash)

            plot!(pl,times,mean(rnn_ϕ,dims=2),ribbon=std(rnn_ϕ,dims=2)/sqrt(J),color=:red,label="",linestyle=:dot)
            plot!(pl,times,mean(rnn_ψ,dims=2),ribbon=std(rnn_ψ,dims=2)/sqrt(J),color=:blue,label="",linestyle=:dot)
            ylims!(pl,0,1.0)

            savefig(pl,joinpath("figures_diala","diagnostics_$(saddle_ix)_$(domain)_$(gamma).pdf"))

        end
end


### Crunch data and make tables

const GAMMAS = [1, 2, 5, 10, 20]
const SADDLES = 1:5
const ALPHAS_RNN = [0.5, 0.6,0.7,0.8, 0.9]
const ALPHAS_GR  = [0.2, 0.1, 0.05,0.02, 0.01]
const DOMAIN = "basin"
const DATADIR = "results_diala"
const TV_THRESHOLD = 0.05
const MIN_BURNIN = 0.5 # picoseconds (ignores convergence signals before this time)

# --- Color Logic ---

function mix_color(r::Int, g::Int, b::Int, p::Float64)
    final_r = floor(Int, r * p)
    final_g = floor(Int, g * p)
    final_b = floor(Int, b * p)
    
    bg_style = "rgb($final_r, $final_g, $final_b)"
    
    # Text contrast: if background is dark, use white text
    # Simple brightness formula
    brightness = (final_r * 299 + final_g * 587 + final_b * 114) / 1000
    text_color = (brightness < 100) ? "white" : "black"
    
    return bg_style, text_color
end

"""
Base color logic for Time Efficiency.
Red (Fast/Dangerous) <-> White (Perfect) <-> Blue (Slow)
"""
function get_style_time(val::Float64, p_success::Float64)
    if p_success == 0.0
        return "black", "white"
    end
    
    r, g, b = 255, 255, 255 # Default White

    if val < 1.0
        # Danger Zone: Orange (255,130,0) mixing with White
        # val=0 -> pure red, val=1 -> white
        factor = clamp(val, 0.0, 1.0)
        # Interpolate:
        g = floor(Int, 130 + 125 * factor)
        b = floor(Int, 255 * factor)
        # r stays 255
    else
        # Waste Zone: Blue (0,0,255) mixing with White
        # val=1 -> white, val>= 5-> pure blue
        factor = clamp((val - 1.0) / 4.0, 0.0, 1.0) 
        # Interpolate white (1.0) to blue (0.0 for R/G)
        r = floor(Int, 255 * (1.0 - factor))
        g = floor(Int, 255 * (1.0 - factor))
        # b stays 255
    end

    return mix_color(r, g, b, p_success)
end

"""
Base color logic for Error metric.
White (Perfect) <-> Red (High Error)
"""
function get_style_err(val::Float64, p_success::Float64)
    if p_success == 0.0
        return "black", "white"
    end

    # < 0.05 error -> White (255,255,255)
    # >0.5 error -> Red (255,0,0)
    
    factor = clamp((val-0.05) / (0.5-0.05), 0.0, 1.0) # 0.0 = Good, 1.0 = Bad
    
    r = 255
    g = floor(Int, 255 * (1.0 - factor))
    b = floor(Int, 255 * (1.0 - factor))
    
    return mix_color(r, g, b, p_success)
end

# --- Data Processing ---

function process_file(filepath, alphas, method)
    if !isfile(filepath)
        return nothing
    end
    d = load(filepath)
    
    times = d["times"]
    tv_ϕ, tv_ψ = d["tv_ϕ"], d["tv_ψ"]
    
    # Ground Truth Time
    start_idx = findfirst(t -> t >= MIN_BURNIN, times)
    if isnothing(start_idx)
        start_idx = 1 # Fallback if times < burnin
    end
    
    # Ground Truth Time
    # Modified to use findnext starting from start_idx to avoid initial false positives
    ix_true_ϕ = findnext(x -> x < TV_THRESHOLD, tv_ϕ, start_idx)
    ix_true_ψ = findnext(x -> x < TV_THRESHOLD, tv_ψ, start_idx)
    
    if isnothing(ix_true_ϕ) || isnothing(ix_true_ψ)
        t_true = times[end]
    else
        t_true = times[max(ix_true_ϕ, ix_true_ψ)]
    end

    

    if method == :RNN
        data_ϕ, data_ψ = d["rnn_ϕ"], d["rnn_ψ"]
        comp = (vals, thr) -> vals .>= thr
    else # GR
        data_ϕ, data_ψ = d["gr_ϕ"], d["gr_ψ"]
        comp = (vals, thr) -> vals .<= thr
    end

    res_time = Dict()
    res_err = Dict()
    res_prob = Dict()

    for α in alphas
        t_ratios = Float64[]
        errors = Float64[]
        success_count = 0
        total_count = size(data_ϕ, 2)

        for i in 1:total_count
            col_ϕ = data_ϕ[:, i]
            col_ψ = data_ψ[:, i]
            
            ix_ϕ = findnext(x -> comp(x, α), col_ϕ, start_idx)
            ix_ψ = findnext(x -> comp(x, α), col_ψ, start_idx)
            
            if !isnothing(ix_ϕ) && !isnothing(ix_ψ)
                # Success case
                success_count += 1
                ix = max(ix_ϕ, ix_ψ)
                push!(t_ratios, times[ix] / t_true)
                push!(errors, max(tv_ϕ[ix], tv_ψ[ix]))
            end
        end
        
        p_success = success_count / total_count
        
        if success_count > 0
            res_time[α] = (mean(t_ratios), std(t_ratios))
            res_err[α]  = (mean(errors), std(errors))
        else
            res_time[α] = (last(times)/t_true, NaN)
            res_err[α]  = (NaN, NaN)
        end
        res_prob[α] = p_success
    end
    
    return res_time, res_err, res_prob
end

# --- HTML Generation ---

function generate_html_table(metrics_time, metrics_err, metrics_prob, alphas, title, type_metric)
    
    io = IOBuffer()
    
    println(io, "<div style='margin-bottom: 30px;text-align: center;'>")
    println(io, "<h3>$title</h3>")
    println(io, "<table style='border-collapse: collapse; font-family: sans-serif; font-size: 0.9em; width: 100%; box-shadow: 0 0 10px rgba(0,0,0,0.1);'>")
    
    println(io, "<thead><tr style='background-color: #333; color: white;'>")
    println(io, "<th style='padding:5px; border:1px solid #555; width: 1%; white-space: nowrap;'>Friction \\ π₀</th>")
    for s in SADDLES
        println(io, "<th style='padding:5px; border:1px solid #555;'>Saddle $s $(s== 5 ? "(minimum)" : "")</th>")
    end
    println(io, "</tr></thead><tbody>")
    
    for γ in GAMMAS
        println(io, "<tr>")

        println(io, "<td style='padding:5px; border:1px solid #ddd; font-weight:bold; background-color:#f9f9f9; white-space: nowrap;'>γ = $γ</td>")
        
        for s in SADDLES
            println(io, "<td style='padding:0; border:1px solid #ddd; vertical-align: top;'>")
            
            key = (γ, s)
            
            if !haskey(metrics_time, key)
                print(io, "<div style='padding:5px;'>N/A</div>")
            else
                data_val = (type_metric == :time) ? metrics_time[key] : metrics_err[key]
                probs    = metrics_prob[key]
                
                # Inner table
                println(io, "<table style='width:100%; border-collapse:collapse; margin:0;'>")
                
                for (i, α) in enumerate(alphas)
                    val_mean, val_std = data_val[α]
                    p_success = probs[α]
                    
                    # Determine Colors
                    if type_metric == :time
                        bg_color, txt_color = get_style_time(val_mean, p_success)
                    else
                        bg_color, txt_color = get_style_err(val_mean, p_success)
                    end
                    
                    border_style = (i == length(alphas)) ? "" : "border-bottom: 1px solid rgba(0,0,0,0.1);"
                    
                    print(io, "<tr style='background-color: $bg_color; color: $txt_color; $border_style'>")
                    
                    # Alpha Label:
                    # - Reduced padding (2px)
                    # - width: 1% and white-space: nowrap ensures it takes minimum space
                    print(io, "<td style='padding:2px 5px 2px 2px; font-size:0.8em; width:1%; white-space: nowrap; opacity:0.8; border-right: 1px solid rgba(0,0,0,0.1);'><strong>$α</strong></td>")
                    
                    # Value:
                    # - white-space: nowrap ensures the value never breaks into two lines
                    # - Reduced padding
                    print(io, "<td style='padding:2px 5px; text-align:center; white-space: nowrap;'>")
                    if p_success == 0.0
                        if type_metric == :time
                            @printf(io, "ncv (> %.2f)",val_mean)
                        else
                            print(io, "ncv")
                        end
                    else
                        if p_success == 1.0
                            @printf(io, "%.2f ± %.2f", val_mean, val_std)
                        else    
                            @printf(io, "%.2f ± %.2f (p=%.2f)", val_mean, val_std, p_success)
                        end
                    end
                    println(io, "</td></tr>")
                end
                println(io, "</table>")
            end
            println(io, "</td>")
        end
        println(io, "</tr>")
    end
    
    println(io, "</tbody></table><h4> α ∈ {",join(alphas,", "),"}</h4></div>")
    return String(take!(io))
end

# --- Main Execution ---

function build_report()
    rnn_time = Dict(); rnn_err = Dict(); rnn_prob = Dict()
    gr_time  = Dict(); gr_err  = Dict(); gr_prob  = Dict()
    
    for γ in GAMMAS, s in SADDLES
        fpath = joinpath(DATADIR, "diagnostics_$(s)_$(DOMAIN)_$(γ).jld2")
        
        # Process RNN
        rt, re, rp = process_file(fpath, ALPHAS_RNN, :RNN)
        if !isnothing(rt)
            rnn_time[(γ, s)] = rt; rnn_err[(γ, s)] = re; rnn_prob[(γ, s)] = rp
        end
        
        # Process GR
        gt, ge, gp = process_file(fpath, ALPHAS_GR, :GR)
        if !isnothing(gt)
            gr_time[(γ, s)] = gt; gr_err[(γ, s)] = ge; gr_prob[(γ, s)] = gp
        end
    end
    
    # Generate HTML
    t1 = generate_html_table(rnn_time, rnn_err, rnn_prob, ALPHAS_RNN, "RNN diagnostic", :time)
    t2 = generate_html_table(rnn_time, rnn_err, rnn_prob, ALPHAS_RNN, "RNN diagnostic", :error)
    t3 = generate_html_table(gr_time, gr_err, gr_prob, ALPHAS_GR, "Gelman-Rubin diagnostic", :time)
    t4 = generate_html_table(gr_time, gr_err, gr_prob, ALPHAS_GR, "Gelman-Rubin diagnostic", :error)

    for (fname,t)=zip(["_rnn_time.md","_rnn_error.md","_gr_time.md","_gr_error.md"],[t1,t2,t3,t4])
        open(fname,"w") do f
            println(f, "```{=html}\n")
            println(f, t)
            println(f, "```")
        end
        println("Table generated: $(fname)")
    end
end

build_report()