using Plots, ProgressMeter, LaTeXStrings
using Statistics, Random
using JSON, Dates

include("./FVDiagnosticTests.jl")
using .FVDiagnosticTests

# trapezoidal integration
@inline trapz(x, y) = sum((y[i] + y[i+1]) / 2 * (x[i+1] - x[i]) for i in 1:length(x)-1)
# binning
@inline get_bin(val,minval,maxval,nbins) = 1+clamp(floor(Int,nbins*(val-minval)/(maxval-minval)),0,nbins-1)

function auc_func(fpr, tpr)
    fpr_full = [1.0; fpr; 0.0]
    tpr_full = [1.0; tpr; 0.0]
    p = sortperm(fpr_full)
    return trapz(fpr_full[p], tpr_full[p])
end

Nreplica = 50
stride = 50

input_dim = Nreplica*stride

βlims = (1.0,3.0) # temperature range

dummy_feature(x,input_dim) = Float32.(x)

function eval_metrics(;
                      num_samples::Int=100,
                      npot_per_chunk::Int=100,
                      ntrace::Int=10,
                      ncut::Int=0,
                      ncorr::Int=2,
                      input_dim=input_dim,
                      time_bin_edges=0.0:0.1:2.0,
                      tol_vals=[0.01,0.02,0.05,0.1],
                      diagnostic = :gr,
                      rng=Random.default_rng(),
                      )

    nbins_time = length(time_bin_edges)-1
    nvals = length(tol_vals)

    roc_thrs = reverse(collect(0.01:0.01:maximum(tol_vals)))
    m_roc = length(roc_thrs)

    decorr_steps = Int[]

    fp_mat = zeros(nvals, num_samples)
    fn_mat = zeros(nvals, num_samples)
    tp_mat = zeros(nvals, num_samples)
    tn_mat = zeros(nvals, num_samples)

    acc_mat = zeros(nvals, num_samples)
    prec_mat = zeros(nvals, num_samples)
    rec_mat = zeros(nvals, num_samples)

    roc_tpr_mat = zeros(m_roc, num_samples)
    roc_fpr_mat = zeros(m_roc, num_samples)
    aucs = zeros(num_samples)

    acc_time = zeros(nbins_time, nvals)
    nsamps_time = zeros(nbins_time, nvals)    

    @showprogress for n = 1:num_samples
        X, Y, mask = get_batch(rng; input_dim=input_dim, ntrace=ntrace, npot=npot_per_chunk, ncut=ncut,feature=dummy_feature, βlims=βlims,ncorr=ncorr,naive=true)
        Y_true = Y[mask]
        Yhat_diagnostic = (diagnostic == :gr) ? X[1,:,:] : X[2,:,:]
        
        npts = length(Y_true)
        ntrue = sum(Y_true .== 1.0f0)
        nfalse = npts - ntrue

        fp = zeros(nvals)
        fn = zeros(nvals)
        tp = zeros(nvals)
        tn = zeros(nvals)

        prec = zeros(nvals)
        rec = zeros(nvals)
        acc = zeros(nvals)


        for (i, th) in enumerate(tol_vals)

            pred_pos = (Yhat_diagnostic[mask] .< th)

            tp_i = sum(@. pred_pos && (Y_true == 1.0f0))
            fp_i = sum(@. pred_pos && (Y_true == 0.0f0))
            fn_i = sum(@. !pred_pos && (Y_true == 1.0f0))
            tn_i = sum(@. !pred_pos && (Y_true == 0.0f0))

            tp[i] = tp_i / npts
            fp[i] = fp_i / npts
            fn[i] = fn_i / npts
            tn[i] = tn_i / npts

            acc[i] = (tp_i + tn_i) / max(1, npts)
            prec[i] = tp_i / (tp_i + fp_i)
            rec[i] = (tp_i / ntrue)

        end

        fp_mat[:, n] .= fp
        fn_mat[:, n] .= fn
        tp_mat[:, n] .= tp
        tn_mat[:, n] .= tn

        acc_mat[:, n] .= acc
        prec_mat[:, n] .= prec
        rec_mat[:, n] .= rec

        ntraj = size(Y, 2)

        for j = 1:ntraj
            decorr_step = findfirst(==(1.0f0),Y[:,j])
            traj_length = sum(mask[:, j])

            if decorr_step != nothing
                push!(decorr_steps, decorr_step)

                for k = 1:traj_length
                    time_bin = get_bin(k / decorr_step, first(time_bin_edges), last(time_bin_edges), nbins_time)
                        for (i, th) in enumerate(tol_vals)
                            nsamps_time[time_bin, i] += 1.0
                            acc_time[time_bin,i] += ((Yhat_diagnostic[k, j] < th) == (Y[k, j] == 1.0f0))
                        end
                end
            end
        end

        for (ridx, rth) in enumerate(roc_thrs)
            pred_pos = Yhat_diagnostic[mask] .< rth

            tp_count = sum(@. pred_pos && (Y_true == 1.0f0))
            fp_count = sum(@. pred_pos && (Y_true == 0.0f0))
            roc_tpr_mat[ridx, n] = tp_count / ntrue
            roc_fpr_mat[ridx, n] = fp_count / nfalse

        end

        aucs[n] = auc_func(roc_fpr_mat[:, n], roc_tpr_mat[:, n])
    end

    mean_fp = vec(mean(fp_mat; dims=2))
    std_fp = vec(std(fp_mat; dims=2))

    mean_fn = vec(mean(fn_mat; dims=2))
    std_fn = vec(std(fn_mat; dims=2))

    mean_tp = vec(mean(tp_mat; dims=2))
    std_tp = vec(std(tp_mat; dims=2))

    mean_tn = vec(mean(tn_mat; dims=2))
    std_tn = vec(std(tn_mat; dims=2))

    mean_acc = vec(mean(acc_mat; dims=2))
    std_acc = vec(std(acc_mat; dims=2))

    mean_prec = vec(mean(prec_mat; dims=2))
    std_prec = vec(std(prec_mat; dims=2))

    mean_rec = vec(mean(rec_mat; dims=2))
    std_rec = vec(std(rec_mat; dims=2))

    mean_roc_tpr = vec(mean(roc_tpr_mat; dims=2))
    mean_roc_fpr = vec(mean(roc_fpr_mat; dims=2))

    std_auc = std(aucs)
    mean_auc = mean(aucs)

    idxs = sortperm(aucs)
    imin = max(1, floor(Int, 0.05 * num_samples))
    imax = min(num_samples, ceil(Int, 0.95 * num_samples))
    lower_idx = idxs[imin]
    upper_idx = idxs[imax]

    ic_roc_tpr = (roc_tpr_mat[:, lower_idx], roc_tpr_mat[:, upper_idx])
    ic_roc_fpr = (roc_fpr_mat[:, lower_idx], roc_fpr_mat[:, upper_idx])

    acc_time .= acc_time ./ nsamps_time

    results = (
        mean_fp = mean_fp, std_fp = std_fp,
        mean_fn = mean_fn, std_fn = std_fn,
        mean_tp = mean_tp, std_tp = std_tp,
        mean_tn = mean_tn, std_tn = std_tn,
        mean_acc = mean_acc, std_acc = std_acc,
        mean_prec = mean_prec, std_prec = std_prec,
        mean_rec = mean_rec, std_rec = std_rec,
        mean_roc_tpr = mean_roc_tpr, ic_roc_tpr = ic_roc_tpr,
        mean_roc_fpr = mean_roc_fpr, ic_roc_fpr = ic_roc_fpr,
        mean_auc = mean_auc, std_auc = std_auc,
        acc_time = acc_time,
        roc_thrs = roc_thrs,
        decorr_steps = decorr_steps
    )

    return results
end

time_bin_edges = 0.0:0.1:2.0
bin_centers = (time_bin_edges[1:end-1] .+ time_bin_edges[2:end]) ./ 2

num_chunks = 1000
tol_vals_gr = 0.01:0.01:0.1

rng = Random.default_rng()
t0 = now()
results = eval_metrics(
    num_samples=num_chunks,
    npot_per_chunk=5,
    ntrace=5,
    ncut=0,
    time_bin_edges=time_bin_edges,
    tol_vals=tol_vals_gr,
    ncorr=2,
    diagnostic=:gr,
    rng=rng
)
t1 = now()


println("Evaluation completed in $(t1 - t0)")

json_results = JSON.json(results,allownan=true)

open("eval_results_gr.json","w") do f
    write(f,json_results)
end

σ = 1.96/sqrt(num_chunks)  # 95% confidence interval scaling

plot(tol_vals_gr, results.mean_acc, ribbon=σ*results.std_acc, xlabel=L"\alpha", ylabel="Metric", label=L"Accuracy $\frac{T_P + T_N}{P+N}$", legend=:bottomleft)
plot!(tol_vals_gr,results.mean_prec, ribbon=σ*results.std_prec, label=L"Precision $\frac{T_P}{T_P+F_P}$")
plot!(tol_vals_gr,results.mean_rec, ribbon=σ*results.std_rec, label=L"Recall $\frac{T_P}{P}$")

savefig("metrics_gr.pdf")

plot(tol_vals_gr, results.mean_fp, ribbon=σ*results.std_fp, xlabel=L"\alpha", ylabel="Frequency", label=L"F_P", legend=:left)
plot!(tol_vals_gr,results.mean_fn, ribbon=σ*results.std_fn, label=L"F_N")
plot!(tol_vals_gr,results.mean_tp, ribbon=σ*results.std_tp, label=L"T_P")
plot!(tol_vals_gr,results.mean_tn, ribbon=σ*results.std_tn, label=L"T_N")

savefig("confusion_matrix_gr.pdf")

plot([1.0;results.mean_roc_fpr;0.0],
     [1.0;results.mean_roc_tpr;0.0],
     color=:black, label="mean",
     xlabel=L"False positive rate $\frac{F_P}{N}$",
     ylabel=L"True positive rate $\frac{T_P}{P}$",
     title="ROC curve (AUC = $(round(results.mean_auc, sigdigits=2)) ± $(round(1.96/sqrt(num_chunks)*results.std_auc, sigdigits=2)))",
     aspectratio=1,
     xlims=(-0.1,1.1),
     ylims=(-0.1,1.1))
plot!([1.0;results.ic_roc_fpr[1];0.0],
      [1.0;results.ic_roc_tpr[1];0.0],
      linestyle=:dash, color=:blue, label="AUC quantiles 0.05/0.95")
plot!([1.0;results.ic_roc_fpr[2];0.0],
      [1.0;results.ic_roc_tpr[2];0.0],
      linestyle=:dash, color=:blue, label="")
savefig("roc_curve_gr.pdf")

heatmap(tol_vals_gr,bin_centers,results.acc_time,xlabel=L"\alpha",ylabel=L"t/t_{\mathrm{corr}}")
hline!([1.0], linestyle=:dash, color=:blue, label=L"t = t_{\mathrm{corr}}")
savefig("acc_time_gr.pdf")


tol_vals_w1 = 0.01:0.01:0.5

rng = Random.default_rng()
t0 = now()
results = eval_metrics(
    num_samples=num_chunks,
    npot_per_chunk=5,
    ntrace=5,
    ncut=0,
    time_bin_edges=time_bin_edges,
    tol_vals=tol_vals_w1,
    ncorr=2,
    diagnostic=:w1,
    rng=rng
)
t1 = now()


println("Evaluation completed in $(t1 - t0)")

json_results = JSON.json(results,allownan=true)

open("eval_results_w1.json","w") do f
    write(f,json_results)
end

σ = 1.96/sqrt(num_chunks)  # 95% confidence interval scaling

plot(tol_vals_w1, results.mean_acc, ribbon=σ*results.std_acc, xlabel=L"\alpha", ylabel="Metric", label=L"Accuracy $\frac{T_P + T_N}{P+N}$", legend=:bottomleft)
plot!(tol_vals_w1,results.mean_prec, ribbon=σ*results.std_prec, label=L"Precision $\frac{T_P}{T_P+F_P}$")
plot!(tol_vals_w1,results.mean_rec, ribbon=σ*results.std_rec, label=L"Recall $\frac{T_P}{P}$")

savefig("metrics_w1.pdf")

plot(tol_vals_w1, results.mean_fp, ribbon=σ*results.std_fp, xlabel=L"\alpha", ylabel="Frequency", label=L"F_P", legend=:left)
plot!(tol_vals_w1,results.mean_fn, ribbon=σ*results.std_fn, label=L"F_N")
plot!(tol_vals_w1,results.mean_tp, ribbon=σ*results.std_tp, label=L"T_P")
plot!(tol_vals_w1,results.mean_tn, ribbon=σ*results.std_tn, label=L"T_N")

savefig("confusion_matrix_w1.pdf")

plot([1.0;results.mean_roc_fpr;0.0],
     [1.0;results.mean_roc_tpr;0.0],
     color=:black, label="mean",
     xlabel=L"False positive rate $\frac{F_P}{N}$",
     ylabel=L"True positive rate $\frac{T_P}{P}$",
     title="ROC curve (AUC = $(round(results.mean_auc, sigdigits=2)) ± $(round(1.96/sqrt(num_chunks)*results.std_auc, sigdigits=2)))",
     aspectratio=1,
     xlims=(-0.1,1.1),
     ylims=(-0.1,1.1))
plot!([1.0;results.ic_roc_fpr[1];0.0],
      [1.0;results.ic_roc_tpr[1];0.0],
      linestyle=:dash, color=:blue, label="AUC quantiles 0.05/0.95")
plot!([1.0;results.ic_roc_fpr[2];0.0],
      [1.0;results.ic_roc_tpr[2];0.0],
      linestyle=:dash, color=:blue, label="")
savefig("roc_curve_w1.pdf")

heatmap(tol_vals_w1,bin_centers,results.acc_time,xlabel=L"\alpha",ylabel=L"t/t_{\mathrm{corr}}")
hline!([1.0], linestyle=:dash, color=:blue, label=L"t = t_{\mathrm{corr}}")
savefig("acc_time_w1.pdf")