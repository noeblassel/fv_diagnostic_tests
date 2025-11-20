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

function w1(X,Y)
    @assert size(X) == size(Y)
    Xs = sort(X)
    Ys = sort(Y)
    return sum(abs,Xs-Ys)
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
                      ncut::Int=1,
                      input_dim=input_dim,
                      time_bin_edges=0.0:0.1:2.0,
                      gr_vals=[0.01,0.02,0.05,0.1],
                      rng=Random.default_rng(),
                      )

    nbins_time = length(time_bin_edges)-1
    nvals_gr = length(gr_vals)

    roc_thrs = collect(0.01:0.01:0.99)
    m_roc = length(roc_thrs)

    decorr_steps = Int[]

    fp_mat_gr = zeros(nvals_gr, num_samples)
    fn_mat_gr = zeros(nvals_gr, num_samples)
    tp_mat_gr = zeros(nvals_gr, num_samples)
    tn_mat_gr = zeros(nvals_gr, num_samples)

    acc_mat_gr = zeros(nvals_gr, num_samples)
    prec_mat_gr = zeros(nvals_gr, num_samples)
    rec_mat_gr = zeros(nvals_gr, num_samples)

    roc_tpr_mat_gr = zeros(m_roc, num_samples)
    roc_fpr_mat_gr = zeros(m_roc, num_samples)
    aucs_gr = zeros(num_samples)

    acc_time_gr = zeros(nbins_time, nvals_gr)
    nsamps_time = zeros(nbins_time, nvals_gr)

    fp_mat_w1 = zeros(nvals_gr, num_samples)
    fn_mat_w1 = zeros(nvals_gr, num_samples)
    tp_mat_w1 = zeros(nvals_gr, num_samples)
    tn_mat_w1 = zeros(nvals_gr, num_samples)

    acc_mat_w1 = zeros(nvals_gr, num_samples)
    prec_mat_w1 = zeros(nvals_gr, num_samples)
    rec_mat_w1 = zeros(nvals_gr, num_samples)

    roc_tpr_mat_w1 = zeros(m_roc, num_samples)
    roc_fpr_mat_w1 = zeros(m_roc, num_samples)
    aucs_w1 = zeros(num_samples)

    acc_time_w1 = zeros(nbins_time, nvals_gr)
    

    @showprogress for n = 1:num_samples
        X, Y, mask = get_batch(rng; input_dim=input_dim, ntrace=ntrace, npot=npot_per_chunk, ncut=ncut,feature=dummy_feature, βlims=βlims)
        Y_true = Y[mask]

        _,max_length,batch_size = size(X)
        X = reshape(X,Nreplica,stride,max_length,batch_size)

        Y_gr = zeros(max_length,batch_size)
        Y_w1 = zeros(max_length,batch_size)

        rel_diffs = zeros(max_length-1,batch_size)

        for j=1:batch_size
            l = sum(mask[:,j])
            for i=1:l
                gr_view = view(X,:,:,1:i,j)
                Y_gr[i,j] = var(gr_view)/mean(var(gr_view,dims=2)) - 1.0

                if i>1
                    Y_w1[i,j] = w1(X[:,:,i-1,j][:],X[:,:,i,j][:])
                else
                    Y_w1[i,j] = NaN
                end
            end

            Y_w1[:,j] /= Y_w1[2,j] # normalize wasserstein by first valid sample
        end

        npts = length(Y_true)
        ntrue = sum(Y_true .== 1.0f0)
        nfalse = npts - ntrue

        fp_gr = zeros(nvals_gr)
        fn_gr = zeros(nvals_gr)
        tp_gr = zeros(nvals_gr)
        tn_gr = zeros(nvals_gr)

        prec_gr = zeros(nvals_gr)
        rec_gr = zeros(nvals_gr)
        acc_gr = zeros(nvals_gr)

        fp_w1 = zeros(nvals_gr)
        fn_w1 = zeros(nvals_gr)
        tp_w1 = zeros(nvals_gr)
        tn_w1 = zeros(nvals_gr)

        prec_w1 = zeros(nvals_gr)
        rec_w1 = zeros(nvals_gr)
        acc_w1 = zeros(nvals_gr)


        for (i, th) in enumerate(gr_vals)

            pred_pos_gr = Y_gr[mask] .< th
            pred_pos_w1 = Y_w1[mask] .< th

            tp_i_gr = sum(@. pred_pos_gr && (Y_true == 1.0f0))
            fp_i_gr = sum(@. pred_pos_gr && (Y_true == 0.0f0))
            fn_i_gr = sum(@. !pred_pos_gr && (Y_true == 1.0f0))
            tn_i_gr = sum(@. !pred_pos_gr && (Y_true == 0.0f0))

            tp_i_w1 = sum(@. pred_pos_w1 && (Y_true == 1.0f0))
            fp_i_w1 = sum(@. pred_pos_w1 && (Y_true == 0.0f0))
            fn_i_w1 = sum(@. !pred_pos_w1 && (Y_true == 1.0f0))
            tn_i_w1 = sum(@. !pred_pos_w1 && (Y_true == 0.0f0))

            tp_gr[i] = tp_i_gr / npts
            fp_gr[i] = fp_i_gr / npts
            fn_gr[i] = fn_i_gr / npts
            tn_gr[i] = tn_i_gr / npts


            tp_w1[i] = tp_i_w1 / npts
            fp_w1[i] = fp_i_w1 / npts
            fn_w1[i] = fn_i_w1 / npts
            tn_w1[i] = tn_i_w1 / npts

            acc_gr[i] = (tp_i_gr + tn_i_gr) / max(1, npts)
            prec_gr[i] = tp_i_gr / (tp_i_gr + fp_i_gr)
            rec_gr[i] = (tp_i_gr / ntrue)

            acc_w1[i] = (tp_i_w1 + tn_i_w1) / max(1, npts)
            prec_w1[i] = tp_i_w1 / (tp_i_w1 + fp_i_w1)
            rec_w1[i] = (tp_i_w1 / ntrue)
        end

        fp_mat_gr[:, n] .= fp_gr
        fn_mat_gr[:, n] .= fn_gr
        tp_mat_gr[:, n] .= tp_gr
        tn_mat_gr[:, n] .= tn_gr

        acc_mat_gr[:, n] .= acc_gr
        prec_mat_gr[:, n] .= prec_gr
        rec_mat_gr[:, n] .= rec_gr

        fp_mat_w1[:, n] .= fp_w1
        fn_mat_w1[:, n] .= fn_w1
        tp_mat_w1[:, n] .= tp_w1
        tn_mat_w1[:, n] .= tn_w1

        acc_mat_w1[:, n] .= acc_w1
        prec_mat_w1[:, n] .= prec_w1
        rec_mat_w1[:, n] .= rec_w1

        ntraj = size(Y, 2)

        for j = 1:ntraj
            decorr_step = findfirst(==(1.0f0),Y[:,j])
            traj_length = sum(mask[:, j])

            if decorr_step != nothing
                push!(decorr_steps, decorr_step)

                for k = 1:traj_length
                    time_bin = get_bin(k / decorr_step, first(time_bin_edges), last(time_bin_edges), nbins_time)
                        for (i, th) in enumerate(gr_vals)
                            nsamps_time[time_bin, i] += 1.0
                            acc_time_gr[time_bin,i] += ((Y_gr[k, j] < th) == (Y[k, j] == 1.0f0))
                            acc_time_w1[time_bin, i] += ((Y_w1[k, j] < th) == (Y[k, j] == 1.0f0))
                        end
                end
            end
        end

        for (ridx, rth) in enumerate(roc_thrs)
            pred_pos_gr = Y_gr[mask] .< rth
            pred_pos_w1 = Y_w1[mask] .< rth

            tp_count_gr = sum(@. pred_pos_gr && (Y_true == 1.0f0))
            fp_count_gr = sum(@. pred_pos_gr && (Y_true == 0.0f0))
            roc_tpr_mat_gr[ridx, n] = tp_count_gr / ntrue
            roc_fpr_mat_gr[ridx, n] = fp_count_gr / nfalse

            tp_count_w1 = sum(@. pred_pos_w1 && (Y_true == 1.0f0))
            fp_count_w1 = sum(@. pred_pos_w1 && (Y_true == 0.0f0))
            roc_tpr_mat_w1[ridx, n] = tp_count_w1 / ntrue
            roc_fpr_mat_w1[ridx, n] = fp_count_w1 / nfalse
        end

        aucs_gr[n] = auc_func(roc_fpr_mat_gr[:, n], roc_tpr_mat_gr[:, n])
        aucs_w1[n] = auc_func(roc_fpr_mat_w1[:, n], roc_tpr_mat_w1[:, n])
    end

    mean_fp_gr = vec(mean(fp_mat_gr; dims=2))
    std_fp_gr = vec(std(fp_mat_gr; dims=2))

    mean_fn_gr = vec(mean(fn_mat_gr; dims=2))
    std_fn_gr = vec(std(fn_mat_gr; dims=2))

    mean_tp_gr = vec(mean(tp_mat_gr; dims=2))
    std_tp_gr = vec(std(tp_mat_gr; dims=2))

    mean_tn_gr = vec(mean(tn_mat_gr; dims=2))
    std_tn_gr = vec(std(tn_mat_gr; dims=2))

    mean_acc_gr = vec(mean(acc_mat_gr; dims=2))
    std_acc_gr = vec(std(acc_mat_gr; dims=2))

    mean_prec_gr = vec(mean(prec_mat_gr; dims=2))
    std_prec_gr = vec(std(prec_mat_gr; dims=2))

    mean_rec_gr = vec(mean(rec_mat_gr; dims=2))
    std_rec_gr = vec(std(rec_mat_gr; dims=2))

    mean_roc_tpr_gr = vec(mean(roc_tpr_mat_gr; dims=2))
    mean_roc_fpr_gr = vec(mean(roc_fpr_mat_gr; dims=2))

    std_auc_gr = std(aucs_gr)
    mean_auc_gr = mean(aucs_gr)

    idxs_gr = sortperm(aucs_gr)
    imin_gr = max(1, floor(Int, 0.05 * num_samples))
    imax_gr = min(num_samples, ceil(Int, 0.95 * num_samples))
    lower_idx_gr = idxs_gr[imin_gr]
    upper_idx_gr = idxs_gr[imax_gr]

    ic_roc_tpr_gr = (roc_tpr_mat_gr[:, lower_idx_gr], roc_tpr_mat_gr[:, upper_idx_gr])
    ic_roc_fpr_gr = (roc_fpr_mat_gr[:, lower_idx_gr], roc_fpr_mat_gr[:, upper_idx_gr])

    acc_time_gr .= acc_time_gr ./ nsamps_time

    mean_fp_w1 = vec(mean(fp_mat_w1; dims=2))
    std_fp_w1 = vec(std(fp_mat_w1; dims=2))

    mean_fn_w1 = vec(mean(fn_mat_w1; dims=2))
    std_fn_w1 = vec(std(fn_mat_w1; dims=2))

    mean_tp_w1 = vec(mean(tp_mat_w1; dims=2))
    std_tp_w1 = vec(std(tp_mat_w1; dims=2))

    mean_tn_w1 = vec(mean(tn_mat_w1; dims=2))
    std_tn_w1 = vec(std(tn_mat_w1; dims=2))

    mean_acc_w1 = vec(mean(acc_mat_w1; dims=2))
    std_acc_w1 = vec(std(acc_mat_w1; dims=2))

    mean_prec_w1 = vec(mean(prec_mat_w1; dims=2))
    std_prec_w1 = vec(std(prec_mat_w1; dims=2))

    mean_rec_w1 = vec(mean(rec_mat_w1; dims=2))
    std_rec_w1 = vec(std(rec_mat_w1; dims=2))

    mean_roc_tpr_w1 = vec(mean(roc_tpr_mat_w1; dims=2))
    mean_roc_fpr_w1 = vec(mean(roc_fpr_mat_w1; dims=2))

    std_auc_w1 = std(aucs_w1)
    mean_auc_w1 = mean(aucs_w1)

    idxs_w1 = sortperm(aucs_w1)
    imin_w1 = max(1, floor(Int, 0.05 * num_samples))
    imax_w1 = min(num_samples, ceil(Int, 0.95 * num_samples))
    lower_idx_w1 = idxs_w1[imin_w1]
    upper_idx_w1 = idxs_w1[imax_w1]

    ic_roc_tpr_w1 = (roc_tpr_mat_w1[:, lower_idx_w1], roc_tpr_mat_w1[:, upper_idx_w1])
    ic_roc_fpr_w1 = (roc_fpr_mat_w1[:, lower_idx_w1], roc_fpr_mat_w1[:, upper_idx_w1])

    acc_time_w1 .= acc_time_w1 ./ nsamps_time

    results = (
        mean_fp_gr = mean_fp_gr, std_fp_gr = std_fp_gr,
        mean_fn_gr = mean_fn_gr, std_fn_gr = std_fn_gr,
        mean_tp_gr = mean_tp_gr, std_tp_gr = std_tp_gr,
        mean_tn_gr = mean_tn_gr, std_tn_gr = std_tn_gr,
        mean_acc_gr = mean_acc_gr, std_acc_gr = std_acc_gr,
        mean_prec_gr = mean_prec_gr, std_prec_gr = std_prec_gr,
        mean_rec_gr = mean_rec_gr, std_rec_gr = std_rec_gr,
        mean_roc_tpr_gr = mean_roc_tpr_gr, ic_roc_tpr_gr = ic_roc_tpr_gr,
        mean_roc_fpr_gr = mean_roc_fpr_gr, ic_roc_fpr_gr = ic_roc_fpr_gr,
        mean_auc_gr = mean_auc_gr, std_auc_gr = std_auc_gr,
        acc_time_gr = acc_time_gr,
        mean_fp_w1 = mean_fp_w1, std_fp_w1 = std_fp_w1,
        mean_fn_w1 = mean_fn_w1, std_fn_w1 = std_fn_w1,
        mean_tp_w1 = mean_tp_w1, std_tp_w1 = std_tp_w1,
        mean_tn_w1 = mean_tn_w1, std_tn_w1 = std_tn_w1,
        mean_acc_w1 = mean_acc_w1, std_acc_w1 = std_acc_w1,
        mean_prec_w1 = mean_prec_w1, std_prec_w1 = std_prec_w1,
        mean_rec_w1 = mean_rec_w1, std_rec_w1 = std_rec_w1,
        mean_roc_tpr_w1 = mean_roc_tpr_w1, ic_roc_tpr_w1 = ic_roc_tpr_w1,
        mean_roc_fpr_w1 = mean_roc_fpr_w1, ic_roc_fpr_w1 = ic_roc_fpr_w1,
        mean_auc_w1 = mean_auc_w1, std_auc_w1 = std_auc_w1,
        acc_time_w1 = acc_time_w1,
        roc_thrs = roc_thrs,
        decorr_steps = decorr_steps
    )

    return results
end

time_bin_edges = 0.0:0.1:3.0
bin_centers = (time_bin_edges[1:end-1] .+ time_bin_edges[2:end]) ./ 2

num_chunks = 1000
gr_vals = 0.0:0.01:0.2

rng = Random.default_rng()
t0 = now()
results = eval_metrics(
    num_samples=num_chunks,
    npot_per_chunk=5,
    ntrace=5,
    ncut=10,
    time_bin_edges=time_bin_edges,
    gr_vals=gr_vals,
    rng=rng
)
t1 = now()


println("Evaluation completed in $(t1 - t0)")

json_results = JSON.json(results,allownan=true)

open("eval_results_naive.json","w") do f
    write(f,json_results)
end

# results = JSON.parsefile("$(prefix)_eval_results.json",allownan=true)

σ = 1.96/sqrt(num_chunks)  # 95% confidence interval scaling

plot(gr_vals, results.mean_acc_gr, ribbon=σ*results.std_acc_gr, xlabel=L"\alpha", ylabel="Metric", label=L"Accuracy $\frac{T_P + T_N}{P+N}$", legend=:bottomleft)
plot!(gr_vals,results.mean_prec_gr, ribbon=σ*results.std_prec_gr, label=L"Precision $\frac{T_P}{T_P+F_P}$")
plot!(gr_vals,results.mean_rec_gr, ribbon=σ*results.std_rec_gr, label=L"Recall $\frac{T_P}{P}$")

savefig("metrics_gr.pdf")

plot(gr_vals, results.mean_fp_gr, ribbon=σ*results.std_fp_gr, xlabel=L"\alpha", ylabel="Frequency", label=L"F_P", legend=:left)
plot!(gr_vals,results.mean_fn_gr, ribbon=σ*results.std_fn_gr, label=L"F_N")
plot!(gr_vals,results.mean_tp_gr, ribbon=σ*results.std_tp_gr, label=L"T_P")
plot!(gr_vals,results.mean_tn_gr, ribbon=σ*results.std_tn_gr, label=L"T_N")

savefig("confusion_matrix_gr.pdf")

plot([1.0;results.mean_roc_fpr_gr;0.0],
     [1.0;results.mean_roc_tpr_gr;0.0],
     color=:black, label="mean",
     xlabel=L"False positive rate $\frac{F_P}{N}$",
     ylabel=L"True positive rate $\frac{T_P}{P}$",
     title="ROC curve (AUC = $(round(results.mean_auc_gr, sigdigits=2)) ± $(round(1.96/sqrt(num_chunks)*results.std_auc_gr, sigdigits=2)))",
     aspectratio=1,
     xlims=(-0.1,1.1),
     ylims=(-0.1,1.1))
plot!([1.0;results.ic_roc_fpr_gr[1];0.0],
      [1.0;results.ic_roc_tpr_gr[1];0.0],
      linestyle=:dash, color=:blue, label="AUC quantiles 0.05/0.95")
plot!([1.0;results.ic_roc_fpr_gr[2];0.0],
      [1.0;results.ic_roc_tpr_gr[2];0.0],
      linestyle=:dash, color=:blue, label="")
savefig("roc_curve_gr.pdf")

heatmap(gr_vals,bin_centers,results.acc_time_gr,xlabel=L"\alpha",ylabel=L"t/t_{\mathrm{corr}}")
hline!([1.0], linestyle=:dash, color=:blue, label=L"t = t_{\mathrm{corr}}")
savefig("acc_time_gr.pdf")

plot(gr_vals, results.mean_acc_w1, ribbon=σ*results.std_acc_w1, xlabel=L"\alpha", ylabel="Metric", label=L"Accuracy $\frac{T_P + T_N}{P+N}$", legend=:bottomleft)
plot!(gr_vals,results.mean_prec_w1, ribbon=σ*results.std_prec_w1, label=L"Precision $\frac{T_P}{T_P+F_P}$")
plot!(gr_vals,results.mean_rec_w1, ribbon=σ*results.std_rec_w1, label=L"Recall $\frac{T_P}{P}$")

savefig("metrics_w1.pdf")

plot(gr_vals, results.mean_fp_w1, ribbon=σ*results.std_fp_w1, xlabel=L"\alpha", ylabel="Frequency", label=L"F_P", legend=:left)
plot!(gr_vals,results.mean_fn_w1, ribbon=σ*results.std_fn_w1, label=L"F_N")
plot!(gr_vals,results.mean_tp_w1, ribbon=σ*results.std_tp_w1, label=L"T_P")
plot!(gr_vals,results.mean_tn_w1, ribbon=σ*results.std_tn_w1, label=L"T_N")

savefig("confusion_matrix_w1.pdf")

plot([1.0;results.mean_roc_fpr_w1;0.0],
     [1.0;results.mean_roc_tpr_w1;0.0],
     color=:black, label="mean",
     xlabel=L"False positive rate $\frac{F_P}{N}$",
     ylabel=L"True positive rate $\frac{T_P}{P}$",
     title="ROC curve (AUC = $(round(results.mean_auc_w1, sigdigits=2)) ± $(round(1.96/sqrt(num_chunks)*results.std_auc_w1, sigdigits=2)))",
     aspectratio=1,
     xlims=(-0.1,1.1),
     ylims=(-0.1,1.1))
plot!([1.0;results.ic_roc_fpr_w1[1];0.0],
      [1.0;results.ic_roc_tpr_w1[1];0.0],
      linestyle=:dash, color=:blue, label="AUC quantiles 0.05/0.95")
plot!([1.0;results.ic_roc_fpr_w1[2];0.0],
      [1.0;results.ic_roc_tpr_w1[2];0.0],
      linestyle=:dash, color=:blue, label="")
savefig("roc_curve_w1.pdf")

heatmap(gr_vals,bin_centers,results.acc_time_w1,xlabel=L"\alpha",ylabel=L"t/t_{\mathrm{corr}}")
hline!([1.0], linestyle=:dash, color=:blue, label=L"t = t_{\mathrm{corr}}")
savefig("acc_time_w1.pdf")