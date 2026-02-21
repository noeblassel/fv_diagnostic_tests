using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))

include(joinpath(@__DIR__, "..", "FVDiagnosticTests.jl"))
using .FVDiagnosticTests
using Statistics, Random, LinearAlgebra, Printf

const N_POT      = 20
const N_IX       = 5
const β_FIXED    = 2.0
const TOL        = 0.05   # matches get_batch default
const TAU_STUDY1 = 0.1
const NGRID_VALUES = [50, 100, 150, 200, 300, 500]
const NGRID_REF    = 500

const NGRID_STUDY2 = 200   # placeholder; set after Study 1 prints results
const TAU_VALUES   = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
const TAU_REF      = 0.01

rng = Xoshiro(2025)

# Generate potentials once, reuse across both studies
pots = [generate_potential(rng=rng) for _ in 1:N_POT]

# Sample physical starting positions in (0,1) once, reuse for consistent comparison
x0s = [rand(rng, N_IX) for _ in 1:N_POT]

# Helper: map physical position x0 ∈ (0,1) to grid index for a given Ngrid
# Interior grid point ix corresponds to physical position (ix+1)/(Ngrid+2)
grid_ix(x0, Ngrid) = clamp(round(Int, x0 * (Ngrid + 2) - 1), 1, Ngrid)

# ─── Study 1: Ngrid convergence ───────────────────────────────────────────────
println("=== Study 1: Ngrid convergence  (tau_gt = $(TAU_STUDY1)) ===")
println()

T_conv_s1    = zeros(length(NGRID_VALUES), N_POT, N_IX)
time_exp_s1  = zeros(length(NGRID_VALUES), N_POT)

for (gi, Ngrid) in enumerate(NGRID_VALUES)
    for (pi, (W, D, W′, D′)) in enumerate(pots)
        L = comp_generator(W, D, β_FIXED, Ngrid)
        ν, _ = comp_qsd(W, β_FIXED, L)
        t = @elapsed P = exp(-TAU_STUDY1 * Matrix(L))
        time_exp_s1[gi, pi] = t

        for (ii, x0) in enumerate(x0s[pi])
            ix = grid_ix(x0, Ngrid)
            k  = conv_tv(P, ν, ix, TOL)
            T_conv_s1[gi, pi, ii] = max(2, k) * TAU_STUDY1
        end
    end
    @printf("  Ngrid = %3d done\n", Ngrid)
end
println()

ref_gi   = findfirst(==(NGRID_REF), NGRID_VALUES)
T_ref_s1 = T_conv_s1[ref_gi, :, :]   # N_POT × N_IX reference matrix

@printf("%-10s  %-18s  %-20s\n", "Ngrid", "Mean Rel. Error", "Mean Exp. Time (s)")
@printf("%-10s  %-18s  %-20s\n", "-----", "---------------", "------------------")
for (gi, Ngrid) in enumerate(NGRID_VALUES)
    rel_err = mean(abs.(T_conv_s1[gi, :, :] .- T_ref_s1) ./ max.(T_ref_s1, 1e-10))
    t_exp   = mean(time_exp_s1[gi, :])
    marker  = (Ngrid == NGRID_REF) ? " (ref)" : ""
    @printf("%-10d  %-18.4f  %-20.6f%s\n", Ngrid, rel_err, t_exp, marker)
end
println()

# Recommendation for Ngrid: smallest value with mean rel. error < 5%
errs_s1   = [mean(abs.(T_conv_s1[gi,:,:] .- T_ref_s1) ./ max.(T_ref_s1, 1e-10)) for gi in 1:length(NGRID_VALUES)]
rec_ngi   = something(findfirst(<(0.05), errs_s1), length(NGRID_VALUES))
rec_ngrid = NGRID_VALUES[rec_ngi]

# ─── Study 2: tau_gt calibration ──────────────────────────────────────────────
println("=== Study 2: tau_gt calibration (Ngrid = $(NGRID_STUDY2)) ===")
println()

T_conv_s2    = zeros(length(TAU_VALUES), N_POT, N_IX)
niter_s2     = zeros(length(TAU_VALUES), N_POT, N_IX)
time_conv_s2 = zeros(length(TAU_VALUES), N_POT)

for (pi, (W, D, W′, D′)) in enumerate(pots)
    L = comp_generator(W, D, β_FIXED, NGRID_STUDY2)
    ν, _ = comp_qsd(W, β_FIXED, L)

    for (ti, tau_gt) in enumerate(TAU_VALUES)
        P = exp(-tau_gt * Matrix(L))

        t = @elapsed begin
            for (ii, x0) in enumerate(x0s[pi])
                ix = grid_ix(x0, NGRID_STUDY2)
                k  = conv_tv(P, ν, ix, TOL)
                niter_s2[ti, pi, ii]  = max(2, k)
                T_conv_s2[ti, pi, ii] = max(2, k) * tau_gt
            end
        end
        time_conv_s2[ti, pi] = t
    end
end
println("  All potentials done")
println()

ref_ti   = findfirst(==(TAU_REF), TAU_VALUES)
T_ref_s2 = T_conv_s2[ref_ti, :, :]   # N_POT × N_IX reference matrix

@printf("%-10s  %-18s  %-14s  %-20s\n", "tau_gt", "Mean Rel. Error", "Mean N Iter", "Mean Conv. Time (s)")
@printf("%-10s  %-18s  %-14s  %-20s\n", "------", "---------------", "-----------", "-------------------")
for (ti, tau_gt) in enumerate(TAU_VALUES)
    rel_err = mean(abs.(T_conv_s2[ti, :, :] .- T_ref_s2) ./ max.(T_ref_s2, 1e-10))
    n_iter  = mean(niter_s2[ti, :, :])
    t_conv  = mean(time_conv_s2[ti, :])
    marker  = (tau_gt == TAU_REF) ? " (ref)" : ""
    @printf("%-10.3f  %-18.4f  %-14.2f  %-20.6f%s\n", tau_gt, rel_err, n_iter, t_conv, marker)
end
println()

# Recommendation for tau_gt: largest value with mean rel. error < 5%
errs_s2  = [mean(abs.(T_conv_s2[ti,:,:] .- T_ref_s2) ./ max.(T_ref_s2, 1e-10)) for ti in 1:length(TAU_VALUES)]
rec_tai  = something(findlast(<(0.05), errs_s2), 1)
rec_tau  = TAU_VALUES[rec_tai]

println("─── Recommendations ─────────────────────────────────────────────────────────")
@printf("Recommended Ngrid_gt : %d  (smallest Ngrid with mean rel. error < 5%%)\n", rec_ngrid)
@printf("Recommended tau_gt   : %.3f  (largest tau_gt with mean rel. error < 5%%)\n", rec_tau)
