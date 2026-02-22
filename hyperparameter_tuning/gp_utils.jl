# ============================================================
# Self-contained Nelder-Mead minimiser (no extra dependencies)
# ============================================================

function _nelder_mead(f, x0::Vector{Float64};
                      max_iter::Int=500, tol::Float64=1e-7)
    d = length(x0)
    α_nm, γ_nm, ρ_nm, σ_nm = 1.0, 2.0, 0.5, 0.5

    simplex = [copy(x0) for _ in 1:d+1]
    for i in 1:d
        simplex[i+1][i] += ifelse(abs(x0[i]) > 1e-8, 0.05*abs(x0[i]), 0.05)
    end
    vals = f.(simplex)

    for _ in 1:max_iter
        p = sortperm(vals); simplex = simplex[p]; vals = vals[p]
        abs(vals[end] - vals[1]) < tol && break
        c  = mean(simplex[1:d])          # centroid (best d points)
        xr = c + α_nm*(c - simplex[end])
        fr = f(xr)
        if fr < vals[1]
            xe = c + γ_nm*(xr - c); fe = f(xe)
            simplex[end], vals[end] = fe < fr ? (xe, fe) : (xr, fr)
        elseif fr < vals[end-1]
            simplex[end], vals[end] = xr, fr
        else
            xc = c + ρ_nm*(simplex[end] - c); fc = f(xc)
            if fc < vals[end]
                simplex[end], vals[end] = xc, fc
            else
                for i in 2:d+1
                    simplex[i] = simplex[1] + σ_nm*(simplex[i] - simplex[1])
                    vals[i]    = f(simplex[i])
                end
            end
        end
    end
    return simplex[1], vals[1]
end
