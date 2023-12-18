
include("Lorenz96.jl")
include("KalmanFilter.jl")
include("ParticleFilter.jl")

# using Lorenz96
# using KalmanFilter
using Plots
using Random
using StatsBase
using LinearAlgebra
using FStrings

const EPSILON = 0.0001

function main()
    println(stderr, "start")

    lorenz_parameter = Lorenz96.Parameter()
    # 乱数のseed値指定
    seed = 0
    if length(ARGS) > 0
        seed = parse(Int, ARGS[1])
        println(stderr, "--- seed = $seed ---")
    end
    rng = MersenneTwister(seed)


    initial_x = ones(lorenz_parameter.num_sites) * lorenz_parameter.F
    # initial_x[20] *= 1.001
    initial_x += randn(rng, 40)
    Start = 365 / 5
    # Goal = 380 / 5
    Goal = 400 / 5
    # Goal = 365 * 2 / 5

    # 真値の生成
    X = get_true_state(lorenz_parameter, initial_x, Start, Goal)
    tn = Start:lorenz_parameter.h:Goal
    X_converted = convert_matrix(X)

    # 観測の生成
    observed_X = make_observation_data(X, length(tn), rng)


    # アトラクタ上からランダムな初期値をとる
    # initial_x = get_random_point_on_attractor(rng, lorenz_parameter)

    # assignment1_SIS(observed_X, X, X_converted, tn, lorenz_parameter, rng; ensemble_size=100000)
    # assignment1_SIR(observed_X, X, X_converted, tn, lorenz_parameter, rng; ensemble_size=100000)
    # assignment1_SIR_anime(observed_X, X, X_converted, tn, lorenz_parameter, rng; ensemble_size=100000, Ne=10, perturb_r=0.5)
    # assignment1_SIR_perturb_vs_ensemble(observed_X, X, X_converted, tn, lorenz_parameter, rng; Ne=100)
    # assignment1_SIR_perturb_vs_Ne(observed_X, X, X_converted, tn, lorenz_parameter, rng; ensemble_size=5000)
    # assignment1_SIR_anime(observed_X, X, X_converted, tn, lorenz_parameter, rng; ensemble_size=50000, Ne=100, perturb_r=0.05)
    # assignment1_SIR_anime(observed_X, X, X_converted, tn, lorenz_parameter, rng; ensemble_size=50000, Ne=100, perturb_r=5.0)
    # assignment1_SIR(observed_X, X, X_converted, tn, lorenz_parameter, rng; ensemble_size=100000, Ne=10, perturb_r=0.5)

    # assignment2_check_TLM_ADJ(lorenz_parameter, rng)
    # assignment2_leading_SV(X, 1, tn, lorenz_parameter, rng, perturb_r=5.0)
    # assignment2_check_Lanczos(X[1], tn[1], lorenz_parameter, rng; itr_max=3000)
    # assignment2_all_SV(X, tn, lorenz_parameter, rng)

    # assignment3_check_LV(X, tn[1:80], lorenz_parameter, rng; alpha=2.0)
    # assignment3_check_BV(X, tn[1:80], lorenz_parameter, rng; alpha=2.0)

    assignment2_check_Ms(X, tn, lorenz_parameter, rng)
end

function assignment2_check_Ms(X, tn, lorenz_parameter, rng)
    Map = KalmanFilter.get_M_by_approx(X[1], Lorenz96.step, tn[1], lorenz_parameter)

    heatmap(Map, color=:cool, clim=(-1.5, 1.5), 
            title="M_approx", 
            ylabel="\n #th Vector", 
            xlabel="#th site", 
            xticks=[1, 5, 10, 15, 20, 25, 30, 35, 40],
            yticks=[1, 5, 10, 15, 20, 25, 30, 35, 40], yflip=true,
            aspect_ratio=:equal, 
            right_margin = 10Plots.mm,
            dpi=600, size=(625, 600))
    savefig("./1212/M_approx.png")

    # SVD(Map)
    svdM = svd(Map)
    println(stderr, "svd(M)")
    # println(stderr, svdM.U, svdM.S, svdM.V)
    println(stderr, "norm(svdM.U * svdM.S * svdM.V' - Map) = $(norm(svdM.U * Diagonal(svdM.S) * svdM.V' - Map, 2))")
    println(stderr, size(svdM.U), size(svdM.S), size(svdM.V))
    println(stderr, svdM.S[1])
    println(stderr, svdM.V[1, 1])

    heatmap(svdM.U, color=:cool, clim=(-1.5, 1.5), 
            title="svd(M_ap) U", 
            xlabel="\n #th Left Singular Vector mode", 
            ylabel="#th site", 
            xticks=[1, 5, 10, 15, 20, 25, 30, 35, 40],
            yticks=[1, 5, 10, 15, 20, 25, 30, 35, 40], yflip=true,
            aspect_ratio=:equal, 
            right_margin = 10Plots.mm,
            dpi=600, size=(625, 600))
    savefig("./1212/svdM_U.png")

    heatmap(Diagonal(svdM.S), color=:cool, clim=(-1.5, 1.5), 
            title="svd(M_ap) diag(S)", 
            ylabel="\n #th Value σ", 
            xlabel="#th Value σ", 
            xticks=[1, 5, 10, 15, 20, 25, 30, 35, 40],
            yticks=[1, 5, 10, 15, 20, 25, 30, 35, 40], yflip=true,
            aspect_ratio=:equal, 
            right_margin = 10Plots.mm,
            dpi=600, size=(625, 600))
    savefig("./1212/svdM_diagS.png")

    heatmap(svdM.V', color=:cool, clim=(-1.5, 1.5), 
            title="svd(M_ap) VT", 
            ylabel="\n #th Right Singular Vector mode", 
            xlabel="#th site", 
            xticks=[1, 5, 10, 15, 20, 25, 30, 35, 40],
            yticks=[1, 5, 10, 15, 20, 25, 30, 35, 40], yflip=true,
            aspect_ratio=:equal, 
            right_margin = 10Plots.mm,
            dpi=600, size=(625, 600))
    savefig("./1212/svdM_VT.png")

    heatmap(svdM.U * Diagonal(svdM.S) * svdM.V', color=:cool, clim=(-1.5, 1.5), 
    title="U⋅S⋅VT",
    ylabel="\n #th Vector", 
    xlabel="#th site", 
    xticks=[1, 5, 10, 15, 20, 25, 30, 35, 40],
    yticks=[1, 5, 10, 15, 20, 25, 30, 35, 40], yflip=true,
    aspect_ratio=:equal, 
    right_margin = 10Plots.mm,
    dpi=600, size=(625, 600))
    savefig("./1212/svdM_check.png")

    # SVD(MapT⋅Map)
    svdMTM = svd(Map' * Map)

    println(stderr, "svd(MT⋅M)")
    # println(stderr, svdMTM.U, svdMTM.S, svdMTM.V)
    println(stderr, "norm(svdMTM.U * svdMTM.S * svdMTM.V' - Map' * Map) = $(norm(svdMTM.U * Diagonal(svdMTM.S) * svdMTM.V' - Map' * Map, 2))")
    println(stderr, size(svdMTM.U), size(svdMTM.S), size(svdMTM.V))

    println(stderr, "!!!!!!! U ?= V ---> norm(U - V) == $(norm(svdMTM.U - svdMTM.V'))")

    heatmap(Map' * Map, color=:cool, clim=(-1.5, 1.5), 
            title="M_approx", 
            ylabel="\n #th Vector", 
            xlabel="#th site", 
            xticks=[1, 5, 10, 15, 20, 25, 30, 35, 40],
            yticks=[1, 5, 10, 15, 20, 25, 30, 35, 40], yflip=true,
            aspect_ratio=:equal, 
            right_margin = 10Plots.mm,
            dpi=600, size=(625, 600))
    savefig("./1212/MTM_approx.png")

    heatmap(svdMTM.U, color=:cool, clim=(-1.5, 1.5), 
            title="svd(MT⋅M) U", 
            xlabel="\n #th Left Singular Vector mode", 
            ylabel="#th site", 
            xticks=[1, 5, 10, 15, 20, 25, 30, 35, 40],
            yticks=[1, 5, 10, 15, 20, 25, 30, 35, 40], yflip=true,
            aspect_ratio=:equal, 
            right_margin = 10Plots.mm,
            dpi=600, size=(625, 600))
    savefig("./1212/svdMTM_U.png")

    heatmap(Diagonal(svdMTM.S), color=:cool, clim=(-1.5, 1.5), 
            title="svd(MT⋅M) diag(S)", 
            ylabel="\n #th Value σ", 
            xlabel="#th Value σ", 
            xticks=[1, 5, 10, 15, 20, 25, 30, 35, 40],
            yticks=[1, 5, 10, 15, 20, 25, 30, 35, 40], yflip=true,
            aspect_ratio=:equal, 
            right_margin = 10Plots.mm,
            dpi=600, size=(625, 600))
    savefig("./1212/svdMTM_diagS.png")

    heatmap(svdMTM.V', color=:cool, clim=(-1.5, 1.5), 
            title="svd(MT⋅M) VT", 
            ylabel="\n #th Right Singular Vector mode", 
            xlabel="#th site", 
            xticks=[1, 5, 10, 15, 20, 25, 30, 35, 40],
            yticks=[1, 5, 10, 15, 20, 25, 30, 35, 40], yflip=true,
            aspect_ratio=:equal, 
            right_margin = 10Plots.mm,
            dpi=600, size=(625, 600))
    savefig("./1212/svdMTM_VT.png")

    heatmap(svdMTM.U * Diagonal(svdMTM.S) * svdMTM.V', color=:cool, clim=(-1.5, 1.5),
            title="U⋅S⋅VT",
            ylabel="\n #th Vector",
            xlabel="#th site",
            xticks=[1, 5, 10, 15, 20, 25, 30, 35, 40],
            yticks=[1, 5, 10, 15, 20, 25, 30, 35, 40], yflip=true,
            aspect_ratio=:equal,
            right_margin = 10Plots.mm,
            dpi=600, size=(625, 600))
    savefig("./1212/svdMTM_check.png")

    # Eigen(MapT⋅Map)
    eigenMTM = eigen(Map' * Map)
    reverse!(eigenMTM.vectors, dims=2)
    reverse!(eigenMTM.values)

    println(stderr, "eigen(MT⋅M)")
    # println(stderr, eigenMTM.vectors, eigenMTM.values)
    println(stderr, size(eigenMTM.vectors), size(eigenMTM.values))

    heatmap(eigenMTM.vectors, color=:cool, clim=(-1.5, 1.5), 
            title="eigen(MT⋅M) vectors", 
            ylabel="\n #th Eigen Vector mode", 
            xlabel="#th site", 
            xticks=[1, 5, 10, 15, 20, 25, 30, 35, 40], 
            yticks=[1, 5, 10, 15, 20, 25, 30, 35, 40], yflip=true,
            aspect_ratio=:equal,
            right_margin = 10Plots.mm,
            dpi=600, size=(625, 600))
    savefig("./1212/eigenMTM_vectors.png")

    heatmap(Diagonal(eigenMTM.values), color=:cool, clim=(-1.5, 1.5), 
            title="eigen(MT⋅M) values", 
            ylabel="\n #th Value λ", 
            xlabel="#th Value λ", 
            xticks=[1, 5, 10, 15, 20, 25, 30, 35, 40], 
            yticks=[1, 5, 10, 15, 20, 25, 30, 35, 40], yflip=true,
            aspect_ratio=:equal,
            right_margin = 10Plots.mm,
            dpi=600, size=(625, 600))
    savefig("./1212/eigenMTM_values.png")

    heatmap(eigenMTM.vectors * Diagonal(eigenMTM.values) * eigenMTM.vectors', color=:cool, clim=(-1.5, 1.5), 
            title="eigen(MT⋅M) vectors * diag(values) * vectors'", 
            ylabel="\n #th Vector", 
            xlabel="#th site", 
            xticks=[1, 5, 10, 15, 20, 25, 30, 35, 40], 
            yticks=[1, 5, 10, 15, 20, 25, 30, 35, 40], yflip=true,
            aspect_ratio=:equal,
            right_margin = 10Plots.mm,
            dpi=600, size=(625, 600))
    savefig("./1212/eigenMTM_check.png")

    """
    VapT = Vap'
    for k in eachindex(VapT[:, 1])
        if VapT[k, 1] < 0
            VapT[k, :] = -VapT[k, :]
        end
    end

    heatmap(VapT, color=:cool, clim=(-1, 1), 
            title="SV by approx", 
            ylabel="\n Singular Vector", 
            xlabel="direction", 
            xticks=[1, 5, 10, 15, 20, 25, 30, 35, 40],
            yticks=[1, 5, 10, 15, 20, 25, 30, 35, 40],
            aspect_ratio=:equal, 
            right_margin = 10Plots.mm,
            dpi=600, size=(625, 600))
    savefig("SingularVector_approx.png")
    """

    plot(1:lorenz_parameter.num_sites, ones(lorenz_parameter.num_sites), ylims=(0.5, 1.6),
         c=:black, linestyle=:dash, label="σ=1.0", 
         xlabel="i", ylabel="Singular Value", title="sigma_k", 
         dpi=600, size=(600, 600))

    plot!(1:lorenz_parameter.num_sites, svdM.S,
         linewidth=2, marker=:rect, markersize=7, 
         label="S of svd(M)", c=:yellow)
    
    plot!(1:lorenz_parameter.num_sites, sqrt.(svdMTM.S),
         linewidth=2, marker=:utriangle, markersize=6, 
         label="sqrt(S) of svd(MT⋅M)", c=:lightgreen)
    
    plot!(1:lorenz_parameter.num_sites, sqrt.(eigenMTM.values),
         linewidth=2, marker=:x, markersize=5, 
         label="sqrt(Λ) of eigen(MT⋅M)", c=:blue)
    

    Stlm, Vtlm = Lorenz96.get_SV_by_Lanczos(X[1], tn[1], lorenz_parameter, rng; itr_max=3000)
    Vtlm = Vtlm'

    plot!(1:lorenz_parameter.num_sites, sqrt.(Stlm),
    linewidth=1, marker=:+, markersize=5, 
    label="sqrt(S) of Lanczos", c=:red)
    
    savefig("./1212/SingularValue_modes.png")


    heatmap(Vtlm, color=:cool, clim=(-1.5, 1.5), 
    title="SV by Lanczos", 
    xlabel="\n #th Singular Vector mode", 
    ylabel="#th site", 
    xticks=[1, 5, 10, 15, 20, 25, 30, 35, 40],
    yticks=[1, 5, 10, 15, 20, 25, 30, 35, 40], yflip=true,
    aspect_ratio=:equal, 
    right_margin = 10Plots.mm,
    dpi=600, size=(625, 600))
    savefig("./1212/SingularVector_TLM.png")

    heatmap(compare(svdM.U, Vtlm), color=:winter, 
    title="diff (svdM.U - Vtlm)", 
    xlabel="\n #th Singular Vector mode", 
    ylabel="#th site", 
    xticks=[1, 5, 10, 15, 20, 25, 30, 35, 40],
    yticks=[1, 5, 10, 15, 20, 25, 30, 35, 40], yflip=true,
    aspect_ratio=:equal, 
    right_margin = 10Plots.mm,
    dpi=600, size=(625, 600))
    savefig("./1212/svdM_U-Lanczos.png")

    heatmap(compare(svdM.V, Vtlm), color=:winter,
    title="diff (svdM.V - Vtlm)", 
    xlabel="\n #th Singular Vector mode", 
    ylabel="#th site", 
    xticks=[1, 5, 10, 15, 20, 25, 30, 35, 40],
    yticks=[1, 5, 10, 15, 20, 25, 30, 35, 40], yflip=true,
    aspect_ratio=:equal, 
    right_margin = 10Plots.mm,
    dpi=600, size=(625, 600))
    savefig("./1212/svdM_V-Lanczos.png")

    heatmap(compare(svdMTM.U, Vtlm), color=:winter, 
    title="diff (svdMTM.U - Vtlm)", 
    xlabel="\n #th Singular Vector mode", 
    ylabel="#th site", 
    xticks=[1, 5, 10, 15, 20, 25, 30, 35, 40],
    yticks=[1, 5, 10, 15, 20, 25, 30, 35, 40], yflip=true,
    aspect_ratio=:equal, 
    right_margin = 10Plots.mm,
    dpi=600, size=(625, 600))
    savefig("./1212/svdMTM_U-Lanczos.png")

    heatmap(compare(svdMTM.V, Vtlm), color=:winter, 
    title="diff (svdMTM.V - Vtlm)", 
    xlabel="\n #th Singular Vector mode", 
    ylabel="#th site", 
    xticks=[1, 5, 10, 15, 20, 25, 30, 35, 40],
    yticks=[1, 5, 10, 15, 20, 25, 30, 35, 40], yflip=true,
    aspect_ratio=:equal, 
    right_margin = 10Plots.mm,
    dpi=600, size=(625, 600))
    savefig("./1212/svdMTM_V-Lanczos.png")

    heatmap(compare(svdMTM.U, svdMTM.V), color=:winter, 
    title="diff (svdMTM.U - svdMTM.V)", 
    xlabel="\n #th Singular Vector mode", 
    ylabel="#th site", 
    xticks=[1, 5, 10, 15, 20, 25, 30, 35, 40],
    yticks=[1, 5, 10, 15, 20, 25, 30, 35, 40], yflip=true,
    aspect_ratio=:equal, 
    right_margin = 10Plots.mm,
    dpi=600, size=(625, 600))
    savefig("./1212/svdMTM_U-svdMTM_V.png")

    heatmap(compare(eigenMTM.vectors, svdMTM.V), color=:winter, 
    title="diff (eigenMTM.V - svdMTM.V)", 
    xlabel="\n #th Singular Vector mode", 
    ylabel="#th site", 
    xticks=[1, 5, 10, 15, 20, 25, 30, 35, 40],
    yticks=[1, 5, 10, 15, 20, 25, 30, 35, 40], yflip=true,
    aspect_ratio=:equal, 
    right_margin = 10Plots.mm,
    dpi=600, size=(625, 600))
    savefig("./1212/eigenMTM_V-svdMTM_V.png")

    heatmap(compare(eigenMTM.vectors, Vtlm), color=:winter, 
    title="diff (eigenMTM.vectors - Vtlm)", 
    xlabel="\n #th Singular Vector mode", 
    ylabel="#th site", 
    xticks=[1, 5, 10, 15, 20, 25, 30, 35, 40],
    yticks=[1, 5, 10, 15, 20, 25, 30, 35, 40], yflip=true,
    aspect_ratio=:equal, 
    right_margin = 10Plots.mm,
    dpi=600, size=(625, 600))
    savefig("./1212/eigenMTM_V-Lanczos.png")

end

function compare(U, V_lanczos)
    diffV = zeros(Float64, size(U))

    for k in eachindex(U[1, :])
        u = U[:, k]
        v = V_lanczos[:, k]
        diffp = u - v
        diffm = u + v
    
        if norm(diffp, 2) < norm(diffm, 2)
            diffV[:, k] = diffp
        else
            diffV[:, k] = diffm
        end
    end

    return diffV
end

function assignment3_check_BV(X, tn, lorenz_parameter, rng ;alpha=0.9, ens_size=100)
    perturbs = get_random_perturb(rng, lorenz_parameter.num_sites, ens_size, 1)

    diffs1   = zeros(Float64, ens_size, 2 * length(tn) - 1)
    diffsabs = zeros(Float64, ens_size, 2 * length(tn) - 1)
    for k in eachindex(perturbs)
        dX = perturbs[k]
        dX = Lorenz96.TangentLinearCode(tn[1], X[1], dX, lorenz_parameter)

        diffs1[k, 1]   = X[1][1] + dX[1]
        diffsabs[k, 1] = norm(dX, 2)
    end

    similarity = zeros(Float64, length(tn))
    similarity[1] = get_similarity(perturbs)

    for i in eachindex(tn[begin:end-1])
        print(stderr, "i = $i, ")
        for k in eachindex(perturbs)
            print(stderr, "$k, ")
            dX = perturbs[k]
            dX = Lorenz96.step(X[i] + dX, tn[i], lorenz_parameter) - X[i+1]

            diffs1[k, 2*i]   = X[i][1] + dX[1]       # grow
            diffsabs[k, 2*i] = norm(dX, 2) # grow

            if norm(dX, 2) > alpha
                perturbs[k] = alpha * dX / norm(dX, 2)
            else
                perturbs[k] = dX
            end

            diffs1[k, 2*i + 1]   = X[i][1] + perturbs[k][1]       # grow
            diffsabs[k, 2*i + 1] = norm(perturbs[k], 2) # grow
        end

        similarity[i+1] = get_similarity(perturbs)
        print(stderr, "\n")
        println(stderr, perturbs[1])
    end

    plot(tn*5, ones(length(tn)) * alpha, 
         c=:blue, linestyle=:dash, label="alpha = $alpha",
         # yscale=:log10, ylims=(1e-18, 10), yticks=[1e-18, 1e-16, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e0, 10], 
         xlabel="Time [days]", ylabel="similarity", title="Bred Vector convergence; α=$alpha", 
         dpi=600, size=(600, 600)
        )
    plot!(tn*5, zeros(length(tn)), 
          c=:black, linestyle=:dash,
         )
    plot!(tn*5, similarity, 
          c=:red, label="similarity"
          )
    savefig("BV_similarity_$alpha.png")
    println(stderr, "similarity done.")

    tnidx = [tn[1]*5]
    for i in eachindex(tn)
        if i == 1
            continue
        end
        push!(tnidx, 5*tn[i])
        push!(tnidx, 5*tn[i])
    end

    println("$(size(tnidx))), $(size(diffs1))")

    X_converted = convert_matrix(X[1:length(tn)])
    plot(tn*5, X_converted[1, :], 
         c=:black, label="Trajectory",
         xlabel="Time [days]", ylabel="1st site value x1", title="Bred Vector growth (x1); α=$alpha", 
         dpi=600, size=(600, 600), legend=false
        )
    for k in eachindex(perturbs)
        plot!(tnidx, diffs1[k, :], 
              c=:red, alpha=0.2, width=0.5)
    end
    println(stderr, "$((diffs1[1, 1], diffs1[10, 10], diffs1[20, 20]))")
    savefig("BV_x1_$alpha.png")
    println(stderr, "x1 done.")

    plot(tn*5, ones(length(tn)) * alpha, 
         c=:blue, label="alpha = $alpha",
         xlabel="Time [days]", ylabel="abs |X|", title="Bred Vector growth (abs |X|); α=$alpha", 
         dpi=600, size=(600, 600), legend=false
        )
    for k in eachindex(perturbs)
        plot!(tnidx, diffsabs[k, :], 
              c=:red, alpha=0.2, width=0.5)
    end
    println(stderr, "$((diffsabs[1, 1], diffsabs[10, 10], diffsabs[20, 20]))")
    savefig("BV_abs_x_$alpha.png")
    println(stderr, "abs x done.")

    for k in eachindex(perturbs)
        if perturbs[k][40] < 0
            perturbs[k] = -perturbs[k]
        end

        perturbs[k] = perturbs[k] / norm(perturbs[k], 2)
    end
    p_converted = convert_matrix(perturbs)
    heatmap(transpose(p_converted), color=:matter, 
            title="Bred Vector; α=$alpha, time=day $(tn[end]*5)", 
            ylabel="\n Ensemble member", 
            xlabel="site", 
            xticks=[1, 5, 10, 15, 20, 25, 30, 35, 40],
            yticks=[1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            # aspect_ratio=:equal, 
            right_margin = 10Plots.mm,
            dpi=600, size=(625, 600))
    savefig("BVs_heatmap_$alpha.png")
    println(stderr, "heatmap done.")

end

function assignment3_check_LV(X, tn, lorenz_parameter, rng ;alpha=0.9, ens_size=100)
    perturbs = get_random_perturb(rng, lorenz_parameter.num_sites, ens_size, 1)

    diffs1   = zeros(Float64, ens_size, 2 * length(tn) - 1)
    diffsabs = zeros(Float64, ens_size, 2 * length(tn) - 1)
    for k in eachindex(perturbs)
        dX = perturbs[k]
        dX = Lorenz96.TangentLinearCode(tn[1], X[1], dX, lorenz_parameter)

        diffs1[k, 1]   = X[1][1] + dX[1]
        diffsabs[k, 1] = norm(dX, 2)
    end

    similarity = zeros(Float64, length(tn))
    similarity[1] = get_similarity(perturbs)

    for i in eachindex(tn[begin:end-1])
        print(stderr, "i = $i, ")
        for k in eachindex(perturbs)
            print(stderr, "$k, ")
            dX = perturbs[k]
            dX = Lorenz96.TangentLinearCode(tn[i], X[i], dX, lorenz_parameter)

            diffs1[k, 2*i]   = X[i][1] + dX[1]       # grow
            diffsabs[k, 2*i] = norm(dX, 2) # grow

            if norm(dX, 2) > alpha
                perturbs[k] = alpha * dX / norm(dX, 2)
            else
                perturbs[k] = dX
            end

            diffs1[k, 2*i + 1]   = X[i][1] + perturbs[k][1]       # grow
            diffsabs[k, 2*i + 1] = norm(perturbs[k], 2) # grow
        end

        similarity[i+1] = get_similarity(perturbs)
        print(stderr, "\n")
        println(stderr, perturbs[1])
    end

    plot(tn*5, ones(length(tn)) * alpha, 
         c=:blue, linestyle=:dash, label="alpha = $alpha",
         # yscale=:log10, ylims=(1e-18, 10), yticks=[1e-18, 1e-16, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e0, 10], 
         xlabel="Time [days]", ylabel="similarity", title="Lyapnov Vector convergence; α=$alpha", 
         dpi=600, size=(600, 600)
        )
    plot!(tn*5, zeros(length(tn)), 
          c=:black, linestyle=:dash,
         )
    plot!(tn*5, similarity, 
          c=:red, label="similarity"
          )
    savefig("LV_similarity_$alpha.png")
    println(stderr, "similarity done.")

    tnidx = [tn[1]*5]
    for i in eachindex(tn)
        if i == 1
            continue
        end
        push!(tnidx, 5*tn[i])
        push!(tnidx, 5*tn[i])
    end

    println("$(size(tnidx))), $(size(diffs1))")

    X_converted = convert_matrix(X[1:length(tn)])
    plot(tn*5, X_converted[1, :], 
         c=:black, label="Trajectory",
         xlabel="Time [days]", ylabel="1st site value x1", title="Lyapnov Vector growth (x1); α=$alpha", 
         dpi=600, size=(600, 600), legend=false
        )
    for k in eachindex(perturbs)
        plot!(tnidx, diffs1[k, :], 
              c=:red, alpha=0.2, width=0.5)
    end
    println(stderr, "$((diffs1[1, 1], diffs1[10, 10], diffs1[20, 20]))")
    savefig("LV_x1_$alpha.png")
    println(stderr, "x1 done.")

    plot(tn*5, ones(length(tn)) * alpha, 
         c=:blue, label="alpha = $alpha",
         xlabel="Time [days]", ylabel="abs |X|", title="Lyapnov Vector growth (abs |X|); α=$alpha", 
         dpi=600, size=(600, 600), legend=false
        )
    for k in eachindex(perturbs)
        plot!(tnidx, diffsabs[k, :], 
              c=:red, alpha=0.2, width=0.5)
    end
    println(stderr, "$((diffsabs[1, 1], diffsabs[10, 10], diffsabs[20, 20]))")
    savefig("LV_abs_x_$alpha.png")
    println(stderr, "abs x done.")

    for k in eachindex(perturbs)
        if perturbs[k][40] < 0
            perturbs[k] = -perturbs[k]
        end

        perturbs[k] = perturbs[k] / norm(perturbs[k], 2)
    end
    p_converted = convert_matrix(perturbs)
    heatmap(transpose(p_converted), color=:matter, 
            title="Lyapnov Vector; α=$alpha, time=day $(tn[end]*5)", 
            ylabel="\n Ensemble member", 
            xlabel="site", 
            xticks=[1, 5, 10, 15, 20, 25, 30, 35, 40],
            yticks=[1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            # aspect_ratio=:equal, 
            right_margin = 10Plots.mm,
            dpi=600, size=(625, 600))
    savefig("LVs_heatmap_$alpha.png")
    println(stderr, "heatmap done.")

end

function get_random_perturb(rng, dimension, ens_size, perturb_r)
    perturb = [randn(rng, Float64, dimension) for _ in 1:ens_size]
    for k in eachindex(perturb)
        perturb[k] = perturb_r * perturb[k] / norm(perturb[k], 2)
    end

    return perturb
end


function assignment2_all_SV(X, tn, lorenz_parameter, rng)
    Map = KalmanFilter.get_M_by_approx(X[1], Lorenz96.step, tn[1], lorenz_parameter)
    Uap, Sap, Vap = KalmanFilter.get_SVD_by_approx(X[1], Lorenz96.step, tn[1], lorenz_parameter)

    println(stderr, Uap, Sap, Vap)
    println(stderr, "norm(Uap * Sap * Vap' - Map' * Map) = $(norm(Uap * Diagonal(Sap) * Vap' - Map' * Map, 2))")
    println(stderr, size(Uap), size(Sap), size(Vap))
    println(stderr, Sap[1])
    println(stderr, Vap[1, 1])

    VapT = Vap'

    for k in eachindex(VapT[:, 1])
        if VapT[k, 1] < 0
            VapT[k, :] = -VapT[k, :]
        end
    end

    heatmap(VapT, color=:cool, clim=(-1, 1), 
            title="SV by approx", 
            ylabel="\n Singular Vector", 
            xlabel="direction", 
            xticks=[1, 5, 10, 15, 20, 25, 30, 35, 40],
            yticks=[1, 5, 10, 15, 20, 25, 30, 35, 40],
            aspect_ratio=:equal, 
            right_margin = 10Plots.mm,
            dpi=600, size=(625, 600))
    savefig("SingularVector_approx.png")

    plot(1:lorenz_parameter.num_sites, ones(lorenz_parameter.num_sites), ylims=(0.5, 1.6),
         c=:black, linestyle=:dash, label="σ=1.0", 
         xlabel="i", ylabel="Singular Value", title="sigma_k", 
         dpi=600, size=(600, 600))

    plot!(1:lorenz_parameter.num_sites, sqrt.(Sap),
         linewidth=2, marker=:x, markersize=5, 
         label="approx", c=:blue)
    

    Stlm, Vtlm = Lorenz96.get_SV_by_Lanczos(X[1], tn[1], lorenz_parameter, rng; itr_max=3000)

    plot!(1:lorenz_parameter.num_sites, sqrt.(Stlm),
    linewidth=1, marker=:+, markersize=5, 
    label="TLM", c=:red)
    
    savefig("SingularValue_modes.png")


    heatmap(Vtlm, color=:cool, clim=(-1, 1), 
    title="SV by TLM", 
    ylabel="\n Singular Vector mode", 
    xlabel="direction", 
    xticks=[1, 5, 10, 15, 20, 25, 30, 35, 40],
    yticks=[1, 5, 10, 15, 20, 25, 30, 35, 40],
    aspect_ratio=:equal, 
    right_margin = 10Plots.mm,
    dpi=600, size=(625, 600))
    savefig("SingularVector_TLM.png")


    heatmap(VapT - Vtlm, color=:winter, 
    title="diff (byApprox - byTLM)", 
    ylabel="\n Singular Vector mode", 
    xlabel="direction", 
    xticks=[1, 5, 10, 15, 20, 25, 30, 35, 40],
    yticks=[1, 5, 10, 15, 20, 25, 30, 35, 40],
    aspect_ratio=:equal, 
    right_margin = 10Plots.mm,
    dpi=600, size=(625, 600))
    savefig("SingularVector_diff.png")
end


function assignment2_check_Lanczos(X0, ti, lorenz_parameter, rng; itr_max=1000)

    V = zeros(Float64, lorenz_parameter.num_sites, lorenz_parameter.num_sites)

    plot(1:itr_max, ones(itr_max) * 1e-16, 
         yscale=:log10, ylims=(1e-18, 10), yticks=[1e-18, 1e-16, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e0, 10], 
         c=:black, linestyle=:dash, label="bottom (1e-16)",
         xlabel="iteration", ylabel="similarity", title="Lanczos convergence", 
         dpi=600, size=(600, 600)
        )

    for i in 1:40
        println(stderr, "i = $i")

        perturbs = [randn(rng, Float64, lorenz_parameter.num_sites) for _ in 1:100]
        perturbs = redux_DOF(perturbs, V, i)

        diffs = zeros(Float64, itr_max)
        print(stderr, "itr = ")
        for itr in 1:itr_max
            perturbs = redux_DOF(perturbs, V, i) # avoid numeric noise

            for k in eachindex(perturbs)
                dX = perturbs[k]

                if dX[1] < 0
                    dX = -dX
                end

                Mx = Lorenz96.TangentLinearCode(ti, X0, dX, lorenz_parameter)
                MTMx = Lorenz96.AdjointCode(ti, X0, Mx, lorenz_parameter)

                perturbs[k] = MTMx / norm(MTMx, 2)
            end

            diffs[itr] = get_similarity(perturbs)
        end

        if i in [1, 2, 3, 4, 5, 10, 20, 30, 40]
            plot!(1:itr_max, diffs, label="mode i = $(i)")
        end

        V[i, :] = get_mean_vector(perturbs)
        print(stderr, "\n")
    end

    println(V[1, :] ⋅ V[2, :])

    savefig("Lanczos_similarity.png")

    S = zeros(Float64, lorenz_parameter.num_sites)
    for i in 1:40
        Mv = Lorenz96.TangentLinearCode(ti, X0, V[i, :], lorenz_parameter)
        MTMv = Lorenz96.AdjointCode(ti, X0, Mv, lorenz_parameter)
        
        S_vec = MTMv ./ V[i, :]
        S[i] = mean(S_vec)
        plot(1:lorenz_parameter.num_sites, S_vec, title="$i")
        plot!(1:lorenz_parameter.num_sites, ones(lorenz_parameter.num_sites) * S[i])

        savefig("SingularValue_mode_S_vec$i.png")
    end

    plot(1:lorenz_parameter.num_sites, ones(lorenz_parameter.num_sites), ylims=(0.3, 2.5),
    c=:black, linestyle=:dash, label="σ=1.0", 
    xlabel="i", ylabel="Singular Value", title="sigma_k", 
    dpi=600, size=(600, 600))

    plot!(1:lorenz_parameter.num_sites, S,
        linewidth=2, marker=:o, markersize=5, 
        label="TLM", c=:red)

    savefig("SingularValue_modes_tlm.png")

end

function redux_DOF(perturbs, V, i)
    if i > 1
        for k in eachindex(perturbs)
            for j in 1:i-1
                perturbs[k] -= (perturbs[k] ⋅ V[j, :]) * V[j, :]
            end
        end
    end

    return perturbs
end

function get_mean_vector(perturbs)
    mean_vector = zeros(Float64, length(perturbs[1]))
    for k in eachindex(perturbs)
        mean_vector += perturbs[k]
    end

    mean_vector = mean_vector / length(perturbs)

    return mean_vector / norm(mean_vector, 2)
end

function get_similarity(perturbs)
    sim_mat = zeros(Float64, length(perturbs))
    for k in eachindex(perturbs)
        if k == 1
            sim_mat[k] = norm(perturbs[k] - perturbs[end], 2)
        else
            sim_mat[k] = norm(perturbs[k] - perturbs[k-1], 2)
        end
    end

    return mean(sim_mat)
end

function assignment2_leading_SV(X, i, tn, lorenz_parameter, rng; perturb_r=1.0)
    sites = 1:lorenz_parameter.num_sites

    perturbs = [ones(lorenz_parameter.num_sites) for _ in 1:1000]

    for k in eachindex(perturbs)
        perturbs[k] = perturb_r * randn(rng, lorenz_parameter.num_sites)
        if perturb[k][1] < 0
            perturb[k] = -perturb[k]
        end
    end

    X0 = X[i]

    for itr in 1:2
        println(stderr, "itr = $itr")

        plot(sites, X0-X0, ylims=(-5, 5),
             label="X0", xlabel="site", ylabel="value", title="X0", 
             dpi=600, size=(600, 600), c=:red)

        for k in eachindex(perturbs)
            dX = perturbs[k]

            Mx = Lorenz96.TangentLinearCode(tn[i], X0, dX, lorenz_parameter)
            MTMx = Lorenz96.AdjointCode(tn[i], X0, Mx, lorenz_parameter)

            MTMx = perturb_r * MTMx / norm(MTMx, 2)

            plot!(sites, MTMx, 
                  c=:cyan, alpha=0.2, width=0.5)
            
            perturbs[k] = MTMx
        end

        plot!(sites, X0-X0, c=:red, width=2,
             label="X0", xlabel="site", title="t = $(5 * tn[i]) (day); (MTM)^$(itr) dX",
             legend=false)

        savefig("0_SingularVector_dX.png")

    end
end

function D(X0, dX, lorenz_parameter)
    top = norm(Lorenz96.step(X0 + dX, 0.0, lorenz_parameter) - Lorenz96.step(X0, 0.0, lorenz_parameter), 2)
    bottom = norm(Lorenz96.TangentLinearCode(0.0, X0, dX, lorenz_parameter), 2)

    return top / bottom
end

function assignment2_check_TLM_ADJ(lorenz_parameter, rng)
    X0 = zeros(lorenz_parameter.num_sites)
    X0[20] += 1.0

    dX0 = 0.01 * randn(rng, lorenz_parameter.num_sites)

    # Lorenz-96 TLM and ADJ
    dV_true = Lorenz96.lorenz_96(X0 + dX0, 0.0, lorenz_parameter) - Lorenz96.lorenz_96(X0, 0.0, lorenz_parameter)
    Lx = Lorenz96.l96_tlm(X0, dX0)
    LTLx = Lorenz96.l96_adj(X0, Lx)

    println(stderr, "Lorenz96: dV⋅dV ~= Lx⋅Lx == x⋅LTLx")
    println(stderr, "$(dV_true ⋅ dV_true) ~= $(Lx ⋅ Lx) == $(dX0 ⋅ LTLx)")

    # Model TLM and ADJ
    dnX_true = Lorenz96.step(X0 + dX0, 0.0, lorenz_parameter) - Lorenz96.step(X0, 0.0, lorenz_parameter)
    Mx = Lorenz96.TangentLinearCode(0.0, X0, dX0, lorenz_parameter)
    MTMx = Lorenz96.AdjointCode(0.0, X0, Mx, lorenz_parameter)

    println(stderr, "Model: dnX⋅dnX ~= Mx⋅Mx == x⋅MTMx")
    println(stderr, "$(dnX_true ⋅ dnX_true) ~= $(Mx ⋅ Mx) == $(dX0 ⋅ MTMx)")


    sites = 1:lorenz_parameter.num_sites

    perturbs = [ones(lorenz_parameter.num_sites) for _ in 1:100]

    for k in eachindex(perturbs)
        perturbs[k] = randn(rng, lorenz_parameter.num_sites)
    end

    alphas = 0.01:0.01:1

    plot([1], [1], c=RGBA(0, 0, 0, 0), 
        xlabel="α", ylabel="D(α)", title="TLM check", 
        legend=false
        )

    for dX in perturbs
        y = []
        for alpha in alphas
            push!(y, D(X0, alpha * dX, lorenz_parameter))
        end

        plot!(alphas, y;
              c=rand(RGB), alpha=0.2, 
              st=:scatter, markersize=3, width = 0.05)
    end

    savefig("TLM_check_alpha.png")

end

function assignment1_SIR_perturb_vs_Ne(observed_X, true_X, true_x_converted ,tn, lorenz_parameter, rng; ensemble_size=10000)
    sir_Xa = [Float64[] for i = 1:length(tn)]
    sir_W = [Float64[] for i = 1:length(tn)]
    sir_N_eff = zeros(Float64, length(tn))

    N = length(tn) - 1

    num_observed = lorenz_parameter.num_sites
    initial_H = Matrix{Int64}(I, num_observed, num_observed)
    initial_R = Matrix{Int64}(I, num_observed, num_observed)

    N_cand = [0, 10, 100, 500, 1000, 10000]
    p_cand = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5]
    result_mat = zeros(Float64, length(N_cand), length(p_cand)) # mean RMSE
    for N_idx in eachindex(N_cand)
        Ne = N_cand[N_idx]

        println(stderr, "m = $(ensemble_size)")

        # make initial ensemble
        initial_Xf = [Float64[] for _ = 1:ensemble_size]
        for k = 1:ensemble_size
            # initial_Xf[k] = get_random_point_on_attractor(rng, lorenz_parameter)
            initial_Xf[k] = observed_X[1] + 1.0 * randn(rng, lorenz_parameter.num_sites)
        end
        println(stderr, "initial ensemble generated")

        for p_idx in eachindex(p_cand)
            perturb_r = p_cand[p_idx]

            particle_parameter = ParticleFilter.Parameter(Lorenz96.step, ensemble_size)
            sir_snap_shot = ParticleFilter.SnapShot(initial_Xf, ones(ensemble_size) / ensemble_size, initial_H, initial_R)
            
            println(stderr, "start assimilation Ne = $(Ne), |η| = $(perturb_r)")
            sir_Xa[1], sir_W[1], sir_N_eff[1] = store_result(sir_snap_shot)
            
            for i in 1:N
                print(stderr, f"t = {tn[i+1]:.2f}; ")
                y_observed = observed_X[i+1]
            
                sir_snap_shot = ParticleFilter.SIR(y_observed, tn[i+1], particle_parameter, lorenz_parameter, sir_snap_shot, rng, 
                                                   Ne=Ne, perturb_r=perturb_r)
            
                sir_Xa[i+1], sir_W[i+1], sir_N_eff[i+1] = store_result(sir_snap_shot)
                print(stderr, "done.\n")
            end

            result_mat[N_idx, p_idx] = mean(get_diffs(lorenz_parameter.num_sites, sir_Xa, true_X))
        end
    end

    println(stderr, "assimilation done; plotting...")

    heatmap(result_mat, color=:matter, 
            title="mean RMSE; ensemble_size = $(ensemble_size) (fixed)", 
            ylabel="\n Resampling Threshold: Ne", 
            yticks=([idx for idx in eachindex(N_cand)], [string(N_cand[idx]) for idx in eachindex(N_cand)]), 
            xlabel="perturb radius: |η|", 
            xticks=([idx for idx in eachindex(p_cand)], [string(p_cand[idx]) for idx in eachindex(p_cand)]),
            dpi=600, size=(900, 600))
    
    fontsize = 12
    nrow, ncol = size(result_mat)
    ann = [(j,i, text(round(result_mat[i,j], digits=2), fontsize, :gray20, :center))
                for i in 1:nrow for j in 1:ncol]
    annotate!(ann, linecolor=:white)

    savefig("SIR_perturb_vs_Ne.png")
end

function assignment1_SIR_perturb_vs_ensemble(observed_X, true_X, true_x_converted ,tn, lorenz_parameter, rng; Ne=100)
    sir_Xa = [Float64[] for i = 1:length(tn)]
    sir_W = [Float64[] for i = 1:length(tn)]
    sir_N_eff = zeros(Float64, length(tn))

    N = length(tn) - 1

    num_observed = lorenz_parameter.num_sites
    initial_H = Matrix{Int64}(I, num_observed, num_observed)
    initial_R = Matrix{Int64}(I, num_observed, num_observed)

    m_cand = [100, 1000, 5000, 10000, 50000, 100000]
    p_cand = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5]
    result_mat = zeros(Float64, length(m_cand), length(p_cand)) # mean RMSE
    for m_idx in eachindex(m_cand)
        ensemble_size = m_cand[m_idx]

        println(stderr, "m = $(ensemble_size)")

        # make initial ensemble
        initial_Xf = [Float64[] for _ = 1:ensemble_size]
        for k = 1:ensemble_size
            # initial_Xf[k] = get_random_point_on_attractor(rng, lorenz_parameter)
            initial_Xf[k] = observed_X[1] + 1.0 * randn(rng, lorenz_parameter.num_sites)
        end
        println(stderr, "initial ensemble generated")

        for p_idx in eachindex(p_cand)
            perturb_r = p_cand[p_idx]

            particle_parameter = ParticleFilter.Parameter(Lorenz96.step, ensemble_size)
            sir_snap_shot = ParticleFilter.SnapShot(initial_Xf, ones(ensemble_size) / ensemble_size, initial_H, initial_R)
            
            println(stderr, "start assimilation m = $(ensemble_size), |η| = $(perturb_r)")
            sir_Xa[1], sir_W[1], sir_N_eff[1] = store_result(sir_snap_shot)
            
            for i in 1:N
                print(stderr, f"t = {tn[i+1]:.2f}; ")
                y_observed = observed_X[i+1]
            
                sir_snap_shot = ParticleFilter.SIR(y_observed, tn[i+1], particle_parameter, lorenz_parameter, sir_snap_shot, rng, 
                                                   Ne=Ne, perturb_r=perturb_r)
            
                sir_Xa[i+1], sir_W[i+1], sir_N_eff[i+1] = store_result(sir_snap_shot)
                print(stderr, "done.\n")
            end

            result_mat[m_idx, p_idx] = mean(get_diffs(lorenz_parameter.num_sites, sir_Xa, true_X))
        end
    end

    println(stderr, "assimilation done; plotting...")

    heatmap(result_mat, color=:matter, 
            title="mean RMSE; Ne = $(Ne) (fixed)", 
            ylabel="\n ensemble size: m", 
            yticks=([idx for idx in eachindex(m_cand)], [string(m_cand[idx]) for idx in eachindex(m_cand)]), 
            xlabel="perturb radius: |η|", 
            xticks=([idx for idx in eachindex(p_cand)], [string(p_cand[idx]) for idx in eachindex(p_cand)]),
            dpi=600, size=(900, 600))
    
    fontsize = 12
    nrow, ncol = size(result_mat)
    ann = [(j,i, text(round(result_mat[i,j], digits=2), fontsize, :gray20, :center))
                for i in 1:nrow for j in 1:ncol]
    annotate!(ann, linecolor=:white)

    savefig("SIR_perturb_vs_ensemble.png")
end

function assignment1_SIR_anime(observed_X, true_X, true_x_converted ,tn, lorenz_parameter, rng; ensemble_size=10000, Ne=100, perturb_r=1.0)
    sis_Xa = [Float64[] for i = 1:length(tn)]
    sis_W = [Float64[] for i = 1:length(tn)]
    sis_N_eff = zeros(Float64, length(tn))

    sir_Xa = [Float64[] for i = 1:length(tn)]
    sir_W = [Float64[] for i = 1:length(tn)]
    sir_N_eff = zeros(Float64, length(tn))

    N = length(tn) - 1

    sis_Xf1 = [Float64[] for i = 1:length(tn)]
    sir_Xf1 = [Float64[] for i = 1:length(tn)]

    # make initial ensemble
    initial_Xf = [Float64[] for _ = 1:ensemble_size]
    for k = 1:ensemble_size
        # initial_Xf[k] = get_random_point_on_attractor(rng, lorenz_parameter)
        initial_Xf[k] = observed_X[1] + 1.0 * randn(rng, lorenz_parameter.num_sites)
    end
    println(stderr, "initial ensemble generated")

    num_observed = lorenz_parameter.num_sites
    initial_H = Matrix{Int64}(I, num_observed, num_observed)
    initial_R = Matrix{Int64}(I, num_observed, num_observed)

    particle_parameter = ParticleFilter.Parameter(Lorenz96.step, ensemble_size)
    sis_snap_shot = ParticleFilter.SnapShot(initial_Xf, ones(ensemble_size) / ensemble_size, initial_H, initial_R)
    sir_snap_shot = ParticleFilter.SnapShot(initial_Xf, ones(ensemble_size) / ensemble_size, initial_H, initial_R)

    println(stderr, "start assimilation")
    sis_Xa[1], sis_W[1], sis_N_eff[1] = store_result(sis_snap_shot)
    sir_Xa[1], sir_W[1], sir_N_eff[1] = store_result(sir_snap_shot)

    sis_Xf1[1] = [initial_Xf[k][1] for k = 1:ensemble_size]
    sir_Xf1[1] = [initial_Xf[k][1] for k = 1:ensemble_size]

    for i in 1:N
        print(stderr, f"t = {tn[i+1]:.2f}; ")
        y_observed = observed_X[i+1]

        sis_snap_shot = ParticleFilter.SIS(y_observed, tn[i+1], particle_parameter, lorenz_parameter, sir_snap_shot)

        sir_snap_shot = ParticleFilter.SIR(y_observed, tn[i+1], particle_parameter, lorenz_parameter, sir_snap_shot, rng, 
                                           Ne=Ne, perturb_r=perturb_r)

        sis_Xa[i+1], sis_W[i+1], sis_N_eff[i+1] = store_result(sis_snap_shot)
        sir_Xa[i+1], sir_W[i+1], sir_N_eff[i+1] = store_result(sir_snap_shot)

        sis_Xf1[i+1] = [sis_snap_shot.Xf[k][1] for k = 1:ensemble_size]
        sir_Xf1[i+1] = [sir_snap_shot.Xf[k][1] for k = 1:ensemble_size]
        print(stderr, "done.\n")
    end

    println(stderr, "assimilation done; plotting...")

    savefig_RMSEvsOBSE(sir_Xa, observed_X, true_X, tn, title="SIR method; \n m = $(ensemble_size), Ne = $(Ne), |η| = $(perturb_r)", filename="SIR_RMSE-$(ensemble_size).png")

    savefig_N_eff_sis_sir(tn, sis_N_eff, sir_N_eff, Ne, title="SIR method; \n m = $(ensemble_size), Ne = $(Ne), |η| = $(perturb_r)", filename="SIR_N_eff-$(ensemble_size).png")

    # plot(x, line=:stem, marker=:star, markersize=20)
    for iframe in 1:N
        print(stderr, "iframe = $iframe; ")

        m = sis_N_eff[1]
        s  = 10 / m
        x  = [i for i in -12:0.01:12]
        y  = observed_X[iframe][1]
        Bp = s * exp.(-0.5 * (y .- x).^2)

        xt = true_X[iframe][1]

        #-SIS store_result--------------------------------
        xa = sis_Xa[iframe][1]
        xt = true_X[iframe][1]
        w  = sis_W[iframe]

        plot( sis_Xf1[iframe], w,   line=:stem, marker=:o, markersize=1, label="SIS weight", color=rgb(82, 114, 242))
        plot!( x,                Bp,  linewidth=3,                 label="Background",  color=rgb(255, 75, 145))
        plot!([y],              [0],  marker=:star, markersize=10, label="observed",    color=rgb(255, 118, 118))
        plot!([xt],             [0],  marker=:star, markersize=10, label="true",        color=rgb(255, 205, 75))
        plot!([xa],             [0],  marker=:star, markersize=10, label="assimilated", color=rgb(8, 2, 163),
              xlims=(-12, 12), ylims=(-s/3, s*1.1),
              xlabel="x", ylabel="weight", title=f"time = {5*tn[iframe]:.2f}; SIS", 
              legend=:bottomleft, dpi=600, size=(600, 600))

        savefig("./anim/$(ensemble_size)_$(perturb_r)_$(iframe)-1_weight_stem.png")

        #-SIR store_result--------------------------------
        xa = sir_Xa[iframe][1]
        w  = sir_W[iframe]

        plot( sir_Xf1[iframe], w,   line=:stem, marker=:o, markersize=1, label="SIR weight", color=rgb(82, 114, 242))
        plot!( x,                Bp,  linewidth=3,                 label="Background",  color=rgb(255, 75, 145))
        plot!([y],              [0],  marker=:star, markersize=10, label="observed",    color=rgb(255, 118, 118))
        plot!([xt],             [0],  marker=:star, markersize=10, label="true",        color=rgb(255, 205, 75))
        plot!([xa],             [0],  marker=:star, markersize=10, label="assimilated", color=rgb(8, 2, 163),
              xlims=(-12, 12), ylims=(-s/3, s*1.1),
              xlabel="x", ylabel="weight", title=f"time = {5*tn[iframe]:.2f}; SIR", 
              legend=:bottomleft, dpi=600, size=(600, 600))

        savefig("./anim/$(ensemble_size)_$(perturb_r)_$(iframe)-2_weight_stem.png")

        print(stderr, "done.\n")
    end

end

function assignment1_SIR(observed_X, true_X, true_x_converted ,tn, lorenz_parameter, rng; ensemble_size=10000, Ne=100, perturb_r=1.0)
    Xa = [Float64[] for i = 1:length(tn)]
    W = [Float64[] for i = 1:length(tn)]
    N_eff = zeros(Float64, length(tn))
    N = length(tn) - 1

    # make initial ensemble
    initial_Xf = [Float64[] for _ = 1:ensemble_size]
    for k = 1:ensemble_size
        # initial_Xf[k] = get_random_point_on_attractor(rng, lorenz_parameter)
        initial_Xf[k] = observed_X[1] + 1.0 * randn(rng, lorenz_parameter.num_sites)
    end
    println(stderr, "initial ensemble generated")

    num_observed = lorenz_parameter.num_sites
    initial_H = Matrix{Int64}(I, num_observed, num_observed)
    initial_R = Matrix{Int64}(I, num_observed, num_observed)

    particle_parameter = ParticleFilter.Parameter(Lorenz96.step, ensemble_size)
    snap_shot = ParticleFilter.SnapShot(initial_Xf, ones(ensemble_size) / ensemble_size, initial_H, initial_R)

    println(stderr, "start assimilation")
    Xa[1], W[1], N_eff[1] = store_result(snap_shot)

    for i in 1:N
        print(stderr, "t = $(tn[i+1]); ")
        y_observed = observed_X[i+1]
        snap_shot = ParticleFilter.SIR(y_observed, tn[i+1], particle_parameter, lorenz_parameter, snap_shot, rng, 
                                       Ne=Ne, perturb_r=perturb_r)

        Xa[i+1], W[i+1], N_eff[i+1] = store_result(snap_shot)
        print(stderr, "done.\n")
    end

    println(stderr, "assimilation done; plotting...")

    savefig_RMSEvsOBSE(Xa, observed_X, true_X, tn, title="SIR method; m = $(ensemble_size)", filename="SIR_RMSE-$(ensemble_size).png")

    savefig_Weight_Hovmollor(W, tn, title="SIR method; m = $(ensemble_size) \n Weight Hovmollor Diagram", filename="SIR_weight_hovmollor-$(ensemble_size).png")

    savefig_N_eff(tn, N_eff, title="SIR method; m = $(ensemble_size) \n N_eff", filename="SIR_N_eff-$(ensemble_size).png")

end


function assignment1_SIS(observed_X, true_X, true_x_converted ,tn, lorenz_parameter, rng; ensemble_size=10000)
    Xa = [Float64[] for i = 1:length(tn)]
    W = [Float64[] for i = 1:length(tn)]
    N_eff = zeros(Float64, length(tn))
    N = length(tn) - 1

    # make initial ensemble
    initial_Xf = [Float64[] for _ = 1:ensemble_size]
    for k = 1:ensemble_size
        # initial_Xf[k] = get_random_point_on_attractor(rng, lorenz_parameter)
        initial_Xf[k] = observed_X[1] + 1.0 * randn(rng, lorenz_parameter.num_sites)
    end
    println(stderr, "initial ensemble generated")

    num_observed = lorenz_parameter.num_sites
    initial_H = Matrix{Int64}(I, num_observed, num_observed)
    initial_R = Matrix{Int64}(I, num_observed, num_observed)

    particle_parameter = ParticleFilter.Parameter(Lorenz96.step, ensemble_size)
    snap_shot = ParticleFilter.SnapShot(initial_Xf, ones(ensemble_size) / ensemble_size, initial_H, initial_R)

    println(stderr, "start assimilation")
    Xa[1], W[1], N_eff[1] = store_result(snap_shot)

    for i in 1:N
        y_observed = observed_X[i+1]
        snap_shot = ParticleFilter.SIS(y_observed, tn[i+1], particle_parameter, lorenz_parameter, snap_shot)

        Xa[i+1], W[i+1], N_eff[i+1] = store_result(snap_shot)

        println(stderr, "t = $(tn[i+1]); done.")
    end

    println(stderr, "assimilation done; plotting...")

    savefig_RMSEvsOBSE(Xa, observed_X, true_X, tn, title="SIS method; m = $(ensemble_size)", filename="SIS_RMSE-$(ensemble_size).png")

    savefig_Weight_Hovmollor(W, tn, title="SIS method; m = $(ensemble_size) \n Weight Hovmollor Diagram", filename="SIS_weight_hovmollor-$(ensemble_size).png")

    savefig_N_eff(tn, N_eff, title="SIS method; m = $(ensemble_size) \n N_eff", filename="SIS_N_eff-$(ensemble_size).png")

end

# utilities --------------------------------

function rgb(r, g, b)
    return RGBA(r/255, g/255, b/255, 1.0)
end

function savefig_RMSEvsOBSE(Xa, observed_X, true_X, tn; title="", filename="RMSEvsOBSE.png", num_sites=40)
    println(stderr, "plotting RMSE vs OBSE...")

    RMSEs = get_diffs(num_sites, Xa, true_X)
    OBSEs = get_diffs(num_sites, observed_X, true_X)

    plot( tn*5, RMSEs,
         label="RMSE", xlabel="day", ylabel="RMSE", ylims=(0, 6), 
         title=title, legend=:best, 
         dpi=600, size=(600, 600))
    plot!(tn*5, OBSEs, label="OBSE")
    savefig(filename)
end


function savefig_Weight_Hovmollor(W, tn; title="Weight Hovmollor Diagram", filename="WeightHovmollor.png")
    println(stderr, "plotting Weight Hovmollor Diagram...")

    # println(stderr, convert_matrix(W)[:, 1])
    # println(stderr, convert_matrix(W)[:, 11])
    # println(stderr, convert_matrix(W)[:, 21])
    # println(stderr, "x: $(ensemble_size) sites, t: $(length(tn)) steps, size: $(size(convert_matrix(W)))")
    ensemble_size = length(W[1])
    heatmap(1:ensemble_size, tn*5, transpose(convert_matrix(W)), 
            color=cgrad(:thermal, 100, categorical = true, scale = :exp10), clim=(0.0, 10/ensemble_size),
            xlabel="ensemble member", ylabel="day", 
            title=title, 
            dpi=600, size=(1000, 1000))
    savefig(filename)
end


function savefig_N_eff(tn, N_eff; title="N_eff", filename="N_eff.png")
    println(stderr, "N_eff plotting...")
    plot(tn*5, N_eff, 
         label="N_eff", xlabel="day", ylabel="N_eff", yscale=:log10, ylims=(0.1, 1000000),
         title=title, legend=:best, 
         dpi=600, size=(600, 600))
    savefig(filename)
end


function savefig_N_eff_sis_sir(tn, sis_N_eff, sir_N_eff, Ne; title="N_eff", filename="N_eff.png")
    println(stderr, "N_eff (SIS --> SIR) plotting...")

    x, y = Float64[], Float64[]

    for i in eachindex(sis_N_eff)
        push!(x, tn[i]*5)
        push!(y, sis_N_eff[i])

        push!(x, tn[i]*5)
        push!(y, sir_N_eff[i])
    end

    plot(x, y, 
         label="N_eff", xlabel="day", ylabel="N_eff", yscale=:log10, ylims=(0.1, 1000000),
         title=title, legend=:best, 
         dpi=600, size=(600, 600))
    plot!(tn*5, [Ne for _ in eachindex(tn)], label="Ne")
    savefig(filename)
end


function convert_matrix(X)
    m = length(X)
    n = length(X[1])
    converted = zeros(n, m)
    for i = eachindex(X)
        for j = eachindex(X[i])
            converted[j, i] = X[i][j] # X[i][j]はi番目の時刻のj番目のsiteの値; X[i] = converted[:, i]
        end
    end
    return converted
end

function get_diffs(Dim, X1, X2)
    diffs = Float64[]
    for i in eachindex(X1)
        diff = get_RMSE(Dim, X1[i], X2[i])
        push!(diffs, diff)
    end
    return diffs
end

function get_RMSE(Dim, x1, x2)
    diff2 = 0
    for i in 1:Dim
        diff2 += (x1[i] - x2[i])^2
    end
    diff = sqrt(diff2 / Dim)
    return diff
end

function get_limited_RMSE(Dim, x1, x2, target; reverse=false)
    """
    target:長さDimの{0,1}-値配列
    """
    diff2 = 0
    count = 0
    for i in 1:Dim
        if (!target[i]) ⊻ reverse
            continue
        end
        count += 1
        diff2 += (x1[i] - x2[i])^2
    end
    diff = 0
    if count > 0
        diff = sqrt(diff2 / count)
    end
    return diff
end


function get_random_point_on_attractor(rng::Random.MersenneTwister, lorenz_parameter::Lorenz96.Parameter)
    initial_x = zeros(lorenz_parameter.num_sites)
    for i = 1:lorenz_parameter.num_sites
        initial_x[i] = lorenz_parameter.F
    end
    initial_x += randn(rng, lorenz_parameter.num_sites)
    loop_num = 1000 # floor(Int, 1000 + rand(rng) * 3000)
    for _ = 1:loop_num
        initial_x = Lorenz96.step(initial_x, 0.0, lorenz_parameter)
    end
    return initial_x
end

function get_true_state(lorenz_parameter::Main.Lorenz96.Parameter, initial_x, Start, Goal)
    t = 0
    while t + EPSILON < Start # get spinned-up state as the initial state of assimilation
        t += lorenz_parameter.h
        initial_x = Lorenz96.step(initial_x, t, lorenz_parameter)
    end
    X = Lorenz96.integrate(initial_x, Start, lorenz_parameter.h, Goal, lorenz_parameter)
    return X
end

function make_L(num_sites, L_scale)
    L = ones(Float64, num_sites, num_sites)
    for i = 1:num_sites
        for j = 1:num_sites
            if i > j
                continue
            end
            d = min((j - i), (40 + i - j))
            L[i, j] = exp(-d^2 / (L_scale^2))
        end
    end
    L = Symmetric(L)
    return L
end

function make_observation_data(X, len_tn, rng::Random.MersenneTwister)
    copy_X = [[X[i][j] for j = eachindex(X[i])] for i = eachindex(X)]
    randns = randn(rng, len_tn, 40)
    for i = eachindex(copy_X)
        for j = eachindex(copy_X[i])
            copy_X[i][j] += randns[i, j]
        end
    end
    return copy_X
end

"""
function integrate(initial_x, Y, Start, Goal, kalman_parameter::KalmanFilter.Parameter, lorenz_parameter::Lorenz96.Parameter)
    a = 5.1
    Rho_a = a^2 * Matrix{Int64}(I, lorenz_parameter.num_sites, lorenz_parameter.num_sites)
    snap_shot = SnapShot(zeros(lorenz_parameter.num_sites), initial_x, zeros(lorenz_parameter.num_sites, lorenz_parameter.num_sites), Rho_a, zeros(lorenz_parameter.num_sites, 1), zeros(1, lorenz_parameter.num_sites), zeros(lorenz_parameter.num_sites, lorenz_parameter.num_sites))
    # snap_shot = SnapShot(zeros(0), initial_x, zeros(0, lorenz_parameter.num_sites),Rho_a,zeros(num_sites))

    tn = Start:lorenz_parameter.h:Goal
    N = length(tn) - 1
    X_a = [Float64[] for i = 1:length(tn)]
    X_a[1] = initial_x
    tr_Rho = zeros(length(tn))
    for i in 1:N
        y_observed = Y[i+1]
        observed = select_sites(lorenz_parameter.num_sites, kalman_parameter.num_observed)
        snap_shot = assimilate(y_observed, observed, tn[i], kalman_parameter, lorenz_parameter, snap_shot)
        X_a[i+1] = snap_shot.x_a
    end
    return X_a
end
"""
function select_sites(num_sites, num_observed)
    # sites = zeros(num_sites)
    sites = [false for _ = 1:num_sites]
    for i = 1:num_observed
        sites[i] = true
    end
    shuffle!(sites)
    return sites
end


function solve_KalmanFilter(rng::MersenneTwister, tn, initial_x, X, Y, kalman_parameter, lorenz_parameter; positive_site_num=40)
    observed = Bool[(i <= positive_site_num) ? true : false for i in 1:lorenz_parameter.num_sites] # positive_site_num個のTrueとそれ以外のFalseを持つ配列

    a = 5.1
    initial_Rho = a^2 * Matrix{Int64}(I, lorenz_parameter.num_sites, lorenz_parameter.num_sites)
    snap_shot = KalmanFilter.SnapShot(x_a=initial_x, x_f=initial_x, Rho_a=initial_Rho, Rho_f=initial_Rho)

    N = length(tn) - 1
    X_a = [Float64[] for i = 1:length(tn)]
    X_a[1] = initial_x
    diff = zeros(Float64, length(tn))
    tr_Rho_a = zeros(length(tn))
    tr_Rho_f = zeros(length(tn))


    # 初期状態のtr_Rho_a,tr_Rho_fを計算
    tr_Rho_a[1] = sqrt(tr(snap_shot.Rho_a) / lorenz_parameter.num_sites)
    tr_Rho_f[1] = sqrt(tr(snap_shot.Rho_f) / lorenz_parameter.num_sites)


    diff = zeros(Float64, length(tn))
    diff[1] = get_RMSE(lorenz_parameter.num_sites, X[1], X_a[1])

    num_selected = zeros(lorenz_parameter.num_sites)

    diverged = false
    for i in 1:N
        # 観測地点を決定
        shuffle!(rng, observed)
        num_selected += observed

        y_observed = Y[i+1]

        snap_shot = KalmanFilter.assimilate(y_observed, observed, tn[i], kalman_parameter, lorenz_parameter, snap_shot)
        X_a[i+1], tr_Rho_a[i+1], tr_Rho_f[i+1] = store_result(snap_shot, lorenz_parameter.num_sites)
        diff[i+1] = get_RMSE(lorenz_parameter.num_sites, X[i+1], X_a[i+1])

        if diff[i+1] > 10a
            diverged = true
            println("diverged at $(tn[i+1])")
            break
        end
    end

    if diverged
        return NaN, diff, tr_Rho_f, tr_Rho_a
    else
        # 300ステップ目以降の誤差平均を出力
        mean_diff = mean(diff[300:end])
        return mean_diff, diff, tr_Rho_f, tr_Rho_a
    end

end

function solve_3DVar(rng::MersenneTwister, tn, initial_x, X, Y, kalman_parameter, lorenz_parameter; positive_site_num=40, beta=0.2)
    observed = Bool[(i <= positive_site_num) ? true : false for i in 1:lorenz_parameter.num_sites] # positive_site_num個のTrueとそれ以外のFalseを持つ配列

    a = 5.1
    B = beta * Matrix{Int64}(I, lorenz_parameter.num_sites, lorenz_parameter.num_sites)
    snap_shot = KalmanFilter.SnapShot(x_a=initial_x, x_f=initial_x, Rho_a=B, Rho_f=B)
    # snap_shot = SnapShot(zeros(0), initial_x, zeros(0, lorenz_parameter.num_sites),Rho_a,zeros(num_sites))

    N = length(tn) - 1
    X_a = [Float64[] for i = 1:length(tn)]
    X_a[1] = initial_x
    diff = zeros(Float64, length(tn))
    tr_Rho_a = zeros(length(tn))
    tr_Rho_f = zeros(length(tn))


    # 初期状態のtr_Rho_a,tr_Rho_fを計算
    tr_Rho_a[1] = sqrt(tr(snap_shot.Rho_a) / lorenz_parameter.num_sites)
    tr_Rho_f[1] = sqrt(tr(snap_shot.Rho_f) / lorenz_parameter.num_sites)


    diff = zeros(Float64, length(tn))
    diff[1] = get_RMSE(lorenz_parameter.num_sites, X[1], X_a[1])

    num_selected = zeros(lorenz_parameter.num_sites)

    diverged = false
    for i in 1:N
        # 観測地点を決定
        shuffle!(rng, observed)
        num_selected += observed

        y_observed = Y[i+1]

        snap_shot = KalmanFilter.assimilate(y_observed, observed, tn[i], kalman_parameter, lorenz_parameter, snap_shot)
        X_a[i+1], tr_Rho_a[i+1], tr_Rho_f[i+1] = store_result(snap_shot, lorenz_parameter.num_sites)
        diff[i+1] = get_RMSE(lorenz_parameter.num_sites, X[i+1], X_a[i+1])

        if diff[i+1] > 10a
            diverged = true
            println("diverged at $(tn[i+1])")
            break
        end
    end

    if diverged
        return NaN, diff, tr_Rho_f, tr_Rho_a
    else
        # 300ステップ目以降の誤差平均を出力
        mean_diff = mean(diff[300:end])
        return mean_diff, diff, tr_Rho_f, tr_Rho_a
    end

end

function solve_EnSRF(rng::MersenneTwister, tn, initial_X, X, Y, kalman_parameter, lorenz_parameter; positive_site_num=40)
    observed = Bool[(i <= positive_site_num) ? true : false for i in 1:lorenz_parameter.num_sites] # positive_site_num個のTrueとそれ以外のFalseを持つ配列

    a = 5.1
    # Rho_a = a^2 * Matrix{Int64}(I, lorenz_parameter.num_sites, lorenz_parameter.num_sites)
    L = zeros(0, 0)
    if kalman_parameter.localize == true
        L = make_L(lorenz_parameter.num_sites, kalman_parameter.L_scale)
    end

    mean_initial_X = reshape(mean(initial_X, dims=2), lorenz_parameter.num_sites)
    snap_shot = KalmanFilter.SnapShot(x_a=mean_initial_X, x_f=mean_initial_X, X_a=initial_X, X_f=initial_X, L=L)

    N = length(tn) - 1
    X_a = [Float64[] for i = 1:length(tn)]

    X_a[1] = mean_initial_X
    tr_Rho_a = zeros(length(tn))
    tr_Rho_f = zeros(length(tn))

    # 初期状態のtr_Rho_a,tr_Rho_fを計算
    initial_delta_rho_a = snap_shot.X_a .- snap_shot.x_a
    initial_delta_rho_f = snap_shot.X_f .- snap_shot.x_f
    m = (length(snap_shot.X_a) == 0) ? 0 : length(snap_shot.X_a[1, :])
    initial_rho_a = 1 / (m - 1) * initial_delta_rho_a * transpose(initial_delta_rho_a)
    initial_rho_f = 1 / (m - 1) * initial_delta_rho_f * transpose(initial_delta_rho_f)
    tr_Rho_a[1] = sqrt(tr(initial_rho_a) / lorenz_parameter.num_sites)
    tr_Rho_f[1] = sqrt(tr(initial_rho_f) / lorenz_parameter.num_sites)


    diff = zeros(Float64, length(tn))
    diff[1] = get_RMSE(lorenz_parameter.num_sites, X[1], X_a[1])

    num_selected = zeros(lorenz_parameter.num_sites)

    diverged = false
    for i in 1:N
        # 観測地点を決定
        shuffle!(rng, observed)
        num_selected += observed

        y_observed = Y[i+1]

        snap_shot = KalmanFilter.assimilate(y_observed, observed, tn[i], kalman_parameter, lorenz_parameter, snap_shot)
        X_a[i+1], tr_Rho_a[i+1], tr_Rho_f[i+1] = store_result(snap_shot, lorenz_parameter.num_sites)
        diff[i+1] = get_RMSE(lorenz_parameter.num_sites, X[i+1], X_a[i+1])

        if diff[i+1] > 10a
            diverged = true
            println("diverged at $(tn[i+1])")
            break
        end
    end

    if diverged
        return NaN, diff, tr_Rho_f, tr_Rho_a
    else
        # 300ステップ目以降の誤差平均を出力
        mean_diff = mean(diff[300:end])
        return mean_diff, diff, tr_Rho_f, tr_Rho_a
    end

end

function store_result(snap_shot::KalmanFilter.SnapShot, num_sites)
    X_a = snap_shot.x_a
    tr_Rho_a = sqrt(tr(snap_shot.Rho_a) / num_sites)
    tr_Rho_f = sqrt(tr(snap_shot.Rho_f) / num_sites)
    return X_a, tr_Rho_a, tr_Rho_f
end

function store_result(snap_shot::ParticleFilter.SnapShot)
    w = snap_shot.w
    Xf_mat = convert_matrix(snap_shot.Xf)
    x_a = Xf_mat * w
    N_eff = snap_shot.N_eff
    return x_a, w, N_eff
end

main()
