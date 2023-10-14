
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
    # Goal = 365 * 2 / 5
    Goal = 380 / 5

    # 真値の生成
    X = get_true_state(lorenz_parameter, initial_x, Start, Goal)
    tn = Start:lorenz_parameter.h:Goal
    X_converted = convert_matrix(X)

    # 観測の生成
    observed_X = make_observation_data(X, length(tn), rng)

    # アトラクタ上からランダムな初期値をとる
    # initial_x = get_random_point_on_attractor(rng, lorenz_parameter)

    # assignment1_SIS(observed_X, X, X_converted, tn, lorenz_parameter, rng; ensemble_size=1000)
    # assignment1_SIR(observed_X, X, X_converted, tn, lorenz_parameter, rng; ensemble_size=100000)
    assignment1_SIR_anime(observed_X, X, X_converted, tn, lorenz_parameter, rng; ensemble_size=1000, Ne=10, perturb_r=0.5)
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

    savefig_N_eff_sis_sir(tn, sis_N_eff, sir_N_eff, Ne, title="SIR method; \n m = $(ensemble_size), Ne = $(Ne),|η| = $(perturb_r)", filename="SIR_N_eff-$(ensemble_size).png")

    # plot(x, line=:stem, marker=:star, markersize=20)
    for iframe in 1:N
        print(stderr, "iframe = $iframe; ")

        s  = maximum(sir_W[iframe])
        x  = [i for i in -12:0.01:12]
        y  = observed_X[iframe][1]
        Bp = s * exp.(-0.5 * (y .- x).^2)

        xt = true_X[iframe][1]

        #-SIS store_result--------------------------------
        xa = sis_Xa[iframe][1]
        xt = true_X[iframe][1]
        w  = sis_W[iframe]

        plot( sis_Xf1[iframe+1], w,   line=:stem, marker=:o, markersize=1, label="SIS weight", color=rgb(82, 114, 242))
        plot!( x,                Bp,  linewidth=2,                 label="Background",  color=rgb(255, 75, 145))
        plot!([y],              [0],  marker=:star, markersize=10, label="observed",    color=rgb(255, 118, 118))
        plot!([xt],             [0],  marker=:star, markersize=10, label="true",        color=rgb(255, 205, 75))
        plot!([xa],             [0],  marker=:star, markersize=10, label="assimilated", color=rgb(8, 2, 163),
              xlims=(-12, 12), ylims=(-s/3, s*1.1),
              xlabel="x", ylabel="weight", title=f"time = {5*tn[iframe]:.2f}; SIS", 
              legend=:bottomleft, dpi=600, size=(600, 600))

        savefig("./anim/$(ensemble_size)_$(iframe)-1_weight_stem.png")

        #-SIR store_result--------------------------------
        xa = sir_Xa[iframe][1]
        w  = sir_W[iframe]

        plot( sir_Xf1[iframe+1], w,   line=:stem, marker=:o, markersize=1, label="SIR weight", color=rgb(82, 114, 242))
        plot!( x,                Bp,  linewidth=2,                 label="Background",  color=rgb(255, 75, 145))
        plot!([y],              [0],  marker=:star, markersize=10, label="observed",    color=rgb(255, 118, 118))
        plot!([xt],             [0],  marker=:star, markersize=10, label="true",        color=rgb(255, 205, 75))
        plot!([xa],             [0],  marker=:star, markersize=10, label="assimilated", color=rgb(8, 2, 163),
              xlims=(-12, 12), ylims=(-s/3, s*1.1),
              xlabel="x", ylabel="weight", title=f"time = {5*tn[iframe]:.2f}; SIR", 
              legend=:bottomleft, dpi=600, size=(600, 600))

        savefig("./anim/$(ensemble_size)_$(iframe)-2_weight_stem.png")

        print(stderr, "done.\n")
    end

end

function assignment1_SIR(observed_X, true_X, true_x_converted ,tn, lorenz_parameter, rng; ensemble_size=10000)
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
                                       Ne=100, perturb_r=1.0)

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
