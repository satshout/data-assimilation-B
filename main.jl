
include("Lorenz96.jl")
include("KalmanFilter.jl")

# using Lorenz96
# using KalmanFilter
using Plots
using Random
using StatsBase
using LinearAlgebra

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
    Goal = 365 * 2 / 5

    # 真値の生成
    X = get_true_state(lorenz_parameter, initial_x, Start, Goal)
    tn = Start:lorenz_parameter.h:Goal
    X_converted = convert_matrix(X)

    # 観測の生成
    observed_X = make_observation_data(X, length(tn), rng)

    # アトラクタ上からランダムな初期値をとる
    initial_x = get_random_point_on_attractor(rng, lorenz_parameter)

end


# utilities --------------------------------

function convert_matrix(X)
    n = length(X)
    m = length(X[1])
    converted = zeros(n, m)
    for i = eachindex(X)
        for j = eachindex(X[i])
            converted[i, j] = X[i][j]
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


main()
