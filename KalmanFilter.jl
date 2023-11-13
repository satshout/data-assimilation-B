
module KalmanFilter

using LinearAlgebra
using Random
using StatsBase

@enum Mode begin
    Mode_KalmanFilter
    Mode_3DVar
    Mode_EnSRF
end

struct Parameter
    forecast::Function # Runge_Kutta_4_get_next
    mode::Mode
    num_observed::Int
    inflation::Float64
    num_ensemble::Int
    localize::Bool
    L_scale::Float64

    function Parameter(forecast::Function, mode::Mode, num_observed::Int; inflation=1.00, num_ensemble=40, localize=false, L_scale=5)
        new(forecast, mode, num_observed, inflation, num_ensemble, localize, L_scale)
    end

    # function Parameter(forecast, mode=KalmanFilter.Mode_KalmanFilter, inflation=1, num_observed=40)
    #     new(forecast, mode, inflation, num_observed)
    # end
    # function Parameter(mode::String, forecast::Function, inflation::Float64)
    #     if mode == "KalmanFilter"
    #         return new(forecast, 0, inflation)
    #     else
    #         if mode == "3D-Var"
    #             return new(forecast, 1, inflation)
    #         end
    #     end
    # end
end

struct SnapShot
    x_f::Vector{Float64}
    x_a::Vector{Float64}
    Rho_f::Matrix{Float64}
    Rho_a::Matrix{Float64}
    K::Matrix{Float64}
    H::Matrix{Int64}
    M::Matrix{Float64}
    X_f::Matrix{Float64}
    X_a::Matrix{Float64}
    L::Matrix{Float64}

    function SnapShot(; x_f=zeros(Float64, 0), x_a=zeros(Float64, 0), Rho_f=zeros(Float64, 0, 0), Rho_a=zeros(Float64, 0, 0), K=zeros(Float64, 0, 0), H=zeros(Int64, 0, 0), M=zeros(Float64, 0, 0), X_f=zeros(Float64, 0, 0), X_a=zeros(Float64, 0, 0), L=zeros(Float64, 0, 0))
        new(x_f, x_a, Rho_f, Rho_a, K, H, M, X_f, X_a, L)
    end

    # function SnapShot(x_f::Vector{Float64}, x_a::Vector{Float64}, Rho_f::Matrix{Float64}, Rho_a::Matrix{Float64}, K::Matrix{Float64}, H::Matrix{Int64}, M::Matrix{Float64})
    #     new(x_f, x_a, Rho_f, Rho_a, K, H, M)
    # end
end

function integrate(initial_x, Y, Start, Goal, kalman_parameter::Parameter, lorenz_parameter::Main.Lorenz96.Parameter)
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
        observed = ones(lorenz_parameter.num_sites)
        snap_shot = assimilate(y_observed, observed, tn[i], kalman_parameter, lorenz_parameter, snap_shot)
        X_a[i+1] = snap_shot.x_a
    end
    return X_a
end


function assimilate(y_observed, observed, t, parameter::Parameter, lorenz_parameter::Main.Lorenz96.Parameter, snap_shot::SnapShot)
    """
    y 観測データ
    observed: 観測ありは1,観測なしは0
    parameter:
    lorenz_parameter:
    snap_shot: 前ステップの情報
    """

    # y,Hを構成
    num_observed = 0
    for i = eachindex(observed)
        if observed[i] == 0
            continue
        else
            num_observed += 1
        end
    end
    y = zeros(Float64, num_observed)
    H = zeros(num_observed, lorenz_parameter.num_sites)
    R = Matrix{Int64}(I, num_observed, num_observed)

    cursor = 1
    for i = eachindex(y_observed)
        if observed[i] == 0
            continue
        else
            y[cursor] = y_observed[i]
            H[cursor, i] = 1
            cursor += 1
        end
    end

    m = (length(snap_shot.X_a) == 0) ? 0 : length(snap_shot.X_a[1, :])

    X_f = zeros(lorenz_parameter.num_sites, m)
    delta_X_f = zeros(0, 0)
    x_f = zeros(0)
    if parameter.mode == Mode_EnSRF
        for k = 1:m
            X_f[:, k] = parameter.forecast(snap_shot.X_a[:, k], t, lorenz_parameter)
        end

        x_f = reshape(mean(X_f, dims=2), lorenz_parameter.num_sites)
        delta_X_f = X_f .- x_f
        delta_X_f *= parameter.inflation

    else
        x_f = parameter.forecast(snap_shot.x_a, t, lorenz_parameter)
    end



    M = zeros(0, 0)
    Rho_f = zeros(lorenz_parameter.num_sites, lorenz_parameter.num_sites)

    if parameter.mode == Mode_3DVar
        Rho_f = snap_shot.Rho_a
    elseif parameter.mode == Mode_EnSRF
        Rho_f = 1 / (m - 1) * delta_X_f * transpose(delta_X_f)
        if parameter.localize
            Rho_f .*= snap_shot.L
        end
        # Rho_f *= parameter.inflation
        # println(Rho_f)

    else
        M = get_M_by_approx(snap_shot.x_a, parameter.forecast, t, lorenz_parameter)
        Rho_f = equation_2(snap_shot.Rho_a, M)
        Rho_f *= parameter.inflation
    end

    K = equation_5(H, R, Rho_f)

    x_a = equation_3(x_f, K, y, H)
    X_a = zeros(0, 0)
    Rho_a = zeros(0, 0)
    if parameter.mode == Mode_3DVar
        Rho_a = Rho_f
    elseif parameter.mode == Mode_EnSRF
        delta_X_a = get_delta_X_a(delta_X_f, H, R, m)
        X_a = delta_X_a .+ x_a
        Rho_a = 1 / (m - 1) * delta_X_a * transpose(delta_X_a)

    else
        Rho_a = equation_4(K, H, Rho_f)
    end

    next_snapshot = SnapShot(x_f=x_f, x_a=x_a, Rho_f=Rho_f, Rho_a=Rho_a, K=K, H=H, M=M, X_f=X_f, X_a=X_a, L=snap_shot.L)
    return next_snapshot

end

function get_M_by_approx(x, stepper, t, lorenz_parameter::Main.Lorenz96.Parameter; delta=0.001)
    M = zeros(length(x), length(x))
    for j in 1:length(x)
        ej = zeros(length(x))
        ej[j] = 1

        M_col_j = (stepper(x + delta * ej, t, lorenz_parameter) - stepper(x, t, lorenz_parameter)) / delta

        M[:, j] = M_col_j
    end

    return M
end

function get_SVD_by_approx(x, stepper::Function, t, lorenz_parameter::Main.Lorenz96.Parameter; delta=0.001)
    M = get_M_by_approx(x, stepper, t, lorenz_parameter; delta=delta)
    U, S, V = svd(M)
    return U, S, V
end

function equation_2(prev_Rho_a, M)
    Rho_f = M * prev_Rho_a * transpose(M)
    return Rho_f
end


function equation_3(x_f, K, y, H)
    x_a = x_f + K * (y - H * x_f)
    return x_a
end

function equation_4(K, H, Rho_f)
    Rho_a = (I - K * H) * Rho_f
    return Rho_a
end

function get_delta_X_a(delta_X_f, H, R, m)
    delta_Y = H * delta_X_f
    A = Symmetric(Matrix{Float64}(I, m, m) - 1 / (m - 1) * transpose(delta_Y) * inv(1 / (m - 1) * delta_Y * transpose(delta_Y) + R) * delta_Y)
    # A=UDU_tとして, U: 直交行列
    F = eigen(A)
    # U = (固有ベクトルを正規化したものを並べる)
    vecD, U = F
    # D=(固有値を並べる)
    # rootD=(固有値のrootを並べる)
    vec_rootD = sqrt.(vecD)
    rootD = diagm(vec_rootD)

    T = U * rootD * transpose(U)
    delta_X_a = delta_X_f * T

    return delta_X_a
end

function equation_5(H, R, Rho_f)
    K = Rho_f * transpose(H) * inv(H * (Rho_f) * transpose(H) + R)
    return K
end



end
