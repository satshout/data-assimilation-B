
module ParticleFilter

using LinearAlgebra
using Random
using StatsBase

struct Parameter
    forecast::Function # Lorenz96.step
    ensemble_size::Int

    function Parameter(forecast::Function, ensemble_size::Int)
        new(forecast, ensemble_size)
    end

end

struct SnapShot
    Xf::Vector{Vector{Float64}}
    w::Vector{Float64}
    H::Matrix{Float64}
    R::Matrix{Float64}
    N_eff::Float64

    function SnapShot(Xf, w, H, R)
        N_eff = 1 / sum(w .^ 2)
        new(Xf, w, H, R, N_eff)
    end

end


function gauss(x::Vector{Float64}, A::Matrix{Float64})
    n = length(x)
    s2 = abs(det(A))
    a = 1/(2 \pi)^(n/2) * 1/sqrt(s2)
    G = exp(-0.5 * transpose(x) * inv(A) * x)
    return a * G
end

function SIS(y_observed, t, parameter::Parameter, lorenz_parameter::Main.Lorenz96.Parameter, snap_shot::SnapShot)
    """
    y_observed: 観測データ
    parameter:
    lorenz_parameter:
    snap_shot: 前ステップの情報
    """
    num_observed = lorenz_parameter.num_sites

    # y,Hを構成
    y = y_observed
    H = snap_shot.H
    R = snap_shot.R

    m = parameter.ensemble_size
    Xf = snap_shot.Xf

    if length(Xf) != m
        error("length(X_f) != m")
    end

    # 重みの計算
    nx_Xf = [Float64[] for k = 1:m]
    nx_w = zeros(m)

    for k in 1:m
        nx_Xf[k] = parameter.forecast(Xf[k], t, lorenz_parameter)
        l_k = gauss(y - H * nx_Xf[k], R)
        nx_w[k] = snap_shot.w[k] * l_k
    end

    # 正規化
    nx_w = nx_w / sum(nx_w)

    nx_snap_shot = SnapShot(nx_Xf, nx_w, H, R)

    return nx_snap_shot
end

function SIR(y_observed, t, parameter::Parameter, lorenz_parameter::Main.Lorenz96.Parameter, snap_shot::SnapShot, rng::Random.AbstractRNG; Ne = 100, perturb_r=0.0 )
    """
    y_observed: 観測データ
    parameter:
    lorenz_parameter:
    snap_shot: 前ステップの情報
    rng: Mersenne Twister
    """

    m = parameter.ensemble_size
    sis_snap_shot = SIS(y_observed, t, parameter, lorenz_parameter, snap_shot)

    # resampling
    if sis_snap_shot.N_eff < Ne
        print(stderr, "resampling; ")

        sis_Xf = sis_snap_shot.Xf
        sis_w = sis_snap_shot.w

        Fw = [sum(sis_w[1:k]) for k in 1:m]

        nx_Xf = [Float64[] for k = 1:m]

        for k in 1:m
            dice = rand(rng)

            nx_Xf[k] = sis_Xf[1]
            for mk in 1:m-1
                if dice > Fw[mk]
                    nx_Xf[k] = sis_Xf[mk+1]
                end
            end

            nx_Xf[k] = nx_Xf[k] + perturb_r * randn(rng, length(nx_Xf[k]))
        end

        nx_w = ones(m) / m

        nx_snap_shot = SnapShot(nx_Xf, nx_w, sis_snap_shot.H, sis_snap_shot.R)

        return nx_snap_shot
    else
        return sis_snap_shot
    end
end

end
