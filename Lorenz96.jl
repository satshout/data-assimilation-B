module Lorenz96

struct Parameter
    F::Float64
    num_sites::Int
    h::Float64
    Parameter() = new(8, 40, 0.05)
end

# ------- Lorenz96 & RK4 ------- #

function adjustIndex(i::Int, max::Int)
    if i > max
        return i - max
    elseif i <= 0
        return i + max
    end
    return i
end

function lorenz_96(x::Array{Float64,1}, t::Float64, parameter::Parameter)
    Dim = parameter.num_sites
    F = parameter.F
    dxdt = zeros(Float64, Dim)
    for i = 1:Dim
        dxdt[i] = (x[adjustIndex(i + 1, Dim)] - x[adjustIndex(i - 2, Dim)]) * x[adjustIndex(i - 1, Dim)] - x[adjustIndex(i, Dim)] + F
    end
    return dxdt
end

function Runge_Kutta_4_get_next(f::Function, x, t::Float64, parameter::Parameter; return_k::Bool=false)
    h = parameter.h

    k1 = f(x, t, parameter)
    k2 = f(x + 0.5h * k1, t + 0.5h, parameter)
    k3 = f(x + 0.5h * k2, t + 0.5h, parameter)
    k4 = f(x + h * k3, t + h, parameter)

    dx = (k1 + 2 * k2 + 2 * k3 + k4) * (h / 6)
    x_next = x + dx
    if return_k
        return x_next, dx, k1, k2, k3, k4
    else
        return x_next
    end
end

function Runge_Kutta_4(f::Function, initial_x, tn, parameter::Parameter)
    # f(x, t, Dim)
    N = length(tn) - 1
    X = [Float64[] for _ = 1:N+1]
    X[1] = initial_x

    for j in 1:N
        X[j+1] = Runge_Kutta_4_get_next(f, X[j], tn[j], parameter)
    end
    return X
end

function integrate(initial_x, t_start::Float64, t_step::Float64, t_goal::Float64, parameter::Parameter)
    tn = collect(t_start:t_step:t_goal)
    X = Runge_Kutta_4(lorenz_96, initial_x, tn, parameter)
    return X
end

function step(x, t::Float64, parameter::Parameter)
    Runge_Kutta_4_get_next(lorenz_96, x, t, parameter)
end


# ------- TLM ------- #

function Model(ti, xi, parameter)
    x_next, dx, k1, k2, k3, k4 = Runge_Kutta_4_get_next(lorenz_96, xi, ti, parameter, return_k=true)
    return k1, k2, k3, k4
end

function L(x::Vector{Float64}, j::Int, k::Int)
    J = length(x)

    if k == adjustIndex(j - 2, J)
        return -x[adjustIndex(j - 1, J)]
    elseif k == adjustIndex(j - 1, J)
        return x[adjustIndex(j + 1, J)] - x[adjustIndex(j - 2, J)]
    elseif k == j
        return -1
    elseif k == adjustIndex(j + 1, J)
        return x[adjustIndex(j - 1, J)]
    else
        return 0
    end
end

function l96_tlm(Xi::Vector{Float64}, dX::Vector{Float64})
    J = length(Xi)
    
    dDX = zeros(Float64, J)
    
    for j in 1:J
        for k in 1:J
            dDX[j] += L(Xi, j, k) * dX[k]
        end
    end

    return dDX
end

function l96_adj(Xi::Vector{Float64}, dDX::Vector{Float64})
    J = length(Xi)
    
    dX = zeros(Float64, J)
    
    for j in 1:J
        for k in 1:J
            dX[j] += L(Xi, k, j) * dDX[k]
        end
    end

    return dX
end

function TangentLinearCode(ti, Xi, dX, parameter) #入力 = 基本場(xi, yi, zi) + 摂動(dxi, dyi, dzi)
    Dt = parameter.h
    
    K1i, K2i, K3i, K4i = Model(ti, Xi, parameter) #基本場の情報を取得
    
    # tlm( RK4[1] )
    X0_rk0 = copy(Xi) #lorenz63_tlm が感じる基本場
    dDX1 = l96_tlm(X0_rk0, dX)
    dX_rk1 = dX + dDX1 * Dt / 2

    # tlm( RK4[2] )
    X0_rk1 = Xi + K1i * Dt / 2 #lorenz63_tlm が感じる基本場
    dDX2 = l96_tlm(X0_rk1, dX_rk1)
    dX_rk2 = dX + dDX2 * Dt / 2

    # tlm( RK4[3] )
    X0_rk2 = Xi + K2i * Dt / 2 #lorenz63_tlm が感じる基本場
    dDX3 = l96_tlm(X0_rk2, dX_rk2)
    dX_rk3 = dX + dDX3 * Dt

    # tlm( RK4[4] )
    X0_rk3 = Xi + K3i * Dt #lorenz63_tlm が感じる基本場
    dDX4 = l96_tlm(X0_rk3, dX_rk3)

    # tlm( RK4[ΔX] )
    dnX = dX + (dDX1 + 2 * dDX2 + 2 * dDX3 + dDX4) * Dt / 6
    return dnX #出力 = 応答
end

function AdjointCode(ti, Xi, dnX, parameter) #入力 = 基本場(xi, yi, zi) + 応答(dxi, dyi, dzi)
    Dt = parameter.h

    K1i, K2i, K3i, K4i = Model(ti, Xi, parameter) #基本場の情報を取得

    # adj( RK4[ΔX] )
    dX   = copy(dnX)
    dDX4 = dnX * Dt / 6
    dDX3 = dnX * Dt / 3
    dDX2 = dnX * Dt / 3
    dDX1 = dnX * Dt / 6

    # adj( RK4[4] )
    X0_rk3 = Xi + K3i * Dt #lorenz63_adj が感じる基本場
    dX_rk3 = l96_adj(X0_rk3, dDX4)
    dDX3 = dDX3 + dX_rk3 * Dt
    dX += dX_rk3

    # adj( RK4[3] )
    X0_rk2 = Xi + K2i * Dt / 2 #lorenz63_adj が感じる基本場
    dX_rk2 = l96_adj(X0_rk2, dDX3)
    dDX2 = dDX2 + dX_rk2 * Dt / 2
    dX += dX_rk2

    # adj( RK4[2] )
    X0_rk1 = Xi + K1i * Dt / 2 #lorenz63_adj が感じる基本場
    dX_rk1 = l96_adj(X0_rk1, dDX2)
    dDX1 = dDX1 + dX_rk1 * Dt / 2
    dX += dX_rk1

    # adj( RK4[1] )
    X0_rk0 = Xi #lorenz63_adj が感じる基本場
    dX_rk0 = l96_adj(X0_rk0, dDX1)
    dX += dX_rk0
    
    return dX #出力 = 摂動
end

end