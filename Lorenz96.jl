module Lorenz96

struct Parameter
    F::Float64
    num_sites::Int
    h::Float64
    Parameter() = new(8, 40, 0.05)
end


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

function Runge_Kutta_4_get_next(f::Function, x, t::Float64, parameter::Parameter)
    h = parameter.h
    Dim = parameter.num_sites
    F = parameter.F
    k1 = f(x, t, parameter)
    k2 = f(x + 0.5h * k1, t + 0.5h, parameter)
    k3 = f(x + 0.5h * k2, t + 0.5h, parameter)
    k4 = f(x + h * k3, t + h, parameter)

    x_next = x + (k1 + 2 * k2 + 2 * k3 + k4) * (h / 6)
    return x_next
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

end
