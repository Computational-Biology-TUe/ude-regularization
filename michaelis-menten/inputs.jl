"""
michaelismenten!(du, u, p, t)

Ground truth model for the Michaelis-Menten kinetics model according to the following ODEs:
    dA/dt = k1*A - (k2*A)/(kM + A)
    dB/dt = (k2*A)/(kM + A) - k3*B
Implementation according to the DifferentialEquations.jl API.
"""
function michaelismenten!(du, u, p, t)
    A, B = u
    k1, k2, kM, k3 = p
    du[1] = k1*A - (k2*A)/(kM+A)
    du[2] = (k2*A)/(kM+A) - k3*B;
end

"""
simulate_inputs(parameters, rng; train_end=30, train_step=5)

Simulate noisy data for the Michaelis-Menten kinetics model according to the following ODEs:
    dA/dt = k1*A - (k2*A)/(kM + A)
    dB/dt = (k2*A)/(kM + A) - k3*B

Returns:
  - data_A: Noisy data for species A
  - data_B: Noisy data for species B
  - times_A: Time points for species A
  - times_B: Time points for species B
  - val_data_A: Validation data for species A
  - val_data_B: Validation data for species B
"""
function simulate_inputs(parameters, rng; train_end=30, train_step=5)
    # parameters, timespan and initial values
    u0 = [2, 0.]
    tspan = (0.,400)

    # ODE Problem and solution
    problem = ODEProblem(michaelismenten!, u0, tspan, parameters)
    solution = solve(problem, Tsit5())

    # Data point times
    times_A = 0:train_step:train_end
    times_B = 0:train_step:train_end

    # Ground truth data
    gtdata_A = solution(times_A, idxs=1)
    gtdata_B = solution(times_B, idxs=2)

    # Validation data
    val_data_A = solution(tspan[1]:0.1:tspan[end], idxs=1)
    val_data_B = solution(tspan[1]:0.1:tspan[end], idxs=2)

    # "Normally" distributed noise
    data_A = max.(0, gtdata_A.u .+ randn(rng, size(gtdata_A.u)).*maximum(val_data_A.u).*0.05)
    data_B = max.(0, gtdata_B.u .+ randn(rng, size(gtdata_B.u)).*maximum(val_data_B.u).*0.05)

    return data_A, data_B, times_A, times_B, val_data_A, val_data_B
end
