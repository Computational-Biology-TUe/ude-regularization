"""
michaelismenten!(du, u, p, t)

Ground truth model for the Michaelis-Menten kinetics model according to the following ODEs:
    dS/dt = kS*S - (kSP*S)/(kM + S)
    dP/dt = (kSP*S)/(kM + S) - kP*P
Implementation according to the DifferentialEquations.jl API.
"""
function michaelismenten!(du, u, p, t)
    A, B = u
    kS, kSP, kM, kP = p
    du[1] = kS*A - (kSP*A)/(kM+A)
    du[2] = (kSP*A)/(kM+A) - kP*B;
end

"""
simulate_inputs(parameters, rng; train_end=30, train_step=5, noise_level=0.05, initial_conditions=[2.0, 0.0])

Simulate noisy data for the Michaelis-Menten kinetics model according to the following ODEs:
    dS/dt = kS*S - (kSP*S)/(kM + S)
    dP/dt = (kSP*S)/(kM + S) - kP*P

Parameters:
  - parameters: Model parameters [kS, kSP, kM, kP]
  - rng: Random number generator
  - train_end: End time for training data
  - train_step: Time step for training data
  - noise_level: Noise level for the data
  - initial_conditions: Initial conditions for species S and P

Returns:
  - data_S: Noisy data for species S
  - data_P: Noisy data for species P
  - times_S: Time points for species S
  - times_P: Time points for species P
  - val_data_S: Validation data for species S
  - val_data_P: Validation data for species P
"""
function simulate_inputs(parameters, rng; train_end=30, train_step=5, noise_level=0.05, initial_conditions=[2.0, 0.0])
    # parameters, timespan and initial values
    u0 = initial_conditions
    tspan = (0.,400)

    # ODE Problem and solution
    problem = ODEProblem(michaelismenten!, u0, tspan, parameters)
    solution = solve(problem, Tsit5())

    # Data point times
    times_S = 0:train_step:train_end
    times_P = 0:train_step:train_end

    # Ground truth data
    gtdata_S = solution(times_S, idxs=1)
    gtdata_P = solution(times_P, idxs=2)

    # Validation data
    val_data_S = solution(tspan[1]:0.1:tspan[end], idxs=1)
    val_data_P = solution(tspan[1]:0.1:tspan[end], idxs=2)

    # "Normally" distributed noise
    data_S = max.(0, gtdata_S.u .+ randn(rng, size(gtdata_S.u)).*maximum(val_data_S.u).*noise_level)
    data_P = max.(0, gtdata_P.u .+ randn(rng, size(gtdata_P.u)).*maximum(val_data_P.u).*noise_level)

    return data_S, data_P, times_S, times_P, val_data_S, val_data_P
end
