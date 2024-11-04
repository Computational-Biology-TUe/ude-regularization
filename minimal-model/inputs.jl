using SpecialFunctions, CSV, DataFrames, Statistics

"""
load_predict_gi(glucose_file::String, insulin_file::String)

Load the predict glucose and insulin data from CSV files and return data as vectors.

Parameters:
  - glucose_file: Path to the glucose data CSV file
  - insulin_file: Path to the insulin data CSV file

Returns:
  - glucose: mean of glucose data
  - glucose_timepoints: Time points for glucose data
  - glucose_std: standard deviation of glucose data
  - insulin: mean of insulin data 
  - insulin_timepoints: Time points for insulin data
  - insulin_std: standard deviation of insulin data
"""
function load_predict_gi(glucose_file::String, insulin_file::String)

    # read glucose and insulin dataframes 
    glucose = DataFrame(CSV.File(glucose_file))
    insulin = DataFrame(CSV.File(insulin_file))

    # extract timepoints
    glucose_timepoints = glucose[!,:time]
    insulin_timepoints = insulin[!,:time]

    # extract std
    glucose_std = glucose[!,:std]
    insulin_std = insulin[!,:std]
    
    # extract glucose and insulin data
    glucose = glucose[!, :glucose]
    insulin = insulin[!, :insulin]

    glucose, glucose_std, glucose_timepoints, insulin, insulin_std, insulin_timepoints
end

"""
insulin_interpolator(mins, insulin_timepoints)

Interpolate the insulin data using cubic spline interpolation.

Parameters:
  - mins: Mean insulin data
  - insulin_timepoints: Time points for insulin data

Returns:
  - CubicSpline object for the interpolated insulin data. Can be called with a time point to get the interpolated insulin value.

"""
function insulin_interpolator(mins, insulin_timepoints)

  steady_state_timepoints_start = [-60, -50, -40, -30]
  insulin_start = repeat([mins[1]], length(steady_state_timepoints_start))

  steady_state_timepoints_end = insulin_timepoints[end] .+ [60, 120, 240, 360, 480]
  insulin_end = repeat([mins[1]], length(steady_state_timepoints_end))

  CubicSpline([insulin_start; mins...; insulin_end], [steady_state_timepoints_start; insulin_timepoints; steady_state_timepoints_end]);

end

"""
rate_of_appearance(σ, k, t, M, γ)

Calculate the rate of appearance of glucose in the blood based on the gamma function estimation.

Parameters:
  - σ: Shape parameter for the gamma function
  - k: Basal rate of appearance
  - t: Time
  - M: Total amount of glucose
  - γ: conversion factor of glucose from mg to mmol divided by the volume of distribution 

Returns:
  - Rate of appearance of glucose in the blood at time t
"""
function rate_of_appearance(σ, k, t, M, γ)
  t >= 0 ? ((γ*M)/gamma(σ))* k^σ * t^(σ-1) * exp(-k*t) : 0.
end

"""
make_minimal_model(I)

Create the minimal model for the oral minimal model of glucose kinetics.

Parameters:
  - I: Insulin interpolator function

Returns:
  - oralminimalmodel! function. Can be used as the ODE function for the minimal model.
"""
function make_minimal_model(I)
  function oralminimalmodel!(du, u, p, t)

    p1, p2, p3, σ, k, M, VG, fG, Qb, Ib = p
    du[1] = - u[1]*u[2] - p3*(u[1]-Qb) + rate_of_appearance(σ, k, t, M, fG/VG)
    du[2] = - p1 * u[2] + p2 * (I(t) - Ib)

  end

  return oralminimalmodel!
end

"""
get_model_parameters(mglc, glucose_timepoints, mins, insulin_timepoints; VG = 18.57, fG = 0.005551, M = 85500.)

Get the model parameters for the minimal model based on the glucose and insulin data.

Parameters:
  - mglc: Mean glucose data
  - glucose_timepoints: Time points for glucose data
  - mins: Mean insulin data
  - insulin_timepoints: Time points for insulin data
  - VG: Volume of distribution of glucose; default is 18.57
  - fG: conversion factor of glucose from mg to mmol; default is 0.005551
  - M: Total amount of glucose in meal; default is 85500.

Returns:
  - Model parameters for the minimal model based on the glucose and insulin data (p1, p2, p3)
"""
function get_model_parameters(mglc, glucose_timepoints, mins, insulin_timepoints; VG = 18.57, fG = 0.005551, M = 85500.)
  Gb = mglc[1][1]
  Ib = mins[1][1]
  tspan = (insulin_timepoints[1], insulin_timepoints[end])
  u0 = [Gb, 0.]
  I = insulin_interpolator(mins, insulin_timepoints)
  p_init = [3e-4, 3e-4, 3e-4, 1.4, 0.014, 86000., VG, fG, u0[1], Ib]

  problem = ODEProblem(make_minimal_model(I), u0, tspan, p_init, sensealg=ForwardDiffSensitivity())

  function loss(p)
    p_model = [p[1], p[2], p[3], 1.4, 0.014, M, VG, fG, u0[1], Ib]
    sol = Array(solve(problem, p=p_model, saveat=glucose_timepoints))
    glucose = sol[1,:]
    g_error = sum(abs2, glucose .- mglc )
    return g_error
  end

  p_estim_init = [1.9e-2, 2.65e-4, 2.60e-2]

  # Container to track the losses
  losses = Float64[]

  callback = function (p, l)
    push!(losses, l)
    if length(losses)%20==0
        println("Current loss after $(length(losses)) iterations: $(losses[end])")
    end
    return false
  end

  adtype = Optimization.AutoForwardDiff()
  optf = Optimization.OptimizationFunction((x,p)->loss(x), adtype)
  optprob = Optimization.OptimizationProblem(optf, p_estim_init, lb=[0., 0.,0.], ub=[5., 5.,5.])
  println("Initial loss is  $(loss(p_estim_init))")
  res1 = Optimization.solve(optprob, LBFGS(), callback=callback, maxiters = 1000)
  println("Training loss after $(length(losses)) iterations: $(losses[end])")

  return res1.u
end

# glucose, glucose_timepoints, insulin, insulin_timepoints = load_predict_gi("minimal-model/data/Predict-Glucose.csv", "minimal-model/data/Predict-Insulin.csv")

# # Only use up until 240 (first meal)
# glucose_timepoints = glucose_timepoints[1:7]
# insulin_timepoints = insulin_timepoints[1:6]

# glucose = Matrix{Float64}(glucose)[:, 1:7]
# insulin = Matrix{Float64}(insulin)[:, 1:6]

# mins = mean(insulin, dims=1)
# stdins = std(insulin, dims=1)
# mglc = mean(glucose, dims=1)
# stdglc = std(glucose, dims=1)

# glucose_df = DataFrame(glucose=mglc[:], time=glucose_timepoints, std=stdglc[:])
# insulin_df = DataFrame(insulin=mins[:], time=insulin_timepoints, std=stdins[:])

# CSV.write("minimal-model/data/mean_glucose.csv", glucose_df)
# CSV.write("minimal-model/data/mean_insulin.csv", insulin_df)

# DataFrame(CSV.File("minimal-model/data/mean_glucose.csv"))[!, :glucose]