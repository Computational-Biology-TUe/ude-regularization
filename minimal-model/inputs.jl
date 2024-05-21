using SpecialFunctions, CSV, DataFrames

function load_predict_gi(glucose_file::String, insulin_file::String)

    # read glucose and insulin dataframes 
    glucose = DataFrame(CSV.File(glucose_file, delim=";", drop=[1], decimal=',', types=Union{Missing,Float64}))
    insulin = DataFrame(CSV.File(insulin_file, delim=";", drop=[1], decimal=',', types=Union{Missing,Float64}))

    # extract first meal
    glucose = glucose[:, 1:7]
    insulin = insulin[:, 1:6]

    # find missing data in both dataframes
    missing_g_rows = any.(eachrow((ismissing.(glucose))))
    missing_i_rows = any.(eachrow((ismissing.(insulin))))

    missing_rows = missing_g_rows .| missing_i_rows

    glucose = glucose[Not(missing_rows), :]
    insulin = insulin[Not(missing_rows), :]

    glucose_timepoints = parse.(Float64, names(glucose))
    insulin_timepoints = parse.(Float64, names(insulin))

    glucose, glucose_timepoints, insulin, insulin_timepoints
end

function insulin_interpolator(mins, insulin_timepoints)

  steady_state_timepoints_start = [-60, -50, -40, -30]
  insulin_start = repeat([mins[1]], length(steady_state_timepoints_start))

  steady_state_timepoints_end = insulin_timepoints[end] .+ [60, 120, 240, 360, 480]
  insulin_end = repeat([mins[1]], length(steady_state_timepoints_end))

  CubicSpline([insulin_start; mins...; insulin_end], [steady_state_timepoints_start; insulin_timepoints; steady_state_timepoints_end]);

end

function rate_of_appearance(σ, k, t, M, γ)
  t >= 0 ? ((γ*M)/gamma(σ))* k^σ * t^(σ-1) * exp(-k*t) : 0.
end

function make_minimal_model(I)
  function oralminimalmodel!(du, u, p, t)

    p1, p2, p3, σ, k, M, VG, fG, Qb, Ib = p
    du[1] = - u[1]*u[2] - p3*(u[1]-Qb) + rate_of_appearance(σ, k, t, M, fG/VG)
    du[2] = - p1 * u[2] + p2 * (I(t) - Ib)

  end

  return oralminimalmodel!
end

function get_model_parameters(mglc, glucose_timepoints, mins, insulin_timepoints; VG = 18.57, fG = 0.005551, M = 85500.)
  Gb = mglc[1][1]
  Ib = mins[1][1]
  tspan = (insulin_timepoints[1], insulin_timepoints[end])
  u0 = [Gb, 0.]
  I = get_insulin_interpolator_fn(mins, insulin_timepoints)
  p_init = [3e-4, 3e-4, 3e-4, 1.4, 0.014, 86000., VG, fG, u0[1], Ib]

  problem = ODEProblem(make_minimal_model(I), u0, tspan, p_init, sensealg=ForwardDiffSensitivity())

  function loss(p)
    p_model = [p[1], p[2], p[3], 1.4, 0.014, M, VG, fG, u0[1], Ib]
    sol = Array(solve(problem, p=p_model, saveat=glucose_timepoints))
    glucose = sol[1,:]
    g_error = sum(abs2, glucose .- mglc[:])
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