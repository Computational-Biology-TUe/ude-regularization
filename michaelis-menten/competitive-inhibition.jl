# Additional analysis of regularization for a competitive inhibition model (extension of the Michaelis-Menten model)


using Distributed

n_cores = 4

println("Setting up parallel pool of $(n_cores) cores.")
# add processes that match the number of cores set
if nprocs()-1 < n_cores
    addprocs(n_cores-nprocs()+1, exeflags="--project")
end

using JLD2

@everywhere begin
  using DifferentialEquations
  using Random, Statistics, LinearAlgebra
  using Lux
  using ComponentArrays
  using Optimization, OptimizationOptimisers, OptimizationOptimJL
  using StableRNGs
  using WeightInitializers
  using SciMLSensitivity
  using LineSearches
  using CairoMakie, ColorSchemes
  using JLD2

  include("ude.jl")
  """
  michaelismenten_inhibition!(du, u, p, t)

  Ground truth model for the Michaelis-Menten kinetics model according to the following ODEs:
      dS/dt = kS*S - (kSP*S)/(kM_app + S)
      dP/dt = (kSP*S)/(kM_app + S) - kP*P
      dI/dt = kP*P - e_I*I

      with kM_app = kM * (1 + I/kI)
  Implementation according to the DifferentialEquations.jl API.
  """
  function michaelismenten_inhibition!(du, u, p, t)
      S, P, I = u
      kS, kSP, kM, kP, kI, e_I = p

      kM_app = kM * (1 + I/kI)

      du[1] = kS*S - (kSP*S)/(kM_app+S)
      du[2] = (kSP*S)/(kM_app+S) - kP*P
      du[3] = kP*P - e_I*I
  end

  """
  simulate_inhibition_data(parameters, rng; train_end=30, train_step=5, noise_level=0.05, initial_conditions=[2.0, 0.0])

  Simulate noisy data for the Michaelis-Menten competitive inhibition
  kinetics model according to the following ODEs:
      dS/dt = kS*S - (kSP*S)/(kM_app + S)
      dP/dt = (kSP*S)/(kM_app + S) - kP*P
      dI/dt = kP*P - e_I*I

      with kM_app = kM * (1 + I/kI)

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
    - data_I: Noisy data for species I
    - times_S: Time points for species S
    - times_P: Time points for species P
    - times_I: Time points for species I
    - val_data_S: Validation data for species S
    - val_data_P: Validation data for species P
    - val_data_I: Validation data for species I
  """
  function simulate_inhibition_data(parameters, rng; train_end=60, train_step=5, noise_level=0.05, initial_conditions=[2.0, 0.0, 0.0], tspan_val_end=800.0)

    tspan = (0.0, tspan_val_end)

    # ODE Problem and solution
    problem = ODEProblem(michaelismenten_inhibition!, initial_conditions, tspan, parameters)
    solution = solve(problem, Tsit5())

    # Data point times
    times_S = 0:train_step:train_end
    times_P = 0:train_step:train_end
    times_I = 0:train_step:train_end

    # Ground truth data
    gtdata_S = solution(times_S, idxs=1)
    gtdata_P = solution(times_P, idxs=2)
    gtdata_I = solution(times_I, idxs=3)

    # Validation data
    val_data_S = solution(tspan[1]:0.1:tspan[end], idxs=1)
    val_data_P = solution(tspan[1]:0.1:tspan[end], idxs=2)
    val_data_I = solution(tspan[1]:0.1:tspan[end], idxs=3)

    # "Normally" distributed noise
    data_S = max.(0, gtdata_S.u .+ randn(rng, size(gtdata_S.u)).*maximum(val_data_S.u).*noise_level)
    data_P = max.(0, gtdata_P.u .+ randn(rng, size(gtdata_P.u)).*maximum(val_data_P.u).*noise_level)
    data_I = max.(0, gtdata_I.u .+ randn(rng, size(gtdata_I.u)).*maximum(val_data_I.u).*noise_level)

    return data_S, data_P, data_I, times_S, times_P, times_I, val_data_S, val_data_P, val_data_I
  end

  """
  michaelismenten_inhibition_ude(net, initial_p, data_S, data_P, data_I, tspan, snn)

  Create a UDE model for the Michaelis-Menten kinetics model according to the following ODEs:
      dS/dt = k1*S - Û
      dP/dt = Û - k3*P
      dI/dt = k3*P - k4*I
  where Û is the output of the neural network.

  Arguments:
      - net: Neural network model
      - initial_p: Initial parameters for the model
      - data_S: Noisy data for species S
      - data_P: Noisy data for species P
      - data_I: Noisy data for species I
      - tspan: Time span for the ODE
      - snn: Neural network state (given by Lux.setup)

  Returns:
      - ODEProblem for the UDE model that can be simulated with DifferentialEquations.jl
  """
  function michaelismenten_inhibition_ude(net, initial_p, data_S, data_P, data_I, tspan, snn)
    
      # Function that defines the differential equations
    function michaelismenten_inhibition_ude!(du, u, p, t, p_true)

      # Calculate the output of the neural network
        Û = net(u, p.ude, snn)[1]
      # Set the state variables (Substrate and Product)
        S, P, I = u
      # Compute the derivatives
        du[1] = p_true[1]*S - Û[1]
        du[2] = Û[1] - p_true[4]*P
        du[3] = p_true[4]*P - p_true[6]*I
      # Return nothing, as the derivatives are updated in-place
        nothing
    end

    # Define a closure to pass the initial parameters to the ODEProblem
    michaelismenten_inhibition_ude!(du, u, p, t) = michaelismenten_inhibition_ude!(du, u, p, t, initial_p)

  # Return the ODEProblem
    return ODEProblem{true, SciMLBase.FullSpecialize}(michaelismenten_inhibition_ude!,[data_S[1], data_P[1], data_I[1]], tspan)
  end

  rbf(x) = exp(-x^2)

end


initial_p = [9e-3, 0.2, 1.1, 0.08, 6e-2, 0.03]
n_starting_points = 50
lambda_values = [0.0, 1e-5, 1e-3, 1.0, 10.0]


rng = StableRNG(1847)

# Define the neural network component
U = Chain(
    Dense(3, 3, rbf), # one additional input
    Dense(3, 3, rbf),
    Dense(3, 3, rbf),
    Dense(3, 1)
)

# Setup the neural network
nn_init, snn = Lux.setup(rng, U)
for lambd in lambda_values
  loc_rng = StableRNG(1234)

  # Simulate the data
  data_S, data_P, data_I, times_S, times_P, times_I, val_data_S, val_data_P, val_data_I = simulate_inhibition_data(
    initial_p, loc_rng; 
    train_end = 200, train_step=10, noise_level=0.05)

  # Create UDE model
  ude_problem = michaelismenten_inhibition_ude(U, initial_p, data_S, data_P, data_I, (0., 100.), snn)

  # Set the initial parameters of the optimization
  initials = initial_parameters(U, n_starting_points, loc_rng)

  # Define the optimization run
  optimizer = setup_inhibition_model_training(
    ude_problem, 
    [data_S, data_P, data_I], 
    times_S, lambd, 
    [val_data_S, val_data_P, val_data_I], 
    0:0.1:800)

  # Optimize for all initial values.
  results = pmap(optimizer, initials)

  println("Saving result.")

  # Save model parameters, training and validation errors
  parameters = [r[1] for r in results]
  training_error = [r[2] for r in results]
  validation_error = [r[3] for r in results]


  jldsave(
          "michaelis-menten/saved_runs_ci/ci_$(lambd).jld2";
          parameters=parameters,
          training_error=training_error,
          validation_error=validation_error)
end

fig_sol = let f = Figure(size=(750,900))
  loc_rng = StableRNG(1234)

  labels = [
    ["A", "B", "C"],
    ["D", "E", "F"],
    ["G", "H", "I"],
    ["J", "K", "L"],
    ["M", "N", "O"]
  ]
  # Simulate the data
  data_S, data_P, data_I, times_S, times_P, times_I, val_data_S, val_data_P, val_data_I = simulate_inhibition_data(
    initial_p, loc_rng; 
    train_end = 200, train_step=10, noise_level=0.05, tspan_val_end=600.0)

  # Create UDE model
  ude_problem = michaelismenten_inhibition_ude(U, initial_p, data_S, data_P, data_I, (0., 100.), snn)

  tA = 0:0.1:600

  for (i,lambd) in enumerate(lambda_values)
    params, val_errors = jldopen("michaelis-menten/saved_runs_ci/ci_$(lambd).jld2") do file
      file["parameters"], file["validation_error"]
    end

    top_25 = partialsortperm(val_errors, 1:25)
    ga = GridLayout(f[i,1])
    gb = GridLayout(f[i,2])
    gc = GridLayout(f[i,3])

    axS = CairoMakie.Axis(ga[1,1], xlabel="Time [min]", ylabel= "S", title="λ = $(lambd)")
    axP = CairoMakie.Axis(gb[1,1], xlabel="Time [min]", ylabel= "P")
    axI = CairoMakie.Axis(gc[1,1], xlabel="Time [min]", ylabel= "I")

    predictions = [predict(ude_problem, param, 0:0.1:600) for param in params][top_25]

    pred_arr = cat(predictions...; dims=3)


    mean_preds = mean(pred_arr, dims=3)[:,:]
    std_preds = std(pred_arr, dims=3)[:,:]
    lines!(axS, tA, mean_preds[1,:], color=colorschemes[:Egypt][1], label="Model (mean ± std)")
    lines!(axP, tA, mean_preds[2,:], color=colorschemes[:Egypt][1])
    lines!(axI, tA, mean_preds[3,:], color=colorschemes[:Egypt][1])

    band!(axS, tA, mean_preds[1,:] .- std_preds[1,:], mean_preds[1,:] .+ std_preds[1,:], color=(colorschemes[:Egypt][1], 0.2), label="Model (mean ± std)")
    band!(axP, tA, mean_preds[2,:] .- std_preds[2,:], mean_preds[2,:] .+ std_preds[2,:], color=(colorschemes[:Egypt][1], 0.2))
    band!(axI, tA, mean_preds[3,:] .- std_preds[3,:], mean_preds[3,:] .+ std_preds[3,:], color=(colorschemes[:Egypt][1], 0.2))

    # for pred in predictions
    #   lines!(axS, tA, pred[1,:], color=colorschemes[:Egypt][1])
    #   lines!(axP, tA, pred[2,:], color=colorschemes[:Egypt][1])
    #   lines!(axI, tA, pred[3,:], color=colorschemes[:Egypt][1])
    # end

    lines!(axS, tA, val_data_S.u, label="Ground Truth")
    lines!(axP, tA, val_data_P.u)
    lines!(axI, tA, val_data_I.u)

    vspan!(axS, [0], [200], color = (:black, 0.2))
    vspan!(axP, [0], [200], color = (:black, 0.2))
    vspan!(axI, [0], [200], color = (:black, 0.2))

    for (label, layout) in zip(labels[i], [ga, gb, gc])
      Label(layout[1, 1, TopLeft()], label,
      fontsize = 15,
      font = :bold,
      padding = (0, 25, 5, 0),
      halign = :right)
    end
    if i == length(lambda_values)
      Legend(f[i+1,1:3], axS, merge=true, orientation=:horizontal)
    end
  end

  f
end

save("figures/others/supp_inhibition.png", fig_sol, px_per_unit=4)