## SETTINGS

# define number of parallel cores 
n_cores = 5

# lambda values for regularization
const lambda_values = [0., 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]

# sampling schedules for training data
const sampling_schedules = [
    (5, 20:10:100),
    (10, 20:10:100),
    (20, 40:20:100)
]

# Number of initial values for optimization
const n_starting_points = 100

# Define the parameter values
const initial_p = [0.05, 0.2, 1.1, 0.08]


## END SETTINGS

println("Now Starting MichaelisMenten Experiment.\n\nSettings:\n\tλ = $(lambda_values)\n\tn_starting_points = $(n_starting_points)")

using Distributed

println("Setting up parallel pool of $(n_cores) cores.")
# add processes that match the number of cores set
if nprocs()-1 < n_cores
    addprocs(n_cores-nprocs()+1, exeflags="--project")
end

using JLD2

@everywhere begin
    using DifferentialEquations                                                       # Solving ODEs
    using Random, Statistics, LinearAlgebra                                    # Basic functionality and plotting
    using Lux                                                                         # Neural Network
    using ComponentArrays                                                             # Parameter Specification
    using Optimization, OptimizationOptimisers, OptimizationOptimJL # Parameter Estimation
    using StableRNGs                                                                  # Random number generator
    using WeightInitializers
    using SciMLSensitivity
    using LineSearches

    include("inputs.jl")
    include("ude.jl")
    rbf(x) = exp(-x^2)
end

# Seed a random number generator for reproducibility
rng = StableRNG(1847)

# Define the neural network component
U = Chain(
    Dense(2, 3, rbf),
    Dense(3, 3, rbf),
    Dense(3, 3, rbf),
    Dense(3, 1)
)

# Setup the neural network
nn_init, snn = Lux.setup(rng, U)

for (step_size, end_times) in sampling_schedules
  for end_time in end_times
    for lambd in lambda_values
      println("Running MSE optimizer for lambda = $(lambd) and end_time = $(end_time)")
      loc_rng = StableRNG(1234)

      # Simulate the data
      data_A, data_B, times_A, times_B, val_data_A, val_data_B = simulate_inputs(
        initial_p, loc_rng; 
        train_end = end_time, train_step=step_size)
      
      # Create UDE model
      ude_problem = michaelismenten_ude(U, initial_p, data_A, data_B, (0., 100.), snn)

      # # Define loss and validation functions 
      # mse_loss = michaelismenten_loss(ude_problem, [data_A, data_B], [[times_A...], [times_B...]])

      # Set the initial parameters of the optimization
      initials = initial_parameters(U, n_starting_points, loc_rng)

      # Define the optimization run
      optimizer = setup_model_training(
        ude_problem, 
        [data_A, data_B], 
        times_A, lambd, 
        [val_data_A, val_data_B], 
        0:0.1:400)

      # Optimize for all initial values.
      results = pmap(optimizer, initials)

      println("Saving result.")

      # Save model parameters, training and validation errors
      parameters = [r[1] for r in results]
      training_error = [r[2] for r in results]
      
      jldsave(
          "michaelis-menten/saved_runs/michaelismenten_$(lambd)_$(step_size)_$(end_time).jld2";
          parameters=parameters,
          training_error=training_error)
    end
  end
end

# Close parallel pool
rmprocs(procs()[2:end])