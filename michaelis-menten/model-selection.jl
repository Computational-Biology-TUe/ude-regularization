

# lambda values for regularization
const lambda_values = [0.0]

# sampling schedules for training data
const sampling_schedules = [
(5, [100])
]

# Number of initial values for optimization
const n_starting_points = 10

# noise level for the data
const noise_level = 0.05

# Define the parameter values
const initial_p = [0.05, 0.2, 1.1, 0.08]

n_cores=5

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
const widths = [2, 3, 4]
const depths = [1, 2, 3, 4]

function generate_chain(width, depth)
    layers = []
    for i in 1:(depth-1)
        push!(layers, Dense(width, width, rbf))
    end
    return Chain(Dense(2, width, rbf), layers..., Dense(width, 1))
end


step_size = 5
end_time = 100
lambd = 0.0

for width in widths
  for depth in depths

    U = generate_chain(width, depth)
    nn_init, snn = Lux.setup(rng, U)

    println("Running MSE optimizer for width = $(width) and depth = $(depth)")
    loc_rng = StableRNG(1234)

    # Simulate the data
    data_A, data_B, times_A, times_B, val_data_A, val_data_B = simulate_inputs(
    initial_p, loc_rng; 
    train_end = end_time, train_step=step_size, noise_level=noise_level)
    
    # Create UDE model
    ude_problem = michaelismenten_ude(U, initial_p, data_A, data_B, (0., 100.), snn)

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
    validation_error = [r[3] for r in results]
    
    jldsave(
        "michaelis-menten/model_selection_runs/michaelismenten_$(width)_$(depth).jld2";
        parameters=parameters,
        training_error=training_error,
        validation_error=validation_error)
  end
end

# Close parallel pool
rmprocs(procs()[2:end])