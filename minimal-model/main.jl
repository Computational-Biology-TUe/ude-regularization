# define number of parallel cores 
n_cores = 8

# lambda values for regularization
const λ_AUC = [0., 1e-2, 1e-1, 1., 10., 100.]
const λ_nonneg = [0., 1e-2, 1e-1, 1., 10., 100.]

# Number of initial values for optimization
const n_initials = 100

OVERWRITE = false

## END SETTINGS

println("Now Starting Minimal Model Experiment.\n\nSettings:\n\tλ_AUC = $(λ_AUC)\n\tλ_nonneg = $(λ_nonneg)\n\tn_starting_points = $(n_initials)")

using Distributed

println("Setting up parallel pool of $(n_cores) cores.")
# add processes that match the number of cores set
if nprocs()-1 < n_cores
    addprocs(n_cores-nprocs()+1, exeflags="--project")
end

using JLD2

@everywhere begin
  using DifferentialEquations
  using Lux
  using ComponentArrays                                                            
  using Optimization, OptimizationOptimisers, OptimizationOptimJL, SciMLSensitivity 
  using LineSearches
  using Statistics
  using StableRNGs
  using DataInterpolations
  using Trapz
  using SciMLBase
  using WeightInitializers

  include("inputs.jl")
  include("ude.jl")

  rbf(x) = exp(-x^2)

  # low std truncated normal initializer
  ude_neural_initializer(rng, dims...) = truncated_normal(rng, Float64, dims...; std = 1e-2)
end

rng = StableRNG(4520)

# Load the data
glucose, glucose_timepoints, insulin, insulin_timepoints = load_predict_gi("minimal-model/data/Predict-Glucose.csv", "minimal-model/data/Predict-Insulin.csv");

# Only use up until 240 (first meal)
glucose_timepoints = glucose_timepoints[1:7]
insulin_timepoints = insulin_timepoints[1:6]

glucose = Matrix{Float64}(glucose)[:, 1:7]
insulin = Matrix{Float64}(insulin)[:, 1:6]

mins = mean(insulin, dims=1)
mglc = mean(glucose, dims=1)

# Get the model parameter fits based on the current meal appearance
println("Obtaining initial model parameters:")
model_parameters = get_model_parameters(mglc, glucose_timepoints, mins, insulin_timepoints)
println("Initial model parameters found. Starting neural UDE optimization.")

# Define the neural network components
U = Chain(
    Dense(1, 3, rbf; init_weight = ude_neural_initializer),
    #Dense(3, 3, rbf; init_weight = ude_neural_initializer),
    Dense(3, 1; init_weight = ude_neural_initializer)
)

# Set the meal size and basal glucose and insulin values
M = 85500.
Qb = mglc[1][1]
Ib = mins[1][1]
I = insulin_interpolator(mins, insulin_timepoints)

# setup the neural network
nn_init, snn = Lux.setup(rng, U);
nn_init = ComponentArray(nn_init);

# define the model
udemodel = minimalmodel_ude(Qb, Ib, M, model_parameters, I, U, snn)
p_estim_temp = ComponentArray(nn = nn_init)
u0 = [Qb, 0.]
UDEproblem = ODEProblem{true, SciMLBase.FullSpecialize}(udemodel,  u0, (0., 480.), p_estim_temp, sensealg=SciMLSensitivity.ForwardDiffSensitivity())

save_timepoints = 0:1:480.
glucose_idx = findall(x -> x ∈ glucose_timepoints, save_timepoints)

# for all regularization strengths run the model
for lAUC in λ_AUC
  for lnonneg in λ_nonneg
    println("Running MSE optimizer for lAUC = $(lAUC) and lnonneg = $(lnonneg)")

    if !OVERWRITE && isfile("minimal-model/saved_runs_smaller_network/minimalmodel_$(lAUC)_$(lnonneg).jld2")
      println("File already exists, skipping.")
      continue
    end

    lossfn = get_ude_loss(UDEproblem, save_timepoints, mglc, glucose_idx, U, snn, lAUC, lnonneg)
    initials = initial_parameters(U, rng)
    p_estim_init = [ComponentArray(nn = initials()) for _ in 1:n_initials];

    fit_function = setup_model_training(p_estim_init, lossfn)

    results = pmap(fit_function, 1:n_initials)

    parameters = [r[1] for r in results]
    training_error = [r[2] for r in results]

    jldsave(
          "minimal-model/saved_runs_smaller_network/minimalmodel_$(lAUC)_$(lnonneg).jld2";
          parameters=parameters,
          training_error=training_error)
  end
end

# Close parallel pool
rmprocs(procs()[2:end])