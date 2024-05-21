using JLD2
using CairoMakie
using ColorSchemes
using DifferentialEquations
using ComponentArrays: ComponentArray

# Set figure theme
figure_theme = Theme(
  fontsize = 12,
  palette = (color = colorschemes[:Egypt], marker = ['o'], patchcolor = [(c, 0.2) for c in colorschemes[:Egypt]]),
  Scatter = (cycle = [:color, :marker], markersize = 12),
  Figure = (size = (100,100),)
  )

set_theme!(figure_theme)

# Load the data
glucose, glucose_timepoints, insulin, insulin_timepoints = load_predict_gi("minimal-model/data/Predict-Glucose.csv", "minimal-model/data/Predict-Insulin.csv");

# Only use up until 240 (first meal)
glucose_timepoints = glucose_timepoints[1:7]
insulin_timepoints = insulin_timepoints[1:6]

glucose = Matrix{Float64}(glucose)[:, 1:7]
insulin = Matrix{Float64}(insulin)[:, 1:6]

mins = mean(insulin, dims=1)
mglc = mean(glucose, dims=1)

stdins = 1.96 .* std(insulin, dims=1) ./ sqrt(size(insulin, 1))
stdglc = 1.96 .* std(glucose, dims=1) ./ sqrt(size(glucose, 1))

# Get the model parameter fits based on the current meal appearance
ode_model_parameters = get_model_parameters(mglc, glucose_timepoints, mins, insulin_timepoints)

M = 85500.
VG = 18.57
fG = 0.005551
Qb = mglc[1][1]
Ib = mins[1][1]
I = get_insulin_interpolator_fn(mins, insulin_timepoints);
u0 = [Qb, 0.]

full_ode_model_pvector = [ode_model_parameters; [1.4, 0.014, 86000., VG, fG, u0[1], Ib]]

original_problem = ODEProblem(make_minimal_model(I), u0, (0., 480.), full_ode_model_pvector, sensealg=ForwardDiffSensitivity())