using JLD2
using CairoMakie
using ColorSchemes
using DifferentialEquations
using ComponentArrays
using DataInterpolations
using SciMLSensitivity
using Optimization, OptimizationOptimisers, OptimizationOptimJL, SciMLSensitivity 
using LineSearches
using Statistics, StableRNGs
using Lux                                                         
using LaTeXStrings
using Printf
using Trapz
include("../michaelis-menten/inputs.jl")
include("../michaelis-menten/ude.jl")

FIGURE_RESOLUTION = 4 # set to 4 for paper resolution

rbf(x) = exp.(-x^2)

rng = StableRNG(1234)

# Set figure theme
figure_theme = Theme(
  fontsize = 12,
  palette = (color = colorschemes[:Egypt], marker = ['o'], patchcolor = [(c, 0.2) for c in colorschemes[:Egypt]]),
  Scatter = (cycle = [:color, :marker], markersize = 12),
  Figure = (size = (100,100),)
  )

set_theme!(figure_theme)

### Michaelis Menten ###


## Fig 0: Example Data
initial_p = [0.05, 0.2, 1.1, 0.08]
data_A, data_B, times_A, times_B, val_data_A, val_data_B = simulate_inputs(
  initial_p, rng)

fig_example_data, axis_example_data, plot_example_data = scatter(
  times_A, data_A;
  axis = (
    xlabel = "Time", 
    ylabel = "Concentration",
    xgridvisible = false,
    ygridvisible = false),
  figure = (; size=(300,300),))
lines!(axis_example_data, 0:0.1:100, Array(val_data_A)[1:1001], label="S")

scatter!(axis_example_data, times_B, data_B)
lines!(axis_example_data, 0:0.1:100, Array(val_data_B)[1:1001], label="P")

axislegend()

save("figures/others/example_data.png", fig_example_data, px_per_unit=FIGURE_RESOLUTION)

fig_example_data

## Fig 2: Model Forecasts 

rng = StableRNG(1234)

# Get the data used for training to obtain initial conditions
initial_p = [0.05, 0.2, 1.1, 0.08]
data_A, data_B, times_A, times_B, val_data_A, val_data_B = simulate_inputs(
  initial_p, rng)

# Setup the problem
# Define the neural network component
U = Chain(
    Dense(2, 3, rbf),
    Dense(3, 3, rbf),
    Dense(3, 3, rbf),
    Dense(3, 1)
)
p_neural_init, snn = Lux.setup(rng, U)
p_neural_init = ComponentArray(p_neural_init)
# Define the hybrid model
ude_problem = michaelismenten_ude(U, initial_p, data_A, data_B, (times_A[1], times_A[end]), snn)

# Load the results
regularization_strengths = [0.0, 1e-5, 1.0]
time_between_samples = 5
selected_sampling_time = 40
michment_results = [jldopen("michaelis-menten/saved_runs/michaelismenten_$(λ)_$(time_between_samples)_$(selected_sampling_time).jld2") for λ in regularization_strengths]

figure_model_forecasts = let f = Figure(size=(750,300))

  ga = f[1, 1] = GridLayout()
  gb = f[1,2] = GridLayout()
  gc = f[1,3] = GridLayout()

  grids = [ga, gb, gc]

  titles = ["No Regularization", "Regularization (λ = 1e-5)", "Regularization (λ = 1)"]

  for (i, (res, title)) in enumerate(zip(michment_results, titles))
    training_errors = res["training_error"]
    best_models = partialsortperm(training_errors, 1:25)

    parameters = res["parameters"][best_models]
    
    model_fit_axis = CairoMakie.Axis(grids[i][1,1], limits=(nothing, (-1, 2.5)), title=title, xlabel="Time [min]", ylabel = "Concentration [mM]")
    As = []
    Bs = []
    for p in parameters
      sol = solve(ude_problem, p=p, saveat=0.1, tspan=(0., 100.))
      push!(As, sol[1,:])
      push!(Bs, sol[2,:])
    end

    lines!(model_fit_axis, 0:0.1:100, Array(val_data_A)[1:1001], color=:black, label="S True")
    lines!(model_fit_axis, 0:0.1:100, Array(val_data_B)[1:1001], color=:black, linestyle=:dash, label="P True")

    vspan!(model_fit_axis, [0], [selected_sampling_time], color = (:black, 0.2))

    # compute the interquartile range
    Am = hcat(As...)
    Bm = hcat(Bs...)
    Alow = Float64[]
    Aup = Float64[]
    Blow = Float64[]
    Bup = Float64[]
    for c in axes(Am, 1)
      upper = quantile(Am[c,:], 0.75)
      lower = quantile(Am[c,:], 0.25)
      push!(Alow, lower)
      push!(Aup, upper)
      upper = quantile(Bm[c,:], 0.75)
      lower = quantile(Bm[c,:], 0.25)
      push!(Blow, lower)
      push!(Bup, upper)
    end

    lines!(model_fit_axis, 0:0.1:100, median(Am, dims=2)[:,1], label="S ± IQR")
    band!(model_fit_axis, 0:0.1:100, Alow, Aup, color=(colorschemes[:Egypt][1], 0.2), label="S ± IQR")
    lines!(model_fit_axis, 0:0.1:100, median(Bm, dims=2)[:,1], label="P ± IQR")
    band!(model_fit_axis, 0:0.1:100, Blow, Bup, color=(colorschemes[:Egypt][2], 0.2), label="P ± IQR")
    if i == 1
      f[2,1:3] = Legend(f, model_fit_axis, "Legend", merge=true, framevisible=false, orientation=:horizontal)
    end
  end

  for (label, layout) in zip(["(A)", "(B)", "(C)"], [ga, gb, gc])
    Label(layout[1, 1, TopLeft()], label,
        fontsize = 15,
        font = :bold,
        padding = (0, 25, 5, 0),
        halign = :right)
  end
  f
end

save("figures/fig_2_model_forecasts.png", figure_model_forecasts, px_per_unit=FIGURE_RESOLUTION)

## Fig 3: Regularization Strength Distributions

lambda_values= [0., 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
time_between_samples = 5
selected_sampling_time = 40
michment_results = [jldopen("michaelis-menten/saved_runs/michaelismenten_$(λ)_$(time_between_samples)_$(selected_sampling_time).jld2") for λ in lambda_values]
mse_vals = vcat([res["validation_error"] for res in michment_results]...)

figure_regularization_strengths = let f = Figure(size=(750,300))
  dotplot_width = 0.1
  categories = [@sprintf "λ = %g" val for val in lambda_values]

  #mse_matrix_5 = [res["validation_error"] for res in michment_results]

  axis_regularization_strength = CairoMakie.Axis(f[1,1];
    xticks = ([1:11...] ./ 1.5, categories),
    xticklabelrotation = pi/6,
    xlabel = "Regularization Strength",
    ylabel = L"\mathrm{Err}_{\mathrm{val}}")
  # ff = plot(;xlabel="Sampling Duration", ylabel="SSE", yaxis=:log, grid=false)
  selected_sampling_time = 40
  labels = ["Unregularized", "Regularized"]
  positions = vcat([repeat([i/1.5], 100) for i in eachindex(lambda_values)]...)
  mse = log10.(mse_vals)

  violin!(axis_regularization_strength, positions, mse; side=:left, color = (colorschemes[:Egypt][1], 0.8))
  scatter!(axis_regularization_strength, positions.+0.08 + rand(length(positions))*dotplot_width, mse; markersize=3, marker=:circle)
  f
end


save("figures/fig_3_model_regularization_strengths.png", figure_regularization_strengths, px_per_unit=FIGURE_RESOLUTION)


# Figure 4

lambda_values = [0., 1e-5, 1.]
sampling_schedules = [5, 10, 20]
end_times_lambdas = [
  20:10:100,
  20:10:100,
  40:20:100
]

figure_sampling_schedules = let f = Figure(size=(750,300))

  ga = f[1, 1] = GridLayout()
  gb = f[1,2] = GridLayout()
  gc = f[1,3] = GridLayout()
  titles = [
    "Sampled every 5 min",
    "Sampled every 10 min",
    "Sampled every 20 min"
  ]
  legend_axis = []

  for (schedule, grid, title, end_times) in zip(sampling_schedules, [ga,gb,gc], titles, end_times_lambdas)

    axis = CairoMakie.Axis(grid[1,1], xlabel="Sampling Duration", ylabel = L"\mathrm{Err}_{\mathrm{val}}", title=title)
    # ff = plot(;xlabel="Sampling Duration", ylabel="SSE", yaxis=:log, grid=false)
    labels = ["λ = 0", "λ = 1e-5", "λ = 1"]
    for (λ, label) in zip(lambda_values, labels)

      michment_results = [jldopen("michaelis-menten/saved_runs/michaelismenten_$(λ)_$(schedule)_$(sampling_time).jld2") for sampling_time in end_times]
      mse_vals = log10.(hcat([res["validation_error"] for res in michment_results]...))
      means = mean(mse_vals, dims=1)[:]
      stds = 1.96 .* std(mse_vals, dims=1)[:] ./ (sqrt(size(mse_vals, 1)))
      #println(size(mse_vals))
      lines!(axis, end_times, means, label=label)
      band!(axis, end_times, means-stds, means+stds; alpha=0.1,label=label, transparency=true)
    end

    if schedule == 5
      push!(legend_axis, axis)
    end
  end

  for (label, layout) in zip(["(A)", "(B)", "(C)"], [ga, gb, gc])
    Label(layout[1, 1, TopLeft()], label,
        fontsize = 15,
        font = :bold,
        padding = (0, 25, 5, 0),
        halign = :right)
  end
  f[2,:] = Legend(f[2,1:3], legend_axis[1], "Legend", merge=true, framevisible=false, orientation=:horizontal)
  
  f

end

save("figures/fig_4_sampling_schedules.png", figure_sampling_schedules, px_per_unit=FIGURE_RESOLUTION)

include("../minimal-model/inputs.jl")
include("../minimal-model/ude.jl")

## Load data
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

stdins = 1.96 .* std(insulin, dims=1) ./ sqrt(size(insulin, 1))
stdglc = 1.96 .* std(glucose, dims=1) ./ sqrt(size(glucose, 1))

# Get the model parameter fits based on the current meal appearance
ode_model_parameters = get_model_parameters(mglc, glucose_timepoints, mins, insulin_timepoints)

M = 85500.
VG = 18.57
fG = 0.005551
Qb = mglc[1][1]
Ib = mins[1][1]
I = insulin_interpolator(mins, insulin_timepoints);
u0 = [Qb, 0.]

full_ode_model_pvector = [ode_model_parameters; [1.4, 0.014, M, VG, fG, u0[1], Ib]]

original_problem = ODEProblem(make_minimal_model(I), u0, (0., 480.), full_ode_model_pvector, sensealg=ForwardDiffSensitivity())

# Define the neural network components
rbf(x) = exp(-x^2)

U = Chain(
    Dense(1, 3, rbf),
    Dense(3, 3, rbf),
 #   Dense(3, 3, rbf),
    Dense(3, 1)
)

nn_init, snn = Lux.setup(rng, U);
nn_init = ComponentArray(nn_init);

udemodel = minimalmodel_ude(Qb, Ib, M, ode_model_parameters, I, U, snn)

function simulate(p)
    UDEproblem = ODEProblem{true, SciMLBase.FullSpecialize}(udemodel,  u0, (0., 480.), saveat=0.1, p, sensealg=SciMLSensitivity.ForwardDiffSensitivity())
    solve(UDEproblem)
end

auc_values = [0., 1e-2, 1e-1, 1., 10., 100.]
nonneg_values = [0., 1e-2, 1e-1, 1., 10., 100.]

model_parameters = Dict()
model_objectives = Dict()

for auc in auc_values
  for nonneg in nonneg_values
    models = jldopen("minimal-model/saved_runs/minimalmodel_$(auc)_$(nonneg).jld2", "r")

    parameters = [isnothing(model) ? nothing : model for model in models["training_error"]]
    objectives = [isnothing(model) ? nothing : model for model in models["training_error"]]

    model_parameters[(auc, nonneg)] = models["parameters"][.!isnothing.(parameters)]
    model_objectives[(auc, nonneg)] = models["training_error"][.!isnothing.(parameters)]
  end
end

fig_model_fits = Figure(resolution=(1200,1200))

for (position, parameters) in enumerate(model_parameters)

  xpos = indexin(parameters[1][1], auc_values)[1]
  ypos = indexin(parameters[1][2], nonneg_values)[1]

  ax_model_fit = CairoMakie.Axis(fig_model_fits[xpos, ypos], limits = (nothing, (4,8)))

  simulations = simulate.(parameters[2])

  for sim in simulations
    lines!(ax_model_fit, sim.t, Array(sim)[1,:], color = (colorschemes[:Egypt].colors[1], 0.1), label="Model")
  end

  scatter!(ax_model_fit, glucose_timepoints, mglc[1,:], color= :black, label="Data", markersize=12, marker='o');
  errorbars!(ax_model_fit,  glucose_timepoints, mglc[1,:], stdglc[1,:], whiskerwidth = 10)
  text!(ax_model_fit, -50, 7.5; text= L"\lambda_{\text{AUC}} = %$(round(parameters[1][1], digits=2)) \text{, } \lambda_{\text{neg}} = %$(round(parameters[1][2], digits=2))")

  if xpos == 1 && ypos == 5
    axislegend(merge=true)
  end

end

save("figures/others/minimal_model_all_fits.png", fig_model_fits, px_per_unit=FIGURE_RESOLUTION)

fig_model_fit_comparison = Figure(resolution=(600,800))

ga = fig_model_fit_comparison[1, 1] = GridLayout()
gb = fig_model_fit_comparison[1,2] = GridLayout()
gc = fig_model_fit_comparison[2, 1] = GridLayout()
gd = fig_model_fit_comparison[2,2] = GridLayout()
ge = fig_model_fit_comparison[4, 1] = GridLayout()
gf = fig_model_fit_comparison[4,2] = GridLayout()

function _meal_appearance_gamma(σ, k, t)
  t >= 0 ? (1/gamma(σ))* k^σ * t^(σ-1) * exp(-k*t) : 0.
end

normal_ra_sol = solve(original_problem, saveat=0.01, tspan=(0, 480));
glc_model = Array(normal_ra_sol)[1,:]

normal_ra = _meal_appearance_gamma.(1.4, 0.014, normal_ra_sol.t)

# no regularization
parameters = model_parameters[(0.,0.)]
objectives = model_objectives[(0.,0.)]

best_objectives = partialsortperm(objectives, 1:25)

ax_model_fit_comparison_noreg = CairoMakie.Axis(
  ga[1,1], 
  limits=(nothing, (4,8)),
  title = "No Regularization",
  xlabel = "Time [min]",
  ylabel = "Glucose [mM]")
ax_ra_comparison_noreg = CairoMakie.Axis(
  gc[1,1],
  limits=(nothing, (-0.005,0.015)),
  xlabel = "Time [min]",
  ylabel = "Rate of Appearance [/min]"
)
ax_ra_variability_noreg = CairoMakie.Axis(
  ge[1,1],
  xlabel = "Time [min]",
  ylabel = "RoA Std [/min]"
)

simulations = simulate.(parameters)

aucs_noreg = []
ra_vals_noreg = []

for obj in best_objectives

  sim = simulations[obj]
  par = parameters[obj]

  # model fit
  lines!(ax_model_fit_comparison_noreg, sim.t, Array(sim)[1,:], color = (colorschemes[:Egypt].colors[1], 0.5), label="UDE Model")
  lines!(ax_model_fit_comparison_noreg, normal_ra_sol.t, glc_model, color = (colorschemes[:Egypt].colors[2]), linestyle=:dash, label="Original Model")

  # rate-of-appearance
  lines!(ax_ra_comparison_noreg, sim.t, [U([t], par.nn, snn)[1][1] for t in sim.t], color = (colorschemes[:Egypt].colors[1], 0.5))
  lines!(ax_ra_comparison_noreg, normal_ra_sol.t, normal_ra, color = (colorschemes[:Egypt].colors[2]), linestyle=:dash, label="Original Model")


  push!(ra_vals_noreg, [U([t], par.nn, snn)[1][1] for t in sim.t])

  push!(aucs_noreg, trapz(sim.t, [U([t], par.nn, snn)[1][1] for t in sim.t]))
end

ra_vals_noreg = hcat(ra_vals_noreg...)

iqr_noreg = [quantile(ra_vals_noreg[ii,:], 0.75) - quantile(ra_vals_noreg[ii,:], 0.25) for ii in axes(ra_vals_noreg,1)]

lines!(ax_ra_variability_noreg, 0:0.1:480, std(ra_vals_noreg, dims=2)[:])
lines!(ax_ra_variability_noreg, 0:0.1:480, iqr_noreg)

scatter!(ax_model_fit_comparison_noreg, glucose_timepoints, mglc[1,:], color= :black, label="Data", markersize=12, marker='o');
errorbars!(ax_model_fit_comparison_noreg,  glucose_timepoints, mglc[1,:], stdglc[1,:], whiskerwidth = 10, color=:black, label="Data")

# regularization
parameters = model_parameters[(1.,100.)]
objectives = model_objectives[(1.,100.)]

best_objectives = partialsortperm(objectives, 1:25)

ax_model_fit_comparison_reg = CairoMakie.Axis(
  gb[1,1], 
  limits=(nothing, (4,8)),
  xlabel = "Time [min]",
  title = "Regularization",
  ylabel = "Glucose [mM]")
ax_ra_comparison_reg = CairoMakie.Axis(
  gd[1,1],
  xlabel = "Time [min]",
  ylabel = "Rate of Appearance [/min]",
  limits=(nothing, (-0.005,0.015))
)
ax_ra_variability_reg = CairoMakie.Axis(
  gf[1,1],
  xlabel = "Time [min]",
  ylabel = "RoA Std [/min]"
)

simulations = simulate.(parameters)

aucs_reg = []
ra_vals_reg = []

for obj in best_objectives
  
  sim = simulations[obj]
  par = parameters[obj]

  # model fit
  lines!(ax_model_fit_comparison_reg, sim.t, Array(sim)[1,:], color = (colorschemes[:Egypt].colors[1], 0.5), label="Model")
  lines!(ax_model_fit_comparison_reg, normal_ra_sol.t, glc_model, color = (colorschemes[:Egypt].colors[2]), linestyle=:dash, label="Assumed Model")


  # rate-of-appearance
  lines!(ax_ra_comparison_reg, sim.t, [U([t], par.nn, snn)[1][1] for t in sim.t], color = (colorschemes[:Egypt].colors[1], 0.5))
  lines!(ax_ra_comparison_reg, normal_ra_sol.t, normal_ra, color = (colorschemes[:Egypt].colors[2]), linestyle=:dash, label="Assumed Model")

  push!(ra_vals_reg, [U([t], par.nn, snn)[1][1] for t in sim.t])

  push!(aucs_reg, trapz(sim.t, [U([t], par.nn, snn)[1][1] for t in sim.t]))

end

ra_vals_reg = hcat(ra_vals_reg...)
iqr_reg = [quantile(ra_vals_reg[ii,:], 0.75) - quantile(ra_vals_reg[ii,:], 0.25) for ii in axes(ra_vals_reg,1)];

lines!(ax_ra_variability_reg, 0:0.1:480, std(ra_vals_reg, dims=2)[:], label="Stdev")
lines!(ax_ra_variability_reg, 0:0.1:480, iqr_reg, label="IQR")
axislegend(ax_ra_variability_reg, merge=true)

scatter!(ax_model_fit_comparison_reg, glucose_timepoints, mglc[1,:], color= :black, label="Data", markersize=12, marker='o');
errorbars!(ax_model_fit_comparison_reg,  glucose_timepoints, mglc[1,:], stdglc[1,:], whiskerwidth = 10, color=:black, label="Data")

fig_model_fit_comparison[3,1:2] = Legend(fig_model_fit_comparison, ax_model_fit_comparison_reg, "Legend (A-D)", merge=true, orientation=:horizontal, framevisible=false)

linkyaxes!(ax_ra_variability_noreg, ax_ra_variability_reg)


for (label, layout) in zip(["(A)", "(B)", "(C)", "(D)", "(E)", "(F)"], [ga, gb, gc, gd, ge, gf])
  Label(layout[1, 1, TopLeft()], label,
      fontsize = 15,
      font = :bold,
      padding = (0, 25, 5, 0),
      halign = :right)
end

save("figures/fig_5_minimal_model_comparison.png", fig_model_fit_comparison, px_per_unit=FIGURE_RESOLUTION)

fig_model_fit_comparison


fig_boxplot = let bxpl = boxplot(
  [repeat([0], length(aucs_noreg)); repeat([1], length(aucs_noreg))], [Float64.(aucs_noreg); Float64.(aucs_reg)],
  color = [repeat([(colorschemes[:Egypt][1],1.)], length(aucs_noreg)); repeat([(colorschemes[:Egypt][2],1.)], length(aucs_noreg))],
  figure = (resolution=(400,200),),
  axis = (limits=((-0.5,1.5), (0, 1.2)), ylabel="AUC", xticks=([0,1], ["λ=0", "λ=1"]))
  )
f,a,p = bxpl
Box(f[1,1], width=Relative(0.95), height=Relative(0.6), valign=3.6/3.7, color=(:white, 0.), linestyle=:dash)
axl = CairoMakie.Axis(f[1,1:2], limits=((0,1), (0,1)))
hidespines!(axl)
hidedecorations!(axl)
ax2 = CairoMakie.Axis(f[1,2],  limits=((-0.5,1.5), (0.5, 1.1)),yaxisposition=:right, height=Relative(0.8), width=Relative(0.8), halign=1, valign=0.98, xticks=([0,1], ["λ=0", "λ=1"]), ylabel="AUC")
boxplot!(ax2, 
  [repeat([0], length(aucs_noreg)); repeat([1], length(aucs_noreg))], [Float64.(aucs_noreg); Float64.(aucs_reg)],
  color = [repeat([(colorschemes[:Egypt][1],0.8)], length(aucs_noreg)); repeat([(colorschemes[:Egypt][2],0.8)], length(aucs_noreg))],
  figure = (resolution=(300,300),)
  )

  xs = [0.40, 0.68]
  ys = [0.38, 0.19]
  linesegments!(axl, xs, ys, color=:black, linewidth=1, linestyle=:dash)
  
  xs2 = [0.4, 0.68]
  ys2 = [0.99, 0.99]
  linesegments!(axl, xs2, ys2, color=:black, linewidth=1, linestyle=:dash)


for (label, layout) in zip(["(A)", "(B)"], [1,2])
  Label(f[1, layout, TopLeft()], label,
      fontsize = 15,
      font = :bold,
      padding = (0, 15, 5, 0),
      halign = :right)
end

f
end
save("figures/fig_6_minimal_model_auc_comparison.png", fig_boxplot, px_per_unit=FIGURE_RESOLUTION)

# Supplementary Figures

## Fig S2: Model Forecasts with more noise

rng = StableRNG(1234)

# Get the data used for training to obtain initial conditions
initial_p = [0.05, 0.2, 1.1, 0.08]
data_A, data_B, times_A, times_B, val_data_A, val_data_B = simulate_inputs(
  initial_p, rng; noise_level=0.10)

# Setup the problem
# Define the neural network component
U = Chain(
    Dense(2, 3, rbf),
    Dense(3, 3, rbf),
    Dense(3, 3, rbf),
    Dense(3, 1)
)
p_neural_init, snn = Lux.setup(rng, U)
p_neural_init = ComponentArray(p_neural_init)
# Define the hybrid model
ude_problem = michaelismenten_ude(U, initial_p, data_A, data_B, (times_A[1], times_A[end]), snn)

# Load the results
regularization_strengths = [0.0, 1e-5]
time_between_samples = 5
selected_sampling_time = 40
michment_results = [jldopen("michaelis-menten/saved_runs_2/michaelismenten_$(λ)_$(time_between_samples)_$(selected_sampling_time).jld2") for λ in regularization_strengths]

figure_model_forecasts = let f = Figure(size=(750,300))

  ga = f[1, 1] = GridLayout()
  gb = f[1,2] = GridLayout()
  gc = f[1,3] = GridLayout()

  grids = [ga, gb, gc]

  titles = ["No Regularization", "Regularization (λ = 1e-5)"]

  for (i, (res, title)) in enumerate(zip(michment_results, titles))
    training_errors = res["training_error"]
    best_models = partialsortperm(training_errors, 1:25)

    parameters = res["parameters"][best_models]
    
    model_fit_axis = CairoMakie.Axis(grids[i][1,1], limits=(nothing, (-1, 2.5)), title=title, xlabel="Time [min]", ylabel = "Concentration [mM]")
    As = []
    Bs = []
    for p in parameters
      sol = solve(ude_problem, p=p, saveat=0.1, tspan=(0., 100.))
      push!(As, sol[1,:])
      push!(Bs, sol[2,:])
    end

    lines!(model_fit_axis, 0:0.1:100, Array(val_data_A)[1:1001], color=:black, label="S True")
    lines!(model_fit_axis, 0:0.1:100, Array(val_data_B)[1:1001], color=:black, linestyle=:dash, label="P True")

    vspan!(model_fit_axis, [0], [selected_sampling_time], color = (:black, 0.2))

    # compute the interquartile range
    Am = hcat(As...)
    Bm = hcat(Bs...)
    Alow = Float64[]
    Aup = Float64[]
    Blow = Float64[]
    Bup = Float64[]
    for c in axes(Am, 1)
      upper = quantile(Am[c,:], 0.75)
      lower = quantile(Am[c,:], 0.25)
      push!(Alow, lower)
      push!(Aup, upper)
      upper = quantile(Bm[c,:], 0.75)
      lower = quantile(Bm[c,:], 0.25)
      push!(Blow, lower)
      push!(Bup, upper)
    end

    lines!(model_fit_axis, 0:0.1:100, median(Am, dims=2)[:,1], label="S ± IQR")
    band!(model_fit_axis, 0:0.1:100, Alow, Aup, color=(colorschemes[:Egypt][1], 0.2), label="S ± IQR")
    lines!(model_fit_axis, 0:0.1:100, median(Bm, dims=2)[:,1], label="P ± IQR")
    band!(model_fit_axis, 0:0.1:100, Blow, Bup, color=(colorschemes[:Egypt][2], 0.2), label="P ± IQR")
    if i == 1
      f[2,1:3] = Legend(f, model_fit_axis, "Legend", merge=true, framevisible=false, orientation=:horizontal)
    end
  end

  for (label, layout) in zip(["(A)", "(B)", "(C)"], [ga, gb, gc])
    Label(layout[1, 1, TopLeft()], label,
        fontsize = 15,
        font = :bold,
        padding = (0, 25, 5, 0),
        halign = :right)
  end
  f
end

save("figures/others/fig_s2_model_forecasts_noise.png", figure_model_forecasts, px_per_unit=FIGURE_RESOLUTION)