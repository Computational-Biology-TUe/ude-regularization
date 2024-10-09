"""
minimalmodel_ude(Qb, Ib, M, p_opt, I, U, snn; VG = 18.57, fG = 0.005551)

create the ude model for the minimal model of glucose kinetics.

Parameters:
  - Qb: Basal glucose
  - Ib: Basal insulin level
  - M: Total amount of glucose
  - p_opt: Model parameters (p1, p2, p3)
  - I: Insulin interpolator function
  - U: Neural network
  - snn: Neural network state

Returns:
  - minimalmodel_ude! function. Can be used as the ODE function for the UDE minimal model.
"""
function minimalmodel_ude(Qb, Ib, M, p_opt, I, U, snn; VG = 18.57, fG = 0.005551)

  # define a closure around the neural network to be used as RA function
  input(τ, p) = U([τ], p.nn, snn)[1][1]

  function minimalmodel_ude!(du, u, p, t)

    p1, p2, p3 = p_opt
    du[1] = - u[1]*u[2] - p3*(u[1]-Qb) + (fG / VG) * M * input(t, p)
    du[2] = - p1 * u[2] + p2 * (I(t) - Ib)
    

  end

  return minimalmodel_ude!
end

"""
initial_parameters(U, rng)

Initialize the neural network parameters.

Parameters:
  - U: Neural network
  - rng: Random number generator

Returns:
  - Function to initialize the neural network parameters and return them as a ComponentArray
"""
function initial_parameters(U, rng)
  function initials()
    nn_init, _ = Lux.setup(rng, U)
    ComponentArray(nn_init)
  end
end
 
"""
setup_model_training(initials, udeloss)

Setup the model training function. 

Parameters:
  - initials: Initial model parameters
  - udeloss: Loss function

Returns:
  - Function to fit the model to the data
"""
function setup_model_training(initials, udeloss)

  adtype = Optimization.AutoForwardDiff()
  optf = Optimization.OptimizationFunction((x,p)->udeloss(x), adtype)

  function fit_function(i)
    try
      optprob = Optimization.OptimizationProblem(optf, initials[i])
      res1 = Optimization.solve(optprob, ADAM(0.01), maxiters = 500)
      println("First stage successfully finished with objective value of $(res1.objective)")
      optprob2 = Optimization.OptimizationProblem(optf, res1.minimizer)
      res2 = Optimization.solve(optprob2, BFGS(initial_stepnorm=0.01), maxiters = 1000, reltol=1e-6)
      println("Optimization successfully finished with objective value of $(res2.objective)")
      return res2.u, res2.objective
    catch
      println("Optimization failed, resampling...")
      return initials[i], NaN
    end
  end

  return fit_function
end

"""
get_ude_loss(UDEproblem, save_timepoints, mglc, glucose_idx, U, snn, λ_AUC, λ_nonneg)

Get the loss function for the UDE model.

Parameters:
  - UDEproblem: ODE problem for the UDE model
  - save_timepoints: Time points to save the solution
  - mglc: Mean glucose data
  - glucose_idx: Index of glucose data
  - U: Neural network
  - snn: Neural network state
  - λ_AUC: Regularization parameter for AUC
  - λ_nonneg: Regularization parameter for non-negativity

Returns:
  - Loss function for the UDE model
"""
function get_ude_loss(UDEproblem, save_timepoints, mglc, glucose_idx, U, snn, λ_AUC, λ_nonneg)

  function udeloss(p)
    sol = Array(solve(UDEproblem, p=p, saveat=save_timepoints))
    glucose = sol[1,:]

    # PI-regularizer
    ra = [U([t], p.nn, snn)[1][1] for t in save_timepoints]
    auc_regularizer = abs(trapz(0:480, ra)-1.) * λ_AUC
    nonnegatives = sum(abs2, min.(0, ra))*λ_nonneg

    return sum(abs2, glucose[glucose_idx] .- mglc[:]) + nonnegatives + auc_regularizer
  end

  return udeloss
end