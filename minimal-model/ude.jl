function minimalmodel_ude(Qb, Ib, M, p_opt, I, U, snn; VG = 18.57, fG = 0.005551)

  input(τ, p) = U([τ], p.nn, snn)[1][1]

  function minimalmodel_ude!(du, u, p, t)

    p1, p2, p3 = p_opt
    du[1] = - u[1]*u[2] - p3*(u[1]-Qb) + (fG / VG) * M * input(t, p)
    du[2] = - p1 * u[2] + p2 * (I(t) - Ib)
    

  end

  return minimalmodel_ude!
end

function initial_parameters(U, rng)
  function initials()
    nn_init, _ = Lux.setup(rng, U)
    ComponentArray(nn_init)
  end
end
  
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