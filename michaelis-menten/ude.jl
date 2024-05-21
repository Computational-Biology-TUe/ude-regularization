function michaelismenten_ude(net, initial_p, data_A, data_B, tspan, snn)
  # Define the hybrid model
  function michaelismenten_ude!(du, u, p, t, p_true)
      Û = net(u, p.ude, snn)[1]
      A, B = u
      du[1] = p_true[1].*A .- Û[1]
      du[2] = Û[1] .- p_true[4].*B;
      nothing
  end

  michaelismenten_ude!(du, u, p, t) = michaelismenten_ude!(du, u, p, t, initial_p)

  return ODEProblem{true, SciMLBase.FullSpecialize}(michaelismenten_ude!,[data_A[1], data_B[1]], tspan)
end

# Setup the initial value generator
function initial_parameters(net, num, rng)
  initials = []
  for _ in 1:num
      nn_init,_ = Lux.setup(rng, net)
      push!(initials, ComponentVector{Float64}(ude = nn_init))
  end
  initials
end

function setup_model_training(loss, validation, λ)
    function fit_model(initial_parameters)

       try
            adtype = Optimization.AutoZygote()
            optf = Optimization.OptimizationFunction(loss, adtype)
            optprob = Optimization.OptimizationProblem(optf, initial_parameters, λ)
            res1 = Optimization.solve(optprob, ADAM(0.01), maxiters = 500)

            # Train with BFGS
            optprob2 = Optimization.OptimizationProblem(optf, res1.minimizer, λ)
            res2 = Optimization.solve(optprob2, Optim.BFGS(initial_stepnorm=0.01), x_tol=1e-6, f_tol=1e-6, maxiters = 1_000)
            
            return res2.minimizer, res2.objective, validation(res2.minimizer, 0.)
       catch
            print("Optimization Failed... Resampling...")
            return initial_parameters, NaN, NaN
       end
    end
end

function michaelismenten_loss(m, y, ts)

  # Define a loss function 
  times = sort(unique([ts[1]; ts[2]]))
  idxs = [
      Vector{Int}(indexin(ts[1], times)),
      Vector{Int}(indexin(ts[2], times))
  ]

  function predict(p̂, t=times)
      _prob = remake(m, tspan = (t[1], t[end]), p = p̂)
      Array(solve(_prob, Vern7(), saveat = t,
                  abstol=1e-6, reltol=1e-6,
                  sensealg = ForwardDiffSensitivity()
                  ))
  end

  function err(p̂, λ)
      ŷ = predict(p̂)
      l = 0.
      for i in axes(ŷ, 1)
          pred = ŷ[i, idxs[i]]
          data = y[i][1:length(pred)]
          l += sum(abs2, data.-pred)
      end

      ŷ_reg = predict(p̂, 0:500)
      l += λ .* sum(abs2, min.(0, ŷ_reg))

      return l
  end
  
  return err
end
  