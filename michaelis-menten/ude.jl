function michaelismenten_ude(net, initial_p, data_A, data_B, tspan, snn)
  # Define the hybrid model
  function michaelismenten_ude!(du, u, p, t, p_true)
      Û = net(u, p.ude, snn)[1]
      A, B = u
      du[1] = p_true[1]*A - Û[1]
      du[2] = Û[1] - p_true[4]*B;
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

function setup_model_training(m, y, ts, λ, y_val, ts_val)
    adtype = Optimization.AutoForwardDiff()
    optf = Optimization.OptimizationFunction(michaelismenten_loss, adtype)

    function fit_model(initial_parameters)
       try
            optprob = Optimization.OptimizationProblem(optf, initial_parameters, (m,y,ts,λ))
            res1 = Optimization.solve(optprob, ADAM(0.01), maxiters = 500)
            println("First Stage successfully finished with objective value of $(res1.objective)")
            # Train with BFGS
            optprob2 = Optimization.OptimizationProblem(optf, res1.u, (m,y,ts,λ))
            res2 = Optimization.solve(optprob2, Optim.BFGS(
                linesearch=LineSearches.BackTracking(order=3), 
                initial_stepnorm=0.01), x_tol=1e-6, f_tol=1e-6, maxiters = 1_000)

            println("Optimization successfully finished with objective value of $(res2.objective)")
            validation_loss = michaelismenten_validation(res2.u, (m,y_val,ts_val))
            return res2.u, res2.objective, validation_loss
       catch
            print("Optimization Failed... Resampling...")
            return initial_parameters, NaN, NaN
       end
    end

    return fit_model
end

function predict(m, p̂, t)
    _prob = remake(m, tspan = (t[1], t[end]), p = p̂)
    Array(solve(_prob, Tsit5(), saveat = t,
                abstol=1e-6, reltol=1e-6,
                sensealg = ForwardDiffSensitivity()
                ))
end

function michaelismenten_validation(p̂, args)
    m, y, ts = args
    ŷ = predict(m, p̂, ts)
    l = 0.
    for i in axes(ŷ, 1)
        pred = ŷ[i, :]
        data = y[i][1:length(pred)]
        l += sum(abs2, data-pred)
    end

    return l
end

function michaelismenten_loss(p̂, args)
    m, y, ts, λ = args
    ŷ = predict(m, p̂, ts)
    l = 0.
    for i in axes(ŷ, 1)
        pred = ŷ[i, :]
        data = y[i][1:length(pred)]
        l += sum(abs2, data-pred)
    end

    ŷ_reg = predict(m, p̂, 0:200)
    l += λ .* sum(abs2, min.(0, ŷ_reg))

    return l
end
  