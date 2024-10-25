"""
michaelismenten_ude(net, initial_p, data_A, data_B, tspan, snn)

Create a UDE model for the Michaelis-Menten kinetics model according to the following ODEs:
    dA/dt = k1*A - Û
    dB/dt = Û - k3*B
where Û is the output of the neural network.

Arguments:
    - net: Neural network model
    - initial_p: Initial parameters for the model
    - data_A: Noisy data for species A
    - data_B: Noisy data for species B
    - tspan: Time span for the ODE
    - snn: Neural network state (given by Lux.setup)

Returns:
    - ODEProblem for the UDE model that can be simulated with DifferentialEquations.jl
"""
function michaelismenten_ude(net, initial_p, data_A, data_B, tspan, snn)
  
    # Function that defines the differential equations
  function michaelismenten_ude!(du, u, p, t, p_true)

    # Calculate the output of the neural network
      Û = net(u, p.ude, snn)[1]
    # Set the state variables (Substrate and Product)
      S, P = u
    # Compute the derivatives
      du[1] = p_true[1]*S - Û[1]
      du[2] = Û[1] - p_true[4]*P;
    # Return nothing, as the derivatives are updated in-place
      nothing
  end

  # Define a closure to pass the initial parameters to the ODEProblem
  michaelismenten_ude!(du, u, p, t) = michaelismenten_ude!(du, u, p, t, initial_p)

# Return the ODEProblem
  return ODEProblem{true, SciMLBase.FullSpecialize}(michaelismenten_ude!,[data_A[1], data_B[1]], tspan)
end

"""
initial_parameters(net, num, rng)

Generate initial parameters for the UDE model.

Arguments:
    - net: Neural network model
    - num: Number of initial parameters to generate
    - rng: Random number generator

Returns:
    - Array of ComponentVector{Float64} with the initial parameters
"""
function initial_parameters(net, num, rng)

# Generate the initial parameters
  initials = []
  for _ in 1:num
      nn_init,_ = Lux.setup(rng, net)
      push!(initials, ComponentVector{Float64}(ude = nn_init))
  end
  initials
end

"""
setup_model_training(m, y, ts, λ, y_val, ts_val)

Setup the model training function for the UDE model.

Arguments:
    - m: UDE model
    - y: Data for the model
    - ts: Time points for the data
    - λ: Regularization parameter
    - y_val: Validation data
    - ts_val: Time points for the validation data

Returns:
    - Function that trains the UDE model given initial parameters
"""
function setup_model_training(m, y, ts, λ, y_val, ts_val)
    # Define the automatic differentiation type; we use forward mode automatic differentiation
    adtype = Optimization.AutoForwardDiff()

    # Define the optimization function
    optf = Optimization.OptimizationFunction(michaelismenten_loss, adtype)

    # Define the function to fit the model
    function fit_model(initial_parameters)
       try
            # Train with ADAM
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
        # Prevent the optimization from quitting for all initial values if one fails
            print("Optimization Failed... Resampling...")
            return initial_parameters, NaN, NaN
       end
    end

    return fit_model
end

"""
predict(m, p̂, t)

Predict the output of the UDE model given the parameters p̂ and time points t.

Arguments:
    - m: UDE model
    - p̂: Parameters for the model
    - t: Time points for the prediction

Returns:
    - Array with the predicted values
"""
function predict(m, p̂, t)
    # Set the parameters for the model and the time span
    _prob = remake(m, tspan = (t[1], t[end]), p = p̂)
    # Solve the ODE
    sol = solve(_prob, Tsit5(), saveat = t,
                abstol=1e-6, reltol=1e-6,
                sensealg = ForwardDiffSensitivity()
                )
    # Check if the ODE solve was successful
    if ~SciMLBase.successful_retcode(sol.retcode)
        throw(ErrorException("ODE Solve failed"))
    end
    # Return the solution
    return Array(sol)
end

"""
michaelismenten_validation(p̂, args)

Calculate the validation loss for the UDE model.

Arguments:
    - p̂: Parameters for the model
    - args: Tuple with the model, data and time points

Returns:
    - Validation loss
"""
function michaelismenten_validation(p̂, args)
    m, y, ts = args
    ŷ = predict(m, p̂, ts)
    l = 0.
    for i in axes(ŷ, 1)
        pred = ŷ[i, :]
        data = y[i][1:length(pred)]
        # Squared error loss
        l += sum(abs2, data-pred)
    end

    return l
end

"""
michaelismenten_loss(p̂, args)

Calculate the loss for the UDE model.

Arguments:
    - p̂: Parameters for the model
    - args: Tuple with the model, data, time points and regularization parameter

Returns:
    - Loss
"""
function michaelismenten_loss(p̂, args)
    m, y, ts, λ = args
    ŷ = predict(m, p̂, ts)
    l = 0.
    for i in axes(ŷ, 1)
        pred = ŷ[i, :]
        data = y[i][1:length(pred)]
        # Squared error loss
        l += sum(abs2, data-pred)
    end

    # Regularization term
    ŷ_reg = predict(m, p̂, 0:200)
    l += λ .* sum(abs2, min.(0, ŷ_reg))

    return l
end

"""
michaelismenten_inhibition_loss(p̂, args)

Calculate the loss for the inhibition UDE model.

Arguments:
    - p̂: Parameters for the model
    - args: Tuple with the model, data, time points and regularization parameter

Returns:
    - Loss
"""
function michaelismenten_inhibition_loss(p̂, args)
    m, y, ts, λ = args
    ŷ = predict(m, p̂, ts)
    l = 0.
    for i in axes(ŷ, 1)
        pred = ŷ[i, :]
        data = y[i][1:length(pred)]
        # Squared error loss
        l += sum(abs2, data-pred)
    end

    # Regularization term
    ŷ_reg = predict(m, p̂, 0:800)
    l += λ .* sum(abs2, min.(0, ŷ_reg))

    return l
end


"""
setup_inhibition_model_training(m, y, ts, λ, y_val, ts_val)

Setup the model training function for the inhibition UDE model.

Arguments:
    - m: UDE model
    - y: Data for the model
    - ts: Time points for the data
    - λ: Regularization parameter
    - y_val: Validation data
    - ts_val: Time points for the validation data

Returns:
    - Function that trains the UDE model given initial parameters
"""
function setup_inhibition_model_training(m, y, ts, λ, y_val, ts_val)
    # Define the automatic differentiation type; we use forward mode automatic differentiation
    adtype = Optimization.AutoForwardDiff()

    # Define the optimization function
    optf = Optimization.OptimizationFunction(michaelismenten_inhibition_loss, adtype)

    # Define the function to fit the model
    function fit_model(initial_parameters)
       try
            # Train with ADAM
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
        # Prevent the optimization from quitting for all initial values if one fails
            print("Optimization Failed... Resampling...")
            return initial_parameters, NaN, NaN
       end
    end

    return fit_model
end