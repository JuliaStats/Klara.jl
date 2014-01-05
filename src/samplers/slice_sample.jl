# Multivariate-capable slice sampler (Neal 2003; MacKay 2003, sec. 29.7)
#
# REQUIRED
#   logdist: log-density function of target distribution
#   initial: initial state
# OPTIONAL
#   widths: step sizes for expanding the slice (D-dim vector)
#   niter: number of iterations/samples to return (default 30, which is often
#          more than enough enough so that the last sample is indep of the init)
#   burnin: set to >0 to run for 'burnin' iterations without recording values
#   step_out: set to false if you're having infinite loop trouble 
#             (but that may indicate a bug in your density function)
#   verbose: set to true for info at every iteration
#
# RETURNS a sampling history.  Pick its last one for an independent sample.
#
# EXAMPLES: take samples from N(3,1), starting from a crappy initializer. Mixes to a good place!
# Get one sample:
#   slice_sample(x-> -0.5(x-3)^2, 42.0)[end]
# Get many samples, discarding first 10 as burnin:
#   slice_sample(x-> -0.5(x-3)^2, 42.0; niter=100)[11:end]
#
# Can use the 'burnin' parameter to be slightly more efficient. Equivalent versions of the above:
#   slice_sample(x-> -0.5(x-3)^2, 42.0; burnin=29, niter=1)[end]
#   slice_sample(x-> -0.5(x-3)^2, 42.0; niter=90, burnin=10)
# (TODO: burnin>0,niter=1 could be special-cased by storing no history at all, for another speedup.)
#
# There are two versions of this procedure: either 
#     univariate:   initial \in R,   logdist: R -> R,   retval \in R^niter
#     multivariate: initial \in R^D, logdist: R^D -> R, retval \in R^(niter x D)
#
# WHERE THIS IS FROM: This is a port of Iain Murray's
# http://homepages.inf.ed.ac.uk/imurray2/teaching/09mlss/slice_sample.m 
# which in turn derives from MacKay.  Murray notes where he found bugs in
# MacKay's pseudocode... good sign for correctness

# (This code is the multivariate version; univariate interfaces are further below.)
function slice_sample(logdist::Function, initial::Array{Float64,1}; 
        niter::Integer = 30,
        widths::Array{Float64,1} = ones(length(initial)),
        step_out = true,
        burnin = 0,
        verbose = false)
    D = length(initial)
    @assert length(widths)==D

    state::Vector{Float64} = copy(initial)
    log_Px::Float64 = logdist(state)

    history = zeros(niter,D)

    for iter=1:(niter+burnin)
        if verbose
            @printf("Slice iter %d state %s log_Px %f\n",iter, join(map(x->@sprintf("%.5f",x),state)," "), log_Px)
        end
        log_uprime = log(rand()) + log_Px

        # Sweep through axes
        for dd=1:D
            x_l  = copy(state)
            x_r  = copy(state)
            xprime = copy(state)
            # Create a horizontal interval (x_l, x_r) enclosing xx
            r = rand()
            x_l[dd] = state[dd] - r*widths[dd]
            x_r[dd] = state[dd] + (1-r)*widths[dd]
            if step_out
                while logdist(x_l) > log_uprime
                    x_l[dd] -= widths[dd]
                end
                while logdist(x_r) > log_uprime
                    x_r[dd] += widths[dd]
                end
            end
            # Inner loop:
            # Propose xprimes and shrink interval until good one is found.
            while true
                xprime[dd] = rand() * (x_r[dd] - x_l[dd]) + x_l[dd];
                log_Px = logdist(xprime)
                if log_Px > log_uprime
                    break
                else
                    if (xprime[dd] > state[dd])
                        x_r[dd] = xprime[dd];
                    elseif (xprime[dd] < state[dd])
                        x_l[dd] = xprime[dd];
                    else
                        @assert false "BUG, shrunk to current position and still not acceptable";
                    end
                end
            end
            state[dd] = xprime[dd]
        end
        if iter > burnin
            history[iter-burnin, :] = state
        end
    end
    history
end



## Univariate formulations

function slice_sample(logdist::Function, initial::Float64; kwargs...)
    history = slice_sample(x-> logdist(x[1]), [initial]; kwargs...)
    reshape(history, (size(history,1),))
end

function slice_sample(logdist::Function, initial::Float64, width::Float64; kwargs...)
    history = slice_sample(x-> logdist(x[1]), [initial]; widths=[width], kwargs...)
    reshape(history, (size(history,1),))
end

