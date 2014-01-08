# Slice sampler (Neal 2003; MacKay 2003, sec. 29.7)
#
# A slice sampler is an adaptive step-size MCMC algorithm for continuous random
# variables, which only requires an unnormalized density function as input.  It
# is a convenient alternative to Metropolis because it often gives good results
# with very little tuning, and works in the case that different parts of the
# distribution require different step sizes.  However, if the distribution's
# widths don't vary too much, and there's a good initialization and proposal,
# Metropolis may be more efficient.
#
# This slice sampler is univariate; for a multivariate problem it just updates
# variables one at a time.  Thus it suffers when variables are correlated.
#
# REQUIRED
#   logdist: Log-density function of target distribution
#   initial: Initial state
#   niter:   Number of iterations (samples to return)
#
# OPTIONAL
#   widths: Step sizes for initially expanding the slice (D-dim vector).
#           There is little harm in making these very large.
#   burnin: Set to >0 to run for 'burnin' iterations without recording values
#   step_out: Protects against the case if you pass in widths that are too small.
#             If you are sure your widths are large enough, can set this to false.
#   verbose: Set to true for information at every iteration
#
# RETURNS a sampling history.  Pick the last one for an independent sample.
#
# EXAMPLES: take samples from N(3,1), starting from a poor initializer. Mixes to a good place!
# Get many samples:
#   slice_sample(x-> -0.5(x-3)^2, 42.0, 30)
# Get one sample:
#   slice_sample(x-> -0.5(x-3)^2, 42.0, 30)[end]
#
# There are two versions of this procedure: either 
#     univariate:   initial \in R,   logdist: R -> R,   retval \in R^niter
#     multivariate: initial \in R^D, logdist: R^D -> R, retval \in R^(niter x D)
#
# This is a port of Iain Murray's
# http://homepages.inf.ed.ac.uk/imurray2/teaching/09mlss/slice_sample.m 

# (This code is the multivariate version; univariate interfaces are further below.)
function slice_sample(logdist::Function, initial::Array{Float64,1}, niter::Integer;
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
            @printf("Slice iter %d state [%s] log_Px %f\n",iter, join(map(x->@sprintf("%.5f",x),state)," "), log_Px)
        end

        # Sweep through axes
        for dd=1:D
            log_uprime = log(rand()) + log_Px
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

function slice_sample(logdist::Function, initial::Float64, niter::Int; kwargs...)
    history = slice_sample(x-> logdist(x[1]), [initial], niter; kwargs...)
    reshape(history, (size(history,1),))
end

