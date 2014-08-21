###########################################################################
#  Slice sampler (SliceSampler)
#
#  References:
#    - Neal R.M. Slice Sampling. The Annals of Statistics, 2003, 31 (3), pp 705–1031
#    - MacKay model.size. Information Theory, Inference, and Learning Algorithms. Cambridge University Press, 2003, section 29.7,
#      pp 374-378
#
#  A slice sampler is an adaptive step-size MCMC algorithm for continuous random variables, which only requires an
#  unnormalized density function as input.  It is a convenient alternative to Metropolis because it often gives good
#  results with very little tuning, and works in the case that different parts of the distribution require different
#  step sizes.  However, if the distribution's widths don't vary too much, and there's a good initialization and
#  proposal, Metropolis may be more efficient. This slice sampler is univariate; for a multivariate problem it just
#  updates variables one at a time.  Thus it suffers when variables are correlated.
#
#  Parameters :
#    - widths: Step sizes for initially expanding the slice (model.size-dim vector).
#    - stepout: Protects against the case of passing in small widths.
#
#  This is a port of Iain Murray's implementation, who kindly accepted to share his code at
#  http://homepages.inf.ed.ac.uk/imurray2/teaching/09mlss/slice_sampler.m
#
###########################################################################

export SliceSampler

###########################################################################
#                  SliceSampler type
###########################################################################

immutable SliceSampler <: MCMCSampler
  widths::Union(Vector{Float64}, Nothing)
  stepout::Bool

  function SliceSampler(widths::Union(Vector{Float64}, Nothing), stepout::Bool)
    # @assert x > 0 "widths should be > 0"
    new(widths, stepout)
  end
end
SliceSampler() = SliceSampler(nothing, true)
SliceSampler(widths::Union(Vector{Float64}, Nothing)) = SliceSampler(widths, true)
SliceSampler(stepout::Bool) = SliceSampler(nothing, stepout)
SliceSampler(; widths::Union(Vector{Float64}, Nothing)=nothing, stepout::Bool=true) = SliceSampler(widths, stepout)

###########################################################################
#                  SliceSampler task
###########################################################################

# SliceSampler sampling
function SamplerTask(model::MCMCModel, sampler::SliceSampler, runner::MCMCRunner)
  local state::Vector{Float64}, x_l::Vector{Float64}, x_r::Vector{Float64}, xprime::Vector{Float64}
  local logTarget::Float64, log_uprime::Float64, r::Float64
  local widths::Vector{Float64}
  if sampler.widths == nothing
    widths = ones(model.size)
  else
    @assert length(sampler.widths)==model.size "Length of step sizes in widths must be equal to model size"
    widths = sampler.widths
  end
  
  state = copy(model.init)
  logTarget = model.eval(state)

  i = 1
  while true
    for dd = 1:model.size
      log_uprime = log(rand()) + logTarget
      x_l  = copy(state)
      x_r  = copy(state)
      xprime = copy(state)

      # Create a horizontal interval (x_l, x_r) enclosing xx
      r = rand()
      x_l[dd] = state[dd] - r*widths[dd]
      x_r[dd] = state[dd] + (1-r)*widths[dd]
      if sampler.stepout
        while model.eval(x_l) > log_uprime
          x_l[dd] -= widths[dd]
        end
        while model.eval(x_r) > log_uprime
          x_r[dd] += widths[dd]
        end
      end

      # Inner loop: propose xprimes and shrink interval until good one is found
      while true
        xprime[dd] = rand() * (x_r[dd] - x_l[dd]) + x_l[dd];
        logTarget = model.eval(xprime)
        if logTarget > log_uprime
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

    ms = MCMCSample(state, logTarget, fill(NaN, model.size), NaN)
    produce(ms)

    i += 1
  end
end
