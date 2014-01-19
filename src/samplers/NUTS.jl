###########################################################################
#
#  NUTS variant Hamiltonian Monte Carlo
#
#  Reference: Hoffman M.D, Gelman A.The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo.
#  arXiv, 2011
#
#  Parameters :
#    - maxdoublings : max number of doublings in inner steps
#    - tuner: for tuning the NUTS parameters
#
###########################################################################

export NUTS

println("Loading NUTS(maxdoublings, tuner) sampler")


###########################################################################
#                  NUTS specific 'tuners'
###########################################################################
abstract NUTSTuner <: MCMCTuner

# TODO : describe here the leapSteps adaptation


###########################################################################
#                  NUTS type
###########################################################################
immutable NUTS <: MCMCSampler
  maxdoublings::Int
  tuner::Union(Nothing, NUTSTuner)

  function NUTS(i::Int, t::Union(Nothing, NUTSTuner))
    @assert i>0 "max doublings should be > 0"
    @assert i<20 "max doublings reasonably be < 20"
    new(i,t)
  end
end
NUTS(i::Int=5) = NUTS(i , nothing)
NUTS(t::NUTSTuner) = NUTS(5, t)
# keyword args version
NUTS(;maxdoublings::Int=5, tuner::Union(Nothing, NUTSTuner)=nothing) = NUTS(maxdoublings, tuner)

###########################################################################
#                  HMC task
###########################################################################

####  Helper functions and types for HMC sampling task
uturn(s1::HMCSample, s2::HMCSample) = dot(s2.pars-s1.pars, s1.m) < 0. || dot(s2.pars-s1.pars, s2.m) < 0.

####  HMC task
function SamplerTask(model::MCMCModel, sampler::NUTS, runner::MCMCRunner)
    local epsilon, u_slice
    local state0  # starting state of each loop
    local scale

 	@assert hasgradient(model) "NUTS sampler requires model with gradient function"

	# hook inside Task to allow remote resetting
	task_local_storage(:reset,
	             (resetPars::Vector{Float64}) -> (state0 = HMCSample(copy(resetPars)); 
	                                              calc!(state0, model.evalallg)) ) 
	
	# initialization
	state0 = HMCSample(copy(model.init))
	calc!(state0, model.evalallg)
  @assert isfinite(state0.logTarget) "Initial values out of model support, try other values"
	scale = model.scale 

	# find initial value for epsilon
	epsilon = 1.
	state0.m = randn(model.size) .* scale
	state1 = leapfrog(state0, epsilon, model.evalallg)

	ratio = exp(state0.H-state1.H)
	a = 2*(ratio>0.5)-1.
	while ratio^a > 2^-a
		epsilon *= 2^a
		state1 = leapfrog(state0, epsilon, model.evalallg)
		ratio = exp(state0.H-state1.H)
	end

	# buidtree function
	function buildTree(state, dir, j, ll)  # TODO : pass espilon, u_slice, state0 ?
		local state1, n1, s1, alpha1, nalpha1
		local state2, n2, s2, alpha2, nalpha2
		local state_plus, state_minus
		local dummy
		const deltamax = 100

		if j == 0
			state1 = leapfrog(state, dir*epsilon, ll)
			n1 = ( u_slice <= -state1.H ) + 0 
			s1 = u_slice < ( deltamax-state1.H )

			return state1, state1, state1, n1, s1, min(1., exp(state0.H-state1.H)), 1
		else
			state_minus, state_plus, state1, n1, s1, alpha1, nalpha1 = buildTree(state, dir, j-1, ll)
			if s1 
				if dir == -1
					state_minus, dummy, state2, n2, s2, alpha2, nalpha2 = buildTree(state_minus, dir, j-1, ll)
	 			else
	 				dummy, state_plus, state2, n2, s2, alpha2, nalpha2 = buildTree(state_plus, dir, j-1, ll)
	 			end
	 			if rand() <= n2/(n2+n1)
	 				state1 = deepcopy(state2)
	 			end

	 			alpha1 += alpha2
	 			nalpha1 += nalpha2
	 			s1 = s2 && !uturn(state_minus, state_plus)
	 			n1 += n2
	 		end

	 		return state_minus, state_plus, state1, n1, s1, alpha1, nalpha1
		end
	end

	### adaptation parameters  # TODO : should be part of the tuner
	const delta = 0.7  # target acceptance
	const nadapt = 1000  # nb of steps to adapt epsilon
	const gam = 0.05
	const kappa = 0.75
	const t0 = 10
	### adaptation inital values
	hbar = 0.
	mu = log(10*epsilon)
	lebar = 0.0

	### main loop
	# using a 'for' loop instead of 'while true' for this sampler
	#  because leapStep adaptation requires a sampling counter 
  i = 1
  while true  
 		local alpha, nalpha, n, s, j, n1, s1
 		local dummy, state_minus, state_plus, state, state1

 		state0.m = randn(model.size) .* scale
 		update!(state0)

 		u_slice  = log(rand())-state0.H # use log ( != paper) to avoid underflow
 		
 		state = deepcopy(state0)
 		state_minus = deepcopy(state0)
 		state_plus = deepcopy(state0)

 		# inner loop
 		j, n = 0, 1
 		s = true
 		while s && j < sampler.maxdoublings
 			dir = randbool() ? 1 : -1
 			if dir == -1
 				state_minus, dummy, state1, n1, s1, alpha, nalpha = buildTree(state_minus, dir, j, model.evalallg)
 			else
 				dummy, state_plus, state1, n1, s1, alpha, nalpha = buildTree(state_plus, dir, j, model.evalallg)
 			end
 			if s1 && rand() < n1/n  # accept 
 				state = deepcopy(state1)
 			end
 			n += n1
 			j += 1
 			s = s1 && !uturn(state_minus, state_plus)
 		end
 		
 		# epsilon adjustment
 		if i <= nadapt  # warming up period
 			hbar = hbar * (1-1/(i+t0)) + (delta-alpha/nalpha)/(i+t0)
			le = mu-sqrt(i)/gam*hbar
			lebar = i^(-kappa) * le + (1-i^(-kappa)) * lebar
			epsilon = exp(le)
		else # post warm up, keep dual epsilon
			epsilon = exp(lebar)
		end

		ms = MCMCSample(state.pars, state.logTarget, state.grad, 
		                state0.pars, state0.logTarget, state0.grad,
		                {"epsilon" => epsilon, "ndoublings" => j} )
		produce(ms)

		state0 = deepcopy(state)

    i += 1
	end

end
