###########################################################################
#
#  Sequential Monte-Carlo 
#
#  Takes MCMCTasks and performs Sequential Monte-Carlo sampling
#   ( a kind of population MC )
#
#  The last of the MCMCTasks should be the target distribution of interest,
#   others can either be the same target or tempered versions (see ref).
#
#  Particles are an arbitrary number n of initial values on which the tasks
#   will be run in succession, the number of tasks (target distributions) is not
#   related to the number of particles.
#
#  Ref : "On population-based simulation for static inference" A.Jasra - 
#          D.Stephens - C.Holmes, p 271
#
###########################################################################

export SeqMC

println("Loading SeqMC(steps, burnin, trigger) runner")

immutable SeqMC <: MCMCRunner
  steps::Int
  burnin::Int
  trigger::Float64

  function SeqMC(steps::Int, burnin::Int, trigger::Float64)
	  @assert burnin >= 0 "Burnin rounds ($burnin) should be >= 0"
	  @assert steps > burnin "Steps ($steps) should be > to burnin ($burnin)"

    new(steps, burnin, trigger)
  end
end

SeqMC(; steps::Int=1, burnin::Int=0, trigger::Float64=1e-10) = SeqMC(steps, burnin, trigger)

function run_seqmc(targets::Array{MCMCTask}; particles::Vector{Vector{Float64}} = [[randn()] for i in 1:100])
	local ntargets = length(targets)
	local npart = length(particles)
	local tsize = targets[end].model.size
  local steps = targets[end].runner.steps
  local burnin = targets[end].runner.burnin
  local trigger = targets[end].runner.trigger

  @assert all(map(t->t.model.size, targets) .== tsize) "Models do not have the same parameter vector size"

	tic() # start timer

	# initialize all tasks
	map(t -> consume(t.task), targets)

	# initialize samples and weights arrays
	samples = fill(NaN, tsize, (steps-burnin)*npart) 
	weights = fill(NaN, (steps-burnin)*npart)

	local pars = deepcopy(particles)
	local logW = zeros(npart)  # log of particle weights
	local logtarget = zeros(npart) # loglik of previous target distrib

	for i in 1:steps  # i = 1

		for t in targets  # t = targets[1]
			# mutate each particle with task t
			for n in 1:npart  #  n = 1
				MCMC.reset(t, pars[n])  # force pars of task #t to particle #n
				sample = consume(t.task)
				pars[n], plogtarget, ll0 = sample.ppars, sample.plogtarget, sample.logtarget
				logW[n] += ll0 - logtarget[n]
				logtarget[n] = plogtarget
			end

			# resample if likelihood variance of particles is too low
			#  TODO : improve, make user-settable, ..
			local W = exp(logW)
			if var(W) < trigger
				cp = cumsum(W) / sum(W)
				rs = fill(0, npart)
				for n in 1:npart  #  n = 1
					l = rand()
					rs[n] = findfirst(p-> (p>=l), cp)
				end
				pars = pars[rs]
				logW = zeros(npart)
				logtarget = logtarget[rs]  
				# println("resampled !")
			end
		end

		println("iter $i, var $(var(exp(logW)))")
		logtarget = zeros(npart)

		if i > burnin # store betas of all particles in the result chain
			pos = (i-burnin-1) * npart
			for n in 1:npart  #  n = 1
				samples[:, pos+n] = pars[n]
				weights[pos+n] = exp(logW[n])
			end
		end
	end

	# generate column names for the samples DataFrame
	cn = ASCIIString[]
	for (k,v) in targets[end].model.pmap
	  if length(v.dims) == 0 # scalar
	    push!(cn, string(k))
	  elseif length(v.dims) == 1 # vector
	    cn = vcat(cn, ASCIIString[ "$k.$i" for i in 1:v.dims[1] ])
	  elseif length(v.dims) == 2 # matrix
	    cn = vcat(cn, ASCIIString[ "$k.$i.$j" for i in 1:v.dims[1], j in 1:v.dims[2] ])
	  end
	end

	# create Chain
	MCMCChain((burnin+1):1:((steps-burnin)*npart),
	  DataFrame(samples', cn),
		DataFrame(),  # TODO, store gradient here, needs to be passed by newprop
		{"weigths" => weights, "particle" => rep([1:npart],(steps-burnin))},  
		targets,
		toq())
end

# TODO: check that all elements of array contain MCMCTasks of the same type
function resume_seqmc(targets::Array{MCMCTask}; steps::Int=100)
	run(MCMCTask[targets[i].model * targets[i].sampler * SeqMC(steps=steps, trigger=targets[i].runner.trigger)
		for i in 1:length(targets)])
end
