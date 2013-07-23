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

export seqMC

function seqMC(targets::Array{MCMCTask}, 
				particles::Vector{Vector{Float64}}; 
				steps::Integer=1, burnin::Integer=0, resTrigger::Float64=1e-10)
	
	local tsize = targets[end].model.size
	local ntargets = length(targets)
	local npart = length(particles)

	assert(burnin >= 0, "Burnin rounds ($burnin) should be >= 0")
	assert(steps > burnin, "Steps ($steps) should be > to burnin ($burnin)")
	assert(all( map(t->t.model.size, targets) .== tsize),
		   "Models do not have the same parameter vector size")

	tic() # start timer

	# initialize all tasks
	map(t -> consume(t.task), targets)

	# initialize MCMCChain result
	res = MCMCChain({:beta => fill(NaN, tsize, (steps-burnin)*npart)},
					fill(NaN, (steps-burnin)*npart), # weight of samples
		            targets[end], NaN)

	local logW = zeros(npart)  # log of particle weights
	local oldll = zeros(npart) # loglik of previous target distrib
	local beta = deepcopy(particles)

	for i in 1:steps  # i = 1
		for t in targets  # t = targets[1]
			# mutate each particle with task t
			for n in 1:npart  #  n = 1
				MCMC.reset(t, beta[n])  # force beta of task #t to particle #n
				sample = consume(t.task)
				beta[n], ll, ll0 = sample.beta, sample.ll, sample.oldll
				logW[n] += ll0 - oldll[n]
				oldll[n] = ll
			end

			# print("1- beta [ $(min(beta)), $(max(beta)) ], ")
			# print("oldll [ $(round(min(oldll),2)), $(round(max(oldll),2)) ], ")
			# println("logW [ $(round(min(logW),2)), $(round(max(logW),2)) ]")

			# resample if likelihood variance of particles is too low
			#  TODO : improve, clarify
			local W = exp(logW)
			if var(W) < resTrigger
				cp = cumsum(W) / sum(W)
				rs = fill(0, npart)
				for n in 1:npart  #  n = 1
					l = rand()
					rs[n] = findfirst(p-> (p>=l), cp)
				end
				beta = beta[rs]
				logW = zeros(npart)
				oldll = oldll[rs]   #zeros(npart)
				# println("resampled !")
				# print("2- beta [ $(min(beta)), $(max(beta)) ], ")
				# print("oldll [ $(round(min(oldll),2)), $(round(max(oldll),2)) ], ")
				# println("logW [ $(round(min(logW),2)), $(round(max(logW),2)) ]")
			end
		end

		println("iter $i, var $(var(exp(logW)))")
		oldll = zeros(npart)

		if i > burnin # store betas of all particles in the result chain
			pos = (i-burnin-1) * npart
			for n in 1:npart  #  n = 1
				res.samples[:beta][:, pos+n] = beta[n]
				res.weights[pos+n] = exp(logW[n])
			end
		end
	end

	res.runTime = toq()
	res
end

