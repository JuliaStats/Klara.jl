###########################################################################
#
#  Serial Tempering Monte-Carlo 
#
#  Takes MCMCTasks and performs Serial Tempering on a set of MCMCTasks
#
#
#  Ref : "Bayes Factors via Serial Tempering" C.Geyer
#
###########################################################################

export serialMC

function serialMC(tasks::Array{MCMCTask}; 
  				  steps::Integer=1, burnin::Integer=0, swapPeriod::Integer=5)
	
	local nmods = length(tasks)
	local tsize = tasks[end].model.size

	assert(burnin >= 0, "Burnin rounds ($burnin) should be >= 0")
	assert(steps > burnin, "Steps ($steps) should be > to burnin ($burnin)")
	assert(all( map(t->t.model.size, tasks) .== tsize),
		   "Models do not have the same parameter vector size")

	tic() # start timer

	# initialize all tasks
	map(t -> consume(t.task), tasks)

	# initialize MCMCChain result (one Chain only)
	res = MCMCChain({:beta => fill(NaN, tsize, steps-burnin)},
		            tasks[end], NaN,
		            {:mod => fill(0, steps-burnin) }) # model running

	local logW = zeros(nmods)  # log of task weights that will be adapted
	local at = 1  # pick starting task
	local s = consume(tasks[at].task)
	local beta, ll, beta0
	beta, ll, beta0 = s.beta, s.ll, s.oldbeta

	for i in 1:steps  # i = 1

		if i % swapPeriod == 0  # attempt a task switch
			# pick another task at random
			at2 = rand(1:(nmods-1))
			at2 = at2 >= at ? at2+1 : at2

			reset(tasks[at2], beta0)
			s2 = consume(tasks[at2].task)
			if rand() < exp(ll - s2.ll + logW[at2] - logW[at]) # accept swap ?
				at, s = at2, s2
			end
		else
			s = consume(tasks[at].task)
		end

		# TODO : add logW adaptation

		beta, ll, beta0 = s.beta, s.ll, s.oldbeta

		if i > burnin # store beta and the model we're on
			pos = i-burnin 
			res.samples[:beta][:, pos] = beta
			res.misc[:mod][pos] = at
		end
	end


	res.runTime = toq()
	res
end

