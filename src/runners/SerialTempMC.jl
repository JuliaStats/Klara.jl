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

export SerialTempMC

println("Loading SerialTempMC(steps, burnin, swapPeriod) runner")

immutable SerialTempMC <: MCMCRunner
  steps::Int
  burnin::Int
  swapPeriod::Int

  function SerialTempMC(steps::Int, burnin::Int, swapPeriod::Int)
	  assert(burnin >= 0, "Burnin rounds ($burnin) should be >= 0")
	  assert(steps > burnin, "Steps ($steps) should be > to burnin ($burnin)")

    new(steps, burnin, swapPeriod)
  end
end

SerialTempMC(; steps::Int=1, burnin::Int=0, swapPeriod::Int=5) = SerialTempMC(steps, burnin, swapPeriod)

function run_serialtempmc(tasks::Array{MCMCTask})
	
	local nmods = length(tasks)
	local tsize = tasks[end].model.size
  local steps = tasks[end].runner.steps
  local burnin = tasks[end].runner.burnin
  local swapPeriod = tasks[end].runner.swapPeriod

	assert(all( map(t->t.model.size, tasks) .== tsize),
		   "Models do not have the same parameter vector size")

	tic() # start timer

	# initialize all tasks
	map(t -> consume(t.task), tasks)

	# initialize MCMCChain result (one Chain only)
  res = MCMCChain(1:1:2, DataFrame(tsize, steps-burnin), tasks[end])

	local logW = zeros(nmods)  # log of task weights that will be adapted
	local at = 1  # pick starting task
	local s = consume(tasks[at].task)
	local ppars, logtarget, pars
	ppars, logtarget, pars = s.ppars, s.logtarget, s.pars

	for i in 1:steps  # i = 1

		if i % swapPeriod == 0  # attempt a task switch
			# pick another task at random
			at2 = rand(1:(nmods-1))
			at2 = at2 >= at ? at2+1 : at2

			reset(tasks[at2], pars)
			s2 = consume(tasks[at2].task)
			if rand() < exp(logtarget - s2.logtarget + logW[at2] - logW[at]) # accept swap ?
				at, s = at2, s2
			end
		else
			s = consume(tasks[at].task)
		end

		# TODO : add logW adaptation

		ppars, logtarget, pars = s.ppars, s.logtarget, s.pars

		if i > burnin # store ppars and the model we're on
			pos = i-burnin 
			res.samples[:, pos] = ppars
			#res.misc[:mod][pos] = at
		end
	end


	res.runTime = toq()
	res
end

# TODO: check that all elements of array contain MCMCTasks of the same type
function resume_serialtempmc(tasks::Array{MCMCTask}; steps::Int=100)
	run(MCMCTask[tasks[i].model * tasks[i].sampler * SerialTempMC(steps=steps, swapPeriod=tasks[i].runner.swapPeriod)
		for i in 1:length(tasks)])
end
