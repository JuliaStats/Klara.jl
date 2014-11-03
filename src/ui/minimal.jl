### An alternative interface is via the function wrappers provided below
### This third interface fulfills the following purposes:
### i) It acts as a simplified interface to facilitate interaction with other packages
### ii) It provides a shorter syntax by creating the unerlying model, sampler and runner types under the hook
### iii) It fits better the mindset of some users

function ARS(f::Function, init::Vector{Float64}, nsteps::Int, burnin::Int,
  logproposal::Function, proposalscale::Float64, jumpscale::Float64;
  nchains::Int=1, thinning::Int=1)
  mcmodel::MCLikelihood = model(f, init=init)
  mcsampler::ARS = ARS(logproposal, proposalscale, jumpscale)
  mcrunner::SerialMC = SerialMC(burnin=burnin, thinning=thinning, nsteps=nsteps)
  mcsamples::Array{Float64, 3} = Array(Float64, length(mcrunner.r), length(init), nchains)

  for i = 1:nchains
    mcsamples[:, :, i] = run(mcmodel, mcsampler, mcrunner).samples
  end

  mcsamples
end

function SliceSampler(f::Function, init::Vector{Float64}, nsteps::Int, burnin::Int, widths::Vector{Float64};
  nchains::Int=1, thinning::Int=1, stepout::Bool=true)
  mcmodel::MCLikelihood = model(f, init=init)
  mcsampler::SliceSampler = SliceSampler(widths, stepout)
  mcrunner::SerialMC = SerialMC(burnin=burnin, thinning=thinning, nsteps=nsteps)
  mcsamples::Array{Float64, 3} = Array(Float64, length(mcrunner.r), length(init), nchains)

  for i = 1:nchains
    mcsamples[:, :, i] = run(mcmodel, mcsampler, mcrunner).samples
  end

  mcsamples
end

function MH(f::Function, init::Vector{Float64}, nsteps::Int, burnin::Int;
  nchains::Int=1,
  thinning::Int=1,
  logproposal::FunctionOrNothing=nothing,
  randproposal::FunctionOrNothing=(x::Vector{Float64} -> rand(IsoNormal(x, 1.))))
  mcmodel::MCLikelihood = model(f, init=init)
  mcsampler::MH
  if logproposal==nothing
    mcsampler = MH(randproposal)
  else
    mcsampler = MH(logproposal, randproposal)
  end
  mcrunner::SerialMC = SerialMC(burnin=burnin, thinning=thinning, nsteps=nsteps)
  mcsamples::Array{Float64, 3} = Array(Float64, length(mcrunner.r), length(init), nchains)

  for i = 1:nchains
    mcsamples[:, :, i] = run(mcmodel, mcsampler, mcrunner).samples
  end

  mcsamples
end

function RAM(f::Function, init::Vector{Float64}, nsteps::Int, burnin::Int;
  nchains::Int=1, thinning::Int=1, jumpscale::Float64=1., targetrate::Float64=0.234)
  mcmodel::MCLikelihood = model(f, init=init)
  mcsampler::RAM = RAM(jumpscale, targetrate)
  mcrunner::SerialMC = SerialMC(burnin=burnin, thinning=thinning, nsteps=nsteps)
  mcsamples::Array{Float64, 3} = Array(Float64, length(mcrunner.r), length(init), nchains)

  for i = 1:nchains
    mcsamples[:, :, i] = run(mcmodel, mcsampler, mcrunner).samples
  end

  mcsamples
end

function HMC(f::Function, g::Function, init::Vector{Float64}, nsteps::Int, burnin::Int;
  nchains::Int=1, thinning::Int=1, nleaps::Int=10, leapstep::Float64=0.1)
  mcmodel::MCLikelihood = model(f, grad=g, init=init)
  mcsampler::HMC = HMC(nleaps, leapstep)
  mcrunner::SerialMC = SerialMC(burnin=burnin, thinning=thinning, nsteps=nsteps)
  mcsamples::Array{Float64, 3} = Array(Float64, length(mcrunner.r), length(init), nchains)

  for i = 1:nchains
    mcsamples[:, :, i] = run(mcmodel, mcsampler, mcrunner).samples
  end

  mcsamples
end

function MALA(f::Function, g::Function, init::Vector{Float64}, nsteps::Int, burnin::Int;
  nchains::Int=1, thinning::Int=1, driftstep::Float64=1.)
  mcmodel::MCLikelihood = model(f, grad=g, init=init)
  mcsampler::MALA = MALA(driftstep=driftstep)
  mcrunner::SerialMC = SerialMC(burnin=burnin, thinning=thinning, nsteps=nsteps)
  mcsamples::Array{Float64, 3} = Array(Float64, length(mcrunner.r), length(init), nchains)

  for i = 1:nchains
    mcsamples[:, :, i] = run(mcmodel, mcsampler, mcrunner).samples
  end

  mcsamples
end

function SMMALA(f::Function, g::Function, h::Function, init::Vector{Float64}, nsteps::Int, burnin::Int;
  nchains::Int=1, thinning::Int=1, driftstep::Float64=1.)
  mcmodel::MCLikelihood = model(f, grad=g, tensor=h, init=init)
  mcsampler::SMMALA = SMMALA(driftstep=driftstep)
  mcrunner::SerialMC = SerialMC(burnin=burnin, thinning=thinning, nsteps=nsteps)
  mcsamples::Array{Float64, 3} = Array(Float64, length(mcrunner.r), length(init), nchains)

  for i = 1:nchains
    mcsamples[:, :, i] = run(mcmodel, mcsampler, mcrunner).samples
  end

  mcsamples
end
