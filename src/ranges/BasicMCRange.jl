### BasicMCRange

# BasicMCRange is used for sampling a single parameter via serial Monte Carlo
# It is the most elementary and typical Markov chain Monte Carlo (MCMC) range
# It contains nagivational info (burnin, thinning, number of steps)

struct BasicMCRange{T<:Integer} <: MCRange{T}
  burnin::T
  thinning::T
  nsteps::T
  postrange::StepRange{T, T}
  npoststeps::T

  function BasicMCRange{T}(postrange::StepRange{T, T}) where T<:Integer
    burnin = first(postrange)-1
    thinning = postrange.step
    nsteps = last(postrange)

    @assert burnin >= 0 "Number of burn-in iterations should be non-negative"
    @assert thinning >= 1 "Thinning should be >= 1"
    @assert nsteps > burnin "Total number of MCMC iterations should be greater than number of burn-in iterations"

    npoststeps = length(postrange)

    new(burnin, thinning, nsteps, postrange, npoststeps)
  end
end

BasicMCRange(postrange::StepRange{T, T}) where T<:Integer = BasicMCRange{T}(postrange)

BasicMCRange(postrange::UnitRange{T}) where {T<:Integer} = BasicMCRange{T}(first(postrange):1:last(postrange))

BasicMCRange(; burnin::T=0, thinning::T=1, nsteps::T=100) where {T<:Integer} = BasicMCRange{T}((burnin+1):thinning:nsteps)

show(io::IO, r::BasicMCRange) =
  print(io, "BasicMCRange: number of steps = $(r.nsteps), burnin = $(r.burnin), thinning = $(r.thinning)")
