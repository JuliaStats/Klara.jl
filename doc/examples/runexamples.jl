files =
  [
    joinpath("BivariateNormal", "AM", "function"),
    joinpath("BivariateNormal", "AM", "pdf"),
    joinpath("BivariateNormal", "MALA", "function", "analytical"),
    joinpath("BivariateNormal", "MALA", "function", "forwarddiff"),
    joinpath("BivariateNormal", "MALA", "function", "reversediff"),
    joinpath("BivariateNormal", "MALA", "pdf", "analytical"),
    joinpath("BivariateNormal", "MALA", "pdf", "forwarddiff"),
    joinpath("BivariateNormal", "MALA", "pdf", "reversediff"),
    joinpath("BivariateNormal", "SMMALA", "analytical"),
    joinpath("BivariateNormal", "SMMALA", "forwarddiff"),
    joinpath("BivariateNormal", "SMMALA", "reversediff"),
    joinpath("BivariateNormal", "Gibbs"),
    joinpath("Gamma", "MH", "transformation"),
    joinpath("Gamma", "MH", "truncation"),
    joinpath("Gamma", "MALA"),
    joinpath("Normal", "AM", "function"),
    joinpath("Normal", "AM", "pdf"),
    joinpath("Normal", "HMC", "function", "analytical"),
    joinpath("Normal", "HMC", "function", "forwarddiff"),
    joinpath("Normal", "HMC", "pdf", "analytical"),
    joinpath("Normal", "HMC", "pdf", "forwarddiff"),
    joinpath("Normal", "MALA", "function", "analytical"),
    joinpath("Normal", "MALA", "function", "forwarddiff"),
    joinpath("Normal", "MALA", "pdf", "analytical"),
    joinpath("Normal", "MALA", "pdf", "forwarddiff"),
    joinpath("Poisson", "MH"),
    joinpath("swiss", "SMMALA", "analytical"),
    joinpath("swiss", "SMMALA", "forwarddiff"),
    joinpath("swiss", "SMMALA", "reversediff"),
    joinpath("swiss", "RAM"),
    joinpath("swiss", "SliceSampler")
  ]

println("Running examples:")

for file in files
  filejl = file*".jl"
  println("  * $(filejl) *")
  include(filejl)
end
