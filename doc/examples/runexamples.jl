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
    joinpath("Normal", "AM", "function")
  ]

println("Running examples:")

for file in files
  filejl = file*".jl"
  println("  * $(filejl) *")
  include(filejl)
end
