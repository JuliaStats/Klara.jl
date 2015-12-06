tests =
  [
    "common",
    "VariableStates",
    "ParameterStates",
    "VariableNStates",
    "ParameterNStates",
    "VariableIOStreams",
    "ParameterIOStreams",
    "variables",
    "BasicContUnvParameter",
    "BasicContMuvParameter",
    "dependencies",
    "GenericModel",
    "generators"
    # "tuners",
    # "VanillaMCTuner",
    # "AcceptanceRateMCTuner",
    # "MH"
  ]

println("Running tests:")

for t in tests
  tfile = t*".jl"
  println("  * $(tfile) *")
  include(tfile)
end
