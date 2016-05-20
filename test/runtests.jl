tests =
  [
    "common",
    "VariableStates",
    "BasicDiscUnvParameterState",
    "BasicContUnvParameterState",
    "BasicContMuvParameterState",
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
  ]

println("Running tests:")

for t in tests
  tfile = t*".jl"
  println("  * $(tfile) *")
  include(tfile)
end
