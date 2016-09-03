tests =
  [
    "common",
    "VariableStates",
    "BasicDiscUnvParameterState",
    "BasicDiscMuvParameterState",
    "BasicContUnvParameterState",
    "BasicContMuvParameterState",
    "VariableNStates",
    "ParameterNStates",
    "VariableIOStreams",
    "ParameterIOStreams",
    "BasicContUnvParameter",
    "BasicContMuvParameter",
    "GenericModel",
    "generators"
  ]

println("Running tests:")

for t in tests
  tfile = t*".jl"
  println("  * $(tfile) *")
  include(tfile)
end
