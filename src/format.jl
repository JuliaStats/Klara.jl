lcover(s::AbstractString, w::AbstractString=" ") = w*s
lcover(s, w::AbstractString=" ") = lcover(string(s), w)

rcover(s::AbstractString, w::AbstractString=" ") = s*w
rcover(s, w::AbstractString=" ") = lcover(string(s), w)

cover(s::AbstractString, w::AbstractString=" ") = w*s*w
cover(s, w::AbstractString=" ") = lcover(string(s), w)

format_iteration(ndigits::Int, ralign::Bool=true) = generate_formatter("%$(ralign ? "" : "-")$(ndigits)d")

format_percentage(precision::Int=2, hundredalign::Bool=true, ralign::Bool=true) =
  generate_formatter("%$(ralign ? "" : "-")$(hundredalign ? 4+precision : 3+precision).$(precision)f")
