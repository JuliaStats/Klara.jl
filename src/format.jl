lcover(s::AbstractString, w::AbstractString=" ") = w*s
rcover(s::AbstractString, w::AbstractString=" ") = s*w
cover(s::AbstractString, w::AbstractString=" ") = w*s*w

format_iteration(ndigits::Int, ralign::Bool=true) = generate_formatter("%$(ralign ? "" : "-")$(ndigits)d")
format_percentage(precision::Int=2, hundredalign::Bool=true, ralign::Bool=true) =
  generate_formatter("%$(ralign ? "" : "-")$(hundredalign ? 6 : 5).$(precision)f")
