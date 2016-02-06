format_iteration(ndigits::Int, ralign::Bool=true) = generate_formatter("%$(ralign ? "" : "-")$(ndigits)d")

format_percentage(precision::Int=2, hundredalign::Bool=true, ralign::Bool=true) =
  generate_formatter("%$(ralign ? "" : "-")$(hundredalign ? 4+precision : 3+precision).$(precision)f")
