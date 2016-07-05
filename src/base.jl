typealias IntegerVector{I<:Integer} Vector{I}
typealias RealVector{N<:Real} Vector{N}
typealias RealMatrix{N<:Real} Matrix{N}

typealias RealLowerTriangular{T<:Real, S<:AbstractMatrix} LowerTriangular{T, S}

# chol(y, Val{:L})
