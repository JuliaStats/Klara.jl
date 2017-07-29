function softabs(hessian::RealMatrix, a::Float64=1000.0)
  λ, Q = eig(hessian)
  Q*diagm(λ./tanh.(a*λ))*Q'
end
