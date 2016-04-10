normalise(d::Truncated{Normal, Continuous}) = d.untruncated.σ*(d.ucdf-d.lcdf)

lognormalise(d::Truncated{Normal, Continuous}) = log(normalise(d))
