### Logistic function in its general form

# See https://en.wikipedia.org/wiki/Logistic_function
# l scales the curve. For l>1, the curve is stretched, whereas for l<1 it is shrinked
# The curve's maximum value coincides with l
# k gives the curve's steepness. For larger k, the curve becomes more steep
# x0 is the x value of the curve's midpoint
# y0 is the y value of the curve's midpoint
# The logistic function has been defined for tuning purposes

logistic(x::Real, l::Real=1., k::Real=1., x0::Real=0., y0::Real=0.) = l/(1+exp(-k*(x-x0)))+y0
