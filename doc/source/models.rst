Models
==========================================================================================

The model component of a job refers in principle to a model on which MCMC inference is performed. Likelihood-based
modelling is currently supported. The package's roadmap plans to provide graph-based model specification in collaboration
with the Julia `PGM <https://github.com/JuliaStats/PGM.jl>`_ package. This intended upgrade will facilitate MCMC
inference on a wider range of models.

At the present time, the ``model()`` function acts as the interface for model specification. Its ``jtype`` keyword argument determines the type of user-provided model. ``jtype`` defaults to ``:likelihood``, which is currently the only supported model type.


.. _likelihood_model:

Likelihood Model
------------------------------------------------------------------------------------------

A *likelihood model* is defined by its log-target and its parameters' initial values. The log-target is a possibly
unnormalized density, which is aimed at approximating the log-likelihood or, in a Bayesian context, the log-posterior. Some samplers additionally require the gradient, metric tensor and tensor derivatives of the log-target.
Such extra information is either provided by the user or is attained via automatic differentiation.

The likelihood model is specified via the ``MCLikelihood()`` constructor or the ``model()`` function, which are both invoked in three possible ways.


Model Specification by Explicitly Defining the Log-Target Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One way of setting the model is to provide the log-target as a function. As a working example, suppose that ineterest is
in sampling from the 3-dimensional standard normal
:math:`\mathbb{N}(\mathbf{0},\mathbf{I})`.
The function
:math:`f:\mathbb{R}^3\rightarrow\mathbb{R}`,
:math:`f(\mathbf{x})=-\mathbf{x}\cdot\mathbf{x}`,
is selected as the unnormalized log-target, with its values being the inner product of its input. The initial parameter
values are set to be
:math:`x=\left[1,1,1\right]^T`.
Any of the following invocations instantiate the desired model:

.. code-block:: julia
  :linenos:

  using MCMC

  # Using a predefined function
  f(x) = -dot(x, x)
  mcmodel = model(f, init=ones(3))

  # Using an anonymous function
  mcmodel = model(x -> -dot(x, x), init=ones(3))

  # Passing an anonymous function to MCLikelihood
  mcmodel = MCLikelihood(x -> -dot(x, x), init=ones(3))

``mcmodel.eval()`` and ``mcmodel.init`` give access to the type fields holding the log-target and the initial parameter
values respectively. For instance:

.. code-block:: julia
  :linenos:

  # Evaluate log-target at initial parameter values
  mcmodel.eval(mcmodel.init)

  # Evaluate log-target at [1., 1.5, 1.]
  mcmodel.eval([1., 1., 1.])

The log-target's gradient is required by some samplers. It can be passed to the model via the ``grad`` keyword argument:

.. code-block:: julia
  :linenos:

  mcmodel = model(x -> -dot(x, x), grad=v -> -2v, init=ones(3))

  # Evaluate log-target's gradient at initial parameter values
  mcmodel.evalg(mcmodel.init)

Here is the complete list of optional keyword arguments available for the likelihood model used in conjunction with
a log-target function:

* ``grad::Union(Nothing, Function)`` - gradient of log-target,
* ``tensor::Union(Nothing, Function)`` - metric tensor of log-target, i.e. expected Fisher information matrix (plus negative Hessian of log-prior in a Bayesian context),
* ``dtensor::Union(Nothing, Function)`` - first-order derivatives of metric tensor of log-target,
* ``init::Union(Float64, Vector{Float64})`` - initial parameter values,
* ``scale::Union(Float64, Vector{Float64})`` - scaling hint for parameters used by some samplers,
* ``allgrad::Union(Nothing, Function)`` - tuple (log-target, grad),
* ``alltensor::Union(Nothing, Function)`` - tuple (log-target, grad, tensor),
* ``alldtensor::Union(Nothing, Function)`` - tuple (log-target, grad, tensor, dtensor).

``allgrad``, ``alltensor`` and ``alldtensor`` are deduced from ``grad``, ``tensor`` and ``dtensor``, but can also be
manually provided.


Model Specification by Explicitly Defining the Log-Target Distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Instead of providing it as a function, the log-target can be defined as a distribution. For example, this is how the
log-target is set to be a standard normal distribution:

.. code-block:: julia
  :linenos:
  
  using Distributions, MCMC

  mcmodel = model(Normal(), init=ones(3))

The functions ``logpdf()`` and ``gradlogpdf()`` of the *Distributions* package are then assigned to ``model.eval()`` and ``model.evalg()`` respectively.

The optional keyword arguments for likelihood models defined by lot-target distributions and lot-target functions are
the same.


Model Specification via Expression Parsing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A third way of specifying the model is via expression parsing. The model is described with the help of the package's
expression-based syntax and the resulting expression is passed to ``model()`` as an argument. The following example of
setting the log-target to be a standard normal distribution exemplifies usage:

.. code-block:: julia
  :linenos:

  modelexpression = quote
    v ~ Normal(0, 1)
  end

  # gradient of log-target is not computed
  mcmodel = model(modelexpression, v=ones(3))

  # gradient of log-target is computed via automatic differentiation
  mcmodel = model(modelexpression, gradient=true, v=ones(3))

In the first ``model()`` call, the ``gradient`` keyword argument defaults to ``false``, so the gradient of the
log-target is not used by the sampler. In the second ``model()`` call, ``gradient`` is set to ``true``, which signifies that the log-target's gradient will be computed via automatic differentiation. In both cases, the initial value of parameter vector ``v`` is set to ``[1, 1, 1]``.

This is the list of optional keyword arguments for likelihood models set via expression parsing:

* ``gradient::Bool`` - indicates whether the log-target's gradient will be computed via automatic differentiation,
* ``scale::Union(Float64, Vector{Float64})`` - scaling hint for parameters used by some samplers,
* keyword arguments representing model parameters followed by their initial values as shown in the example.
