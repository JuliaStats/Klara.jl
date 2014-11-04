Samplers
==========================================================================================

This section discusses the Monte Carlo samplers implemented by the *MCMC* package. Each sampler is an immutable type.
The type's fields correspond to the defining attributes of the sampler. Knowledge of the samplers is assumed. The
interested user is referred to relevant literature for more background information.

The table below summarizes up to which order of log-target derivatives are required by each sampler. First, second and
third order derivatives correspond to the ``model()`` keyword arguments ``grad``, ``tensor`` and ``dtensor``.

+---------------+---------+---------+---------+
| Sampler       | grad    | tensor  | dtensor |
+===============+=========+=========+=========+
| ARS           |    ✗    |    ✗    |    ✗    |
+---------------+---------+---------+---------+
| SliceSampler  |    ✗    |    ✗    |    ✗    |
+---------------+---------+---------+---------+
| MH            |    ✗    |    ✗    |    ✗    |
+---------------+---------+---------+---------+
| RAM           |    ✗    |    ✗    |    ✗    |
+---------------+---------+---------+---------+
| HMC           |    ✓    |    ✗    |    ✗    |
+---------------+---------+---------+---------+
| HMCDA         |    ✓    |    ✗    |    ✗    |
+---------------+---------+---------+---------+
| NUTS          |    ✓    |    ✗    |    ✗    |
+---------------+---------+---------+---------+
| MALA          |    ✓    |    ✗    |    ✗    |
+---------------+---------+---------+---------+
| SMMALA        |    ✓    |    ✓    |    ✗    |
+---------------+---------+---------+---------+
| RMHMC         |    ✓    |    ✓    |    ✓    |
+---------------+---------+---------+---------+
| PMMALA        |    ✓    |    ✓    |    ✓    |
+---------------+---------+---------+---------+


.. _ars:

Acceptance-Rejection Sampler (ARS)
------------------------------------------------------------------------------------------

The *acceptance-rejection method*, also known as *rejection sampling*, was introduced by :cite:`neu:var` . For a more
recent treatment see for example :cite:`rob:cas:mon,bol:und`. Suppose that it is diffcult to attain samples from a
target distribution
:math:`f(x)`.
Instead, assume that it is easier to sample from an alternative proposal density
:math:`g(x)` which satisfies
:math:`f(x)<cg(x)` for some constant
:math:`c>1`.
Due to this inequality,
:math:`cg(x)`
is also called the *envelope*. Rejection sampling allows to sample from
:math:`f(x)`
indirectly by sampling instead from the instrumental proposal
:math:`g(x)`.

The table below provides the fields of ``ARS``.

+---------------+--------------+---------+---------+---------------+
| ARS field     | Field type   | Required/optional | Default value |
+===============+==============+=========+=========+===============+
| logproposal   | Function     | Required          | ✗             |
+---------------+--------------+---------+---------+---------------+
| proposalscale | Float64      | Optional          | 1.0           |
+---------------+--------------+---------+---------+---------------+
| jumpscale     | Float64      | Optional          | 1.0           |
+---------------+--------------+---------+---------+---------------+

``logproposal`` refers to the log-proposal
:math:`g(x)`.
``proposalscale`` is a scale factor to ensure the scaled-up logproposal covers target.
``jumpscale`` is a scale factor for adapting the jump size.

Below is a demonstration of how the ``ARS`` constructor can be invoked:

.. code-block:: julia
  :linenos:

  ARS(x -> -dot(x, x)/2)

  ARS(x -> -dot(x, x)/2; proposalscale=1.5, jumpscale=0.9)


.. _slice_sampler:

Slice Sampler (SliceSampler)
------------------------------------------------------------------------------------------


.. _mh:

Metropolis-Hastings (MH)
------------------------------------------------------------------------------------------


.. _ram:

Robust Adaptive Metropolis (RAM)
------------------------------------------------------------------------------------------


.. _hmc:

Hamiltonian Monte Carlo (HMC)
------------------------------------------------------------------------------------------


.. _hmcda:

Adaptive Hamiltonian Monte Carlo with Dual averaging (HMCDA)
------------------------------------------------------------------------------------------


.. _nuts:

No-U-Turn Sampler (NUTS)
------------------------------------------------------------------------------------------


.. _mala:

Metropolis-Adjusted Langevin Algorithm (MALA)
------------------------------------------------------------------------------------------


.. _smmala:

Simplified Manifold Metropolis-Adjusted Langevin Algorithm (SMMALA)
------------------------------------------------------------------------------------------


.. _rmhmc:

Riemannian Manifold Hamiltonian Monte Carlo (RMHMC)
------------------------------------------------------------------------------------------


.. _pmala:

Position-Dependent Metropolis adjusted Langevin algorithm (PMALA)
------------------------------------------------------------------------------------------
