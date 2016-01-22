.. _introduction:

Introduction
------------------------------------------------------------------------------------------

.. _principles:

Main principles of development
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Julia *Lora* package provides an interoperable generic engine for a breadth of Markov Chain Monte Carlo (MCMC) methods.

The idea that there exists no unique optimal MCMC methodology for all purposes has been a development cornerstone. Along
these lines, interest is in providing a wealth of Monte Carlo strategies and options, letting the user decide which
algorithm suits their use case. Such "agnostic" approach to coding drives *Lora* from top to bottom, offering a variety of
methods and detailed configuration, ultimately leading to rich functionality. *Lora*'s wide range of functionality makes
itself useful in applications and as a test bed for comparative methodological research. It also offers the flexibility to
connect *Lora* with various other packages, exploiting different levels of ongoing developments in them.

In fact, interoperability has been another central principle of development. A high-level API enables the user to implement
their application effectively, while it facilitates connectivity to Julia packages. Whenever deemed necessary, minor wrappers
are provided to integrate *Lora* with other packages such as *ReverseDiffSource* and *ForwardDiff*.

The high-level API sits atop of a low-level one. The latter aims at providing an extensible codebase for developers
interested in adding new functionality and offers an alternative interface for users who prefer more hands-on access to the
underlying routines.

Speed of execution has been another motivation behind the low-level API. Passing from the higher to the lower level interface
allows to exploit Julia's meta-programming capabilities by generating code dynamically and by substituting dictionaries by
vectors internally. Memory footprint and garbage collection have been kept to a minimum without compromising ease of use
thanks to the duality of higher and lower level APIs.

.. _features:

Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A summary of *Lora*'s main features includes:

* *Graph-based model specification*. Representing the model as a graph widens the scope of accommodated models and enables \
  exploiting graph algorithms from the *Graphs* package.

* *Diverse options for defining model parameters*. Parameters can be defined on the basis of a log-target or they can be
  introduced in a Bayesian fashion via their log-likelihood and log-prior. Parameter targets, likelihoods and priors can be
  specified via functions or distributions. *Lora*'s integration with the *Distributions* package facilitates
  parameter definition via distributions.

* *Job-centric simulations*. The concept of MCMC simulation has been separated from that of model specification. Job types
  indicate the context in which a model is simulated. For example, a *BasicMCJob* instance determines how to sample an MCMC
  chain for a model with a single parameter, whereas a *GibbsJob* provide Gibbs sampling for more complex models involving
  several parameters.

* *Customized job flow*. Job control flow comes in two flavors, as it can be set to ordinary loop-based flow or it can be
  managed by Julia's tasks (coroutines). Job management with tasks allows MCMC simulations to be suspended and resumed in
  a flexible manner.

* *Wide range of Monte Carlo samplers*. A range of MCMC samplers is available, including accept-reject and slice sampling,
  variations of the Metropolis-Hastings algorithm, No-U-Turn (NUTS) sampling, geometric MCMC schemes, such as Riemann
  manifold Langevin and Hamiltonian Monte Carlo. Adaptive samplers, such as random adaptive Metropolis, and empirical tuning
  are included in *Lora* as alternative means to expedite convergence. It is noted that most of these samplers need to be
  ported from the older version of Lora, which is work in progress.

* *MCMC summary statistics and convergence diagnostics*. Main routines for computing the effective sampling size and
  integrated autocorrelation time have been coded, while there is a roadmap to provide more convergence diagnostics tools
  (note to user; this functionality will also be ported soon from the older version of *Lora*).

* *States and chains*. Proposed Monte Carlo samples are organized systematically with the help of a state and chain type
  system. This way, values can be passed around and stored without re-allocating memory. At the same time, the state/chain
  type system offers scope for extending the current functionality if storing of less usual components is intended.

* *Detailed configuration of output storage in memory or in file*. The chain resulting from a Monte Carlo simulation can be
  saved in memory or can be written directly to a file stream. Detailed output configuration is possible, allowing to
  select which elements to save and which to omit from the final output.

* *Automatic differentiation for MCMC sampling*. Some Monte Carlo methods require the gradient or higher order derivatives of
  the log-target. If these derivatives are not user-inputted explicitly, *Lora* can optionally compute them using reverse or
  forward mode automatic differentiation. For this purpose, *Lora* uses *ReverseDiffSource* and *ForwardDiff* under the
  hood.
