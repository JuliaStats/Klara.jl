.. _introduction:

Introduction
------------------------------------------------------------------------------------------

The Julia *Lora* package provides a generic engine for Markov Chain Monte Carlo (MCMC) inference.


.. _features:

Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Briefly, *Lora* implements:

* imperative model specification,
* a range of Monte Carlo samplers,
* serial and sequential Monte Carlo methods,
* tuning of the samplers' parameters,
* various job managers for controlling the flow of simulations,
* descriptive statistics for MCMC and MCMC diagnostic tools,
* output managemement,
* resuming Monte Carlo simulations,
* Monte Carlo sampling with the help of automatic differentiation.


.. _outline:

Outline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Jobs* are the central input entities for handling MCMC simulations. A job is first instantiated to delineate the MCMC
configuration. The main defining components of a job are the *model*, *sampler* and *runner*. Once set up, the job can
be run or resumed.

*Chains* form the building block for managing the output of MCMC simulations. Jobs return chains. Descriptive
statistics, MCMC diagnostics and output processing can be performed on chains.

More elaborate usage information is provided in the following sections.
