.. MCMC.jl documentation master file, created by
   sphinx-quickstart on Wed Oct 29 10:51:54 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


MCMC.jl Package Documentation
==========================================================================================


Features
------------------------------------------------------------------------------------------

The Julia *MCMC* package provides a generic engine for Markov Chain Monte Carlo (MCMC) inference. Briefly, *MCMC*
implements:

* imperative model specification,
* a range of Monte Carlo samplers,
* serial and sequential Monte Carlo methods,
* tuning of the samplers' parameters,
* various job managers for controlling the flow of simulations,
* descriptive statistics for MCMC and MCMC diagnostic tools,
* output managemement,
* resuming Monte Carlo simulations,
* Monte Carlo sampling with the help of automatic differentiation.


Outline
------------------------------------------------------------------------------------------

*Jobs* are the central input entities for handling MCMC simulations. A job is first instantiated to delineate the MCMC
configuration. The main defining components of a job are the *model*, *sampler* and *runner*. Once set up, the job can
be run or resumed.

*Chains* form the building block for managing the output of MCMC simulations. Jobs return chains. Descriptive
statistics, MCMC diagnostics and output processing can be performed on chains.

More elaborate usage information is provided in the following sections.


Contents:

.. toctree::
   :maxdepth: 2
   
   models
   samplers
   runners
   tuners
   jobs
   chains
   output_analysis
   output_management
   minimal_interface
   examples
   xrefs
   

Indices and tables
==========================================================================================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
