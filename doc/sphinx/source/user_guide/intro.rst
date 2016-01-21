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
are provided to integrate *Lora* with other packages such as *ForwardDiff* and *ReverseDiffSource*.

The high-level API sits atop of a low-level one. The latter aims at providing an extensible codebase for developers
interested in adding new functionality and offers an alternative interface for users who prefer more hands-on access to the
underlying routines.

.. _features:

Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

List to appear soon:

* item 1
* item 2
