Introduction
==================================

GEMAct is an **actuarial package**, based on the collective risk theory framework, that offers a comprehensive set of tools for **non-life** (re)insurance **costing**, stochastic **claims reserving**, and **loss aggregation**.

The variety of available functionalities makes GEMAct modeling very flexible and provides users with a powerful tool that fits into the expanding community of **Python** programming language.

Scope
--------

* A collective risk model apparatus for costing non-life (re)insurance contracts.
* Extend the set of distributions available in scipy to actuarial scientists. GEMAct provides the first Python implementation of the `(a, b, 0) <https://en.wikipedia.org/wiki/(a,b,0)_class_of_distributions>`_ and (a, b, 1) distribution classes.
* Popular copulas with improved functionalities, e.g. the Student t copula cumulative distribution function can be numerically approximated.
* A loss reserve estimation tool.
* The first open-source Python implementation of the AEP algorithm.

Reinsurance Contracs
~~~~~~~~~~~~~~~~~~~~~~~~~~~

GEMAct can be used for costing the following reinsurance contracts and their combinations.

* Excess-of-Loss (XL) including:
    * aggregate conditions (cover and deductible),
    * reinstatements.
* **Quota-share (QS)**.
* **Stop-loss (SL)**.

Computational methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GEMAct offers multiple numerical methods used within the context of loss distributions approximation.

* Collective risk model:
    * Recursive method (Panjer recursion),
    * Discrete Fourier transform, via the Fast Fourier Transform (FFT) algorithm,
    * Monte Carlo simulation.
* Loss (model) aggregation:
    * AEP algorithm,
    * Monte Carlo simulation.
