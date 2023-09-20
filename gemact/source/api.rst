API reference guide
========================

``LossModel``
---------------

.. automodule:: gemact.lossmodel
    :members:


(Re)insurance costing
~~~~~~~~~~~~~~~~~~~~~~~~~~~

GEMAct costing model is based on the collective risk theory.

.. math:: X=\sum_{i=1}^{N} Z_i
   :label: crm

Denote the pure premium as :math:`P= \mathbb{E}\left( X \right)`

In order to cost (re)insurance contracts we model the aggregate loss under the following assumptions.

* :math:`X \sim g(X)` and :math:`Z_i \sim f(Z)`
*  Given :math:`N=n`,  the severity :math:`Z_1,\ldots,Z_n` is i.i.d and does not depend on :math:`n`.
* :math:`N` does not depend on :math:`Z_1,\ldots,Z_n` in any way.

Risk costing
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following table gives the correspondence between the ``LossModel`` class attributes and our pricing model as presented below.

+------------------------+-----------------------------------------+
| Costing model notation | Parametrization in ``LossModel``        |
|                        |                                         |
+========================+=========================================+
|:math:`d`               |      deductible                         |
+------------------------+-----------------------------------------+
| :math:`u`              | deductible+cover                        |
+------------------------+-----------------------------------------+
| :math:`L`              | aggr_deductible                         |
+------------------------+-----------------------------------------+
| :math:`K`              |       n_reinst                          |
+------------------------+-----------------------------------------+
|:math:`c_{k}`           |      reinst_percentage                  |
+------------------------+-----------------------------------------+
| :math:`\alpha`         | share                                   |
+------------------------+-----------------------------------------+

Assume:

.. math:: X^{\prime}=\min (\max (0, X-L), (K+1) \cdot u)
   :label: crmRE

Given :math:`X=\sum_{i=1}^{N} Y_{i}`, where :math:`Y_i` is defined in equation :eq:`crmRE`.

.. math:: Y_{i}=\min \left(\max \left(0, Z_{i}-d\right), u\right)
   :label: XLpremium

Equation :eq:`XLpremium` shows the pure premium for the excess of loss contract. It is possible to obtain a (plain-vanilla) XL with :math:`L=0`, :math:`K=+\infty` and :math:`c=0`.

.. math:: P=\frac{D_{L K}}{1+\frac{1}{m} \sum_{k=1}^{K} c_{k} d_{L, k-1}}
   :label: reinstatementsLayer

This costing approach is based on  :cite:t:`b:kp` and  :cite:t:`sundt`.

Refer to :cite:t:`b:kp` for the recursive formula and to :cite:t:`embrechts` for the fast Fourier transform to approximate the discrete Fourier transform.

Example
^^^^^^^^^^^^^^^^^^^^^^^^^^

``LossModel`` can be used for costing purposes.

An example of costing with reinstatements::

   from gemact.lossmodel import Frequency, Severity, PolicyStructure, LossModel
   lossmodel_RS = LossModel(
    frequency=Frequency(
        dist='poisson',
        par={'mu': .5}
    ),
    severity=Severity(
        par= {'loc': 0,
              'scale': 83.34,
              'c': 0.834},
        dist='genpareto'
    ),
    policystructure=PolicyStructure(
        layers=Layer(
            cover=100,
            deductible=0,
            aggr_deductible=0,
            reinst_share=0.5,
            n_reinst=1
        )
    ),
    aggr_loss_dist_method='fft',
    sev_discr_method='massdispersal',
    n_aggr_dist_nodes=int(100000)
    )
    lossmodel_RS.print_costing_specs()

``LossModel`` can be used to model the aggregate loss of claims::

    from gemact.lossmodel Frequency,Severity, LossModel
    frequency = Frequency(
        dist='poisson',
        par={'mu': 4}
        )

    # define a Generalized Pareto severity model
    severity = Severity(
        dist='genpareto',
        par={'c': .2, 'scale': 1})

    lossmodel_dft = LossModel(
    frequency=frequency,
    severity=severity,
    aggr_loss_dist_method='fft',
    n_sev_discr_nodes=int(1000),
    sev_discr_step=1,
    n_aggr_dist_nodes=int(10000)
    )

    print('FFT', lm_fft.aggr_loss_mean())

Severity discretization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When passing from a continuous distribution to an arithmetic distribution, it is important to preserve the distribution properties, either locally or globally.
Given a bandiwith (or discretization step) :math:`h` and a number of nodes :math:`M`, in :cite:t:`b:kp` the method of mass dispersal and the method of local moments matching work as follows.

**Method of mass dispersal**

.. math:: f_{0}=\operatorname{Pr}\left(Y<\frac{h}{2}\right)=F_{Y}\left(\frac{h}{2}-0\right)
   :label: md1

.. math:: f_{j}=F_{Y}\left(j h+\frac{h}{2}-0\right)-F_{Y}\left(j h-\frac{h}{2}-0\right), \quad j=1,2, \ldots, M-1
   :label: md2


.. math:: f_{M}=1-F_{X}[(M-0.5) h-0]
   :label: md3

**Method of local moments matching**

The following approach is applied to preserve the global mean of the distribution.

.. math:: f_0 = m^0_0
   :label: lm1

.. math:: f_j = m^{j}_0+ m^{j-1}_1 , \quad j=0,1, \ldots, M
   :label: lm2

.. math:: \sum_{j=0}^{1}\left(x_{k}+j h\right)^{r} m_{j}^{k}=\int_{x_{k}-0}^{x_{k}+ h-0} x^{r} d F_{X}(x), \quad r=0,1
   :label: lm3

.. math:: m_{j}^{k}=\int_{x_{k}-0}^{x_{k}+p h-0} \prod_{i \neq j} \frac{x-x_{k}-i h}{(j-i) h} d F_{X}(x), \quad j=0,1
   :label: lm4

In addition to these two methods, our package also provides the methods of upper and lower discretizations.

Example
^^^^^^^^^^^^^^^^^^^^^^^^^^

An example of code to implement severity discretization is given below::

    from gemact.lossmodel import Severity
    import numpy as np

    severity=Severity(
        par= {'loc': 0, 'scale': 83.34, 'c': 0.834},
        dist='genpareto'
    )

    massdispersal = severity.discretize(
    discr_method='massdispersal',
    n_discr_nodes=50000,
    discr_step=.01,
    deductible=0
    )

    localmoments = severity.discretize(
    discr_method='localmoments',
    n_discr_nodes=50000,
    discr_step=.01,
    deductible=0
    )

    meanMD = np.sum(massdispersal['sev_nodes'] * massdispersal['fj'])
    meanLM = np.sum(localmoments['sev_nodes'] * localmoments['fj'])

    print('Original mean: ', severity.model.mean())
    print('Mean (mass dispersal): ', meanMD)
    print('Mean (local moments): ', meanLM)

``LossReserve``
------------------

.. automodule::  gemact.lossreserve
   :members:


Claims reserving
~~~~~~~~~~~~~~~~~~~~~~~~~~~

GEMAct provides a software implementation of average cost methods for claims reserving based on the collective risk model framework.

The methods implemented are the Fisher-Lange in :cite:t:`fisher99` the collective risk model for claims reserving in :cite:t:`ricotta16`.

It allows for tail estimates and assumes the triangular inputs to be provided as a ``numpy.ndarray`` with two equal dimensions ``(I,J)``, where ``I=J``.

The aim of average cost methods is to model incremental payments as in equation :eq:`acmethods`.

.. math:: P_{i,j}=n_{i,j} \cdot m_{i,j}
   :label: acmethods

where :math:`n_{i,j}` is the number of payments in the cell :math:`i,j` and :math:`m_{i,j}` is the average cost in the cell :math:`i,j`.

Example
^^^^^^^^^^^^^^^^^^^^^^^^^^

It is possible to use the module ``gemdata`` to test GEMAct average cost methods::

    from gemact import gemdata
    ip_ = gemdata.incremental_payments
    in_ = gemdata.incurred_number
    cp_ = gemdata.cased_payments
    cn_= gemdata.cased_number
    reported_ = gemdata.reported_claims
    claims_inflation = gemdata.claims_inflation


An example of Fisher-Lange implementation::

    from gemact.lossreserve import AggregateData, ReservingModel
    from gemact import gemdata

    ip_ = gemdata.incremental_payments
    in_ = gemdata.incurred_number
    cp_ = gemdata.cased_payments
    cn_= gemdata.cased_number
    reported_ = gemdata.reported_claims
    claims_inflation = gemdata.claims_inflation

    ad = AggregateData(
      incremental_payments=ip_,
      cased_payments=cp_,
      cased_number=cn_,
      reported_claims=reported_,
      incurred_number=in_
      )

    rm = ReservingModel(
      tail=True,
      reserving_method="fisher_lange",
      claims_inflation=claims_inflation
      )

    lm = gemact.LossReserve(
      data=ad,
      reservingmodel=rm
      )


Observe the CRM for reserving requires different model assumptions::

    from gemact import gemdata

    ip_ = gemdata.incremental_payments
    in_ = gemdata.incurred_number
    cp_ = gemdata.cased_payments
    cn_= gemdata.cased_number
    reported_ = gemdata.reported_claims
    claims_inflation = gemdata.claims_inflation

    from gemact.lossreserve import AggregateData, ReservingModel, LossReserve
    ad = AggregateData(
        incremental_payments=ip_,
        cased_payments=cp_,
        cased_number=cn_,
        reported_claims=reported_,
        incurred_number=in_)


    mixing_fq_par = {'a': 1 / .08 ** 2,  # mix frequency
                         'scale': .08 ** 2}

    mixing_sev_par = {'a': 1 / .08 ** 2, 'scale': .08 ** 2}  # mix severity
    czj = gemdata.czj
    claims_inflation= gemdata.claims_inflation

    rm =  ReservingModel(tail=True,
             reserving_method="crm",
             claims_inflation=claims_inflation,
             mixing_fq_par=mixing_fq_par,
             mixing_sev_par=mixing_sev_par,
             czj=czj)

    #Loss reserving: instance lr
    lm = LossReserve(data=ad,
                     reservingmodel=rm)


``LossAggregation``
---------------------

.. autoclass::  gemact.lossaggregation.LossAggregation
   :members:

Loss aggregation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math:: P\left[ X_1 +\ldots +X_d \right] \approx P_n(s)
   :label: pty

The probability in equation :eq:`pty` can be approximated iteratively via the AEP algorithm, which is implemented for the first time in the GEMAct package, under the following assumptions:

Assuming:

* :math:`𝑋=(X_i, \ldots, X_d)` vector of strictly positive random components.

* The joint c.d.f. :math:`H(x_1,…,x_d )=P\left[ X_1 +\ldots +X_d \right]` is known or it can be computed numerically.

Refer to :cite:t:`arbenz11` for an extensive explanation on the AEP algorithm.
It is possible to use a MC simulation for comparison.

Example
^^^^^^^^^^^^^^^^^^^^^^^^^^

Example code under a clayton assumption::

    from gemact.lossaggregation import LossAggregation, Copula, Margins

    lossaggregation = LossAggregation(
    margins=Margins(
    dist=['genpareto', 'lognormal'],
    par=[{'loc': 0, 'scale': 1/.9, 'c': 1/.9}, {'loc': 0, 'scale': 10, 'shape': 1.5}],
    ),
    copula=Copula(
    dist='frank',
    par={'par': 1.2, 'dim': 2}
    ),
    n_sim=500000,
    random_state=10,
    n_iter=8
    )
    s = 300 # arbitrary value
    p_aep = lossaggregation.cdf(x=s, method='aep')
    print('P(X1+X2 <= s) = ', p_aep)
    p_mc = lossaggregation.cdf(x=s, method='mc')
    print('P(X1+X2 <= s) = ', p_mc)

The ``distributions`` module
------------------------------------

``Poisson``
~~~~~~~~~~~~~~~~~~~~

.. autoclass::  gemact.distributions.Poisson
   :members:
   :inherited-members:


Parametrization
^^^^^^^^^^^^^^^^^^^^^^^^^^

For a more detailed explanation refer to `SciPy Poisson distribution documentation <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.poisson.html>`_, see :cite:t:`scipy`.

.. math:: f(k)=\exp (-\mu) \frac{\mu^{k}}{k !}
   :label: poisson

Given :math:`\mu \geq 0` and :math:`k \geq 0`.

Example code on the usage of the Poisson class::

    from gemact import distributions
    import numpy as np
    mu = 4
    dist = distributions.Poisson(mu=mu)
    seq = np.arange(0,20)
    nsim = int(1e+05)

    # Compute the mean via pmf
    mean = np.sum(dist.pmf(seq)*seq)
    variance = np.sum(dist.pmf(seq)*(seq-mean)**2)

    print('Mean',mean)
    print('Theoretical mean', dist.mean())
    print('Variance', variance)
    print('Theoretical variance', dist.var())
    #compare with simulations
    print('Simulated mean', np.mean(dist.rvs(nsim)))
    print('Simulated variance', np.var(dist.rvs(nsim)))

Then, use ``matplotlib`` library to show the pmf and the cdf of a Poisson::

    import matplotlib.pyplot as plt

    #pmf
    plt.vlines(x=seq, ymin=0, ymax=dist.pmf(seq))
    plt.title('Probability mass function, ~Poisson(mu=4)')

    #cdf
    plt.step(np.arange(-2,20), dist.cdf(np.arange(-2,20)),'-', where='pre')
    plt.title('Cumulative distribution function, ~ Poisson(mu=4)')

.. image:: images/pmfPois.png
  :width: 400

.. image:: images/cdfPois.png
  :width: 400

``Binom``
~~~~~~~~~~~~~~~~~~~~

.. autoclass::  gemact.distributions.Binom
   :members:
   :inherited-members:

Parametrization
^^^^^^^^^^^^^^^^^^^^^^^^^^


For a more detailed explanation refer to `SciPy Binomial distribution documentation <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binom.html>`_, see :cite:t:`scipy`.

.. math:: f(k)=\left(
   \begin{array}{l}
      n \\
      k
   \end{array} \right) p^{k}(1-p)^{n-k}
   :label: binomial

for :math:`k \in\{0,1, \ldots, n\}, 0 \leq p \leq 1`

Example code on the usage of the Binom class::

    from gemact import distributions
    import numpy as np

    n = 10
    p = 0.5
    dist = distributions.Binom(n=n, p=p)
    seq = np.arange(0,20)
    nsim = int(1e+05)

    # Compute the mean via pmf
    mean=np.sum(dist.pmf(seq)*seq)
    variance=np.sum(dist.pmf(seq)*(seq-mean)**2)

    print('Mean',mean)
    print('Theoretical mean',dist.mean())
    print('Variance',variance)
    print('Theoretical variance',dist.var())
    #compare with simulations
    print('Simulated mean',np.mean(dist.rvs(nsim)))
    print('Simulated variance',np.var(dist.rvs(nsim)))

Then, use ``matplotlib`` library to show the pmf and the cdf of a Poisson::

    import matplotlib.pyplot as plt
    #pmf
    plt.vlines(x=seq, ymin=0, ymax=dist.pmf(seq))
    plt.title('Probability mass function, ~Binom(n=10,p=0.5)')

    #cdf
    plt.step(np.arange(-2,20),dist.cdf(np.arange(-2,20)),'-', where='pre')
    plt.title('Cumulative distribution function, ~Binom(n=10,p=0.5)')

.. image:: images/pmfBinom.png
  :width: 400

.. image:: images/cdfBinom.png
  :width: 400

``Geom``
~~~~~~~~~~~~~~~~~~~~

.. autoclass::  gemact.distributions.Geom
   :members:
   :inherited-members:

Parametrization
^^^^^^^^^^^^^^^^^^^^^^^^^^

For a more detailed explanation refer to the `SciPy Geometric distribution documentation <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.geom.html>`_, see :cite:t:`scipy`..

.. math:: f(k)=(1-p)^{k-1} p
   :label: geom

for :math:`k \geq 1,0<p \leq 1`

Example code on the usage of the Geom class::

    from gemact import distributions
    import numpy as np

    p=0.8
    dist=distributions.Geom(p=p)
    seq=np.arange(0,100,.001)
    nsim=int(1e+05)

    # Compute the mean via pmf
    mean=np.sum(dist.pmf(seq)*seq)
    variance=np.sum(dist.pmf(seq)*(seq-mean)**2)

    print('Mean',mean)
    print('Theoretical mean',dist.mean())
    print('Variance',variance)
    print('Theoretical variance',dist.var())
    #compare with simulations
    print('Simulated mean',np.mean(dist.rvs(nsim)))
    print('Simulated variance',np.var(dist.rvs(nsim)))

Then, use ``matplotlib`` library to show the pmf and the cdf of a Geom::

    import matplotlib.pyplot as plt

    #pmf
    plt.vlines(x=seq, ymin=0, ymax=dist.pmf(seq))
    plt.title('Probability mass function, ~Geom(p=0.8)')

    #cdf
    plt.step(np.arange(-2,20),dist.cdf(np.arange(-2,20)),'-', where='pre')
    plt.title('Cumulative distribution function, ~Geom(p=0.8)')

.. image:: images/pmfGeom.png
  :width: 400

.. image:: images/cdfGeom.png
  :width: 400

``NegBinom``
~~~~~~~~~~~~~~~~~~~~

.. autoclass::  gemact.distributions.NegBinom
   :members:
   :inherited-members:

Parametrization
^^^^^^^^^^^^^^^^^^^^^^^^^^

For a more detailed explanation refer to the `SciPy Negative Binomial documentation <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.nbinom.html>`_, see :cite:t:`scipy`.

.. math:: f(k)=\left(\begin{array}{c}
      k+n-1 \\
      n-1
   \end{array}\right) p^{n}(1-p)^{k}
   :label: nbinomial

for :math:`k \geq 0,0<p \leq 1`

Example code on the usage of the NegBinom class::

    from gemact import distributions
    import numpy as np

    n = 10
    p = .5
    dist = distributions.NegBinom(n=n, p=p)
    seq = np.arange(0,100,.001)
    nsim = int(1e+05)

    # Compute the mean via pmf
    mean = np.sum(dist.pmf(seq)*seq)
    variance = np.sum(dist.pmf(seq)*(seq-mean)**2)

    print('Mean', mean)
    print('Theoretical mean', dist.mean())
    print('Variance', variance)
    print('Theoretical variance', dist.var())
    #compare with simulations
    print('Simulated mean', np.mean(dist.rvs(nsim)))
    print('Simulated variance', np.var(dist.rvs(nsim)))

Then, use ``matplotlib`` library to show the pmf and the cdf of a NegBinom::

    import matplotlib.pyplot as plt
    #pmf
    plt.vlines(x=seq, ymin=0, ymax=dist.pmf(seq))
    plt.title('Probability mass function, ~NegBinom(n=10,p=0.8)')

    #cdf
    plt.step(np.arange(-2,20),dist.cdf(np.arange(-2,20)),'-', where='pre')
    plt.title('Cumulative distribution function, ~NegBinom(n=10,p=0.8)')

.. image:: images/pmfNBinom.png
  :width: 400

.. image:: images/cdfNBinom.png
  :width: 400

``Logser``
~~~~~~~~~~~~~~~~~~~~

.. autoclass::  gemact.distributions.Logser
   :members:
   :inherited-members:

Parametrization
^^^^^^^^^^^^^^^^^^^^^^^^^^

For a more detailed explanation refer to the `SciPy Logarithmic distribution documentation <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.logser.html>`_, see :cite:t:`scipy`.


.. math:: f(k)=-\frac{p^{k}}{k \log (1-p)}
   :label: logser

Given :math:`0 < p < 1` and :math:`k \geq 1`.

Example code on the usage of the Logser class::

    from gemact import distributions
    import numpy as np

    dist=distributions.Logser(p=.5)
    seq=np.arange(0,30,.001)
    nsim=int(1e+05)

    # Compute the mean via pmf
    mean=np.sum(dist.pmf(seq)*seq)
    variance=np.sum(dist.pmf(seq)*(seq-mean)**2)

    print('Mean',mean)
    print('Variance',variance)
    #compare with simulations
    print('Simulated mean',np.mean(dist.rvs(nsim)))
    print('Simulated variance',np.var(dist.rvs(nsim)))

Then, use ``matplotlib`` library to show the pmf and the cdf of a Logser::

    import matplotlib.pyplot as plt

    #pmf
    plt.vlines(x=seq, ymin=0, ymax=dist.pmf(seq))
    plt.title('Probability mass function, ~Logser(p=.5)')

    #cdf
    plt.step(np.arange(0,20),dist.cdf(np.arange(0,20)),'-', where='pre')
    plt.title('Cumulative distribution function,  ~Logser(p=.5)')



.. image:: images/pmfLogser.png
  :width: 400

.. image:: images/cdfLogser.png
  :width: 400

``ZTPoisson``
~~~~~~~~~~~~~~~~~~~~

.. autoclass::  gemact.distributions.ZTPoisson
   :members:


Parametrization
^^^^^^^^^^^^^^^^^^^^^^^^^^

Let :math:`P(z)` denote the probability generating function of a Poisson distribution. 

:math:`p_{0}` is :math:`p_{0}=P[z=0]`.

The probability generating function of a Zero-Truncated poisson is defined in equation :eq:`ztpois`.

.. math:: P^{T}(z)=\frac{P(z)-p_{0}}{1-p_{0}}
   :label: ztpois


Example code on the usage of the ZTPoisson class::

    from gemact import distributions
    import numpy as np

    mu=2
    dist=distributions.ZTpoisson(mu=mu)
    seq=np.arange(0,30,.001)
    nsim=int(1e+05)

    # Compute the mean via pmf
    mean=np.sum(dist.pmf(seq)*seq)
    variance=np.sum(dist.pmf(seq)*(seq-mean)**2)

    print('Mean',mean)
    print('Variance',variance)
    #compare with simulations
    print('Simulated mean',np.mean(dist.rvs(nsim)))
    print('Simulated variance',np.var(dist.rvs(nsim)))

Then, use ``matplotlib`` library to show the pmf and the cdf of a ZTPoisson::

    import matplotlib.pyplot as plt

    #pmf
    plt.vlines(x=seq, ymin=0, ymax=dist.pmf(seq))
    plt.title('Probability mass function, ~ZTPoisson(mu=2)')
    #cdf
    plt.step(np.arange(0,20),dist.cdf(np.arange(0,20)),'-', where='pre')
    plt.title('Cumulative distribution function,  ~ZTPoisson(mu=2)')

.. image:: images/pmfZTPois.png
  :width: 400

.. image:: images/cdfZTPois.png
  :width: 400


``ZMPoisson``
~~~~~~~~~~~~~~~~~~~~

.. autoclass::  gemact.distributions.ZMPoisson
   :members:

Parametrization
^^^^^^^^^^^^^^^^^^^^^^^^^^

Let :math:`P(z)` denote the probability generating function of a Poisson distribution. 

:math:`p_{0}` is :math:`p_{0}=P[z=0]`.

The probability generating function of a Zero-Truncated poisson is defined in equation :eq:`zmpois`

.. math:: P^{M}(z)=p_{0}^{M}+\frac{1-p_{0}^{M}}{1-p_{0}}\left[P(z)-p_{0}\right]
   :label: zmpois

Example code on the usage of the ZMPoisson class::

    from gemact import distributions
    import numpy as np

    mu=2
    p0m=0.1
    dist=distributions.ZMPoisson(mu=mu,p0m=p0m)
    seq=np.arange(0,30,.001)
    nsim=int(1e+05)

    # Compute the mean via pmf
    mean=np.sum(dist.pmf(seq)*seq)
    variance=np.sum(dist.pmf(seq)*(seq-mean)**2)

    print('Mean',mean)
    print('Variance',variance)
    #compare with simulations
    print('Simulated mean',np.mean(dist.rvs(nsim)))
    print('Simulated variance',np.var(dist.rvs(nsim)))

Then, use ``matplotlib`` library to show the pmf and the cdf of a ZMPoisson::

    import matplotlib.pyplot as plt

    #pmf
    plt.vlines(x=seq, ymin=0, ymax=dist.pmf(seq))
    plt.title('Probability mass function, ~ZMPoisson(mu=2,p0m=.1)')

    #cdf
    plt.step(np.arange(0,20),dist.cdf(np.arange(0,20)),'-', where='pre')
    plt.title('Cumulative distribution function,  ~ZMPoisson(mu=2,p0m=.1)')

.. image:: images/pmfZMPois.png
  :width: 400

.. image:: images/cdfZMPois.png
  :width: 400

``ZTBinom``
~~~~~~~~~~~~~~~~~~~~

.. autoclass::  gemact.distributions.ZTBinom
   :members:

Parametrization
^^^^^^^^^^^^^^^^^^^^^^^^^^

Let :math:`P(z)` denote the probability generating function of a binomial distribution. 

:math:`p_{0}` is :math:`p_{0}=P[z=0]`.

The probability generating function of a Zero-Truncated binomial is defined in equation :eq:`ztbinom`.

.. math:: P^{T}(z)=\frac{P(z)-p_{0}}{1-p_{0}}
   :label: ztbinom

Example code on the usage of the ZTBinom class::

    from gemact import distributions
    import numpy as np

    n=10
    p=.2
    dist=distributions.ZTBinom(n=n, p=p)
    seq=np.arange(0,30,.001)
    nsim=int(1e+05)

    # Compute the mean via pmf
    mean=np.sum(dist.pmf(seq)*seq)
    variance=np.sum(dist.pmf(seq)*(seq-mean)**2)

    print('Mean',mean)
    print('Variance',variance)
    #compare with simulations
    print('Simulated mean',np.mean(dist.rvs(nsim)))
    print('Simulated variance',np.var(dist.rvs(nsim)))

Then, use ``matplotlib`` library to show the pmf and the cdf of a ZTBinom::

    import matplotlib.pyplot as plt

    #pmf
    plt.vlines(x=seq, ymin=0, ymax=dist.pmf(seq))
    plt.title('Probability mass function, ~ZTBinom(n=10 p=.2)')


    #cdf
    plt.step(np.arange(0,20),dist.cdf(np.arange(0,20)),'-', where='pre')
    plt.title('Cumulative distribution function,  ~ZTBinom(n=10 p=.2)')


.. image:: images/pmfZTBinom.png
  :width: 400

.. image:: images/cdfZTBinom.png
  :width: 400



``ZMBinom``
~~~~~~~~~~~~~~~~~~~~

.. autoclass::  gemact.distributions.ZMBinom
   :members:

Parametrization
^^^^^^^^^^^^^^^^^^^^^^^^^^

Let :math:`P(z)` denote the probability generating function of a binomial distribution. 

:math:`p_{0}` is :math:`p_{0}=P[z=0]`.

The probability generating function of a Zero-Truncated binomial is defined in equation :eq:`zmbinom`

.. math:: P^{M}(z)=p_{0}^{M}+\frac{1-p_{0}^{M}}{1-p_{0}}\left[P(z)-p_{0}\right]
   :label: zmbinom

Example code on the usage of the ZMBinom class::

    from gemact import distributions
    import numpy as np

    n=10
    p=.2
    p0m=.05
    dist=distributions.ZMBinom(n=n,p=p,p0m=p0m)
    seq=np.arange(0,100,.001)
    nsim=int(1e+05)

    # Compute the mean via pmf
    mean=np.sum(dist.pmf(seq)*seq)
    variance=np.sum(dist.pmf(seq)*(seq-mean)**2)

    print('Mean',mean)
    print('Variance',variance)
    #compare with simulations
    print('Simulated mean',np.mean(dist.rvs(nsim)))
    print('Simulated variance',np.var(dist.rvs(nsim)))

Then, use ``matplotlib`` library to show the pmf and the cdf of a ZMPoisson::

    import matplotlib.pyplot as plt

    #pmf
    plt.vlines(x=seq, ymin=0, ymax=dist.pmf(seq))
    plt.title('Probability mass function, ~ZMBinom(n=10,p=.5,p0m=.05)')

    #cdf
    plt.step(np.arange(0,20),dist.cdf(np.arange(0,20)),'-', where='pre')
    plt.title('Cumulative distribution function,  ~ZMBinom(n=10,p=.5,p0m=.05)')

.. image:: images/pmfZMBinom.png
  :width: 400

.. image:: images/cdfZMBinom.png
  :width: 400

``ZTGeom``
~~~~~~~~~~~~~~~~~~~~

.. autoclass::  gemact.distributions.ZTGeom
   :members:

Parametrization
^^^^^^^^^^^^^^^^^^^^^^^^^^

Let :math:`P(z)` denote the probability generating function of a geometric distribution. 

:math:`p_{0}` is :math:`p_{0}=P[z=0]`.

The probability generating function of a Zero-Truncated geometric is defined in equation :eq:`ztgeom`.

.. math:: P^{T}(z)=\frac{P(z)-p_{0}}{1-p_{0}}
   :label: ztgeom

Example code on the usage of the ZTGeom class::

    from gemact import distributions
    import numpy as np

    p=.2
    dist=distributions.ZTGeom(p=p)
    seq=np.arange(0,100,.001)
    nsim=int(1e+05)

    # Compute the mean via pmf
    mean=np.sum(dist.pmf(seq)*seq)
    variance=np.sum(dist.pmf(seq)*(seq-mean)**2)

    print('Mean',mean)
    print('Variance',variance)
    #compare with simulations
    print('Simulated mean',np.mean(dist.rvs(nsim)))
    print('Simulated variance',np.var(dist.rvs(nsim)))

Then, use ``matplotlib`` library to show the pmf and the cdf of a ZTGeom::

    import matplotlib.pyplot as plt

    #pmf
    plt.vlines(x=seq, ymin=0, ymax=dist.pmf(seq))
    plt.title('Probability mass function, ~ZTGeom(p=.2)')
    #cdf
    plt.step(np.arange(0,20),dist.cdf(np.arange(0,20)),'-', where='pre')
    plt.title('Cumulative distribution function,  ~ZTGeom(p=.2)')

.. image:: images/pmfZTGeom.png
  :width: 400

.. image:: images/cdfZTGeom.png
  :width: 400

``ZMGeom``
~~~~~~~~~~~~~~~~~~~~

.. autoclass::  gemact.distributions.ZMGeom
   :members:

Parametrization
^^^^^^^^^^^^^^^^^^^^^^^^^^

Let :math:`P(z)` denote the probability generating function of a geometric distribution. 

:math:`p_{0}` is :math:`p_{0}=P[z=0]`.

The probability generating function of a Zero-Modified geometric is defined in equation :eq:`zmgeom`.

.. math:: P^{M}(z)=p_{0}^{M}+\frac{1-p_{0}^{M}}{1-p_{0}}\left[P(z)-p_{0}\right]
   :label: zmgeom

Example code on the usage of the ZMGeom class::

    from gemact import distributions
    import numpy as np

    p=.2
    p0m=.01
    dist=distributions.ZMGeom(p=p,p0m=p0m)
    seq=np.arange(0,100,.001)
    nsim=int(1e+05)

    # Compute the mean via pmf
    mean=np.sum(dist.pmf(seq)*seq)
    variance=np.sum(dist.pmf(seq)*(seq-mean)**2)

    print('Mean',mean)
    print('Variance',variance)
    #compare with simulations
    print('Simulated mean',np.mean(dist.rvs(nsim)))
    print('Simulated variance',np.var(dist.rvs(nsim)))

Then, use ``matplotlib`` library to show the pmf and the cdf of a ZMGeom::

    import matplotlib.pyplot as plt

    #pmf
    plt.vlines(x=seq, ymin=0, ymax=dist.pmf(seq))
    plt.title('Probability mass function, ~ZMGeom(p=.2,p0m=.01)')

    #cdf
    plt.step(np.arange(0,20),dist.cdf(np.arange(0,20)),'-', where='pre')
    plt.title('Cumulative distribution function,  ~ZMGeom(p=.2,p0m=.01)')

.. image:: images/pmfZMGeom.png
  :width: 400

.. image:: images/cdfZMGeom.png
  :width: 400

``ZTNegBinom``
~~~~~~~~~~~~~~~~~~~~

.. autoclass::  gemact.distributions.ZTNegBinom
   :members:

Parametrization
^^^^^^^^^^^^^^^^^^^^^^^^^^

Let :math:`P(z)` denote the probability generating function of a negative binomial distribution. 

:math:`p_{0}` is :math:`p_{0}=P[z=0]`.

The probability generating function of a Zero-Truncated negative binomial is defined in equation :eq:`ztnbinom`.

.. math:: P^{T}(z)=\frac{P(z)-p_{0}}{1-p_{0}}
   :label: ztnbinom

Example code on the usage of the ZTNegBinom class::

    from gemact import distributions
    import numpy as np

    p=.2
    n=10
    dist=distributions.ZTNegBinom(p=p,n=n)
    seq=np.arange(0,100,.001)
    nsim=int(1e+05)

    # Compute the mean via pmf
    mean=np.sum(dist.pmf(seq)*seq)
    variance=np.sum(dist.pmf(seq)*(seq-mean)**2)

    print('Mean',mean)
    print('Variance',variance)
    #compare with simulations
    print('Simulated mean',np.mean(dist.rvs(nsim)))
    print('Simulated variance',np.var(dist.rvs(nsim)))

Then, use ``matplotlib`` library to show the pmf and the cdf of a ZTNbinom::

    import matplotlib.pyplot as plt

    #pmf
    plt.vlines(x=seq, ymin=0, ymax=dist.pmf(seq))
    plt.title('Probability mass function, ~ZTNegBinom(p=.2,n=10)')

    #cdf
    plt.step(np.arange(0,100),dist.cdf(np.arange(0,100)),'-', where='pre')
    plt.title('Cumulative distribution function,  ~ZTNegBinom(p=.2,n=10)')

.. image:: images/pmfZTnbinom.png
  :width: 400

.. image:: images/cdfZTNbinom.png
  :width: 400


``ZMNegBinom``
~~~~~~~~~~~~~~~~~~~~

.. autoclass::  gemact.distributions.ZMNegBinom
   :members:

Parametrization
^^^^^^^^^^^^^^^^^^^^^^^^^^

Let :math:`P(z)` denote the probability generating function of a negative binomial distribution. 

:math:`p_{0}` is :math:`p_{0}=P[z=0]`.

The probability generating function of a Zero-Modified negative binomial is defined in equation :eq:`zmnbinom`.

.. math:: P^{M}(z)=p_{0}^{M}+\frac{1-p_{0}^{M}}{1-p_{0}}\left[P(z)-p_{0}\right]
   :label: zmnbinom

Example code on the usage of the ZMNegBinom class::

    from gemact import distributions
    import numpy as np

    p=.2
    n=100
    p0M=.2
    dist=distributions.ZMNegBinom(n=50,
                                p=.8,
                                p0m=.01)
    seq=np.arange(0,100,.001)
    nsim=int(1e+05)

    # Compute the mean via pmf
    mean=np.sum(dist.pmf(seq)*seq)
    variance=np.sum(dist.pmf(seq)*(seq-mean)**2)

    print('Mean',mean)
    print('Variance',variance)
    #compare with simulations
    print('Simulated mean',np.mean(dist.rvs(nsim)))
    print('Simulated variance',np.var(dist.rvs(nsim)))

Then, use ``matplotlib`` library to show the pmf and the cdf of a ZMNbinom::

    import matplotlib.pyplot as plt

    #pmf
    plt.vlines(x=seq, ymin=0, ymax=dist.pmf(seq))
    plt.title('Probability mass function, ~ZMNegBinom(p=.2,n=10)')

    #cdf
    plt.step(np.arange(0,100),dist.cdf(np.arange(0,100)),'-', where='pre')
    plt.title('Cumulative distribution function,  ~ZMNegBinom(p=.2,n=10)')

.. image:: images/pmfZMNegBinom.png
  :width: 400

.. image:: images/cdfZMNegBinom.png
  :width: 400

``ZMLogser``
~~~~~~~~~~~~~~~~~~~~

.. autoclass::  gemact.distributions.ZMLogser
   :members:

Parametrization
^^^^^^^^^^^^^^^^^^^^^^^^^^

Let :math:`P(z)` denote the probability generating function of a logarithmic distribution. 

:math:`p_{0}` is :math:`p_{0}=P[z=0]`.

The probability generating function of a Zero-Modified logarithmic is defined in equation :eq:`zmlogser`.

.. math:: P^{M}(z)=p_{0}^{M}+\frac{1-p_{0}^{M}}{1-p_{0}}\left[P(z)-p_{0}\right]
   :label: zmlogser


Example code on the usage of the ZMLogser class::

    from gemact import distributions
    import numpy as np

    p=.2
    p0m=.01
    dist=distributions.ZMlogser(p=p,p0m=p0m)
    seq=np.arange(0,100,.001)
    nsim=int(1e+05)

    # Compute the mean via pmf
    mean=np.sum(dist.pmf(seq)*seq)
    variance=np.sum(dist.pmf(seq)*(seq-mean)**2)

    print('Mean',mean)
    print('Variance',variance)
    #compare with simulations
    print('Simulated mean',np.mean(dist.rvs(nsim)))
    print('Simulated variance',np.var(dist.rvs(nsim)))

Then, use ``matplotlib`` library to show the pmf and the cdf of a ZMlogser::

    import matplotlib.pyplot as plt

    #pmf
    plt.vlines(x=seq, ymin=0, ymax=dist.pmf(seq))
    plt.title('Probability mass function, ~ZMlogser(p=.2, p0m=.01)')
    #cdf
    plt.step(np.arange(0,10),dist.cdf(np.arange(0,10)),'-', where='pre')
    plt.title('Cumulative distribution function,  ~ZMlogser(p=.2, p0m=.01)')

.. image:: images/pmfZMLogser.png
  :width: 400

.. image:: images/cdfZMLogser.png
  :width: 400


``Gamma``
~~~~~~~~~~~~~~~~~~~~

.. autoclass::  gemact.distributions.Gamma
   :members:
   :inherited-members:

Parametrization
^^^^^^^^^^^^^^^^^^^^^^^^^^

For a better explanation refer to the `SciPy Gamma distribution documentation <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gamma.html>`_, see :cite:t:`scipy`.

.. math:: f(x, a)=\frac{x^{a-1} e^{-x}}{\Gamma(a)}
   :label: gamma

Given :math:`f(x, a)=\frac{x^{a-1} e^{-x}}{\Gamma(a)}` and :math:`x \geq 0, a>0`. :math:`\Gamma(a)` refers to the gamma function.

Example code on the usage of the Gamma class::

    from gemact import distributions
    import numpy as np

    a=2
    b=5
    dist=distributions.Gamma(a=a,
                             beta=b)
    seq=np.arange(0,100,.1)
    nsim=int(1e+05)

    print('Theorical mean',dist.mean())
    print('Theorical variance',dist.var())
    #compare with simulations
    print('Simulated mean',np.mean(dist.rvs(nsim)))
    print('Simulated variance',np.var(dist.rvs(nsim)))

Then, use ``matplotlib`` library to show the pmf and the cdf of a Gamma::

    import matplotlib.pyplot as plt

    #pmf
    plt.hist(dist.rvs(nsim),
             bins=100,
             density=True)
    plt.title('Probability density function, ~Gamma(a=2, beta=5)')
    #cdf
    plt.plot(seq,dist.cdf(seq))
    plt.title('Cumulative distribution function,  ~Gamma(a=2, beta=5)')

.. image:: images/pdfGamma.png
  :width: 400

.. image:: images/cdfGamma.png
  :width: 400

``Exponential``
~~~~~~~~~~~~~~~~~~~~

.. autoclass::  gemact.distributions.Exponential
   :members:
   :inherited-members:

Parametrization
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. math:: f(x, \theta)=\theta \cdot e^{-\theta x}
   :label: exponential

Given :math:`\theta \geq 0`. 

Example code on the usage of the Exponential class::

    from gemact import distributions
    import numpy as np

    theta=.1
    dist=distributions.Exponential(theta=theta)
    seq=np.arange(0,200,.1)
    nsim=int(1e+05)

    print('Theorical mean',dist.mean())
    print('Theorical variance',dist.var())
    #compare with simulations
    print('Simulated mean',np.mean(dist.rvs(nsim)))
    print('Simulated variance',np.var(dist.rvs(nsim)))

Then, use ``matplotlib`` library to show the pmf and the cdf of an Exponential::

    import matplotlib.pyplot as plt

    #pmf
    plt.hist(dist.rvs(nsim),
             bins=100,
             density=True)
    plt.title('Probability density function, ~Gamma(a=2, beta=5)')
    #cdf
    plt.plot(seq,dist.cdf(seq))
    plt.title('Cumulative distribution function,  ~Gamma(a=2, beta=5)')

.. image:: images/pdfExponential.png
  :width: 400

.. image:: images/cdfExponential.png
  :width: 400

``InvGauss``
~~~~~~~~~~~~~~~~~~~~

.. autoclass::  gemact.distributions.InvGauss
   :members:
   :inherited-members:

Parametrization
^^^^^^^^^^^^^^^^^^^^^^^^^^

For a better explanation refer to the `SciPy InvGauss distribution documentation <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.invgauss.html>`_, see :cite:t:`scipy`.

.. math:: f(x, \mu)=\frac{1}{\sqrt{2 \pi x^{3}}} \exp \left(-\frac{(x-\mu)^{2}}{2 x \mu^{2}}\right)
   :label: invgauss


``InvGamma``
~~~~~~~~~~~~~~~~~~~~

.. autoclass::  gemact.distributions.InvGamma
   :members:
   :inherited-members:

Parametrization
^^^^^^^^^^^^^^^^^^^^^^^^^^

For a better explanation refer to the `SciPy Inverse Gamma distribution documentation <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.invgamma.html>`_, see :cite:t:`scipy`.
:cite:t:`scipy`.

.. math:: f(x, a)=\frac{x^{-a-1}}{\Gamma(a)} \exp \left(-\frac{1}{x}\right)
   :label: invgamma

Given :math:`x>=0, a>0 `. :math:`\Gamma(a)` refers to the gamma function.

``GenPareto``
~~~~~~~~~~~~~~~~~~~~

.. autoclass::  gemact.distributions.GenPareto
   :members:
   :inherited-members:

Parametrization
^^^^^^^^^^^^^^^^^^^^^^^^^^

For a better explanation refer to the `SciPy documentation <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.genpareto.html>`_, see :cite:t:`scipy`.


.. math:: f(x, c)=(1+c x)^{-1-1 / c}
   :label: genpareto

Given :math:`x \geq 0` if :math:`c \geq 0`. :math:`0 \leq x \leq-1 / c if c<0`.

Example code on the usage of the GenPareto class::

    from gemact import distributions
    import numpy as np

    c=.4
    loc=0
    scale=0.25

    dist=distributions.Genpareto(c=c,
                                loc=loc,
                                scale=scale)

    seq=np.arange(0,100,.1)
    nsim=int(1e+05)

    print('Theorical mean',dist.mean())
    print('Theorical variance',dist.var())
    #compare with simulations
    print('Simulated mean',np.mean(dist.rvs(nsim)))
    print('Simulated variance',np.var(dist.rvs(nsim)))

Then, use ``matplotlib`` library to show the pmf and the cdf of a GenPareto::

    import matplotlib.pyplot as plt

    #pmf
    plt.hist(dist.rvs(nsim),
             bins=50,
             density=True)
    plt.title('Probability density function, ~GenPareto(a=2, beta=5)')

    #cdf
    plt.plot(seq,dist.cdf(seq))
    plt.title('Cumulative distribution function,  ~GenPareto(a=2, beta=5)')

.. image:: images/pdfGenPareto.png
  :width: 400

.. image:: images/cdfGenPareto.png
  :width: 400


``Pareto2``
~~~~~~~~~~~~~~~~~~~~

.. autoclass::  gemact.distributions.Pareto2
   :members:
   :inherited-members:

Parametrization
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. math:: f(x,s,m,p)=\frac{1/s}{\theta[1+(x-m) / p]^{1/s+1}}
   :label: pareto2

Given :math:`x>m,-\infty<m<\infty, 1/s>0  p>0`.

Example code on the usage of the Pareto2 class::

    from gemact import distributions
    import numpy as np

    s=2
    dist=distributions.Pareto2(s=.9)

    seq=np.arange(0,100,.1)
    nsim=int(1e+08)

``Pareto1``
~~~~~~~~~~~~~~~~~~~~

.. autoclass::  gemact.distributions.Pareto1
   :members:
   :inherited-members:

``Lognormal``
~~~~~~~~~~~~~~~~~~~~

.. autoclass::  gemact.distributions.Lognormal
   :members:
   :inherited-members:

Parametrization
^^^^^^^^^^^^^^^^^^^^^^^^^^

For a better explanation refer to the `SciPy Lognormal distribution documentation <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lognorm.html>`_, see :cite:t:`scipy`.

.. math:: f(x, s)=\frac{1}{s x \sqrt{2 \pi}} \exp \left(-\frac{\log ^{2}(x)}{2 s^{2}}\right)
   :label: lognorm

Given :math:`x>0, s>0`.

Example code on the usage of the Lognormal class::

    from gemact import distributions
    import numpy as np

    s=2
    dist=distributions.Lognorm(s=s)

    seq=np.arange(0,100,.1)
    nsim=int(1e+08)

    print('Theorical mean',dist.mean())
    print('Theorical variance',dist.var())
    #compare with simulations
    print('Simulated mean',np.mean(dist.rvs(nsim)))
    print('Simulated variance',np.var(dist.rvs(nsim)))

Then, use ``matplotlib`` library to show the pmf and the cdf of a LogNorm::

    import matplotlib.pyplot as plt
    #pmf
    plt.hist(dist.rvs(nsim),
             bins=20,
             density=True)
    plt.title('Probability density function, ~LogNormal(s=2)')
    #cdf
    plt.plot(seq,dist.cdf(seq))
    plt.title('Cumulative distribution function, ~LogNormal(s=2)')

.. image:: images/pdfLogNormal.png
  :width: 400

.. image:: images/cdfLogNormal.png
  :width: 400

``Burr12``
~~~~~~~~~~~~~~~~~~~~

.. autoclass::  gemact.distributions.Burr12
   :members:
   :inherited-members:

Parametrization
^^^^^^^^^^^^^^^^^^^^^^^^^^

For a better explanation refer to the `Scipy Burr12 distribution documentation <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.burr12.html>`_ , see :cite:t:`scipy`.

.. math:: f(x, c, d)=c d x^{c-1} /\left(1+x^{c}\right)^{d+1}
   :label: burr12

Given :math:`x>=0 and c, d>0`.

Example code on the usage of the Burr12 class::

    from gemact import distributions
    import numpy as np

    c=2
    d=3
    dist=distributions.Burr12(c=c,
                              d=d)

    seq=np.arange(0,100,.1)
    nsim=int(1e+05)

    print('Theorical mean',dist.mean())
    print('Theorical variance',dist.var())
    #compare with simulations
    print('Simulated mean',np.mean(dist.rvs(nsim)))
    print('Simulated variance',np.var(dist.rvs(nsim)))

Then, use ``matplotlib`` library to show the pmf and the cdf of a Burr12::

    import matplotlib.pyplot as plt

    #pmf
    plt.hist(dist.rvs(nsim),
             bins=100,
             density=True)
    plt.title('Probability density function, ~Burr12(c=c,d=d)')

    #cdf
    plt.plot(seq,dist.cdf(seq))
    plt.title('Cumulative distribution function,  ~Burr12(c=c,d=d)')

.. image:: images/pdfBurr12.png
  :width: 400

.. image:: images/cdfBurr12.png
  :width: 400

``Paralogistic``
~~~~~~~~~~~~~~~~~~~~

.. autoclass::  gemact.distributions.Paralogistic
   :members:
   :inherited-members:

Parametrization
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. math:: f(x, a)=a^2 x^{a-1} /\left(1+x^{a}\right)^{a+1}
   :label: paralogistic

Given :math:`x>=0 and a >0`.

Example code on the usage of the Paralogistic class::

    from gemact import distributions
    import numpy as np

    a=2

    dist=distributions.Paralogistic(a=a)

    seq=np.arange(0,100,.1)
    nsim=int(1e+05)

    print('Theorical mean',dist.mean())
    print('Theorical variance',dist.var())
    #compare with simulations
    print('Simulated mean',np.mean(dist.rvs(nsim)))
    print('Simulated variance',np.var(dist.rvs(nsim)))

Then, use ``matplotlib`` library to show the pmf and the cdf of a Burr12::

    import matplotlib.pyplot as plt

    #pmf
    plt.hist(dist.rvs(nsim),
             bins=100,
             density=True)
    plt.title('Probability density function, ~Paralogistic(a=a)')

    #cdf
    plt.plot(seq,dist.cdf(seq))
    plt.title('Cumulative distribution function,  ~Paralogistic(a=a)')

.. image:: images/pdfParalogistic.png
  :width: 400

.. image:: images/cdfParalogistic.png
  :width: 400

``Dagum``
~~~~~~~~~~~~~~~~~~~~

.. autoclass::  gemact.distributions.Dagum
   :members:
   :inherited-members:

Parametrization
^^^^^^^^^^^^^^^^^^^^^^^^^^

For a better explanation refer to the `SciPy Dagum distribution documentation <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mielke.html>`_ ,see :cite:t:`scipy`.

.. math:: f(x, k, s)=\frac{k x^{k-1}}{\left(1+x^{s}\right)^{1+k / s}}
   :label: dagum

Given :math:`x>0` , :math:`k,s>0`.


Example code on the usage of the Dagum class::

    from gemact import distributions
    import numpy as np

    d=2
    s=4.2
    dist=distributions.Dagum(d=d,
                             s=s)

    seq=np.arange(0,3,.001)
    nsim=int(1e+05)

    print('Theorical mean',dist.mean())
    print('Theorical variance',dist.var())
    #compare with simulations
    print('Simulated mean',np.mean(dist.rvs(nsim)))
    print('Simulated variance',np.var(dist.rvs(nsim)))

Then, use ``matplotlib`` library to show the pmf and the cdf of a Dagum::

    import matplotlib.pyplot as plt

    #pmf
    plt.hist(dist.rvs(nsim),
             bins=100,
             density=True)
    plt.title('Probability density function, ~Dagum(d=4,s=2)')

    #cdf
    plt.plot(seq,dist.cdf(seq))
    plt.title('Cumulative distribution function,  ~Dagum(d=c,s=4.2)')

.. image:: images/pdfDagum.png
  :width: 400

.. image:: images/cdfDagum.png
  :width: 400


``Inverse Paralogistic``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass::  gemact.distributions.InvParalogistic
   :members:
   :inherited-members:

Parametrization
^^^^^^^^^^^^^^^^^^^^^^^^^^


.. math:: f(x, b)=\frac{b x^{b-1}}{\left(1+x^{b}\right)^{2}}
   :label: invparalogistic

Given :math:`x>0` , :math:`b>0`.


Example code on the usage of the InvParalogistic class::

    from gemact import distributions
    import numpy as np

    b=2
    dist=distributions.InvParalogistic(b=b)

    seq=np.arange(0,3,.001)
    nsim=int(1e+05)

    print('Theorical mean',dist.mean())
    print('Theorical variance',dist.var())
    #compare with simulations
    print('Simulated mean',np.mean(dist.rvs(nsim)))
    print('Simulated variance',np.var(dist.rvs(nsim)))

Then, use ``matplotlib`` library to show the pmf and the cdf of a Dagum::

    import matplotlib.pyplot as plt

    #pmf
    plt.hist(dist.rvs(nsim),
             bins=100,
             density=True)
    plt.title('Probability density function, ~InvParalogistic(b=2)')

    #cdf
    plt.plot(seq,dist.cdf(seq))
    plt.title('Cumulative distribution function,  ~InvParalogistic(b=2)')

.. image:: images/pdfInvParalogistic.png
  :width: 400

.. image:: images/cdfInvParalogistic.png
  :width: 400

``Weibull``
~~~~~~~~~~~~~~~~~~~~

.. autoclass::  gemact.distributions.Weibull
   :members:
   :inherited-members:

Parametrization
^^^^^^^^^^^^^^^^^^^^^^^^^^

For a better explanation refer to the `SciPy Weibull distribution documentation <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.weibull_min.html>`_ , see :cite:t:`scipy`.

.. math:: f(x, c)=c x^{c-1} \exp \left(-x^{c}\right)
   :label: weibull_min

Given :math:`x>0` , :math:`c>0`.


Example code on the usage of the Weibull class::

    from gemact import distributions
    import numpy as np

    c=2.2
    dist=distributions.Weibull_min(c=c)

    seq=np.arange(0,3,.001)
    nsim=int(1e+05)

    print('Theorical mean',dist.mean())
    print('Theorical variance',dist.var())
    #compare with simulations
    print('Simulated mean',np.mean(dist.rvs(nsim)))
    print('Simulated variance',np.var(dist.rvs(nsim)))


Then, use ``matplotlib`` library to show the pmf and the cdf of a Weibull_min::

    import matplotlib.pyplot as plt

    #pmf
    plt.hist(dist.rvs(nsim),
             bins=100,
             density=True)
    plt.title('Probability density function, ~Weibull_min(s=2.2)')

    #cdf
    plt.plot(seq,dist.cdf(seq))
    plt.title('Cumulative distribution function,  ~Invweibull(c=7.2)')

.. image:: images/pdfWeibull_min.png
  :width: 400

.. image:: images/cdfWeibull_min.png
  :width: 400

``InvWeibull``
~~~~~~~~~~~~~~~~~~~~

.. autoclass::  gemact.distributions.InvWeibull
   :members:
   :inherited-members:

Parametrization
^^^^^^^^^^^^^^^^^^^^^^^^^^

For a better explanation refer to the`Scipy Inverse Weibull distribution documentation <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.invweibull.html>`_   :cite:t:`scipy` .

.. math:: f(x, c)=c x^{-c-1} \exp \left(-x^{-c}\right)
   :label: invweibull

Given :math:`x>0` , :math:`c>0`.


Example code on the usage of the InvWeibull class::

    from gemact import distributions
    import numpy as np

    c=7.2
    dist=distributions.Invweibull(c=c)

    seq=np.arange(0,3,.001)
    nsim=int(1e+05)

    print('Theorical mean',dist.mean())
    print('Theorical variance',dist.var())
    #compare with simulations
    print('Simulated mean',np.mean(dist.rvs(nsim)))
    print('Simulated variance',np.var(dist.rvs(nsim)))

Then, use ``matplotlib`` library to show the pmf and the cdf of a Invweibull::

    import matplotlib.pyplot as plt

    #pmf
    plt.hist(dist.rvs(nsim),
             bins=100,
             density=True)
    plt.title('Probability density function, ~Invweibull(c=7.2)')
    #cdf
    plt.plot(seq,dist.cdf(seq))
    plt.title('Cumulative distribution function,  ~Invweibull(c=7.2)')

.. image:: images/pdfInvweibull.png
  :width: 400

.. image:: images/cdfInvweibull.png
  :width: 400


``Beta``
~~~~~~~~~~~~~~~~~~~~

.. autoclass::  gemact.distributions.Beta
   :members:
   :inherited-members:

Parametrization
^^^^^^^^^^^^^^^^^^^^^^^^^^

For a better explanation refer to the `SciPy Beta distribution documentation <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.beta.html>`_, :cite:t:`scipy` .

.. math:: f(x, a, b)=\frac{\Gamma(a+b) x^{a-1}(1-x)^{b-1}}{\Gamma(a) \Gamma(b)}
   :label: beta

Given :math:`0<=x<=1, a>0, b>0` , :math:`\Gamma` is the gamma function.

Example code on the usage of the Beta class::

    from gemact import distributions
    import numpy as np

    a=2
    b=5
    dist=distributions.Beta(a=a,
                            b=b)

    seq=np.arange(0,1,.001)
    nsim=int(1e+05)

    print('Theorical mean',dist.mean())
    print('Theorical variance',dist.var())
    #compare with simulations
    print('Simulated mean',np.mean(dist.rvs(nsim)))
    print('Simulated variance',np.var(dist.rvs(nsim)))

Then, use ``matplotlib`` library to show the pmf and the cdf of a Beta::

    import matplotlib.pyplot as plt

    #pmf
    plt.hist(dist.rvs(nsim),
             bins=100,
             density=True)
    plt.title('Probability density function, ~Beta(a=2,b=5)')
    #cdf
    plt.plot(seq,dist.cdf(seq))
    plt.title('Cumulative distribution function,  ~Beta(a=2,b=5)')

.. image:: images/pdfBeta.png
  :width: 400

.. image:: images/cdfBeta.png
  :width: 400

``Loggamma``
~~~~~~~~~~~~~~~~~~~~

.. autoclass::  gemact.distributions.LogGamma
   :members:
   :inherited-members:

Parametrization
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. math:: f(x)=\frac{\lambda^\alpha}{\Gamma(\alpha)} \frac{(\log x)^{\alpha-1}}{x^{\lambda / 1}}
   :label: loggamma

where :math:`x>0, \alpha>0, \lambda >0` and :math:`\Gamma(\alpha)` is the Gamma function.

``GenBeta``
~~~~~~~~~~~~~~~~~~~~

.. autoclass::  gemact.distributions.GenBeta
   :members:
   :inherited-members:

Parametrization
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. math:: f(x)=\frac{\Gamma(shape_1+shape_2)}{\Gamma(shape_1) \Gamma(shape_2)}(\frac{x}{scale})^{shape_1 shape_3}\left(1-(\frac{x}{scale})^{shape_3}\right)^{shape_2-1} \frac{shape_3}{x}
   :label: genbeta

Given :math:`0<=x<=scale, shape_1>0, shape_2>0` , :math:`\Gamma` is the gamma function.

Example code on the usage of the GenBeta class::

    from gemact import distributions
    import numpy as np

    shape1=1
    shape2=2
    shape3=3
    scale=4

    dist=GenBeta(shape1=shape1,
                 shape2=shape2,
                 shape3=shape3,
                 scale=scale)

    seq=np.arange(0,1,.001)
    nsim=int(1e+05)

    print('Theorical mean',dist.mean())
    print('Theorical variance',dist.var())
    #compare with simulations
    print('Simulated mean',np.mean(dist.rvs(nsim)))
    print('Simulated variance',np.var(dist.rvs(nsim)))


Then, use ``matplotlib`` library to show the pmf and the cdf of a GenBeta::

    import matplotlib.pyplot as plt

    #pmf
    plt.hist(dist.rvs(nsim),
             bins=100,
             density=True)
    plt.title('Probability density function, ~GenBeta(shape1=1,shape2=2,shape3=3,scale=4)')
    #cdf
    plt.plot(np.arange(-1,8,.001),dist.cdf(np.arange(-1,8,.001)))
    plt.title('Cumulative distribution function,  ~GenBeta(shape1=1,shape2=2,shape3=3,scale=4)')


.. image:: images/pdfGenBeta.png
  :width: 400

.. image:: images/cdfGenBeta.png
  :width: 400

The ``copulas`` module
---------------------------

Guide to copulas

``ClaytonCopula``
~~~~~~~~~~~~~~~~~~~~

.. autoclass::  gemact.copulas.ClaytonCopula
   :members:

Parametrization
^^^^^^^^^^^^^^^^^^^^^^

Clayton copula cumulative density function:

.. math:: C_{\delta}^{C l}\left(u_{1}, \ldots, u_{d}\right)=\left(u_{1}^{-\delta}+u_{2}^{-\delta}+\cdots+u_{d}^{-\delta}-d+1\right)^{-1 / \delta}

Given :math:`u_{k} \in[0,1], k=1, \ldots, d` , :math:`\delta > 0` .


Example code on the usage of the ClaytonCopula class::

    from gemact import copulas

    clayton_c=copulas.ClaytonCopula(par=1.4,
                                    dim=2)
    clayton_c.rvs(size=100,
                  random_state= 42)



``FrankCopula``
~~~~~~~~~~~~~~~~~~~~


.. autoclass::  gemact.copulas.FrankCopula
   :members:

Parametrization
^^^^^^^^^^^^^^^^^^^^^^^^^^



Frank copula cumulative density function:

.. math:: C_{\theta}^{F r}\left(u_{1}, \ldots, u_{d}\right)=-\frac{1}{\delta} \log \left(1+\frac{\prod_{i}\left(\exp \left(-\delta u_{i}\right)-1\right)}{\exp (-\delta)-1}\right)

Given :math:`u_{k} \in[0,1], k=1, \ldots, d` , :math:`\delta >= 0` .


Example code on the usage of the FrankCopula class::

    from gemact import copulas
    frank_c=copulas.FrankCopula(par=1.4,
                                dim=2)
    frank_c.rvs(size=100,
                  random_state= 42)

``GumbelCopula``
~~~~~~~~~~~~~~~~~~~~

.. autoclass::  gemact.copulas.GumbelCopula
   :members:

Parametrization
^^^^^^^^^^^^^^^^^^^^^^^^^^

Gumbel copula cumulative density function:

.. math:: C_{\theta}^{G u}\left(u_{1}, \ldots, u_{d}\right)=\exp \left(-\left(\sum_{i}\left(-\log u_{i}\right)^{\delta}\right)^{1 / \delta}\right)

Given :math:`u_{k} \in[0,1], k=1, \ldots, d` , :math:`\delta >= 1` .


Example code on the usage of the GumbelCopula class::

    from gemact import copulas

    gumbel_c=copulas.GumbelCopula(par=1.4,
                                  dim=2)
    gumbel_c.rvs(size=100,
                  random_state= 42)


``GaussCopula``
~~~~~~~~~~~~~~~~~~~



.. autoclass::  gemact.copulas.GaussCopula
   :members:

Parametrization
^^^^^^^^^^^^^^^^^^^^^^^^^^

Gauss copula cumulative density function is based on the scipy function `multivariate_normal`.
For a better explanation refer to the `SciPy Multivariate Normal documentation <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multivariate_normal.html>`_, :cite:t:`scipy` .

.. math:: f(x)=\frac{1}{\sqrt{(2 \pi)^{k} \operatorname{det} \Sigma}} \exp \left(-\frac{1}{2}(x-\mu)^{T} \Sigma^{-1}(x-\mu)\right)


Example code on the usage of the GaussCopula class::

    from gemact import copulas

    corr_mx=np.array([[1,.4],[.4,1]])

    gauss_c=copulas.GaussCopula(corr=corr_mx)
    gauss_c.rvs(size=100,
                  random_state= 42)

``TCopula``
~~~~~~~~~~~~~~~~~~~

.. autoclass::  gemact.copulas.TCopula
   :members:

Parametrization
^^^^^^^^^^^^^^^^^^^^^^^^^^


T copula cumulative density function:

.. math:: C_{v, C}^{t}\left(u_{1}, \ldots, u_{d}\right)=\boldsymbol{t}_{v, C}\left(t_{v}^{-1}\left(u_{1}\right), \ldots, t_{v}^{-1}\left(u_{d}\right)\right)

Given :math:`u_{k} \in[0,1], k=1, \ldots, d`. :math:`t_{v}^{-1}` is the inverse student’s t function and :math:`\boldsymbol{t}_{v, C}` is the cumulative distribution function of the multivariate student’s t distribution with a given mean and correlation matrix C.
Degrees of freedom :math:`v>0` .

Example code on the usage of the TCopula class::

    from gemact import copulas

    corr_mx=np.array([[1,.4],[.4,1]])

    t_c=copulas.TCopula(corr=corr_mx,
                        df=5)
    t_c.rvs(size=100,
            random_state= 42)

``Independence Copula``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass::  gemact.copulas.IndependenceCopula
   :members:

Parametrization
^^^^^^^^^^^^^^^^^^^^^^^^^^


Independence Copula.

.. math:: C^{U}\left(u_{1}, \ldots, u_{d}\right)=\prod^d_{k=1} u_k

Given :math:`u_{k} \in[0,1], k=1, \ldots, d`.

Example code on the usage of the IndependentCopula class::

    from gemact import copulas

    i_c = copulas.IndependentCopula(dim=2)
    i_c.rvs(1,random_state=42)

``Fréchet–Hoeffding Lower Bound``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass::  gemact.copulas.FHLowerCopula
   :members:

Parametrization
^^^^^^^^^^^^^^^^^^^^^^^^^^

The Fréchet–Hoeffding Lower Bound Copula.

.. math:: C^{FHL}\left(u_{1},u_{2}\right)=\max (u+v-1,0)

Given :math:`u_{k} \in[0,1], k=1, 2`.

Example code on the usage of the FHLowerCopula class::

    from gemact import copulas

    fhl_c = copulas.FHLowerCopula
    fhl_c.rvs(10,
            random_state=42)



``Fréchet–Hoeffding Upper Bound``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass::  gemact.copulas.FHUpperCopula
   :members:

Parametrization
^^^^^^^^^^^^^^^^^^^^^^^^^^

The Fréchet–Hoeffding Upper Bound Copula.

.. math:: C^{FHU}\left(u_{1},u_{2}\right)=\min (u,v)

Given :math:`u_{k} \in[0,1], k=1, 2`.

Example code on the usage of the FHUpperCopula class::

    from gemact import copulas

    fhu_c = copulas.FHUpperCopula
    fhu_c.rvs(10,
            random_state=42)





References
=================
.. bibliography::



