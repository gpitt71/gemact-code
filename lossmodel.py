import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from . import helperfunctions as hfns
from . import distributions as distributions

from twiggy import quick_setup,log
quick_setup()
logger = log.name('lossmodel')

## Actual
class Frequency:
    """
    This class computes the frequency underlying the collective risk model.

    :param \**kwargs:
    See below

    :Keyword Arguments:
        * *fdist* (``str``) --
          Name of the frequency distribution. See the distribution module for the discrete distributions supported in GEMAct.
        * *fpar* (``dict``) --
          parameters of the frequency distribution.

    """

    def __init__(self,**kwargs):

        ## initializing the attribute
        # self.stv=structureVar
        # self.stvpar=svpar
        self.fdist = kwargs['fdist']
        self.fpar = kwargs['fpar']

        try:
            self.fq = self.fdist(**self.fpar)
        except:
            logger.error("The frequency distribution is not parametrized correctly \n See our documentation: https://gem-analytics.github.io/gemact/")

        if self.fdist.name in ['ZMpoisson','ZMbinom','ZMgeom','ZMnbinom']:
            self.p0= self.fpar['p0M']
        else:
            self.p0=None
    #
    #
    # @property
    # def stvpar(self):
    #     return self.__stvpar
    #
    # @stvpar.setter
    # def stvpar(self, var):
    #     if not var is None:
    #         assert isinstance(var,dict), logger.error('The structure variable distribution parameters must be given as a dictionary')
    #     self.__stvpar = var

    @property
    def fdist(self):
        return self.__fdist

    @fdist.setter
    def fdist(self, var):
        #this is the set of possible distributions
        frdict = {'poisson': distributions.Poisson,
                  'ZTpoisson':distributions.ZTpoisson,
                  'ZMpoisson': distributions.ZMpoisson,
                  'binom': distributions.Binom,
                  'ZTbinom': distributions.ZTbinom,
                  'ZMbinom': distributions.ZMbinom,
                  'geom':distributions.Geom,
                  'ZTgeom': distributions.ZTgeom,
                  'ZMgeom': distributions.ZMgeom,
                  'nbinom':distributions.Nbinom,
                  'ZTnbinom': distributions.ZTnbinom,
                  'ZMnbinom': distributions.ZMnbinom
                  }

        assert var in frdict.keys(), "%r is not supported \n See our documentation: https://gem-analytics.github.io/gemact/" % var
        # if self.stvpar is None:
        self.__fdist = frdict[var]
        # else:
        # assert var in ['poisson','ZTpoisson','ZMpoisson'], logger.error('Models with structure variable are available only for poisson, ZTpoisson and ZMpoisson distributions')
        # if var == 'poisson':
        #     self.__fdist = frdict['nbinom']
        # elif var == 'ZTpoisson':
        #     logger.info(
        #         "The zero-truncation with the presence of a structure variable is applied to the a posteriori model")
        #     self.__fdist = frdict['ZTnbinom']
        # elif var == 'ZMpoisson':
        #     logger.info(
        #         "The zero-modification with the presence of a structure variable is applied to the a posteriori model")
        #     self.__fdist = frdict['ZMnbinom']

            #caso binomiale e geometrica dobbiamo parlare perchè nel caso serve ricavare tutte le relative ab0 o scegliere di non portare
            #recurrent e fft

    @property
    def fpar(self):
        return self.__fpar

    @fpar.setter
    def fpar(self, var):
        assert isinstance(var,dict), logger.info('The frequency distribution parameters must be given as a dictionary')
        if self.fdist.name in ['geom','ZTgeom','ZMgeom']:
            if not 'loc' in var.keys():
                var['loc'] = -1
        self.__fpar = var
        # if self.stvpar is None:
        #     self.__fpar = var
        # else:
        #         if self.fdist in ['poisson','ZMpoisson','ZTpoisson']: ## modificato
        #             try:
        #                 self.__fpar={'n':self.stvpar['a'],'p':self.stvpar['scale']/(1+self.stvpar['scale'])}
        #             except:
        #                 logger.error("You did not parametrize the distribution correctly. Check the gamma parameters are provided as 'a' and 'scale'" )

    @property
    def p0(self):
        return self.__p0

    @p0.setter
    def p0(self, var):
        if not var is None:
            try:
                if not isinstance(var, float) or isinstance(var, int):
                    logger.warning('Converting %s to a float'%var)
                    var = float(var)
                assert var <= 1 and var >= 0,logger.error('Please provide a value for p0 between 0 and 1')
                if var == 0:
                    logger.info('As you provided p0=%s, this is equivalent to a Zero-Truncated Poisson'%var)
                self.__p0=var
            except:
                logger.error('Please provide p0 as an integer or a float between 0 and 1')
        else:
            self.__p0 = var

    # def fgp(self,f,a=0,b=0):
    #     """
    #     It computes the probability generating function of the discrete distribution given the (a,b,1) parametrization.
    #
    #     :param f: argument of the probability generating function.
    #               It will be severity discrete distribution when computing the collective risk model.
    #     :type f: numpy.ndarray
    #     :param a: a parameter of the distribution.
    #     :param b: b parameter of the distribution.
    #     :return: probability generated in f.
    #     """
    #
    #     if self.fdist.name == 'poisson':
    #         return np.exp(b*(f-1))
    #     if self.fdist.name== 'ZTpoisson':
    #         return (np.exp(b*f)-1)/(np.exp(b)-1)
    #     if self.fdist.name== 'ZMpoisson':
    #         return self.p0+(1-self.p0)*((np.exp(b*f)-1)/(np.exp(b)-1))
    #
    #     if self.fdist.name =='binom':
    #         return (1+a/(a-1)*(f-1))**(-b/a-1)
    #     if self.fdist.name =='ZTbinom':
    #         return ((1+a/(a-1)*(f-1))**(-b/a-1)-(1-a)**(b/a+1))/(1-(1-a)**(b/a+1))
    #     if self.fdist.name== 'ZMbinom':
    #         return self.p0+(1-self.p0)*((1+a/(a-1)*(f-1))**(-b/a-1)-(1-a)**(b/a+1))/(1-(1-a)**(b/a+1))
    #
    #     if self.fdist.name =='geom':
    #         return (1-a/(1-a)*(f-1))**(-1)
    #     if self.fdist.name =='ZTgeom':
    #         return (1/(1-(f-1)/(1-a))-1+a)/a
    #     if self.fdist.name == 'ZMgeom':
    #         return self.p0+(1-self.p0)*(1/(1-(f-1)/(1-a))-1+a)/a
    #
    #     if self.fdist.name =='nbinom':
    #         return (1-a/(1-a)*(f-1))**(-b/a-1)
    #     if self.fdist.name =='ZTnbinom':
    #         return ((1/(1-(f-1)*a/(1-a)))**(b/a+1)-(1-a)**(b/a+1))/(1-(1-a)**(b/a+1))
    #     if self.fdist.name == 'ZMnbinom':
    #         return self.p0+(1-self.p0)*((1/(1-(f-1)*a/(1-a)))**(b/a+1)-(1-a)**(b/a+1))/(1-(1-a)**(b/a+1))

    def abp0g0(self,fj):
        """
        It returns the parameters of the discrete distribution according to the (a,b,k) parametrization,
        the probability generating function computed in zero given the discrete severity fj,
        and the probability of the distribution in zero.

        :param fj: discretized severity distribution probabilities.
        :type fj: numpy.ndarray

        """
        a,b,p0=self.fq.abk()
        return a,b,p0,[self.fq.pgf(fj[0])]

        # if self.fdist.name in ['poisson', 'binom', 'geom', 'nbinom'] and (self.p0 is not None):
        #     logger.info(
        #         "The parameter p0 is arbitrary only for Zero-Modified distributions. \n See our documentation: https://UnGiornoCiFacciamoPureIlSito.html ")
        #
        # # it returns a,b, p0,g
        # if self.fdist.name == 'poisson':
        #     return 0, \
        #            self.fpar['mu'], \
        #             np.array(np.exp(-self.fpar['mu'])),\
        #            [self.fgp(f=fj[0],a=0,b=self.fpar['mu'])]
        #
        # if self.fdist.name == 'ZTpoisson':
        #     return 0, \
        #            self.fpar['mu'], \
        #            0,\
        #            [self.fgp(f=fj[0],a=0,b=self.fpar['mu'])]
        #
        # if self.fdist.name == 'ZMpoisson':
        #     return 0, \
        #            self.fpar['mu'], \
        #            self.p0, \
        #            [self.fgp(f=fj[0],a=0,b=self.fpar['mu'])]
        #
        # if self.fdist.name == 'binom':
        #     return -self.fpar['p']/(1-self.fpar['p']), \
        #            (self.fpar['n']+1)*(self.fpar['p']/(1-self.fpar['p'])),\
        #            (1-self.fpar['p'])**self.fpar['n'],\
        #            [self.fgp(f=fj[0],a=-self.fpar['p']/(1-self.fpar['p']),b=(self.fpar['n']+1)*(self.fpar['p']/(1-self.fpar['p'])))]
        #
        # if self.fdist.name == 'ZTbinom':
        #     return -self.fpar['p'] / (1 - self.fpar['p']), \
        #            (self.fpar['n'] + 1) * (self.fpar['p'] / (1 - self.fpar['p'])), \
        #            0,\
        #            [self.fgp(f=fj[0],a=-self.fpar['p']/(1-self.fpar['p']),b=(self.fpar['n']+1)*(self.fpar['p']/(1-self.fpar['p'])))]
        #
        # if self.fdist.name == 'ZMbinom':
        #     return -self.fpar['p'] / (1 - self.fpar['p']), \
        #            (self.fpar['n'] + 1) * (self.fpar['p'] / (1 - self.fpar['p'])), \
        #            self.p0,\
        #            [self.fgp(f=fj[0],a=-self.fpar['p']/(1-self.fpar['p']),b=(self.fpar['n']+1)*(self.fpar['p']/(1-self.fpar['p'])))]
        #
        # if self.fdist.name == 'geom':
        #     return 1-self.fpar['p'], \
        #            0,\
        #            np.array([((1-self.fpar['p'])/self.fpar['p'])**-1]), \
        #            [self.fgp(f=fj[0], a=1-self.fpar['p'],b=0)]
        #
        # if self.fdist.name == 'ZTgeom':
        #     return 1-self.fpar['p'], \
        #            0,\
        #            np.array([0]),\
        #            [self.fgp(f=fj[0], a=1-self.fpar['p'],b=0)]
        #
        # if self.fdist.name == 'ZMgeom':
        #     return 1-self.fpar['p'], \
        #            0, \
        #            np.array([self.p0]), \
        #            [self.fgp(f=fj[0], a=1-self.fpar['p'],b=0)]
        #
        # if self.fdist.name == 'nbinom':
        #     return 1-self.fpar['p'], \
        #            (self.fpar['n']-1)*(1-self.fpar['p']), \
        #            np.array([self.fpar['p']**self.fpar['n']]), \
        #            [self.fgp(f=fj[0], a=1-self.fpar['p'],b=(self.fpar['n']-1)*(1-self.fpar['p']))]
        #
        # if self.fdist.name == 'ZTnbinom':
        #     return 1-self.fpar['p'], \
        #            (self.fpar['n']-1)*(1-self.fpar['p']),\
        #            np.array([0]), \
        #            [self.fgp(f=fj[0], a=1 - self.fpar['p'], b=(self.fpar['n'] - 1) * (1 - self.fpar['p']))]
        #
        # if self.fdist.name == 'ZMnbinom':
        #     return 1-self.fpar['p'], \
        #            (self.fpar['n']-1)*(1-self.fpar['p']),\
        #            np.array([self.p0]), \
        #            [self.fgp(f=fj[0], a=1 - self.fpar['p'], b=(self.fpar['n'] - 1) * (1 - self.fpar['p']))]


class Severity:
    """
    Computes the severity distribution of the underlying collective risk model.

    :param h: severity discretization step.
    :type h: ``float``
    :param m: number of points of the discrete severity.
    :type m: ``int``
    :param d: deductible, it is set to zero when not present.
    :type d: ``int`` or ``float``
    :param u: upper priority, individual.
    :type u: ``int`` or ``float``
    :param \**kwargs:
        See below

    :Keyword Arguments:
        * *sdist* (``str``) --
          Name of the severity distribution. See the distribution module for the continuous distributions supported in GEMAct.
        * *spar* (``dict``) --
          parameters of the severity distribution.
    """
    def __init__(self, h=None, m=None,d=0, u=float('inf'),**kwargs):

        ## initializing the attribute
        self.sdist = kwargs['sdist']
        self.spar = kwargs['spar']
        self.d=d
        self.u=u
        self.m = m
        self.h = h

        if 'loc' in self.spar.keys():
            self.loc_ = self.spar['loc']
        else:
            self.loc_=.0

        try:
            self.sv = self.sdist(**self.spar)
        except:
            print(
                "The severity distribution is not parametrized correctly \n See our documentation: https://gem-analytics.github.io/gemact/ ")


    @property
    def sdist(self):
        return self.__sdist

    @sdist.setter
    def sdist(self, var):
        # this is the set of possible distributions
        svdict = {'gamma': distributions.Gamma,
                  'lognorm':distributions.Lognorm,
                  'exponential':distributions.Exponential,
                  'genpareto': distributions.Genpareto,
                  'burr12': distributions.Burr12,
                  'dagum': distributions.Dagum,
                  'invgamma': distributions.Invgamma,
                  'weibull_min':distributions.Weibull_min,
                  'invweibull':distributions.Invweibull,
                  'beta':distributions.Beta}
        assert var in svdict.keys(), "%r is not supported \n See our documentation: https://gem-analytics.github.io/gemact/ " % var
        self.__sdist = svdict[var]

    @property
    def spar(self):
        return self.__spar

    @spar.setter
    def spar(self, var):
        assert isinstance(var, dict), 'The severity distribution parameters must be given as a dictionary'
        self.__spar = var

    @property
    def d(self):
        return self.__d

    @d.setter
    def d(self, var):
        assert var >= 0, logger.error('The lower priority for the severity must be higher or equal to zero')
        self.__d = float(var)

    @property
    def u(self):
        return self.__u

    @u.setter
    def u(self, var):
        assert var > 0, logger.error('The upper priority for the severity strictly positive')
        self.__u = float(var)

    @property
    def m(self):
        return self.__m

    @m.setter
    def m(self, var):
        if var is None:
            self.__m = var
        else:
            try:
                if not type(var) == type(int(var)):
                    assert var > 0, logger.error('Please specify the severity nodes as a postive integer')
                    logger.warning('I turned the nodes you gave me into an integer')
                    out= int(var)
                else:
                    out = var
                if self.u==float('inf'):
                   self.__m=out
                else:
                   self.__m = out-1
            except:
                logger.error('Please specify the severity nodes as a postive integer')

    @property
    def h(self):
        return self.__h

    @h.setter
    def h(self, var):
        if var is None:
            assert(self.m is None),logger.error("You provided the severity number of nodes, please provide a value for h")
            self.__h = var
        else:
            if self.u != float('inf'):
                logger.info('As you specified an upper priority for the severity distribution, h is set to (u-d)/m')
                self.__h = (self.u -self.d)/ (self.m+1)
            else:
                assert var > 0, logger.error('The discretization step must be set to a positive float')
                self.__h = float(var)

    def massDispersal(self):
        """
        Severity discretization according to the mass dispersal method.

        :return: discrete severity
        :rtype: ``numpy.ndarray``
        """
        f0=(self.sv.cdf(self.d+self.h/2)-self.sv.cdf(self.d))/(1-self.sv.cdf(self.d))
        seq_=np.arange(0,self.m)+.5
        fj = np.append(f0, (self.sv.cdf(self.d + seq_ * self.h)[1:] - self.sv.cdf(self.d + seq_ * self.h)[:-1]) / (
                    1 - self.sv.cdf(self.d)))
        if self.u != float('inf'):
            fj = np.append(fj, (1 - self.sv.cdf(self.u - self.h / 2)) / (1 - self.sv.cdf(self.d)))

        seq=self.loc_ + np.arange(0, self.m) * self.h

        if self.u!= float('inf'):
            seq=np.concatenate((seq,[seq[-1]+self.h]))

        return {'fj':fj,'severity_seq':seq}

    def __umhoverd(self):
        """
        Helper for the local moments discretization.
        In case an upper priority on the severity is placed, the last point of the sequence is adjusted by this method.

        :return: probability mass in (u-d/h)*m
        :rtype: ``numpy.ndarray``
        """
        if self.u == float('inf'):
            return np.array([])
        else:
            return (self.sv.Emv(self.u - self.loc_)-self.sv.Emv(self.u - self.loc_ - self.h))/(self.h*self.sv.den(d=self.d,loc=self.loc_))


    def localMoments(self):
        """
        Severity discretization according to the local moments method.

        :return: discrete severity.
        :rtype: ``numpy.ndarray``

        """
        v_ = self.__umhoverd()
        n = self.sv.Emv(self.d + self.h - self.loc_) - self.sv.Emv(self.d - self.loc_)
        den=self.h*self.sv.den(d=self.d,loc=self.loc_)
        nj = 2 * self.sv.Emv(self.d - self.loc_ + np.arange(1, self.m) * self.h) - self.sv.Emv(
            self.d - self.loc_ + np.arange(0, self.m - 1) * self.h) - self.sv.Emv(
            self.d - self.loc_ + np.arange(2, self.m + 1) * self.h)

        ## keep the following two lines outside the if cycles
        fj = np.append(1 - n / den, nj / den)

        seq=self.loc_ + np.arange(0, self.m) * self.h
        if self.u!= float('inf'):
            seq=np.concatenate((seq,[seq[-1]+self.h]))
        return {'fj':np.append(fj, v_), 'severity_seq': seq}


class LossModel(Frequency,Severity):
    """
    It computes the collective risk model given the frequency and severity assumptions.
    It allows for (re)insurance pricing and risk modeling.

    :param method: computational method for the aggregate cost. Available choices are Fast Fourier Transform ('fft'), recursive method ('recursive') and Monte Carlo simulation ('mcsimulation').
    :type method: ``str``
    :param nsim: number of simulations, MC.
    :type nsim: ``int``
    :param tilt: whether tilting is present or not when computing the discrete Fourier transform.
    :type tilt: ``bool`` or ``float``
    :param setseed: seed parameter for MC.
    :type setseed: ``int``
    :param discretizationmethod:  discretization method for the severity. Default is 'localmoments'.
    :type discretizationmethod: ``str``
    :param n: number of final points in the aggregate cost sequence.
    :type n: ``int``
    :param L: stop loss priority.
    :type L: ``float``
    :param K: number of reinstatements layers.
    :type K: ``float``
    :param c: reinstatements layers loading.
    :type c: ``float``
    :param alphaqs: quota share.
    :type alphaqs: ``float``

    Inherits from Frequency:

    :param fdist: name of the frequency distribution. See the distributions module for the discrete distributions supported in GEMAct.
    :type fdist: ``str``
    :param fpar: parameters of the frequency distribution.
    :type fpar: ``dict``

    Inherits from Severity:

    :param h: severity discretization step.
    :type h: ``float``
    :param m: number of points of the discrete severity.
    :type m: ``int``
    :param d: deductible, it is set to zero when not present.
    :type d: ``int`` or ``float``
    :param u: upper priority, individual.
    :type u: ``int`` or ``float``
    :param sdist: name of the severity distribution. See the distribution module for the continuous distributions supported in GEMAct.
    :type sdist: ``str``
    :param spar: parameters of the severity distribution.
    :type spar: ``dict``

    """
    def __init__(self, method=None,
                 nsim=10000,
                 tilt=True,
                 setseed=42,
                 discretizationmethod = 'localmoments',
                 n=None,
                 L = 0,
                 K = float('inf'),
                 c = 0,
                 alphaqs=1,
                 **kwargs):
        Frequency.__init__(self,**kwargs)
        Severity.__init__(self, **kwargs)
        self.__nu = 1-self.sv.cdf(self.d)

        if self.d > 0.:
            self.__franchise()
            self.fq = self.fdist(**self.fpar)

        self.method = method
        self.nsim = nsim
        self.setseed = setseed
        self.discretizationmethod = discretizationmethod
        self.n = n
        self.tilt = tilt
        self.L=L
        self.K = K
        self.c=c
        self.alphaqs=alphaqs
        #filled attributes
        self.lossModel = self.lossModelComputation()


        if self.method in ['mcsimulation']: #,'qmcsimulation'
            self.__algtype='simulative'
        else:
            self.__algtype = 'notsimulative'

    @property
    def method(self):
        return self.__method

    @method.setter
    def method(self, var):
        if var is None:
            self.__method = 'mcsimulation'
        else:
            assert isinstance(var,str), '%r is not a string' %var
            assert var.lower() in ['mcsimulation','recursive','fft'], "%r is not supported in our package \n See our documentation: https://gem-analytics.github.io/gemact/" % var
            # ,'qmcsimulation'
            if not var.lower() == var:
                logger.warning("What did you mean with %r? 'method' is set to %r." % (var, var.lower()))
            self.__method = var.lower()

    @property
    def nsim(self):
        return self.__nsim

    @nsim.setter
    def nsim(self, var):
        try:
            var=int(var)
            assert isinstance(var, int)
        except:
            logger.error('nsim=%r must be set to an integer' % var)
        self.__nsim= var


    @property
    def setseed(self):
        return self.__setseed

    @setseed.setter
    def setseed(self, var):
        assert isinstance(var, int), 'setseed=%r must be set to an integer' % var
        self.__setseed = var

    @property
    def discretizationmethod(self):
        return self.__discretizationmethod

    @discretizationmethod.setter
    def discretizationmethod(self, var):
        assert isinstance(var,str), '%r is not a string' %var
        assert var.lower() in ['localmoments','massdispersal'], '%r is not a supported discretization method must be' %var
        if not var.lower() == var:
            logger.warning("What did you mean with %r? 'discretizationmethod' is set to %r." % (var, var.lower()))
        self.__discretizationmethod = var.lower()
    @property
    def n(self):
        return self.__n

    @n.setter
    def n(self, var):
        if var is None:
            self.__n = var
        else:
            try:
                if not type(var) == type(int(var)):
                    assert var > 0, 'Please specify the aggregate model nodes as a postive integer'
                    logger.warning('I turned the nodes you gave me into an integer')
                    self.__n = int(var)
                else:
                    self.__n = var
            except:
                logger.error('Please specify the aggregate model nodes as a postive integer')

    @property
    def tilt(self):
        return self.__tilt

    @tilt.setter
    def tilt(self, var):
        if isinstance(var,bool):
            self.__tilt = var
        else:
            self.__tilt = float(var)

    @property
    def L(self):
        return self.__L

    @L.setter
    def L(self, var):
        assert isinstance(var, float) or isinstance(var, int), print(
            'Please provide the aggregate priority as a float or an integer')
        assert var >= 0, print('The aggregate priority must be positive')
        self.__L = var

    @property
    def K(self):
        return self.__K

    @K.setter
    def K(self, var):
        assert isinstance(var, float) or isinstance(var, int), print(
            'Please provide the number of reinstatements as a positive float or a positive integer')
        assert var >= 0, print('The aggregate priority must be positive')
        self.__K = var

    @property
    def c(self):
        return self.__c

    @c.setter
    def c(self, var):
        if isinstance(var, float) or isinstance(var, int):
            assert var >= 0, logger.error('The loading for the reinstatement layers must be positive')
            if self.K == float('inf'):
                out =np.repeat(var,0)
            else:
                out = np.repeat(var, self.K)
        else:
            if isinstance(var, np.ndarray):
                if self.K != float('inf'):
                    assert var.shape[0] == self.K, logger.error(
                        'The premium share for the reinstatement layers should be %s dimensional' % str(self.K))
                    assert np.sum(var < 0) == 0, logger.error('The premium share for the reinstatement layers must be positive')
                else:
                    logger.info("You specified an array for the layers loading which will be disregarded as: \n K=float('inf') is a stop loss contract")
                out=var
        self.__c = out

    @property
    def alphaqs(self):
        return self.__alphaqs

    @alphaqs.setter
    def alphaqs(self, var):
        assert isinstance(var,int) or isinstance(var,float), logger.error('The retained percentage of the quota share treaty must be an integer or a float')
        assert var <=1 and var>0, logger.error('The retained percentage of the quota share treaty must be in (0,1]')
        self.__alphaqs=var

    def lossModelComputation(self):
        """
        It returns the aggregate cost computed with the proper method.

        :return: aggregate cost empirical pdf, aggregate cost empirical cdf,aggregate cost values sequence
        :rtype: ``dict``
        """
        if self.method == 'mcsimulation':
            return self.__MCsimulation()
        if self.method == 'recursive':
            return self.__recursive()
        # if self.method == 'qmcsimulation':
        #     return self.__QMCsimulation()
        if self.method == 'fft':
            return self.__fft()

    def __franchise(self):
        """
        Frequency parameters correction in case of franchise.

        :return: correct frequency parameters.
        :rtype: ``dict``
        """

        if self.fdist.name == 'poisson' or self.fdist.name == 'ZTpoisson':
            self.fpar['mu']=self.__nu*self.fpar['mu']

        if self.fdist.name == 'ZMpoisson':
            self.fpar['p0M']=(self.fpar['p0M']-np.exp(-self.fpar['mu'])+np.exp(-self.__nu*self.fpar['mu'])-self.fpar['p0M']*np.exp(-self.__nu*self.fpar['mu']))/(1-np.exp(-self.fpar['mu']))
            self.fpar['mu']= self.__nu*self.fpar['mu']

        if self.fdist.name == 'binom' or self.fdist.name == 'ZTbinom':
            self.fpar['p']=self.__nu*self.fpar['p']

        if self.fdist.name == 'ZMbinom':
            self.fpar['p0M']=(self.fpar['p0M']-(1-self.fpar['p'])**self.fpar['n']+(1-self.__nu*self.fpar['p'])**self.fpar['n']-self.fpar['p0M']*(1-self.__nu*self.fpar['p'])**self.fpar['n'])/(1-(1-self.fpar['p'])**self.fpar['n'])
            self.fpar['p']=self.__nu*self.fpar['p']

        if self.fdist.name == 'geom' or self.fdist.name == 'ZTgeom':
            beta=(1-self.fpar['p'])/self.fpar['p']
            self.fpar['p']=1/(1+self.__nu*beta)

        if self.fdist.name == 'ZMgeom':
            beta=(1-self.fpar['p'])/self.fpar['p']
            self.fpar['p0M']=(self.fpar['p0M']-(1+beta)**-1+(1+self.__nu*beta)**-1-self.fpar['p0M']*(1+self.__nu*beta)**-1)/(1-(1+beta)**-1)
            self.fpar['p']=1/(1+self.__nu*beta)

        if self.fdist.name == 'nbinom' or self.fdist.name == 'ZTnbinom':
            beta=(1-self.fpar['p'])/self.fpar['p']
            self.fpar['p']=1/(1+self.__nu*beta)

        if self.fdist.name == 'ZMnbinom':
            beta=(1-self.fpar['p'])/self.fpar['p']
            self.fpar['p0M']=(self.fpar['p0M']-(1+beta)**(-self.fpar['n'])+(1+self.__nu*beta)**-self.fpar['n']-self.fpar['p0M']*(1+self.__nu*beta)**-self.fpar['n'])/(1-(1+beta)**-self.fpar['n'])
            self.fpar['p']=1/(1+self.__nu*beta)


    def __fft(self, tiltingp=0):
        """
        Aggregate cost computation according to discrete Fourier transform via the fast Fourier transform computation.

        :param tiltingp: tilting parameter is set to zero and corrected in case the user chooses a different specification.
        :type tiltingp: ``bool`` or ``float``
        :return: aggregate cost empirical pdf, aggregate cost empirical cdf,aggregate cost values sequence
        :rtype: ``dict``

        """
        logger.info('..Computing the distribution via FFT..')

        if self.discretizationmethod == 'localmoments':
            sevdict = self.localMoments()
        else:
            sevdict = self.massDispersal()

        fj=sevdict['fj']

        if isinstance(self.tilt,bool):
            if self.tilt == True:
                tiltingp=20/self.n
        else:
            tiltingp=self.tilt

        # a,b,p0,g=self.abp0g0(fj)

        if self.u == float('inf'):
            fj=np.append(fj,np.repeat(0,self.n-self.m))
        else:
            fj = np.append(fj, np.repeat(0, self.n - self.m -1))

        f_hat=fft(np.exp(-tiltingp*np.arange(0, self.n, step=1))*fj)
        g_hat = self.fq.pgf(f=f_hat)#self.fgp(f=f_hat,a=a,b=b)
        g=np.exp(tiltingp*np.arange(0, self.n, step=1))*np.real(ifft(g_hat))

        logger.info('..Distribution via FFT completed..')

        return{'epdf': g,
               'ecdf':np.cumsum(g),
               'sequence': self.h*np.arange(0, self.n, step=1)}

    def __recursive(self):
        """
        Aggregate cost computation via the Panjer recursion.

        :return: aggregate cost empirical pdf, aggregate cost empirical cdf,aggregate cost values sequence
        :rtype: ``dict``
        """
        logger.info('..Computing the recursive distribution..')

        if self.discretizationmethod == 'localmoments':
            sevdict = self.localMoments()
        else:
            sevdict = self.massDispersal()

        fj=sevdict['fj']
        a, b, p0, g = self.abp0g0(fj)

        if self.u == float('inf'):
            fj=np.append(fj,np.repeat(0,self.n-self.m))
        else:
            fj = np.append(fj, np.repeat(0, self.n - self.m -1))

        for j in range(1, self.n):
            z = np.arange(1, min(j, len(fj) - 1) + 1)
            g.append((1 / (1 - a * fj[0])) * ((self.fq.pmf(1)-(a+b)*p0)*fj[z[-1]]+np.sum(((a + b * z / j) * fj[z] * np.flip(g)[:len(z)]))))

        logger.info('..Recursive distribution completed..')

        return {'epdf':g,
                'ecdf':np.cumsum(g),
                'sequence':self.h*np.arange(0, self.n, step=1)}

    def __MCsimulation(self):
        """
        Aggregate cost computation according to the Monte Carlo simulation.

        :return: aggregate cost empirical pdf, aggregate cost empirical cdf,aggregate cost simulated values.
        :rtype: ``dict``
        """
        logger.info('..Simulating the aggregate distribution..')

        if self.d > 1e-05:
            p0 = self.sv.cdf(self.d)
        else:
            p0 = 0.

        fqsample = self.fq.rvs(self.nsim, random_state=self.setseed)
        svsample = self.sv.rvs(int(np.sum(fqsample) + np.ceil(p0 * self.nsim)), random_state=self.setseed)

        # fqsample = self.fq.rvs(self.nsim, random_state=self.setseed)
        # svsample = self.sv.rvs(np.sum(fqsample), random_state=self.setseed)

        if self.d > 1e-05:
            np.random.seed(self.setseed)
            svsample = svsample[svsample > self.d]
            while svsample.shape[0] < self.nsim:
                n0 = np.ceil(p0 * (self.nsim - svsample.shape[0]))
                svsample = np.concatenate((svsample, self.sv.rvs(int(n0))))
                svsample = svsample[svsample > self.d]
            svsample = svsample - self.d

        if self.u < float('inf'):
            svsample = np.minimum(svsample, self.u)

        cs = np.cumsum(fqsample).astype(int)

        if fqsample[0:1] == 0:
            xsim = np.array([0])
        else:
            xsim = np.array([np.sum(svsample[0:cs[0]])])

        for i in range(0, self.nsim - 1):
            if cs[i] == cs[i + 1]:
                # xsim.append(0)
                xsim=np.concatenate((xsim,[0]))
            else:
                # xsim.append(np.sum(svsample[cs[i]:cs[i + 1]]))
                xsim = np.concatenate((xsim, [np.sum(svsample[cs[i]:cs[i + 1]])]))

        logger.info('..Simulation completed..')

        x_,ecdf = hfns.ecdf(xsim)
        epdf= np.repeat(1 / self.nsim, self.nsim)

        return {'epdf': epdf,
                'ecdf':ecdf,
                'sequence':x_}

    # def __QMCsimulation(self):
    #     """
    #     Aggregate cost computation according to the Quasi-Monte Carlo simulation.
    #
    #     :return: aggregate cost empirical pdf, aggregate cost empirical cdf,aggregate cost simulated values.
    #     :rtype: ``dict``
    #     """
    #     if self.d > 1e-05:
    #         p0 = self.sv.cdf(self.d)
    #     else:
    #         p0 = 0
    #
    #     logger.info('..QMC Simulating the aggregate distribution..')
    #
    #     fqsample = self.fq.rvs(self.nsim, random_state=self.setseed)
    #     # fss = hfns.sobol_generator(n=self.nsim, dim=1,skip=int(0+self.skip)).flatten()
    #     # fqsample = self.fq.ppf(fss)
    #     # sss = hfns.sobol_generator(n=int(np.sum(fqsample) + 1.1*np.ceil(p0 * self.nsim)), dim=1,skip=int(self.nsim+self.skip)).flatten()
    #     sssgen = qmc.Sobol(d=1, scramble=True,seed=self.setseed)
    #     sss=sssgen.random(n=int(np.sum(fqsample) + 1.1*np.ceil(p0 * self.nsim))).flatten()
    #     svsample = self.sv.ppf(sss)
    #
    #     if self.d > 1e-05:
    #         np.random.seed(self.setseed)
    #         svsample = svsample[svsample > self.d]
    #         while svsample.shape[0] < self.nsim:
    #             n0 = np.ceil(p0 * (self.nsim - svsample.shape[0]))
    #             sssgen = qmc.Sobol(d=1, scramble=True)
    #             sss = sssgen.random(n=int(1.1*n0)).flatten()
    #             # sss = hfns.sobol_generator(n=int(1.1*n0), dim=1,skip=self.skip).flatten()
    #             svsample = np.concatenate((svsample, self.sv.ppf(sss)))
    #             svsample = svsample[svsample > self.d]
    #             # timeel_=round((datetime.now()-now).seconds/60)
    #
    #             # if timeel_%3 == 0 and timeel_!=0 and count_ ==0:
    #             #     logger.warning('Pay attention: when a deductible is present, QMC simulation can be very slow.\n %s minutes elapsed, already'%str((datetime.now()-now).seconds/60))
    #             #     count_=count_
    #
    #         svsample = svsample - self.d
    #
    #     if self.u < float('inf'):
    #         svsample = np.minimum(svsample, self.u)
    #
    #     # random.seed(setseed) elimino: trovo seed nelle funzioni dopo
    #     # fss = hfns.sobol_generator(n=self.nsim, dim=1).flatten() #sobol_seq.i4_sobol_generate(1, self.nsim)
    #     # update frequency parameters
    #     # sample frequency
    #     # fqsample = self.fq.ppf(fss)
    #     # sss = hfns.sobol_generator(n=int(np.sum(fqsample)), dim=1).flatten()
    #     # sss = sobol_seq.i4_sobol_generate(1, int(np.sum(fqsample))).flatten()
    #
    #     # svsample = self.sv.ppf(sss)
    #     cs = np.cumsum(fqsample).astype(int)
    #
    #     if fqsample[0:1] == 0:
    #         xsim = np.array([0])
    #     else:
    #         xsim = np.array([np.sum(svsample[0:cs[0]])])
    #
    #     for i in range(0, self.nsim - 1):
    #         if cs[i] == cs[i + 1]:
    #             # xsim.append(0)
    #             xsim=np.concatenate((xsim,[0]))
    #         else:
    #             # xsim.append(np.sum(svsample[cs[i]:cs[i + 1]]))
    #             xsim = np.concatenate((xsim, [np.sum(svsample[cs[i]:cs[i + 1]])]))
    #
    #     # cs = np.cumsum(fqsample).astype('int')
    #     #
    #     # if fqsample[0:1] == 0:
    #     #     xsim = [0]
    #     # else:
    #     #     xsim = [np.sum(svsample[0:int(cs[0])])]
    #     #
    #     # for i in range(0, self.nsim - 1):
    #     #     if cs[i] == cs[i + 1]:
    #     #         xsim.append(0)
    #     #     else:
    #     #         xsim.append(np.sum(svsample[int(cs[i]):int(cs[i + 1])]))
    #
    #     logger.info('..QMC Simulation completed..')
    #
    #     x_, ecdf = hfns.ecdf(xsim)
    #     epdf = np.repeat(1 / self.nsim, self.nsim)
    #
    #     return {'epdf': epdf,
    #             'ecdf':ecdf,
    #             'sequence':x_}

        # return {'epdf': epdf(np.arange(0, np.max(xsim))),'ecdf':ECDF(xsim).y[1:],'sequence':ECDF(xsim).x[1:]}

    def empiricalmoments(self,central=False,order=1):

        """
        Empirical moments computed on the aggregate cost.

        :param central: whether the moment is central or not.
        :type central: ``bool``
        :param order: order of the moment
        :type order: ``int``
        :return: emprical moment.
        :rtype: ``numpy.float64``
        """
        assert isinstance(central,bool),'The parameter central must be either True or False'
        assert order > 0,'The parameter order must be a positive integer'
        assert isinstance(order,int), 'The parameter order must be a positive integer'

        if self.__algtype=='notsimulative':
            lmmean=np.sum(self.lossModel['epdf']*self.lossModel['sequence'])
            return np.sum(self.lossModel['epdf']*((self.lossModel['sequence']-(central*lmmean))**order))
        else:
            return np.mean((self.lossModel['sequence']-central*np.mean(self.lossModel['sequence']))**order)

    def ppf(self,probs=None):
        """
        Quantiles of the aggregate cost distribution computed in probs.

        :param probs: vector of probabilities.
        :type probs: ``float`` or ``numpy.ndarray``
        :return: vector of quantiles.
        :rtype: ``numpy.ndarray``

        """
        try:
            if self.__algtype == 'notsimulative':
                q=np.array([])
                probs=list(probs)
                for i in probs:
                    q=np.append(q,[self.lossModel['sequence'][self.lossModel['ecdf'] >= i][0]])
                return q
            else:
                return np.quantile(self.lossModel['sequence'],q=probs)
        except:
            logger.error('Please provide the values for the quantiles in a list')

    def plot(self, quantiles=[0.05, 0.5, 0.995]):
        """
        Plot of the aggregate cost empirical cdf and the aggregate cost empirical pdf.
        Quantiles are added to both plots.

        :param quantiles: probabilities in which to compute the quantiles.

        :return: cdf and pdf plots.
        """
        fig, axs = plt.subplots(1, 2)
        axs[0].plot(self.lossModel['sequence'],self.lossModel['ecdf'],color='b', label= '%r ecdf'%self.method)
        axs[0].axvline(x=self.empiricalmoments(), ymin=0,color='g', ymax=1, label='mean')
        axs[0].vlines(self.ppf(quantiles), 0, 1, colors='r', linestyles='--',
               label='%r quantiles' %quantiles)
        axs[0].legend()
        axs[1].set_ylim([0, np.max(self.lossModel['epdf'])])
        if self.__algtype == 'notsimulative':
            axs[1].bar(self.lossModel['sequence'],self.lossModel['epdf'],width=0.03,color='b', label= '%r epdf'%self.method)
        else:
            axs[1].hist(self.lossModel['sequence'],density=True, bins=100,color='b', label= '%r epdf'%self.method)
        axs[1].vlines(self.ppf(quantiles), ymin=0, ymax=np.max(self.lossModel['epdf']), colors='r', linestyles='--',
               label='%r quantiles' %quantiles)
        axs[1].axvline(x=self.empiricalmoments(), ymin=0,color='g', label='mean')
        axs[1].legend()
        plt.show()

    def StopLoss(self,t):
        """
        It computes the expected value of a stop loss treaty with priority t.

        :param t: priority.
        :type t: ``float``
        :return: expected value of the SL treaty.

        """
        t = np.repeat([t],1)
        x = self.lossModel['sequence']
        if self.__algtype == 'simulative':
            if t.shape[0] == 1:
               return(np.mean(np.maximum(x - t, 0)))
            else:
                lg = t.shape[0]
                obs = x.shape[0]
                v1_ = np.tile(x, lg).reshape(lg, -1) - np.repeat(t, obs).reshape(lg, -1)
                v1_[v1_ < 1e-6] = .0
                return (np.apply_along_axis(arr=v1_, func1d=np.mean, axis=1))
        else:
            probs=self.lossModel['epdf']
            if t.shape[0] == 1:
               return(np.sum(np.maximum(x - t, 0) * probs))
            else:
                lg=t.shape[0]
                obs=x.shape[0]
                v1_=np.tile(x, lg).reshape(lg, -1) - np.repeat(t, obs).reshape(lg, -1)
                v1_[v1_ < 1e-6] = .0 #svolgo il massimo
                v2_=v1_ * np.tile(probs, lg).reshape(lg, -1)
                return(np.apply_along_axis(arr=v2_,func1d=np.sum,axis=1))

    def Reinstatements(self):
        """
        Reinstatements pricing.

        :return: final reinstatements pricing.
        :rtupe: ``numpy.ndarray``
        """
        DLK=self.StopLoss(self.L)-self.StopLoss(self.L+(self.K+1)*self.u)
        if self.K==0:
            return(DLK)
        else:
            lowerk_=np.linspace(start=0,stop=self.K,num=self.K+1)
            dLk = (self.StopLoss(self.L+lowerk_[:-1]*self.u) - self.StopLoss(self.L+lowerk_[1:]*self.u))
            den=1+np.sum(dLk*self.c)/self.u
            return(DLK/den)

    def Pricing(self):
        """
        It allows to price contracts with proportional and non-proportional reinsurance.
        GemAct supports quota share, excess of loss, deductibles, stop-loss and reinstatements.

        :return: treaty final pricing.
        """
        #no aggregate conditions
        if self.K == float('inf'):
            P=self.StopLoss(t=self.L)
        if self.K != float('inf'):
            P=self.Reinstatements()

        data = [
            ['deductible','d', self.d],
            ['priority (severity)','u', self.u],
            ['priority (aggregate)','L', self.L],
            ['alpha (qs)','alphaqs', self.alphaqs]]
        if self.K != float('inf'):
            data.extend([['reinstatements','K',self.K]])

        data.append(['Pure premium','P',self.alphaqs*P])
        print("{: >20} {: >20} {: >20} {: >20}".format(" ",*['Contractual limits','parameter','value']))
        print("{: >20} {: >20}".format(" ",*[" =================================================================="]))
        for row in data:
            print("{: >20} {: >20} {: >20} {: >20}".format("", *row))
        print('\n layers loading c: ', self.c)
        if self.__algtype=='simulative':
            print(self.method, '\t nsim: ', self.nsim)
        else:
            print(self.method, '\t m: ', self.m,'\t n: ',self.n)











