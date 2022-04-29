import numpy as np
from scipy import stats
from scipy.special import factorial
from . import helperfunctions as hfns
from . import distributions as distributions

from twiggy import quick_setup,log
quick_setup()
logger = log.name('lossaggregation')


class LossAggregation:
    """
        This class computes the probability that the sum of random variables
        with a dependence structure specified by a copula with the AEP algorithm.

        :param d_la: dimension of the AEP algorithm.
        :type d_la: ``int``
        :param s_la: probability threshold.
        :type s_la: ``float``
        :param n_la: number of iteration of the AEP algorithm.
        :type n_la: ``int``
        :param mdict: dictionary of the supported distributions.
        :type mdict: ``dict``
        :param  m1:  first marginal distribution, provided as a string and assigned to a scipy object in properties
        :type m1: ``str``
        :param  m2:  second marginal distribution, provided as a string and assigned to a scipy object in properties
        :type m2: ``str``
        :param m3:   third marginal distribution, provided as a string and assigned to a scipy object in properties. Optional.
        :type m3: ``str``
        :param  m4:  fourth marginal distribution, provided as a string and assigned to a scipy object in properties. Optional.
        :type m4: ``str``
        :param  m5:  fifth marginal distribution, provided as a string and assigned to a scipy object in properties. Optional.
        :type m5: ``str``
        :param m1par:  parameters of the first marginal, provided as a dictionary according to the scipy parametrization.
        :type m1par: ``dict``
        :param m2par:   parameters of the first marginal, provided as a dictionary according to the scipy parametrization.
        :type m2par: ``dict``
        :param m3par: parameters of the third marginal, provided as a dictionary according to the scipy parametrization. Optional.
        :type m3par: ``dict``
        :param m4par: parameters of the fourth marginal, provided as a dictionary according to the scipy parametrization. Optional.
        :type m4par: ``dict``
        :param m5par: parameters of the fifth marginal, provided as a dictionary according to the scipy parametrization. Optional.
        :type m5par: ``dict``
        :param  dist1:  distribution from which first marginal cdf are computed.
        :type dist1: ``scipy.stats._distn_infrastructure.rv_frozen``
        :param  dist2:  distribution from which second marginal cdf are computed.
        :type dist2: ``scipy.stats._distn_infrastructure.rv_frozen``
        :param dist3: distribution from which third marginal cdf are computed.Optional.
        :type dist3: ``scipy.stats._distn_infrastructure.rv_frozen``
        :param  dist4:  distribution from which fourth marginal cdf are computed.Optional.
        :type dist4: ``scipy.stats._distn_infrastructure.rv_frozen``
        :param dist5: distribution from which fifth marginal cdf are computed.Optional.
        :type dist5: ``scipy.stats._distn_infrastructure.rv_frozen``
        :param alpha_la: AEP algorithm alpha parameter.
        :type alpha_la: ``float``
        :param b_la: vector b of the AEP algorithm.
        :type b_la: ``numpy.ndarry``
        :param h_la: vector h of the AEP algorithm.
        :type h_la: ``numpy.ndarry``
        :param Mat: Matrix of the vectors in the {0,1}**d_la space.
        :type Mat: ``numpy.ndarry``
        :param Card: It contains the cardinality of the matrix Mat.
        :type Card: ``numpy.ndarray``
        :param m_la: It contains +1,-1,0 values that indicate whether the new simpleces origined from sN_ must be summed, subtracted or ignored.
        :type m_la: ``numpy.ndarry``
        :param N_la: Number of new simpleces received in each step.
        :type N_la: ``float``
        :param sN_: It contains +1,-1,0 values that indicate whether a volume must be summed, subtracted or ignored.
        :type sN_: ``numpy.ndarry``
        :param ext: probability correction in convergence
        :type ext: ``float``
        :param __s:  It contains either +1 or -1. It indicates whether to sum or subtract a volume. Private.
        :type __s: ``numpy.ndarray``
        :param out: It contains the result of the AEP algorithm. It must be between zero and one as this is a probability.
        :type out: ``float``

        """
    def __init__(self,m3=None,
                 m4=None,
                 m5=None,
                 m3par=None,
                 m4par=None,
                 m5par=None,**kwargs):
        #setted by the user
        self.__dceiling = 5
        self.d_la = 2
        self.s_la = kwargs['s_la']
        self.n_la = kwargs['n_la']
        # self.copula_la = kwargs['copula_la']
        # self.copulaPar_la = kwargs['copulaPar_la']
        ## marginals
        ##available marginals
        self.mdict = {'gamma': distributions.Gamma,
                  'lognorm': distributions.Lognorm,
                  'exponential':distributions.Exponential,
                  'genpareto': distributions.Genpareto,
                  'burr12': distributions.Burr12,
                  'dagum': distributions.Dagum,
                  'invgamma': distributions.Invgamma,
                  'weibull_min': distributions.Weibull_min,
                  'invweibull': distributions.Invweibull,
                  'beta': distributions.Beta}

        ### distributions
        self.m1 = kwargs['m1']
        self.m2 = kwargs['m2']
        self.m3 = m3
        self.m4 = m4
        self.m5 = m5
        ### parameters
        self.m1par = kwargs['m1par']
        self.m2par = kwargs['m2par']
        self.m3par = m3par
        self.m4par = m4par
        self.m5par = m5par

        ### define it
        try:
            self.dist1 = self.m1(**self.m1par)
            self.dist2 = self.m2(**self.m2par)
            self.marginals= hfns.MarginalCdf2D(dist1=self.dist1,
                                               dist2=self.dist2,
                                               d_la=self.d_la).marginal_cdf
        except:
            logger.error('The marginal distributions in m1,m2 are not parametrized correctly.')

        if self.m3 is not None:
            try:
                self.dist3 = self.m3(**self.m3par)
            except:
                logger.error('The marginal distribution in m3 is not parametrized correctly.')
            self.d_la=3
            self.marginals = hfns.MarginalCdf3D(dist1=self.dist1,
                                                dist2=self.dist2,
                                                dist3=self.dist3,
                                                d_la=self.d_la).marginal_cdf

        if self.m4 is not None:
            try:
                self.dist4 = self.m4(**self.m4par)
            except:
                logger.error('The marginal distribution in m4 is not parametrized correctly.')
            self.d_la = 4
            self.marginals = hfns.MarginalCdf4D(dist1=self.dist1,
                                                dist2=self.dist2,
                                                dist3=self.dist3,
                                                dist4=self.dist4,
                                                d_la=self.d_la).marginal_cdf

        if self.m5 is not None:
            try:
                self.dist5 = self.m5(**self.m5par)
            except:
                logger.error('The marginal distribution in m4 is not parametrized correctly.')
            self.d_la = 5
            self.marginals = hfns.MarginalCdf5D(dist1=self.dist1,
                                                dist2=self.dist2,
                                                dist3=self.dist3,
                                                dist4=self.dist4,
                                                dist5=self.dist5,
                                                d_la=self.d_la).marginal_cdf

        self.copdist= kwargs['copdist']
        self.coppar = kwargs['coppar']

        #setted by us
        ## about copulas
        try:
            self.copula=self.copdist(**self.coppar)
        except:
            logger.error("The copula distribution is not parametrized correctly \n See our documentation: https://gem-analytics.github.io/gemact/ ")


        ## about the AEP algorithm
        self.alpha_la = 2. / (self.d_la + 1)
        self.b_la = np.tile(np.array([[0]]), self.d_la).T
        self.h_la = np.array(self.s_la)
        self.Mat = self.iMat(self.d_la)
        self.Card = self.iCardinality()
        self.m_la = self.m_j()
        self.N_la = 2.0 ** self.d_la - 1
        self.sN_ = self.s_new(np.array([1]))#self.s_la
        self.ext=((self.d_la+1)**self.d_la)/(factorial(self.d_la)*2**self.d_la)
        self.__s = (-1) ** (self.d_la - np.sum(self.Mat, axis=0))

        #result
        self.out = self.v_h(b=self.b_la, h=self.alpha_la * self.h_la,  Mat=self.Mat) #Copula=self.Cop,
        self.__AEP()

    @property
    def d_la(self):
        return self.__d_la

    @d_la.setter
    def d_la(self, var):
        if isinstance(var,float):
            logger.info("The value you provided for your problem dimensions will be converted to an integer")
            var=int(var)
        else:
            assert isinstance(var,int), logger.error('Please provide d_la as a positive integer between 1 and 5')

        assert var > 0 and var <= self.__dceiling, logger.error('Please provide d_la as a positive integer between 1 and 5')
        self.__d_la = var

    @property
    def s_la(self):
        return self.__s_la

    @s_la.setter
    def s_la(self, var):
        assert isinstance(var,int) or isinstance(var,float), logger.error("Please provide the quantile s_la as an integer or a flaot")
        assert var > 0, logger.error("Please provide the quantile s_la as a positive value")
        self.__s_la = [[var]] #ricordiamoci che prima facevamo questo step a monte. Ora sfruttiamo le proprietà di questo attributo

    @property
    def n_la(self):
        return self.__n_la

    @n_la.setter
    def n_la(self, var):
        assert isinstance(var,int) and var > 0, logger.error("Please provide the number of steps n_la as a positive integer")
        self.__n_la = var

    @property
    def copdist(self):
        return self.__copdist

    @copdist.setter
    def copdist(self, var):
        assert isinstance(var,str), logger.error('The copula name must be given as a string')
        # this is the set of possible distributions
        cdict = {'clayton': hfns.ClaytonCDF,
                  'frank': hfns.FrankCDF,
                  'gumbel':hfns.GumbelCDF}
        assert var in cdict.keys(), "%r is not supported \n See our documentation: https://gem-analytics.github.io/gemact/" % var
        self.__copdist = cdict[var]

    @property
    def coppar(self):
        return self.__coppar

    @coppar.setter
    def coppar(self, var):
        assert isinstance(var, dict), 'The copula distribution parameters must be given as a dictionary'
        # if 'dim' not in var.keys():
        #     var['dim'] = self.d_la
        # else:
        #     if var['dim'] != self.d_la:
        #         logger.warning('The dimension you specified inside your copula will be disregarded.')
        #         var['dim'] = self.d_la
        self.__coppar = var

    @property
    def m1(self):
        return self.__m1

    @m1.setter
    def m1(self, var):
        assert var in self.mdict.keys(), "%r is not supported \n See our documentation: https://gem-analytics.github.io/gemact/" % var
        self.__m1 = self.mdict[var]

    @property
    def m2(self):
        return self.__m2

    @m2.setter
    def m2(self, var):
        assert var in self.mdict.keys(), "%r is not supported \n See our documentation: https://gem-analytics.github.io/gemact/" % var
        self.__m2 = self.mdict[var]

    @property
    def m3(self):
        return self.__m3

    @m3.setter
    def m3(self, var):
        if var is not None:
            assert var in self.mdict.keys(), "%r is not supported \n See our documentation: https://gem-analytics.github.io/gemact/ " % var
            self.__m3 = self.mdict[var]
        else:
            assert var is None, logger.error('Please provide in m3 a marginal supported distribution or drop the argument.')
            self.__m3 = var

    @property
    def m4(self):
        return self.__m4

    @m4.setter
    def m4(self, var):
        if var is not None:
            assert var in self.mdict.keys(), "%r is not supported \n See our documentation: https://gem-analytics.github.io/gemact/" % var
            self.__m4 = self.mdict[var]
        else:
            assert var is None, logger.error(
                'Please provide in m4 a marginal supported distribution or drop the argument.')
            self.__m4 = var

    @property
    def m5(self):
        return self.__m5

    @m5.setter
    def m5(self, var):
        if var is not None:
            assert var in self.mdict.keys(), "%r is not supported \n See our documentation: https://gem-analytics.github.io/gemact/" % var
            self.__m5 = self.mdict[var]
        else:
            assert var is None, logger.error(
                'Please provide in m5 a marginal supported distribution or drop the argument.')
            self.__m5 = var

    @property
    def m1par(self):
        return self.__m1par

    @m1par.setter
    def m1par(self, var):
        assert isinstance(var, dict), 'The m1 distribution parameters must be given as a dictionary'
        self.__m1par = var

    @property
    def m2par(self):
        return self.__m2par

    @m2par.setter
    def m2par(self, var):
        assert isinstance(var, dict), 'The m2 distribution parameters must be given as a dictionary'
        self.__m2par = var

    @property
    def m3par(self):
        return self.__m3par

    @m3par.setter
    def m3par(self, var):
        if var is not None:
            assert isinstance(var, dict), 'The m3 distribution parameters must be given as a dictionary'
            self.__m3par = var
        else:
            assert var is None, logger.error('Please provide the m3 distribution parameters as a dictionary or drop the argument.')
            self.__m3par = var

    @property
    def m4par(self):
        return self.__m4par

    @m4par.setter
    def m4par(self, var):
        if var is not None:
            assert isinstance(var, dict), 'The m4 distribution parameters must be given as a dictionary'
            self.__m4par = var
        else:
            assert var is None, logger.error('Please provide the m4 distribution parameters as a dictionary or drop the argument.')
            self.__m4par = var

    @property
    def m5par(self):
        return self.__m5par

    @m5par.setter
    def m5par(self, var):
        if var is not None:
            assert isinstance(var, dict), 'The m4 distribution parameters must be given as a dictionary'
            self.__m5par = var
        else:
            assert var is None, logger.error('Please provide the m5 distribution parameters as a dictionary or drop the argument.')
            self.__m5par = var

    def iMat(self,d):
        """
        It computes the matrix points in which copulas must be computed.

        :param d: dimension.
        :type d: ``int``
        :return: matrix of points.
        :rtype:``numpy.ndarray``
        """
        if d == 1:
            return np.array([0, 1])
        else:
            temp = self.iMat(d - 1)
            if d - 1 == 1:
                ncol = temp.shape[0]
            else:
                ncol = temp.shape[1]
            return np.concatenate(
                (np.repeat([0, 1], ncol).reshape(1, 2 * ncol), np.tile(temp, 2).reshape(d - 1, 2 * ncol)),
                axis=0)

    def MarginalCdf(self,x):
        out_=np.zeros(self.d_la)
        out_[0]= self.dist1.cdf(x[0])
        out_[1] = self.dist2.cdf(x[1])
        # out_=np.append(x1, x2)

        if self.d_la > 2:
            out_[2]=self.dist3.cdf(x[2])
            # out_ = np.append(out_,x3)
        if self.d_la > 3:
            out_[3]=self.dist4.cdf(x[3])
            # out_ = np.append(out_,x4)
        if self.d_la > 4:
            out_[4]=self.dist5.cdf(x[4])
            # out_ = np.append(out_,x5)
        return out_

    # def CopulaCDF(self,x):
    #     return hfns.ClaytonCDF(self.MarginalCdf(x),par=self.coppar['theta'],d=self.d_la)
    #
    # def Cop(self,x):  # input matrice le cui colonne sono punti.
    #     return np.apply_along_axis(func1d=self.CopulaCDF, arr=x, axis=0)

    def v_h(self,b, h, Mat): #Copula
        # arg_= b.reshape(self.d_la, 1) + h * Mat
        # idx = np.argwhere(np.any(arg_[..., :] == 0, axis=0))
        v_=np.transpose((b.reshape(self.d_la, 1) + h * Mat)) #*self.__s * (np.sign(h) ** self.d_la))
        return np.sum(self.copula.cdf(self.marginals(v_))*(self.__s * (np.sign(h) ** self.d_la)))
        # return np.sum(Copula(b.reshape(self.d_la, 1) + h * Mat) * self.__s*(np.sign(h) ** self.d_la))
        # return np.sum(Copula(np.delete(arg_, idx, axis=1)) * np.delete(self.__s, idx, axis=0))

    def iCardinality(self):
        return np.sum(self.Mat, axis=0)[1:]

    def m_j(self):
        mx_ = self.Card.copy()
        greater = np.where(mx_ > (1 / self.alpha_la))
        equal = np.where(mx_ == (1 / self.alpha_la))
        lower = np.where(mx_ < (1 / self.alpha_la))
        mx_[greater] = (-1) ** (self.d_la + 1 - mx_[greater])
        mx_[equal] = 0
        mx_[lower] = (-1) ** (1 + mx_[lower])
        return mx_

    def s_new(self,s_old):
        return np.repeat(s_old, self.N_la) * np.tile(self.m_la, len(s_old))

    def h_new(self,h_old):
        return (1 - np.tile(self.Card, len(h_old)) * self.alpha_la) * np.repeat(h_old, len(self.Card))

    def b_new(self,b_old, h_old):
        Mat = self.Mat[:, 1:]
        h = np.repeat(h_old, self.N_la)
        times = len(h) / len(Mat[0])
        return np.repeat(b_old, self.N_la, axis=1) + self.alpha_la * np.tile(h, (self.d_la, 1)) * np.tile(Mat, int(times))

    def __AEP(self):
        for j in range(1, self.n_la):
            # ix= self.sN_>0
            self.b_la = self.b_new(b_old=self.b_la, h_old=self.h_la)
            self.h_la = self.h_new(h_old=self.h_la)
            # btemp = self.b_la[:,ix]
            # htemp = self.h_la[ix]
            volumes = np.array([])
            for i in range(0, self.b_la.shape[1]):
                volumes=np.append(volumes,[self.v_h(b=self.b_la[:, i].reshape(self.d_la, 1), h= self.alpha_la *self.h_la[i], Mat=self.Mat)]) #, Copula=self.Cop
            # for i in range(0, self.b_la.shape[1]):
            #     volumes=np.append(volumes,[self.v_h(b=self.b_la[:, i].reshape(self.d_la, 1), h= self.alpha_la *self.h_la[i], Copula=self.Cop, Mat=self.Mat)])
            self.out = self.out + np.sum(self.sN_ * volumes)
            self.sN_ = self.s_new(self.sN_)
            # print(self.out)
        return 0
