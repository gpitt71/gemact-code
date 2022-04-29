import sys
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from . import helperfunctions as hfns

from twiggy import quick_setup,log
quick_setup()
logger = log.name('lossreserve')

class LossReserve:
    """
    Class to compute the loss reserve. The available models are the deterministic Fisher-Lange and the collective risk model.
    Input company data must be ``numpy.ndarray`` data on numbers and payments must be in triangular form:
    two dimensional ``numpy.ndarray`` with shape (I,J) with I=J.

    :param tail: set it to True when the tail estimate is required. Default False.
    :type tail: ``bool``
    :param reserving_method: one of the reserving methods supported by the GemAct package.
    :type reserving_method: ``str``
    :param claims_inflation: claims inflation. In case no tail is present and the triangular data IxJ matrices,
                            claims_inflation must be J-1 dimensional. When a tail estimate is required, it must be
                            J dimensional. In case no tail is present it must be J-1 dimensional.
    :type claims_inflation: ``numpy.ndarray``
    :param czj: severity coefficient of variation by development year. It is set to None in case the crm is selected as
                reserving method. When a tail estimate is required, it must be J dimensional.
                In case no tail is present it must be J-1 dimensional.
    :type czj: ``numpy.ndarray``

    :param ntr_sim: Number of simulated triangles in the c.r.m reserving method.
    :type ntr_sim: ``int``
    :param setseed: Simulation seed to make the c.r.m reserving method results reproducible.
    :type setseed: ``int``
    :param mixingfpar: Mixing frequency parameters.
    :type mixingfpar: ``dict``
    :param mixingspar: Mixing severity parameters.
    :type mixingspar: ``dict``
    :param custom_alphas: optional, custom values for the alpha parameters.
    :type custom_alphas: ``numpy.ndarray``
    :param custom_ss: optional, custom values for the settlement speed.
    :type custom_ss: ``numpy.ndarray``

    :param \**kwargs:
        See below

    :Keyword Arguments:
        * *ip_tr* = (``numpy.ndarray``) --
            Incremental payments triangle
        * *cumulatives_tr* = (``numpy.ndarray``) --
            Cumulative payments triangle
        * *cp_tr* = (``numpy.ndarray``) --
            Cased payments triangle
        * *in_tr* = (``numpy.ndarray``) --
            Incurred number
        * *cn_tr* = (``numpy.ndarray``) --
            Cased number

    """
    def __init__(self,tail=False,
                 claims_inflation=None,
                 custom_alphas=None,
                 custom_ss = None,
                 reserving_method="fisherlange",
                 ntr_sim=1000,
                 mixingfpar=None,
                 czj=None,
                 setseed=42,
                 cumulatives_tr=None,
                 mixingspar=None,**kwargs):

        self.tail=tail
        self.reserving_method=reserving_method

        try:
            self.j= kwargs['incremental_payments'].shape[1]
            self.ix = np.tile(np.arange(0, self.j), self.j).reshape(self.j, self.j) + np.tile(np.arange(1, self.j + 1), self.j).reshape(self.j, self.j).T
        except:
            logger.error('The Incremental Payments triangle must be a two dimensional numpy.ndarray with same shape on both dimensions')

        # 1d-arrays
        self.claims_inflation = claims_inflation
        self.reported_claims= kwargs['reported_claims']
        self.czj = czj
        self.czj_t = np.tile(self.czj, self.j).reshape(self.j, -1)

        #triangles
        self.ip_tr = kwargs['incremental_payments']
        self.cp_tr = kwargs['cased_payments']
        self.in_tr = kwargs['incurred_number']
        self.cn_tr = kwargs['cased_number']
        self.ap_tr = None
        self.cumulatives_tr = cumulatives_tr

        #variables and other objects
        if self.reserving_method == 'crm':
            self.ntr_sim = ntr_sim
            self.setseed = setseed
            self.mixingfpar = mixingfpar
            self.mixingspar = mixingspar
            self.gamma1 = stats.gamma(**self.mixingfpar)
            self.gamma2 = stats.gamma(**self.mixingspar)
            self.gamma3 = stats.gamma
            self.pois = stats.poisson

        #attributes with opportunities of not standard customization
        if not custom_alphas == None:
            self.alphaFl_ = custom_alphas
        else:
            self.alphaFl_= self.__AlphaComputer()

        if not custom_ss == None:
            self.ssFl_ = custom_alphas
        else:
            self.ssFl_= self.__SsComputer()

        # results
        self.ss_tr=self.__SsTriangle() #triangle of settlement speed
        self.predicted_Inumbers = self.__fillNumbers() #individual payments triangle

        self.flreserve=self.__Flreserve() #fisher-lange reserve

        if self.reserving_method=='crm':
            self.crmreserve,self.msep,self.skewness=self.__StochasticCRM() #crm reserving: value of the reserve, mean-squared error of prediction and skewness.

    @property
    def tail(self):
        return self.__tail

    @tail.setter
    def tail(self, var):
        assert isinstance(var,bool), logger.error("tail must be a boolean")
        self.__tail=var

    @property
    def reserving_method(self):
        return self.__reserving_method

    @reserving_method.setter
    def reserving_method(self, var):
        assert var.lower() in ['chl','fisherlange','crm'], logger.error("%r is not supported from our package"%var)
        assert isinstance(var,str), logger.error("%r must be a string"%var)
        if not var.lower() == var:
            logger.warning("What did you mean with %r? 'reservingmethod' is set to %r." % (var, var.lower()))
        self.__reserving_method=var.lower()

    # 1-d arrays
    @property
    def claims_inflation(self):
        return self.__claims_inflation

    @claims_inflation.setter
    def claims_inflation(self, var):
        if var is None:
            self.__claims_inflation = np.repeat(1,self.j)
        else:
            try:
                var = np.array(var,dtype=float)
            except:
                logger.error("Provide a numpy.ndarray for claims inflation")

            if self.tail==True:
                assert var.shape[0]==self.j, logger.error("The vector of inflation you provided is not correctly shaped. \n Length must be %s"%(self.j))
            else:
                assert var.shape[0] == self.j-1, logger.error("The vector of inflation you provided is not correctly shaped. \n Length must be %s"%(self.j-1))

            self.__claims_inflation =var

    @property
    def reported_claims(self):
        return self.__reported_claims

    @reported_claims.setter
    def reported_claims(self, var):
        var=np.array(var)
        assert var.shape[0]==self.j, logger.error('Reported Claims must be provided as a \n 1-d array with length %s'%self.j)
        logger.info("A correct 1-d vector for Reported claims was provided. \n Make sure the reported claims on the last position corresponds to the most recent data")
        self.__reported_claims = var

    @property #devi dargli in pasto solo i coefficienti non zero
    def czj(self):
        return self.__czj

    @czj.setter
    def czj(self, var):
        if self.reserving_method in ['chl','fisherlange']:
            self.__czj = None
        else:
            if var is None:
                self.__czj = np.repeat(.001,self.j+self.tail)
            else:
                try:
                    var = np.array(var,dtype=float)
                except:
                    logger.error("Provide a numpy.ndarray for czj")

                if self.tail==True:
                    assert var.shape[0]==self.j, logger.error("The vector of czj you provided is not correctly shaped. \n Length must be %s"%(self.j))
                    var = np.concatenate(([.0], var))
                else:
                    assert var.shape[0] == self.j-1, logger.error("The vector of czj you provided is not correctly shaped. \n Length must be %s"%(self.j-1))
                    var = np.concatenate(([.0], var))

                self.__czj =var

    #Triangles
    @property
    def ip_tr(self):
        return self.__ip_tr

    @ip_tr.setter
    def ip_tr(self, var):
        if self.reserving_method in ['chl']:
            if var is None:
                self.__ip_tr = var
            else:
                var = np.array(var).astype(float)
                assert type(var) == np.ndarray, logger.error(
                    'The Incremental Payments triangle must be a two dimensional numpy.ndarray')
                assert var.shape[0] == var.shape[1], logger.error(
                    'The Incremental Payments triangle must be a two dimensional with same shape on both dimensions. \n The triangle shape is: %s' % str(
                        var.shape))

                nans = np.isnan(
                    var)  # sia con np.nan di numpy che con NaN di python te li trova. Ricordiamoci che è un punto delicato

                if np.sum(nans) > 0:
                    assert np.min(self.ix[nans]) > self.j, logger.error(
                        'Check your Incremental Payments input.\n Not valid values in the upper triangle.')
                self.__ip_tr = var
        else:
            var = np.array(var).astype(float)
            assert type(var) == np.ndarray, logger.error('The Incremental Payments triangle must be a two dimensional numpy.ndarray')
            assert var.shape[0] == var.shape[1], logger.error('The Incremental Payments triangle must be a two dimensional with same shape on both dimensions. \n The triangle shape is: %s'%str(var.shape))

            nans=np.isnan(var) #sia con np.nan di numpy che con NaN di python te li trova. Ricordiamoci che è un punto delicato

            if np.sum(nans) > 0:
                assert np.min(self.ix[nans]) > self.j, logger.error('Check your Incremental Payments input.\n Not valid values in the upper triangle.')
            self.__ip_tr = var

    @property
    def cumulatives_tr(self):
        return self.__cumulatives_tr

    @cumulatives_tr.setter
    def cumulatives_tr(self, var):
        if self.reserving_method in ['chl']:
            if self.ip_tr is None:
                var = np.array(var).astype(float)
                assert type(var) == np.ndarray, logger.error(
                    'The Incremental Payments triangle must be a two dimensional numpy.ndarray')
                assert var.shape[0] == var.shape[1], logger.error(
                    'The Incremental Payments triangle must be a two dimensional with same shape on both dimensions. \n The triangle shape is: %s' % str(
                        var.shape))

                nans = np.isnan(
                    var)  # sia con np.nan di numpy che con NaN di python te li trova. Ricordiamoci che è un punto delicato

                if np.sum(nans) > 0:
                    assert np.min(self.ix[nans]) > self.j, logger.error(
                        'Check your Incremental Payments input.\n Not valid values in the upper triangle.')
                self.__cumulatives_tr = var
            else:
                cumuls=self.ip_tr.copy()
                for i in range(1, cumuls.shape[1]):
                    cumuls[:, i] = cumuls[:, i] + cumuls[:, i - 1]
                self.__cumulatives_tr = cumuls
        else:
            self.__cumulatives_tr = None

    @property
    def cp_tr(self):
        return self.__cp_tr

    @cp_tr.setter
    def cp_tr(self, var):
        if self.reserving_method in ['fisherlange', 'crm']:
            var = np.array(var).astype(float)
            assert type(var) == np.ndarray, logger.error('The Cased Payments triangle must be a two dimensional numpy.ndarray')
            assert var.shape[0] == var.shape[1], logger.error('The Cased Payments triangle must be a two dimensional with same shape on both dimensions. \n The triangle shape is: %s'%str(var.shape))

            nans=np.isnan(var) #sia con np.nan di numpy che con NaN di python te li trova. Ricordiamoci che è un punto delicato

            if np.sum(nans) > 0:
                assert np.min(self.ix[nans]) > self.j, logger.error('Check your Cased Payments input.\n Not valid values in the upper triangle.')
            self.__cp_tr = var
        else:
            self.__cp_tr = var

    @property
    def in_tr(self):
        return self.__in_tr

    @in_tr.setter
    def in_tr(self, var):
        if self.reserving_method in ['fisherlange', 'crm']:
            var=np.array(var).astype(float)
            assert type(var) == np.ndarray, logger.error('The Incurred Number triangle must be a two dimensional numpy.ndarray')
            assert var.shape[0] == var.shape[1], logger.error('The Incurred Number triangle must be a two dimensional with same shape on both dimensions. \n The triangle shape is: %s'%str(var.shape))

            nans=np.isnan(var) #sia con np.nan di numpy che con NaN di python te li trova. Ricordiamoci che è un punto delicato

            if np.sum(nans) > 0:
                assert np.min(self.ix[nans]) > self.j, logger.error('Check your Incurred Number input. \n Not valid values in the upper triangle.')
            self.__in_tr = var
        else:
            self.__in_tr = None

    @property
    def cn_tr(self):
        return self.__cn_tr

    @cn_tr.setter
    def cn_tr(self, var):
        if self.reserving_method in ['fisherlange', 'crm']:
            var = np.array(var).astype(float)
            assert type(var) == np.ndarray, logger.error('The Cased Number triangle must be a two dimensional numpy.ndarray')
            assert var.shape[0] == var.shape[1], logger.error('The Cased Number triangle must be a two dimensional with same shape on both dimensions. \n The triangle shape is: %s'%str(var.shape))

            nans=np.isnan(var) #sia con np.nan di numpy che con NaN di python te li trova. Ricordiamoci che è un punto delicato

            if np.sum(nans) > 0:
                assert np.min(self.ix[nans]) > self.j, logger.error('Check your Cased Number input.\n Not valid values in the upper triangle.')
            self.__cn_tr = var
        else:
            self.__cn_tr = None

    @property
    def ntr_sim(self):
        return self.__ntr_sim

    @ntr_sim.setter
    def ntr_sim(self, var):
        try:
            var = int(var)
            assert isinstance(var,int), logger.error("The number of simulated triangles for the crm reserving model must be provided as an integer")
            self.__ntr_sim=var
        except:
            logger.error("Please provide the number of simulated triangles for the crm reserving as an integer")

    @property
    def setseed(self):
        return self.__setseed

    @setseed.setter
    def setseed(self, var):
        try:
            var = int(var)
            assert isinstance(var,int), logger.error("The seed number for the crm reserving model must be provided as an integer")
            self.__setseed=var
        except:
            logger.error("Please provide the seed number for the crm reserving as an integer")

    @property
    def mixingfpar(self):
        return self.__mixingfpar

    @mixingfpar.setter
    def mixingfpar(self, var):
        if self.reserving_method == "crm":
            try:
                assert isinstance(var,dict), logger.error("The frequency mixing parameters for the crm reserving model must be provided as a dictionary")
                assert all(item in list(var.keys()) for item in ["scale","a"]), logger.error("The mixing frequency parameters for the crm reserving model must be provided as 'a' and 'scale'. See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gamma.html")
                self.__mixingfpar=var
            except:
                logger.error("Please provide the mixing frequency parameters for the crm reserving as a dictionary")
        else:
            self.__mixingfpar=var

    @property
    def mixingspar(self):
        return self.__mixingspar

    @mixingspar.setter
    def mixingspar(self, var):
        if self.reserving_method == "crm":
            try:
                assert isinstance(var, dict), logger.error(
                    "The severity mixing parameters for the crm reserving model must be provided as a dictionary")
                assert all(item in list(var.keys()) for item in ["scale", "a"]), logger.error(
                    "The mixing severity parameters for the crm reserving model must be provided as 'a' and 'scale'. See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gamma.html")
                self.__mixingspar = var
            except:
                logger.error("Please provide the mixing frequency parameters for the crm reserving as a dictionary")
        else:
            self.__mixingspar=var
    @property
    def ap_tr(self):
        return self.__ap_tr

    @ap_tr.setter
    def ap_tr(self, var): #scelgo di utilizzare cicli for. Si può fare in modo estremamente complicato vettoriale.
        # Visto che il pacchetto andrà almeno un minimo mantenuto conviene evitarsi rogne e usare i più comprensibili cicli.
        if var is None:
            temp = self.ip_tr / self.in_tr
            n=self.j-1 # valore finale sul triangolo
            for j in range(1, self.j):  # non serve fare nulla in zero. Quindi si parte da 1
                temp[n - j + 1:, j] = temp[n - j, j] * np.cumprod(self.claims_inflation[:j])

            if self.tail == True:
                self.__ap_tr=np.column_stack((temp,temp[0,-1]*np.cumprod(self.claims_inflation)))
            else:
                self.__ap_tr=temp

    #attributes with not-standard customization
    @property
    def alphaFl_(self):
        return self.__alphaFl_

    @alphaFl_.setter
    def alphaFl_(self, var):
        assert var.shape[0] == self.j+self.tail-1, logger.error("The alpha vector length must be %s"%(self.j+self.tail-1)) #trick per evitare un ciclo if
        self.__alphaFl_= np.array(var)

    @property
    def ssFl_(self):
        return self.__ssFl_

    @ssFl_.setter
    def ssFl_(self, var):
        var = np.array(var)
        assert var.shape[0] == self.j + self.tail - 1, logger.error("The alpha vector length must be %s" % (self.j + self.tail - 1))  # trick per evitare un ciclo if
        assert np.abs(np.sum(var) - 1) < 1e+04, logger.error("The Settlement Speed vector must sum to one")
        self.__ssFl_=var

    #methods
    def __AlphaComputer(self):
        """
        It computes Fisher-Lange alpha. Given a JxJ triangle, this is going to be
        J-1 dimensional in case no tail is present and J dimensional in case of tail estimates.

        :return: vectors of alpha
        :rtype: ``numpy.ndarray``
        """
        tempIn_=self.in_tr.copy()
        tempCn_ = self.cn_tr.copy()
        tempIn_[self.ix > self.j]= 0.
        tempCn_[self.ix > self.j] = np.nan
        v_=np.flip(np.apply_along_axis(arr=tempIn_,func1d=np.sum,axis=1))
        #create a matrix with the cumulative number of claims in the diagonal
        totInc= np.rot90(np.array([v_,]*self.j))
        #create a matrix with cumulative sums
        cumInc=np.apply_along_axis(arr=tempIn_, func1d=np.cumsum, axis=1)
        #create a matrix
        dg_ = np.flip(np.diag(np.rot90(tempCn_)))
        mx2_ = np.rot90(np.array([dg_, ]*self.j))
        #create the matrix of claims incurred in the future years
        mx1_=(totInc-cumInc+mx2_)/tempCn_

        mx1_[self.ix >= self.j]= np.nan

        if self. tail == False:
            return np.apply_along_axis(arr=mx1_[:,:-1],func1d=np.nanmean,axis=0)
        else:
            return np.concatenate((np.apply_along_axis(arr=mx1_[:, :-1], func1d=np.nanmean, axis=0),[1.]))

    def __SsComputer(self):
        """
        It computes Fisher-Lange settlement speeds. Given a JxJ triangle, this is going to be
        J-1 dimensional in case no tail is present and J dimensional in case of tail estimates.

        :return: settlement speed
        :rtype: ``numpy.ndarray``
        """
        temp = np.flip((np.diag(np.rot90(self.in_tr))*self.reported_claims[-1]/self.reported_claims)[:-1])

        if self.tail==True:
            temp = np.concatenate((temp,[self.cn_tr[0,-1]*self.reported_claims[-1]/self.reported_claims[0]]))

        return temp/np.sum(temp)

    def __SsTriangle(self):
        """
        It turns Fisher-Lange settlement speed array into a triangle. Given a JxJ triangle, this is going to be
        JxJ-1 dimensional in case no tail is present and JxJ dimensional in case of tail estimates.

        :return: settlement speed triangle
        :rtype: ``numpy.ndarray``
        """
        mx1_ = np.array([np.concatenate(([0.],self.ssFl_)),]*self.j)
        if self.tail==True:
            newix=np.concatenate((self.ix <= self.j, np.repeat(False, self.j).reshape(self.j, 1)), axis=1)
            mx1_[newix]=np.nan
        else:
            mx1_[self.ix <= self.j] = np.nan

        return np.apply_along_axis(arr=mx1_,func1d=hfns.normalizerNans,axis=1)

    def __fillNumbers(self):
        """
        It computes the lower triangle of numbers. Given a JxJ triangle, this is going to be
        JxJ dimensional in case no tail is present and Jx(J+1) dimensional in case of tail estimates.

        :return: number of payments
        :rtype: ``numpy.ndarray``
        """
        # aliquote
        if self.tail == False:
            v_ = np.concatenate((self.alphaFl_, [1.]))
        else:
            v_ = self.alphaFl_
        alq_ = np.flip(np.array([v_, ] * self.j).T)

        # diagonale
        dg_ = np.diag(np.flip(np.rot90(self.cn_tr)))
        amts_ = np.flip(np.array([dg_, ] * self.j).T)

        if self.tail==True:
            alq_=np.column_stack((np.ones(self.j), alq_))
            amts_ = np.column_stack((np.ones(self.j), amts_))

        # sviluppo finale
        nsdv_ = self.ss_tr * amts_ * alq_

        if self.tail==True:
            newix=np.concatenate((self.ix > self.j, np.repeat(True, self.j).reshape(self.j, 1)), axis=1)
            temp_=np.column_stack((self.in_tr,np.ones(self.j)))
            temp_[newix] = nsdv_[~np.isnan(nsdv_)]
            return temp_
        else:
            temp_=self.in_tr.copy()
            temp_[self.ix > self.j] = nsdv_[~np.isnan(nsdv_)]
            return temp_

    def __Flreserve(self):
        """
        It computes the reserve via the Fisher-Lange reserving model.

        :return: fisher-lange reserve
        :rtype: ``numpy.float64``
        """
        # self.predicted_Inumbers=self.__fillNumbers()
        self.predicted_Ipayments=self.predicted_Inumbers*self.ap_tr

        if self.tail==True:
            newix=np.concatenate((self.ix > self.j, np.repeat(True, self.j).reshape(self.j, 1)), axis=1)
            self.predicted_Ipayments[0,-1]=self.cp_tr[0,-1]
            return np.sum(self.predicted_Ipayments[newix])
        else:
            return np.sum(self.predicted_Ipayments[self.ix > self.j])

    def __StochasticCRM(self):
        """
        It computes the reserve according to the collective risk model based on the Fisher-Lange.

        :return: reserve prediction (simulations mean), reserve msep prediction, reserve skewness
        :rtype:``numpy.float64``,``numpy.float64``,``numpy.float64``

        """
        flag_ = np.repeat('ay' + str(0), self.ap_tr.shape[1]) #create a flag that will be used to pick the correct ay
        for ay in range(1,self.ap_tr.shape[0]):
            cell_= 'ay' + str(ay)
            temp_ = np.repeat(cell_, self.ap_tr.shape[1])
            flag_ =np.vstack((flag_,temp_))

        if self.tail == False:

            v1_ = self.predicted_Inumbers[self.ix > self.j] #numbers lower triangle
            v2_ = self.ap_tr[self.ix > self.j] #average payments lower triangle
            czj_v = self.czj_t[self.ix > self.j] #coefficient of variation lower triangle
            flag_v = flag_[self.ix > self.j]
        else:
            newix = np.concatenate((self.ix > self.j, np.repeat(True, self.j).reshape(self.j, 1)), axis=1) #lower triangle and tail
            v1_ = self.predicted_Inumbers[newix] #numbers lower triangle and tail
            v2_ = self.ap_tr[newix] #average payments lower triangle and tail
            czj_v = self.czj_t[newix]  #coefficient of variation lower triangle and tail
            flag_v = flag_[newix]

        np.random.seed(self.setseed)
        output = np.array([])
        # now = datetime.now()
        self.crmsep_ay=np.array([],dtype=np.float64) #to store the mean squared error of prediction
        self.crmul_ay = np.array([],dtype=np.float64) #to store the ultimate cost
        self.ay_reservecrm= np.array([],dtype=np.float64) #to store the reserve by time period
        mseptemp_=np.array([],dtype=np.float64)
        ultimatemp_=np.array([],dtype=np.float64)
        for i in range(0, len(v1_)): #crm computed on each lower triangle cell

            f_= flag_v[i] #flag the cell with the correspondent ay

            try:
                fp_= flag_v[i+1] #save next year flag. In case this is different from f_ the reserve variability is computed for the ay.
            except:
                fp_= 'stop'

            p1_ = v1_[i] #cell numbers
            p2_ = v2_[i] #cell average payment
            p3_ = czj_v[i] #cell coefficient of variation

            vec1_ = p1_ * self.gamma1.rvs(self.ntr_sim)
            vec2_ = p2_ ** 2 / (p3_ * self.gamma2.rvs(self.ntr_sim))
            vec3_ = p3_ * self.gamma2.rvs(self.ntr_sim) / p2_

            vec4_ = np.apply_along_axis(func1d=hfns.lrcrm_f1, arr=vec1_.reshape(-1, 1), axis=1, dist=self.pois).reshape(-1, ) #simulate all the CRMs for the cell
            mx_ = np.array([vec4_, vec2_, vec3_]).T #create a matrix of parameters
            temp_ = np.apply_along_axis(axis=1, arr=mx_, func1d=hfns.lrcrm_f2, dist=self.gamma3) #simulate the reserves
            if i == 0:
                output = np.array(temp_).reshape(-1, 1)
            else:
                output = np.column_stack((output, np.array(temp_).reshape(-1, 1)))

            mseptemp_=np.concatenate((mseptemp_,temp_)) #add to the ay estimate the simulated CRMs. It will be used for the
                                                        # mean reserve as well
            ultimatemp_=np.concatenate((ultimatemp_,[np.mean(temp_)]))
            if f_ != fp_: # in case next cell belongs to another ay, reserve variability is computed
                self.crmsep_ay=np.concatenate((self.crmsep_ay,[np.std(mseptemp_)]))
                self.ay_reservecrm=np.concatenate((self.ay_reservecrm,[np.sum(ultimatemp_)]))
                self.crmul_ay = np.concatenate((self.crmul_ay, [np.cumsum(ultimatemp_)[-1]]))
                mseptemp_=np.array([],dtype=np.float64)
                ultimatemp_= np.array([],dtype=np.float64)

            sys.stdout.write("\r")
            v = round((i + 1) / len(v1_) * 100, 3)
            str1 = ("[%-" + str(len(v1_)) + "s] %d%%")
            sys.stdout.write(str1 % ('=' * i, v))
            sys.stdout.flush()

        # then = datetime.now()
        print("")
        # logger.info('Time elapsed %s' % str(then - now))
        if self.tail == False:
            self.ay_reservecrm=np.concatenate(([0],self.ay_reservecrm))
            self.crmsep_ay=np.concatenate(([0],self.crmsep_ay))
            self.crmul_ay=np.concatenate(([0],self.crmul_ay))
        else: # I need to fill the last line of the ultimate in advance in case of CRM
            diagcml_ = self.predicted_Ipayments[-1,0]
            self.crmul_ay[self.predicted_Ipayments.shape[0]-1] = self.crmul_ay[self.predicted_Ipayments.shape[0]-1] + diagcml_

        for ay in range(0,self.predicted_Ipayments.shape[0]-self.tail): #add to the estimated random cumulative payments the upper triangle amounts
            diagcml_=np.cumsum(self.predicted_Ipayments[ay,:(self.j-ay-1-self.tail+1)])[-1] #separate it to make it readable
            self.crmul_ay[ay]=self.crmul_ay[ay]+diagcml_

        reserves_ = np.apply_along_axis(arr=output, func1d=np.sum, axis=1)
        return np.mean(reserves_),np.std(reserves_),stats.skew(reserves_)

    def SSPlot(self,start_=0):
        """
        It plots the settlement speed vector for each accident year.
        :param start_: starting accident year from which to plot.
        :type start_: ``int``
        """
        x_ = np.arange(0, self.j+self.tail)
        plt.title('Plot of the settlement speed from accident year %s' % start_)
        plt.xlabel('Development Year')
        plt.ylabel('Settlement Speed')
        for i in range(start_, self.j):
            plt.plot(x_, self.ss_tr[i, :], '-.',label='AY %s' % i)
            plt.legend()
        plt.show()

    def AverageCostPlot(self):
        """
        It plots the mean average cost for each development year.
        """
        x_ = np.arange(0, self.j+self.tail)
        plt.title('Plot of the Average Cost (mean of each DY, data and predictions)')
        plt.xlabel('Development Year')
        plt.ylabel('Average Cost')
        y_=np.apply_along_axis(arr=self.ap_tr,func1d=np.mean,axis=0)
        plt.plot(x_, y_,'-.', label='Mean Average Cost')
        plt.show()

    def AlphaPlot(self):
        """
        It plots the Fisher-Lange alpha.
        """
        x_ = np.arange(0, self.j+self.tail-1)
        plt.title('Plot of Alpha')
        plt.plot(x_,self.alphaFl_,'-.', label='Alpha')
        plt.xlabel('Development Year')
        plt.ylabel('Alpha')
        plt.show()

    def __reservebyAYFL(self):
        """
        It computes the Fisher-Lange reserve for each accident year and the Fisher-Lange ultimate cost for each accident year.

        :return: reserve for each accident year,ultimate cost for each accident year
        :rtype: ``numpy.ndarray``, ``numpy.ndarray``
        """
        self.ay_reserveFL=np.array([])
        self.ay_ultimateFL = np.array([])
        for ay in range(0, self.predicted_Ipayments.shape[0]):
            v_=self.predicted_Ipayments[ay,:]
            self.ay_reserveFL=np.concatenate((self.ay_reserveFL,[np.sum(v_[(self.j - ay):])]))
            self.ay_ultimateFL = np.concatenate((self.ay_ultimateFL,[np.cumsum(v_)[-1]]))

    def claimsreserving(self):
        """
        Table with claims reserve results. When the stochastic reserve according to the collective risk model is computed the results
        are compared with the Fisher-Lange.

        """
        self.__reservebyAYFL()
        ay_=np.arange(0,self.predicted_Ipayments.shape[0])
        data = np.dstack((ay_, self.ay_ultimateFL,self.ay_reserveFL)).reshape(-1, 3)
        if self.reserving_method == 'crm':
            data2=np.dstack((self.crmul_ay,self.ay_reservecrm,self.crmsep_ay)).reshape(-1, 3)
            data= np.column_stack((data,data2))
        l_ = ['time', 'ultimate FL', 'reserve FL']
        if self.reserving_method == 'crm':
            l2_=['ultimate CRM','reserve CRM','msep CRM']
            l_.extend(l2_)
        s_ = "{: >20} {: >20} {: >20} {: >20}"
        if self.reserving_method == 'crm':
            s_=s_+ " {: >20} {: >20} {: >20}"
        print(s_.format(" ", *l_))
        print("{: >20} {: >20}".format(" ", *[" ===================================================================================================================================="]))
        for row in data:
            print(s_.format("", *row))
        print('\n FL reserve: ', self.flreserve)
        if self.reserving_method == 'crm':
            print('\n CRM reserve: ', self.crmreserve)
            print('\n CRM msep: ', self.msep)

