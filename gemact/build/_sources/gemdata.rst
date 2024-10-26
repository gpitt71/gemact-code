Data sets included in GEMAct
====================================

GEMAct includes varous data sets for users interested in exploring our examples and functionalities. 
The current version of the package has data sets for claims reserving. 

Data sets for loss reserving
--------------------------------

The SIFA data set
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The first data set is the SIFA data set from :cite:t:`savelliDATA`.

* ``incremental_payments``: Triangle of individual payments

* ``incurred_number``: Triangle of claim payments numbers

* ``cased_number``: Triangle of open numbers

* ``cased_payments``: Triangle of cased amounts

* ``reported_claims``: Reported claims by accident period

* ``claims_inflation``: Vector of inflation from :cite:t:`savelliDATA`.

* ``czj``: Coefficients of variation by development period from :cite:t:`savelliDATA`.

Data set simulated with SYNTHEtic and SPLICE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We also include the output of a data set simulate with the claims simulator available in the R package `SYNTHEtic <https://cran.rstudio.com/web/packages/SynthETIC/index.html>`_. 
This data set was used in the illustration of the ``LossReserve``  `in the manuscript pre-print <https://arxiv.org/abs/2303.01129>`_.

* ``incremental_payments_sim``: Triangle of individual payments

* ``incurred_number_sim``: Triangle of claim payments numbers

* ``cased_number_sim``: Triangle of open numbers

* ``cased_payments_sim``: Triangle of cased amounts

* ``reported_claims_sim``: Reported claims by accident period

* ``czj_sim``: Coefficients of variation by development period from :cite:t:`savelliDATA`.


Below we provide the R code for interested readers::

    library(SynthETIC)
    library(SPLICE)
    library(data.table)
    library(tidyr)
    library(reticulate)
    library(ChainLadder)


    clean.lt <- function(x){
    J=dim(x)[2]
    for(j in 1:J){
    for(i in 1:J){

        if((i+j) > (J+1)){x[i,j]=NA}

    }}
    return(x)}

    years <- 9

    claims_generator_mdn <- function(random_state,years){

    set.seed(random_state)
    set_parameters(ref_claim = 200000, time_unit = 1)
    ref_claim <- return_parameters()[1]
    time_unit <- return_parameters()[2]

    I <- years / time_unit
    E <- c(rep(200000, I)) # effective annual exposure rates
    lambda <- c(rep(0.1, I))

    n_vector <- claim_frequency(I = I, E = E, freq = lambda)
    # n_vector

    # Occurrence time of each claim r, for each period i
    occurrence_times <- claim_occurrence(frequency_vector = n_vector)

    S_df <- function(s) {
    # truncate and rescale
    if (s < 30) {
    return(0)
    } else {
    p_trun <- pnorm(s^0.2, 9.5, 3) - pnorm(30^0.2, 9.5, 3)
    p_rescaled <- p_trun/(1 - pnorm(30^0.2, 9.5, 3))
    return(p_rescaled)
    }
    }

    claim_sizes <- claim_size(frequency_vector = n_vector,
                            simfun = S_df, type = "p", range = c(0, 1e24))



    notidel_param <- function(claim_size, occurrence_period) {
    # NOTE: users may add to, but not remove these two arguments (claim_size,
    # occurrence_period) as they are part of SynthETIC's internal structure

    # specify the target mean and target coefficient of variation
    target_mean <- 0.4930517
    target_cv <- 1.919494
    # convert to Weibull parameters
    shape <- get_Weibull_parameters(target_mean, target_cv)[1]
    scale <- get_Weibull_parameters(target_mean, target_cv)[2]

    c(shape = shape, scale = scale)
    }

    ## output
    notidel <- claim_notification(n_vector, claim_sizes,
                                rfun = rweibull, paramfun = notidel_param)




    setldel_param <- function(claim_size, occurrence_period) {
    # NOTE: users may add to, but not remove these two arguments (claim_size,
    # occurrence_period) as they are part of SynthETIC's internal structure

    # specify the target Weibull mean

    target_mean <- 6.575582

    # specify the target Weibull coefficient of variation
    target_cv <- 1.02259

    c(shape = get_Weibull_parameters(target_mean, target_cv)[1, ],
    scale = get_Weibull_parameters(target_mean, target_cv)[2, ])
    }


    setldel <- claim_closure(n_vector, claim_sizes, rfun = rweibull, paramfun = setldel_param)


    benchmark_1 <- 0.0375 * ref_claim
    benchmark_2 <- 0.075 * ref_claim
    rmixed_payment_no <- function(n, claim_size, claim_size_benchmark_1, claim_size_benchmark_2) {
    # construct the range indicators
    test_1 <- (claim_size_benchmark_1 < claim_size & claim_size <= claim_size_benchmark_2)
    test_2 <- (claim_size > claim_size_benchmark_2)

    # if claim_size <= claim_size_benchmark_1
    no_pmt <- sample(c(1, 2), size = n, replace = T, prob = c(1/2, 1/2))
    # if claim_size is between the two benchmark values
    no_pmt[test_1] <- sample(c(2, 3), size = sum(test_1), replace = T, prob = c(1/3, 2/3))
    # if claim_size > claim_size_benchmark_2
    no_pmt_mean <- pmin(8, 4 + log(claim_size/claim_size_benchmark_2))
    prob <- 1 / (no_pmt_mean - 3)
    no_pmt[test_2] <- stats::rgeom(n = sum(test_2), prob = prob[test_2]) + 4

    no_pmt
    }

    no_payments <- claim_payment_no(n_vector,
                                claim_sizes,
                                rfun = rmixed_payment_no,
                                claim_size_benchmark_1 = 0.0375 * ref_claim,
                                claim_size_benchmark_2 = 0.075 * ref_claim)

    rmixed_payment_size <- function(n, claim_size) {
    # n = number of simulations, here n should be the number of partial payments
    if (n >= 4) {
    # 1) Simulate the "complement" of the proportion of total claim size
    #    represented by the last two payments
    p_mean <- 1 - min(0.95, 0.75 + 0.04*log(claim_size/(0.10 * ref_claim)))
    p_CV <- 0.20
    p_parameters <- get_Beta_parameters(target_mean = p_mean, target_cv = p_CV)
    last_two_pmts_complement <- stats::rbeta(
        1, shape1 = p_parameters[1], shape2 = p_parameters[2])
    last_two_pmts <- 1 - last_two_pmts_complement

    # 2) Simulate the proportion of last_two_pmts paid in the second last payment
    q_mean <- 0.9
    q_CV <- 0.03
    q_parameters <- get_Beta_parameters(target_mean = q_mean, target_cv = q_CV)
    q <- stats::rbeta(1, shape1 = q_parameters[1], shape2 = q_parameters[2])

    # 3) Calculate the respective proportions of claim amount paid in the
    #    last 2 payments
    p_second_last <- q * last_two_pmts
    p_last <- (1-q) * last_two_pmts

    # 4) Simulate the "unnormalised" proportions of claim amount paid
    #    in the first (m - 2) payments
    p_unnorm_mean <- last_two_pmts_complement/(n - 2)
    p_unnorm_CV <- 0.10
    p_unnorm_parameters <- get_Beta_parameters(
        target_mean = p_unnorm_mean, target_cv = p_unnorm_CV)
    amt <- stats::rbeta(
        n - 2, shape1 = p_unnorm_parameters[1], shape2 = p_unnorm_parameters[2])

    # 5) Normalise the proportions simulated in step 4
    amt <- last_two_pmts_complement * (amt/sum(amt))
    # 6) Attach the last 2 proportions, p_second_last and p_last
    amt <- append(amt, c(p_second_last, p_last))
    # 7) Multiply by claim_size to obtain the actual payment amounts
    amt <- claim_size * amt

    } else if (n == 2 | n == 3) {
    p_unnorm_mean <- 1/n
    p_unnorm_CV <- 0.10
    p_unnorm_parameters <- get_Beta_parameters(
        target_mean = p_unnorm_mean, target_cv = p_unnorm_CV)
    amt <- stats::rbeta(
        n, shape1 = p_unnorm_parameters[1], shape2 = p_unnorm_parameters[2])
    # Normalise the proportions and multiply by claim_size to obtain the actual payment amounts
    amt <- claim_size * amt/sum(amt)

    } else {
    # when there is a single payment
    amt <- claim_size
    }
    return(amt)
    }

    ## output
    payment_sizes <- claim_payment_size(n_vector, claim_sizes, no_payments,
                                    rfun = rmixed_payment_size)


    r_pmtdel <- function(n, claim_size, setldel, setldel_mean) {
    result <- c(rep(NA, n))

    # First simulate the unnormalised values of d, sampled from a Weibull distribution
    if (n >= 4) {
    # 1) Simulate the last payment delay
    unnorm_d_mean <- (1 / 4) / time_unit
    unnorm_d_cv <- 0.20
    parameters <- get_Weibull_parameters(target_mean = unnorm_d_mean, target_cv = unnorm_d_cv)
    result[n] <- stats::rweibull(1, shape = parameters[1], scale = parameters[2])

    # 2) Simulate all the other payment delays
    for (i in 1:(n - 1)) {
        unnorm_d_mean <- setldel_mean / n
        unnorm_d_cv <- 0.35
        parameters <- get_Weibull_parameters(target_mean = unnorm_d_mean, target_cv = unnorm_d_cv)
        result[i] <- stats::rweibull(1, shape = parameters[1], scale = parameters[2])
    }

    } else {
    for (i in 1:n) {
        unnorm_d_mean <- setldel_mean / n
        unnorm_d_cv <- 0.35
        parameters <- get_Weibull_parameters(target_mean = unnorm_d_mean, target_cv = unnorm_d_cv)
        result[i] <- stats::rweibull(1, shape = parameters[1], scale = parameters[2])
    }
    }

    # Normalise d such that sum(inter-partial delays) = settlement delay
    # To make sure that the pmtdels add up exactly to setldel, we treat the last one separately
    result[1:n-1] <- (setldel/sum(result)) * result[1:n-1]
    result[n] <- setldel - sum(result[1:n-1])

    return(result)
    }

    param_pmtdel <- function(claim_size, setldel, occurrence_period) {
    # mean settlement delay
    if (claim_size < (0.10 * ref_claim) & occurrence_period >= 21) {
    a <- min(0.85, 0.65 + 0.02 * (occurrence_period - 21))
    } else {
    a <- max(0.85, 1 - 0.0075 * occurrence_period)
    }
    mean_quarter <- a * min(25, max(1, 6 + 4*log(claim_size/(0.10 * ref_claim))))
    target_mean <- mean_quarter / 4 / time_unit

    c(claim_size = claim_size,
    setldel = setldel,
    setldel_mean = target_mean)
    }


    ## output
    payment_delays <- claim_payment_delay(
    n_vector, claim_sizes, no_payments, setldel,
    rfun = r_pmtdel, paramfun = param_pmtdel,
    occurrence_period = rep(1:I, times = n_vector))


    # payment times on a continuous time scale
    payment_times <- claim_payment_time(n_vector, occurrence_times, notidel, payment_delays)
    # payment times in periods
    payment_periods <- claim_payment_time(n_vector, occurrence_times, notidel, payment_delays,
                                        discrete = TRUE)

    demo_rate <- (1 + 0.02)^(1/4) - 1
    base_inflation_past <- rep(demo_rate, times = 40)
    base_inflation_future <- rep(demo_rate, times = 40)
    base_inflation_vector <- c(base_inflation_past, base_inflation_future)

    # Superimposed inflation:
    # 1) With respect to occurrence "time" (continuous scale)
    SI_occurrence <- function(occurrence_time, claim_size) {
    {1}
    }
    # 2) With respect to payment "time" (continuous scale)
    # -> compounding by user-defined time unit
    SI_payment <- function(payment_time, claim_size) {
    period_rate <- (1 + 0)^(time_unit) - 1
    beta <- period_rate
    (1 + beta)^payment_time
    }


    payment_inflated <- claim_payment_inflation(
    n_vector,
    payment_sizes,
    payment_times,
    occurrence_times,
    claim_sizes,
    base_inflation_vector,
    SI_occurrence,
    SI_payment
    )


    # construct a "claims" object to store all the simulated quantities
    all_claims <- claims(
    frequency_vector = n_vector,
    occurrence_list = occurrence_times,
    claim_size_list = claim_sizes,
    notification_list = notidel,
    settlement_list = setldel,
    no_payments_list = no_payments,
    payment_size_list = payment_sizes,
    payment_delay_list = payment_delays,
    payment_time_list = payment_times,
    payment_inflated_list = payment_inflated
    )


    transaction_dataset <- generate_transaction_dataset(
    all_claims,
    adjust = FALSE # to keep the original (potentially out-of-bound) simulated payment times
    )

    # SPLICE ----

    no_majRev_param <- function(claim_size) {
    majRevNo_mean <- pmax(1, log(claim_size / 15000) - 2)
    c(lambda = majRevNo_mean)
    }

    ## implementation and output
    # major_test <- claim_majRev_freq(
    #   test_claims, rfun = actuar::rztpois, paramfun = no_majRev_param)

    # major revisions
    major <- claim_majRev_freq(all_claims, rfun = actuar::rztpois, paramfun = no_majRev_param)
    major <- claim_majRev_time(all_claims, major)
    major <- claim_majRev_size(major)

    # minor revisions
    minor <- claim_minRev_freq(all_claims)
    minor <- claim_minRev_time(all_claims, minor)
    minor <- claim_minRev_size(all_claims, major, minor)

    # development of case estimates
    test <- claim_history(all_claims, major, minor)

    return(list(synthOUT=transaction_dataset,
            spliceOUT=test))

    }


    # Generate data sets ----
    out0 <- claims_generator_mdn(random_state=1,years=years)

    transaction_dataset = out0$synthOUT
    test = out0$spliceOUT
    i_tri = out0$CL_input
    dt_synthetic <- as.data.table(transaction_dataset)[,payment_period:=payment_period-occurrence_period]


    # incrementals
    incremental_payments <- dt_synthetic[,
                                        .(incremental_payments=sum(payment_size)),
                                        by=.(occurrence_period,payment_period)]

    incremental_payments_tr <- ChainLadder::as.triangle(incremental_payments,
                                                    origin = 'occurrence_period',
                                                    value = 'incremental_payments',
                                                    dev='payment_period')[,1:years]

    #closed claims
    closed_claims <- dt_synthetic[,
                                .(closing_time=ceiling(notidel[1])+ floor(setldel[1])),
                                by=.(claim_no,
                                    occurrence_period)][,
                                                        .(closed_claims=.N),
                                                        by=.(occurrence_period,
                                                            closing_time)]

    closed_claims_tr <- ChainLadder::as.triangle(closed_claims,
                                                origin = 'occurrence_period',
                                                value = 'closed_claims',
                                                dev='closing_time')[,1:years]

    #payments numbers
    number_payments <- dt_synthetic[,
                                .(number_of_payments=.N),
                                by=.(occurrence_period,
                                        payment_period)][order(occurrence_period,payment_period),
                                                        .(occurrence_period,
                                                        number_of_payments,
                                                        payment_period=payment_period)]

    number_payments_tr <- ChainLadder::as.triangle(number_payments,
                                                origin = 'occurrence_period',
                                                value = 'number_of_payments',
                                                dev='payment_period')[,1:years]
    #reported claims
    reports_number <- dt_synthetic[,
                                .(reporting_time=ceiling(notidel[1])),
                                by=.(claim_no,
                                    occurrence_period)][,
                                                        .(reported_claims=.N),
                                                        by=.(occurrence_period,
                                                                reporting_time)]

    reports_number_tr <- ChainLadder::as.triangle(reports_number,
                                                origin = 'occurrence_period',
                                                value = 'reported_claims',
                                                dev='reporting_time')[,1:years]

    reports_number_v <- reports_number[(reporting_time+occurrence_period-1)<=max(occurrence_period),
    ][,.(reported_claims=sum(reported_claims)),
    by=.(occurrence_period,
        reporting_time)][,.(reported_claims=sum(reported_claims)),by=.(occurrence_period)]

    # Open claims

    open_tr<-matrix(NA,
                nrow=years,
                ncol=years)

    open_tr[,1]=unname(reports_number_tr[,1]-closed_claims_tr[,1])

    # reports_number_tr = cbind(reports_number_tr,
    #                           matrix(NA,
    #                                  nrow=years,
    #                                  ncol=years-ncol(reports_number_tr)))


    reports_number_tr <- reports_number_tr[,1:years]

    reports_number_tr[is.na(reports_number_tr)]=0

    # reports_number_tr <- cbind(reports_number_tr, matrix(0,nrow=years,ncol = years-ncol(reports_number_tr)))

    for(col in 2:years){

    open_tr[,col]=open_tr[,col-1]+reports_number_tr[,col]-closed_claims_tr[,col]}

    # czj
    czj<-dt_synthetic[,.(payment_size=payment_size,
                        occurrence_period=occurrence_period,
                        payment_period=pmin(payment_period,max(occurrence_period))),][(payment_period+occurrence_period)<=max(occurrence_period)][order(payment_period),.(czj=sd(payment_size)/mean(payment_size)),.(payment_period)]


    czj[is.na(czj)]=1

    # cased payments

    cased_amount <- output_incurred(test, incremental = T)
