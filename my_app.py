import gemdata as gemdata
# Dati Fisher-Lange
ip_= gemdata.incremental_payments # Incremental payments
in_= gemdata.incurred_number # numbers
cp_= gemdata.cased_payments # reserved amount
cn_= gemdata.cased_number # reserved numbers
reported_= gemdata.reported_claims # reported claims vector
infl_= gemdata.claims_inflation # claims inflation
rm_='fisher_lange' # reserving model
tail_=True #boolean

from lossreserve import LossReserve
lr = LossReserve(tail=tail_,
               incremental_payments=ip_,
               cased_payments=cp_,
               cased_number=cn_,
               reported_claims=reported_,
               incurred_number=in_,
               reserving_method=rm_,
               claims_inflation=infl_)


lr.loss_reserving()

lr.alpha_plot()
lr.ss_plot(start_=7)

lr_crm = LossReserve(tail=True,
incremental_payments=ip_,
cased_payments=cp_,
cased_number=cn_,
reported_claims=reported_,
incurred_number=in_,
reserving_method='crm',
claims_inflation=infl_,
ntr_sim=int(10),
mixing_fq_par={'a':1/.08**2,'scale':.08**2},
mixing_sev_par={'a':1/.08**2,'scale':.08**2})
