# Gao and Wakefield (2024), author implementation of the LGREG first stage.
# Author code: peteragao/SMA-SAE, commit 29a89ec264d41b1638780db7689bfe43f609fc1e.
# The operational estimator is mean_U(qhat) + Hajek_s(y-qhat), not a population
# total divided by an estimated population size. Population X must be known.
# Residual variance uses the full observed first-stage survey frame and a ratio
# linearization, matching the author's svyby(..., svymean) implementation.

gao_ratio_variance <- function(sample, frame, values, means, denominators,
                               area_names, singleton="adjust") {
  stopifnot(singleton %in% c("adjust","certainty"))
  f <- unique(frame[,c("psu_key","stratum")])
  stopifnot(!anyDuplicated(f$psu_key))
  psu_index <- match(sample$psu_key,f$psu_key)
  stopifnot(!anyNA(psu_index))
  strata <- split(seq_len(nrow(f)),f$stratum)
  out <- rep(NA_real_,length(area_names))
  for (a in seq_along(area_names)) {
    z <- which(sample$area==area_names[a] & is.finite(values))
    if (!length(z) || !is.finite(denominators[a]) || denominators[a]<=0) next
    score <- numeric(nrow(f))
    v <- sample$weight[z]*(values[z]-means[a])/denominators[a]
    t <- rowsum(v,psu_index[z],reorder=FALSE)
    score[as.integer(rownames(t))] <- t[,1]
    out[a] <- sum(vapply(strata,function(ii) {
      m<-length(ii);u<-score[ii]
      if(m>1) m/(m-1)*(sum(u*u)-sum(u)^2/m)
      else if(singleton=="adjust") sum(u*u) else 0
    },numeric(1)))
  }
  out
}

gao_stage1 <- function(sample, population_X, frame, outcome, covariates,
                       population_weight="population_weight",
                       singleton="adjust") {
  required <- c("area","weight","psu","stratum","psu_key",outcome,covariates)
  stopifnot(all(required %in% names(sample)),
            all(c("area",covariates,population_weight) %in% names(population_X)))
  stopifnot(all(is.finite(sample$weight)&sample$weight>0),
            all(is.finite(population_X[[population_weight]])&population_X[[population_weight]]>0))
  complete <- is.finite(sample[[outcome]]) & apply(is.finite(as.matrix(sample[covariates])),1,all)
  if(!any(complete)) stop("No measured observations with complete working covariates")
  if(any(!apply(is.finite(as.matrix(population_X[covariates])),1,all)))
    stop("Known population X distribution has missing covariates")
  formula <- reformulate(covariates,response=outcome)
  # Complete survey frame is retained as nonresponse placeholders. These rows
  # enter the design structure, never the working model estimating equations.
  dat <- sample[complete,required,drop=FALSE]
  phantom <- as.data.frame(matrix(NA_real_,nrow(frame),ncol(dat)))
  names(phantom)<-names(dat)
  phantom$area<-as.character(frame$area)
  phantom$weight<-frame$weight;phantom$psu<-frame$psu
  phantom$stratum<-frame$stratum;phantom$psu_key<-frame$psu_key
  dat<-rbind(dat,phantom)
  options(survey.lonely.psu=singleton)
  des<-survey::svydesign(ids=~psu,strata=~stratum,weights=~weight,
                         data=dat,nest=TRUE)
  fit<-survey::svyglm(formula,design=des,family=quasibinomial(),na.action=na.omit)
  if(any(!is.finite(coef(fit)))) stop("Nonfinite or rank-deficient working fit")
  qpop<-as.numeric(predict(fit,newdata=population_X,type="response"))
  s<-sample[complete,,drop=FALSE]
  qsample<-as.numeric(predict(fit,newdata=s,type="response"))
  residual<-s[[outcome]]-qsample
  areas<-sort(unique(as.character(population_X$area)))
  weighted_mean <- function(x,weights,groups) {
    num<-tapply(weights*x,groups,sum);den<-tapply(weights,groups,sum)
    as.numeric((num/den)[areas])
  }
  working<-weighted_mean(qpop,population_X[[population_weight]],population_X$area)
  rbar<-weighted_mean(residual,s$weight,s$area)
  direct<-weighted_mean(s[[outcome]],s$weight,s$area)
  den<-as.numeric(tapply(s$weight,s$area,sum)[areas])
  var<-gao_ratio_variance(s,frame,residual,rbar,den,areas,singleton)
  dvar<-gao_ratio_variance(s,frame,s[[outcome]],direct,den,areas,singleton)
  n<-as.numeric(table(factor(s$area,levels=areas)))
  qrange<-vapply(areas,function(a) {
    z<-qsample[s$area==a];if(!length(z)) NA_real_ else diff(range(z))
  },numeric(1))
  out<-data.frame(area=areas,est=working+rbar,var=var,direct=direct,
                   direct_var=dvar,working_mean=working,residual_mean=rbar,
                   denominator=den,n=n,q_range_in_sample=qrange)
  list(estimates=out,fit=fit,
       metadata=list(outcome=outcome,covariates=covariates,
         estimator="population-weighted mean(qhat) + sample-weighted mean(y-qhat)",
         variance="Full-frame first-stage WR ratio linearization of residual means; beta estimation ignored",
         singleton=singleton,working_model_converged=isTRUE(fit$converged),
         author_code_commit="29a89ec264d41b1638780db7689bfe43f609fc1e"))
}
