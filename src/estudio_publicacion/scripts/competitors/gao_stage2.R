# Spatial Gaussian-logit BYM2 stage, following the Gao/Wakefield author code.
# Numerical posterior marginal integration replaces their 1000 random draws;
# the model and PC priors are unchanged, interval level is explicitly 95%.
gao_cached_phi_prior <- function(adjacency,u=.5,alpha=2/3) {
  # This is the exact table generated inside INLA::inla.ffield.section for
  # BYM2 prior='pc', not an alternative prior. It depends only on the graph.
  a<-as(adjacency,"dgCMatrix")
  key<-digest::digest(list(dim=a@Dim,p=a@p,i=a@i,x=a@x,u=u,alpha=alpha,
    scale.model=TRUE,adjust.for.con.comp=TRUE,
    INLA=as.character(utils::packageVersion("INLA"))),algo="sha256")
  cache_dir<-getOption("endes.gao.prior_cache",file.path(tempdir(),"gao_pc_phi_cache"))
  dir.create(cache_dir,recursive=TRUE,showWarnings=FALSE)
  path<-file.path(cache_dir,paste0("pc_phi_",key,".rds"))
  if(file.exists(path)) {
    prior<-readRDS(path);stopifnot(identical(prior$graph_key,key))
    return(prior)
  }
  table<-INLA:::inla.pc.bym.phi(graph=INLA::inla.read.graph(a),u=u,alpha=alpha,
    scale.model=TRUE,return.as.table=TRUE,adjust.for.con.comp=TRUE)
  prior<-list(table=table,graph_key=key,table_sha256=digest::digest(table,algo="sha256",serialize=FALSE),
    official_function="INLA:::inla.pc.bym.phi",u=u,alpha=alpha,
    INLA_version=as.character(utils::packageVersion("INLA")))
  tmp<-paste0(path,".",Sys.getpid(),".tmp")
  saveRDS(prior,tmp);stopifnot(file.rename(tmp,path))
  prior
}

gao_stage2 <- function(estimates, area, adjacency, covariates=character(),
                       observation_mode="author_strict", level=.95, seed=20261005) {
  stopifnot(observation_mode %in% c("author_strict","common_continuity"))
  stopifnot(nrow(adjacency)==nrow(area),all(c("area")%in%names(area)))
  if(!requireNamespace("INLA",quietly=TRUE))stop("INLA required in the project-local R library")
  # The INLA call argument controls its C backend; these global process-local
  # settings also prevent R-side posterior postprocessing from forking ten cores.
  INLA::inla.setOption(num.threads="1:1",blas.num.threads=1)
  options(mc.cores=1L)
  cache_start<-proc.time()[[3]]
  phi_prior<-gao_cached_phi_prior(adjacency)
  cache_seconds<-proc.time()[[3]]-cache_start
  n<-nrow(area);i<-match(as.character(area$area),as.character(estimates$area))
  p<-estimates$est[i];v<-estimates$var[i]
  if(observation_mode=="author_strict") {
    use<-is.finite(p)&p>0&p<1&is.finite(v)&v>1e-5
    z<-qlogis(p);vz<-v/(p*(1-p))^2
  } else {
    # Extension only: caller supplies the same transformed observation and
    # regularized sampling variance as the primary area models, fitted on train.
    z<-estimates$z[i];vz<-estimates$var_z[i]
    use<-is.finite(z)&is.finite(vz)&vz>0
  }
  z[!use]<-NA_real_
  # Preserve the author's arithmetic order for precision. The inverse of the
  # algebraically identical variance can perturb the INLA optimizer at 1e-6.
  precision<-if(observation_mode=="author_strict") (p^2*(1-p)^2)/v else 1/vz
  precision[!use]<-1
  dat<-area;dat$z<-z;dat$region<-seq_len(n)
  stopifnot(all(is.finite(as.matrix(dat[covariates]))))
  hyper<-list(prec=list(prior="pc.prec",param=c(5,.01)),
              phi=list(prior=phi_prior$table,param=numeric(0)))
  formula<-as.formula(paste("z ~",paste(c(covariates,
     "f(region,model='bym2',graph=adjacency,hyper=hyper,scale.model=TRUE)"),collapse=" + ")))
  environment(formula)<-environment()
  set.seed(seed);start<-proc.time()[[3]]
  fit<-INLA::inla(formula,data=dat,family="gaussian",scale=precision,
    control.family=list(hyper=list(prec=list(initial=0,fixed=TRUE))),
    control.fixed=list(mean=0,prec=0,mean.intercept=0,prec.intercept=0),
    control.predictor=list(compute=TRUE),
    control.compute=list(config=FALSE,return.marginals.predictor=TRUE),
    num.threads="1:1")
  elapsed<-proc.time()[[3]]-start
  marg<-fit$marginals.linear.predictor
  if(length(marg)!=n)stop("Missing posterior predictor marginals")
  alpha<-(1-level)/2
  pm<-vapply(marg,function(m)INLA::inla.emarginal(plogis,m),numeric(1))
  lo<-vapply(marg,function(m)plogis(INLA::inla.qmarginal(alpha,m)),numeric(1))
  hi<-vapply(marg,function(m)plogis(INLA::inla.qmarginal(1-alpha,m)),numeric(1))
  pv<-vapply(marg,function(m)INLA::inla.emarginal(function(x)plogis(x)^2,m),numeric(1))-pm^2
  result<-data.frame(area=as.character(area$area),p_mean=pm,lo95=lo,hi95=hi,
    p_var=pmax(pv,0),eta_mean=fit$summary.linear.predictor$mean,
    eta_sd=fit$summary.linear.predictor$sd,observed_in_stage2=use,
    has_stage1_estimate=is.finite(p),author_filter_excluded=is.finite(p)&!use)
  stopifnot(all(is.finite(pm)),all(pm>=0&pm<=1),all(lo<=hi),all(lo>=0&hi<=1))
  status<-list(INLA_version=as.character(utils::packageVersion("INLA")),
    elapsed_seconds=elapsed,observation_mode=observation_mode,
    sampling_observations=sum(use),excluded_measured=sum(is.finite(p)&!use),
    predictions=n,predictions_without_usable_observation=sum(!use),
    fixed_covariates=covariates,
    pc_sigma="P(sigma > 5) = 0.01",pc_phi="P(phi < 0.5) = 2/3",
    pc_phi_cache_seconds=cache_seconds,pc_phi_table_sha256=phi_prior$table_sha256,
    pc_phi_graph_key=phi_prior$graph_key,
    pc_phi_cache_method="Exact response-independent INLA table cached; same official function as native prior='pc'",
    fixed_prior="flat improper prior; precision 0 including intercept",
    graph="provided symmetric binary adjacency; ICAR scaled in INLA, sum-to-zero constraint",
    posterior="INLA marginal numerical integration; author code instead draws 1000 joint posterior samples",
    level=level,INLA_ok=identical(fit$ok,TRUE),
    author_code_commit="29a89ec264d41b1638780db7689bfe43f609fc1e",
    design_consistency_scope="Unsampled or excluded areas have model-only predictions; no design-consistency claim",
    hyperparameters=fit$summary.hyperpar)
  list(predictions=result,status=status,fit=fit)
}
