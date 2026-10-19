argv<-commandArgs(trailingOnly=FALSE);self<-sub("^--file=","",argv[grepl("^--file=",argv)])
study<-normalizePath(file.path(dirname(self),"../.."))
.libPaths(c(file.path(study,".Rlib"),.libPaths()))
options(endes.gao.prior_cache=file.path(study,"resultados/competidores_gao/prior_cache"))
library(Matrix);library(INLA)
INLA::inla.setOption(num.threads="1:1")
source(file.path(study,"scripts/competitors/gao_stage2.R"))
gao_assess_stage2 <- function(v) {
  # Repeated sparse determinant calculations vary at about 1e-11, which can
  # perturb the finite-difference INLA hyperparameter optimizer at about 1e-6.
  # Require the same prior to 1e-9 and predictor moments to 1e-5 logit units.
  v$tolerances<-list(prior_grid=1e-9,prior_logdensity=1e-9,
     predictor_mean=1e-5,predictor_sd=1e-5,hyperparameter_mean=1e-4)
  v$prior_construction_pass<-v$max_prior_grid_difference<1e-9&&v$max_prior_logdensity_difference<1e-9
  v$stage2_model_pass<-isTRUE(v$prior_construction_pass)&&
     v$max_linear_predictor_mean_difference<1e-5&&
     v$max_linear_predictor_sd_difference<1e-5&&v$max_hyperparameter_mean_difference<1e-4
  v
}
if("--evaluate-existing"%in%commandArgs(trailingOnly=TRUE)) {
  path<-file.path(study,"resultados/competidores_gao/author_stage2_validation.json")
  v<-gao_assess_stage2(jsonlite::read_json(path,simplifyVector=TRUE))
  jsonlite::write_json(v,path,pretty=TRUE,auto_unbox=TRUE,digits=16)
  print(v);quit(save="no",status=if(isTRUE(v$stage2_model_pass))0L else 1L)
}
author<-new.env(parent=globalenv())
source(file.path(study,"fuentes/competidores/gao_wakefield/author_code/analysis/models.R"),local=author)
# Only elementary transformation aliases and an early return of the fitted
# model are changed. Original author source on disk remains untouched. The
# author's Monte Carlo summarization is not needed for comparing fitted models.
b<-paste(deparse(body(author$get_bym2_sdir_logit)),collapse="\n")
b<-gsub("SUMMER::logit","stats::qlogis",b,fixed=TRUE)
b<-gsub("SUMMER::expit","stats::plogis",b,fixed=TRUE)
b<-gsub("return(out_dat)","return(list(output=out_dat,fit=sd_fit))",b,fixed=TRUE)
b<-sub("sd_fit_sample <-", "return(list(fit=sd_fit))\n    sd_fit_sample <-",b,fixed=TRUE)
body(author$get_bym2_sdir_logit)<-parse(text=b)[[1]]
indir<-file.path(study,"resultados/resultados_comparadores/inputs/sparse_srs_2025/rep_00000/model_inputs")
area<-read.csv(file.path(indir,"area.csv"),colClasses=c(area="character"))
edges<-read.csv(file.path(indir,"adjacency_edges.csv"),colClasses="character")
adj<-sparseMatrix(i=match(edges$area1,area$area),j=match(edges$area2,area$area),x=1,
                  dims=c(nrow(area),nrow(area)))
adj<-pmax(adj,t(adj));rownames(adj)<-colnames(adj)<-area$area
e<-read.csv(file.path(study,"resultados/competidores_gao/pilot_sparse_srs/stage1_anemia.csv"),colClasses=c(area="character"))
e$region<-e$area;e$method<-"GREG"
e<-e[match(area$area,e$area),]
set.seed(20261005)
trace("inla.pc.bym.phi",where=asNamespace("INLA"),print=FALSE,
  exit=quote(assign("gao_native_prior",list(table=returnValue(),u=u,alpha=alpha),envir=.GlobalEnv)))
pub<-author$get_bym2_sdir_logit(e,adj,CI=.95)
untrace("inla.pc.bym.phi",where=asNamespace("INLA"))
adapt<-gao_stage2(e,area,adj)
cached<-gao_cached_phi_prior(adj)
nt<-matrix(scan(text=sub("table:","",gao_native_prior$table,fixed=TRUE),quiet=TRUE),ncol=2,byrow=TRUE)
ct<-matrix(scan(text=sub("table:","",cached$table,fixed=TRUE),quiet=TRUE),ncol=2,byrow=TRUE)
validation<-list(
 author_commit="29a89ec264d41b1638780db7689bfe43f609fc1e",
 comparison="Executed author BYM2 stage with only logit/expit aliases and exposed fitted object",
 max_linear_predictor_mean_difference=max(abs(pub$fit$summary.linear.predictor$mean-adapt$fit$summary.linear.predictor$mean)),
 max_linear_predictor_sd_difference=max(abs(pub$fit$summary.linear.predictor$sd-adapt$fit$summary.linear.predictor$sd)),
 max_hyperparameter_mean_difference=max(abs(pub$fit$summary.hyperpar$mean-adapt$fit$summary.hyperpar$mean)),
 native_pc_u=gao_native_prior$u,native_pc_alpha=gao_native_prior$alpha,
 cached_pc_u=cached$u,cached_pc_alpha=cached$alpha,
 prior_table_identical=identical(gao_native_prior$table,cached$table),
 max_prior_grid_difference=max(abs(nt[,1]-ct[,1])),
 max_prior_logdensity_difference=max(abs(nt[,2]-ct[,2])),
 numerical_summarization="Author's original fitted model exposed before its 1000-draw summarization. Posterior targets matched; marginal quadrature replaces Monte Carlo summaries.",
 stage2_model_pass=FALSE)
validation<-gao_assess_stage2(validation)
jsonlite::write_json(validation,file.path(study,"resultados/competidores_gao/author_stage2_validation.json"),
 pretty=TRUE,auto_unbox=TRUE,digits=16)
print(validation)
stopifnot(validation$stage2_model_pass)
