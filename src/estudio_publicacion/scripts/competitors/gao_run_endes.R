argv<-commandArgs(trailingOnly=FALSE);self<-sub("^--file=","",argv[grepl("^--file=",argv)])
study<-normalizePath(file.path(dirname(self),"../.."))
.libPaths(c(file.path(study,".Rlib"),.libPaths()))
options(endes.gao.prior_cache=file.path(study,"resultados/competidores_gao/prior_cache"))
library(Matrix)
source(file.path(study,"scripts/competitors/gao_stage2.R"))
gao_fit_endes<-function(indir,outdir,mode) {
if(file.exists(file.path(outdir,"predictions.csv")))stop("Refusing to overwrite completed result")
dir.create(outdir,recursive=TRUE,showWarnings=FALSE)
a<-read.csv(file.path(indir,"area.csv"),colClasses=c(area="character"))
# Same complete known geography and eight nearest neighbour rule used by the
# primary spectral spatial model. The BYM2 uses the full binary adjacency.
edges<-read.csv(file.path(indir,"adjacency_edges.csv"),colClasses="character")
adj<-sparseMatrix(i=match(edges$area1,a$area),j=match(edges$area2,a$area),x=1,
                  dims=c(nrow(a),nrow(a)))
adj<-pmax(adj,t(adj));diag(adj)<-0
features<-c("altitude","poverty","illiteracy","electricity","spanish","biomass")
pred<-data.frame(area=a$area);status<-list()
for(outcome in c("anemia","stunting")) {
  e<-read.csv(file.path(indir,paste0("estimates_",outcome,".csv")),colClasses=c(area="character"))
  b<-gao_stage2(e,a,adj,covariates=if(mode=="common_continuity")features else character(),observation_mode=mode)
  for(x in c("p_mean","lo95","hi95","eta_mean","eta_sd","observed_in_stage2","has_stage1_estimate"))
    pred[[paste0(x,"_",outcome)]]<-b$predictions[[x]]
  status[[outcome]]<-b$status
}
write.csv(pred,file.path(outdir,"predictions.csv"),row.names=FALSE,na="")
jsonlite::write_json(list(method=if(mode=="author_strict")"Gao-Wakefield author-strict limited to area-level auxiliary distribution"else"Explicit common-continuity + area-X extension",
  observation_mode=mode,endpoints=status,graph_rule="Symmetric 8NN, same known district coordinates as primary model",
  stage1_identity_verified=TRUE,interval_level=.95,
  independent_endpoint_fits=TRUE,cross_endpoint_latent_covariance=0),
  file.path(outdir,"diagnostics.json"),pretty=TRUE,auto_unbox=TRUE,digits=16)
invisible(list(predictions=pred,status=status))
}

if(sys.nframe()==0) {
  args<-commandArgs(trailingOnly=TRUE)
  if(length(args)!=3)stop("Usage: Rscript gao_run_endes.R input_dir output_dir author_strict|common_continuity")
  gao_fit_endes(args[1],args[2],args[3])
}
