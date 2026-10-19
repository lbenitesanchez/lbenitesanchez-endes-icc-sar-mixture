argv<-commandArgs(trailingOnly=FALSE);self<-sub("^--file=","",argv[grepl("^--file=",argv)])
study<-normalizePath(file.path(dirname(self),"../.."))
.libPaths(c(file.path(study,".Rlib"),.libPaths()))
options(endes.gao.prior_cache=file.path(study,"resultados/competidores_gao/prior_cache"))
library(Matrix)
source(file.path(study,"scripts/competitors/gao_stage1.R"))
source(file.path(study,"scripts/competitors/gao_stage2.R"))
gao_fit_synthetic <- function(indir,outdir,variant="area_only",stage2_mode="author_strict",
                              extension_file=NULL) {
indir<-normalizePath(indir)
stopifnot(variant %in% c("area_only","full_population_X"))
stopifnot(stage2_mode %in% c("author_strict","common_continuity"))
if(stage2_mode=="common_continuity" && variant!="area_only")
  stop("Full-H and common-continuity are separate experiments")
if(basename(indir)!="model_inputs")stop("Only blinded model_inputs directory is permitted")
dir.create(outdir,recursive=TRUE,showWarnings=FALSE)
if(file.exists(file.path(outdir,"predictions.csv")))stop("Refusing to overwrite completed comparator output")
sample<-read.csv(file.path(indir,"sample.csv"),colClasses=c(area="character",psu="character",stratum="character"))
area<-read.csv(file.path(indir,"area.csv"),colClasses=c(area="character"))
pop<-read.csv(file.path(indir,"population_X.csv"),colClasses=c(area="character"))
edges<-read.csv(file.path(indir,"adjacency_edges.csv"),colClasses="character")
sample$psu_key<-paste(sample$stratum,sample$psu,sep="|")
frame<-sample;frame$weight<-sample$base_weight
features<-c("X1","X2")
if(variant=="full_population_X" && "H" %in% names(pop)) features<-c(features,"H")
if(variant=="full_population_X" && !("H" %in% names(pop)))
  stop("Full unit-level variant requires a unit-varying H distribution; area-only scenario has no full variant")
adj<-Matrix::sparseMatrix(i=match(edges$area1,area$area),j=match(edges$area2,area$area),
                          x=1,dims=c(nrow(area),nrow(area)))
adj<-pmax(adj,t(adj));diag(adj)<-0
extension<-NULL
if(stage2_mode=="common_continuity") {
  if(is.null(extension_file))stop("Explicit extension observation file required")
  extension<-read.csv(extension_file,colClasses=c(area="character",outcome="character"))
}
pred<-data.frame(area=area$area);status<-list();start<-proc.time()[[3]]
for(outcome in c("anemia","stunting")) {
  s<-sample;s$weight<-s[[paste0("weight_",outcome)]]
  # An unmeasured outcome carries no estimating equation. A base design weight
  # here retains the child's PSU in the full frame, rather than imputing y.
  s$weight[!is.finite(s$weight)]<-s$base_weight[!is.finite(s$weight)]
  a<-gao_stage1(s,pop,frame,paste0("y_",outcome),features)
  write.csv(a$estimates,file.path(outdir,paste0("stage1_",outcome,".csv")),row.names=FALSE,na="")
  e<-a$estimates
  if(stage2_mode=="common_continuity") {
    obs<-extension[extension$outcome==outcome,];j<-match(e$area,obs$area)
    e$z<-obs$z[j];e$var_z<-obs$var_z[j]
  }
  b<-gao_stage2(e,area,adj,covariates=if(stage2_mode=="common_continuity") c("X1","X2") else character(),
                observation_mode=stage2_mode)
  j<-match(pred$area,b$predictions$area)
  for(v in c("p_mean","lo95","hi95"))pred[[paste0(v,"_",outcome)]]<-b$predictions[[v]][j]
  pred[[paste0("observed_in_stage2_",outcome)]]<-b$predictions$observed_in_stage2[j]
  status[[outcome]]<-modifyList(a$metadata,b$status)
}
write.csv(pred,file.path(outdir,"predictions.csv"),row.names=FALSE,na="")
jsonlite::write_json(list(method=if(stage2_mode=="author_strict")"Gao-Wakefield BYM2 author-strict" else "Gao BYM2 common-continuity + area-X extension",
  variant=variant,stage2_mode=stage2_mode,
  endpoints=status,elapsed_seconds=proc.time()[[3]]-start,
  truth_access=FALSE,uses_response_probability_IPW=any(sample$pi_response_anemia<1 | sample$pi_response_stunting<1,na.rm=TRUE),
  interval_level=.95,input_dir_basename=basename(dirname(indir))),
  file.path(outdir,"diagnostics.json"),auto_unbox=TRUE,pretty=TRUE,digits=16)
cat("Comparator complete",variant,proc.time()[[3]]-start,"seconds\n")
invisible(list(predictions=pred,status=status))
}

if(sys.nframe()==0) {
  args<-commandArgs(trailingOnly=TRUE)
  if(length(args)<2)stop("Usage: Rscript gao_run_synthetic.R model_inputs output_dir [area_only|full_population_X] [author_strict|common_continuity] [extension_csv]")
  gao_fit_synthetic(args[1],args[2],if(length(args)>2)args[3]else"area_only",
    if(length(args)>3)args[4]else"author_strict",if(length(args)>4)args[5]else NULL)
}
