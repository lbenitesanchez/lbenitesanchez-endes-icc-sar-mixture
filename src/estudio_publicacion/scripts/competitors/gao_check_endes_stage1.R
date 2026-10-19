argv<-commandArgs(trailingOnly=FALSE);self<-sub("^--file=","",argv[grepl("^--file=",argv)])
study<-normalizePath(file.path(dirname(self),"../.."))
.libPaths(c(file.path(study,".Rlib"),.libPaths()))
source(file.path(study,"scripts/competitors/gao_stage1.R"))
features<-c("altitude","poverty","illiteracy","electricity","spanish","biomass")
outdir<-file.path(study,"resultados/competidores_gao")
dir.create(outdir,recursive=TRUE,showWarnings=FALSE)
checks<-list()
for(year in c(2019,2025)) {
  inp<-readRDS(file.path(study,"datos/private_competitors",paste0("private_inputs_",year,".rds")))
  s<-inp$child;s$area<-s$ubigeo;f<-inp$household;f$area<-f$ubigeo
  p<-inp$area_aux;p$area<-p$ubigeo;p$population_weight<-1
  center<-vapply(p[features],mean,numeric(1));scale<-vapply(p[features],sd,numeric(1))
  # A constant within-area working X has a known degenerate distribution at
  # that area's X; no population child totals or age/sex margins are fabricated.
  for(x in features) {
    p[[x]]<-(p[[x]]-center[[x]])/scale[[x]]
    s[[x]]<-(s[[paste0("aux_",x)]]-center[[x]])/scale[[x]]
  }
  frozen<-read.csv(file.path(study,"datos/processed",paste0("areas_",year,"_6_59.csv")),
                   colClasses=c(ubigeo="character"))
  for(outcome in c("anemia_who2024","stunting")) {
    z<-gao_stage1(s,p,f,paste0("y_",outcome),features)
    e<-z$estimates;j<-match(e$area,frozen$ubigeo)
    observed<-which(e$n>0)
    checks[[length(checks)+1]]<-data.frame(year=year,outcome=outcome,
      measured=sum(e$n),sampled_areas=length(observed),working_fit_converged=z$metadata$working_model_converged,
      max_point_difference=max(abs(e$est[observed]-e$direct[observed])),
      max_variance_difference=max(abs(e$var[observed]-e$direct_var[observed])),
      max_frozen_point_difference=max(abs(e$est[observed]-frozen[[paste0("y_",outcome)]][j[observed]])),
      max_frozen_variance_difference=max(abs(e$var[observed]-frozen[[paste0("var_",outcome)]][j[observed]])),
      unsupported_children=sum(!s$auxiliary_match),
      author_boundary_or_lowvariance_excluded=sum(e$n>0 & (e$est<=0|e$est>=1|e$var<=1e-5)))
    stopifnot(checks[[length(checks)]]$max_point_difference<1e-12,
      checks[[length(checks)]]$max_variance_difference<1e-12,
      checks[[length(checks)]]$max_frozen_variance_difference<1e-12)
    write.csv(e,file.path(outdir,paste0("stage1_",year,"_",outcome,".csv")),row.names=FALSE,na="")
  }
}
write.csv(do.call(rbind,checks),file.path(outdir,"stage1_equivalence_checks.csv"),row.names=FALSE)
print(do.call(rbind,checks))
