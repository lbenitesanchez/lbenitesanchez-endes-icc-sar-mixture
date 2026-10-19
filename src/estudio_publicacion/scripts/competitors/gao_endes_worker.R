argv<-commandArgs(trailingOnly=FALSE);self<-sub("^--file=","",argv[grepl("^--file=",argv)])
source(file.path(dirname(self),"gao_run_endes.R"))
args<-commandArgs(trailingOnly=TRUE)
if(length(args)!=1)stop("Usage: Rscript gao_endes_worker.R tasks.csv")
tasks<-read.csv(args[1],stringsAsFactors=FALSE)
for(i in seq_len(nrow(tasks))) {
  t<-tasks[i,];cat("TASK",i,nrow(tasks),t$output_dir,"\n");flush.console()
  if(file.exists(file.path(t$output_dir,"predictions.csv"))&&file.exists(file.path(t$output_dir,"diagnostics.json"))) {
    d<-jsonlite::read_json(file.path(t$output_dir,"diagnostics.json"),simplifyVector=TRUE)
    stopifnot(identical(d$observation_mode,t$stage2_mode),
       all(vapply(d$endpoints,function(e)isTRUE(e$INLA_ok),logical(1))))
    cat("TASK_SKIPPED_ALREADY_COMPLETE",i,"\n");next
  }
  ok<-tryCatch({gao_fit_endes(t$input_dir,t$output_dir,t$stage2_mode);TRUE},error=function(e){
    dir.create(t$output_dir,recursive=TRUE,showWarnings=FALSE)
    jsonlite::write_json(list(error=conditionMessage(e),completed=FALSE),
      file.path(t$output_dir,"failure.json"),auto_unbox=TRUE,pretty=TRUE)
    cat("FAILED",conditionMessage(e),"\n");FALSE})
  gc();cat("TASK_END",i,ok,"\n");flush.console()
}
