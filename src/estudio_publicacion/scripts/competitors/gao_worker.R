# Persistent single-thread INLA worker. Tasks contain no outcome truth paths.
argv<-commandArgs(trailingOnly=FALSE);self<-sub("^--file=","",argv[grepl("^--file=",argv)])
source(file.path(dirname(self),"gao_run_synthetic.R"))
args<-commandArgs(trailingOnly=TRUE)
if(length(args)!=1)stop("Usage: Rscript gao_worker.R tasks.csv")
tasks<-read.csv(args[1],stringsAsFactors=FALSE)
stopifnot(all(c("model_inputs","output_dir","variant","stage2_mode")%in%names(tasks)))
for(i in seq_len(nrow(tasks))) {
  t<-tasks[i,];cat("TASK",i,nrow(tasks),t$output_dir,"\n");flush.console()
  if(file.exists(file.path(t$output_dir,"predictions.csv")) &&
     file.exists(file.path(t$output_dir,"diagnostics.json"))) {
    old<-jsonlite::read_json(file.path(t$output_dir,"diagnostics.json"),simplifyVector=TRUE)
    if(identical(old$variant,t$variant)&&identical(old$stage2_mode,t$stage2_mode)&&
       all(vapply(old$endpoints,function(e)isTRUE(e$INLA_ok),logical(1)))) {
      cat("TASK_SKIPPED_ALREADY_COMPLETE",i,"\n");next
    }
    stop("Existing output fails integrity/mode check")
  }
  ok<-tryCatch({
    gao_fit_synthetic(t$model_inputs,t$output_dir,t$variant,t$stage2_mode,
      if("extension_file"%in%names(t)&&nzchar(t$extension_file))t$extension_file else NULL)
    TRUE
  },error=function(e){
    dir.create(t$output_dir,recursive=TRUE,showWarnings=FALSE)
    jsonlite::write_json(list(error=conditionMessage(e),completed=FALSE,
      task_index=i,stage2_mode=t$stage2_mode,variant=t$variant),
      file.path(t$output_dir,"failure.json"),pretty=TRUE,auto_unbox=TRUE)
    cat("FAILED",conditionMessage(e),"\n");FALSE
  })
  gc();cat("TASK_END",i,ok,"\n");flush.console()
}
