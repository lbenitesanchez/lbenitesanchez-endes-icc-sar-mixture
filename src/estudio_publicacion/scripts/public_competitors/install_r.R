# Explicit local installation; this script is never invoked by packaging.
argv<-commandArgs(trailingOnly=FALSE)
self<-sub('^--file=','',argv[grepl('^--file=',argv)])
root<-normalizePath(file.path(dirname(self),'../..'))
lib<-file.path(root,'.Rlib');dir.create(lib,showWarnings=FALSE)
.libPaths(c(lib,.libPaths()))
repos<-c(CRAN='https://cloud.r-project.org',INLA='https://inla.r-inla-download.org/R/stable')
deps<-c('haven','survey','jsonlite','fmesher','MatrixModels','sp','splancs')
if('--author-validation'%in%commandArgs(trailingOnly=TRUE))deps<-c(deps,'dplyr','tidyr','sn')
missing<-deps[!vapply(deps,requireNamespace,logical(1),quietly=TRUE)]
if(length(missing))install.packages(missing,lib=lib,repos=repos)
if(!requireNamespace('INLA',quietly=TRUE)||as.character(packageVersion('INLA'))!='26.08.07'){
  # The exact observed Mac ARM binary is available through the official repo.
  # Other platforms need the same-version official compatible distribution.
  if(Sys.info()[['sysname']]=='Darwin'&&grepl('arm64|aarch64',R.version$arch)){
    if(getRversion()<'4.6.0')stop('Observed INLA Mac ARM binary requires R4.6; use a compatible official26.08.07 build.')
    install.packages('https://inla.r-inla-download.org/R/stable/bin/macosx/sonoma-arm64/contrib/4.6/INLA_26.08.07.tgz',
                     repos=NULL,type='binary',lib=lib)
  }else{
    stop('Install official INLA26.08.07 for this platform into .Rlib, then rerun. No newer-version substitution is silent.')
  }
}
stopifnot(as.character(packageVersion('INLA'))=='26.08.07')
capture.output(sessionInfo(),file=file.path(root,'R_competitor_sessionInfo.txt'))
cat('Local R library ready; official INLA26.08.07 verified.\n')
