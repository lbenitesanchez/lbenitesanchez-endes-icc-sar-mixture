# Independent Taylor-ratio validation against survey's full household design.
# Development years only; no new prevalence model is fitted or scored.
args<-commandArgs(trailingOnly=TRUE)
year<-as.integer(args[1]);stopifnot(year %in% c(2019,2025))
phase<-normalizePath('.')
study<-dirname(phase)
.libPaths(c(file.path(study,'.Rlib'),.libPaths()))
suppressPackageStartupMessages(library(survey))
options(survey.lonely.psu='adjust')
h<-read.csv(file.path(study,'datos/private_competitors',paste0('households_',year,'.csv')),
            colClasses=c(ubigeo='character',household_id='character',psu_key='character'))
u<-read.csv(file.path(study,'datos/private_competitors',paste0('children_',year,'.csv')),
            colClasses=c(ubigeo='character',household_id='character',psu_key='character'))
ids<-read.csv(file.path(phase,'datos/diseno',paste0(year,'_full'),'area_ids.csv'),colClasses='character')
des<-svydesign(ids=~psu,strata=~stratum,weights=~weight,data=h,nest=TRUE)
out<-list()
for(name in c('anemia','stunting')) {
  yc<-if(name=='anemia') 'y_anemia_who2024' else 'y_stunting'
  prepared<-read.csv(file.path(phase,'datos/diseno',paste0(year,'_full'),paste0(name,'_calibration.csv')))
  # Deterministic geography ordering plus all full-singleton contributing
  # domains. Neither validation ranks candidates nor uses held-out outcomes.
  chosen<-unique(c(seq_len(min(8,nrow(prepared))),tail(seq_len(nrow(prepared)),8),
                    which(prepared$singleton_Taylor_component>0)))
  for(i in chosen) {
    ub<-ids$ubigeo[prepared$area[i]+1L]
    child<-u[u$ubigeo==ub & !is.na(u[[yc]]),,drop=FALSE]
    numer<-denom<-numeric(nrow(h))
    if(nrow(child)>0) {
      nn<-rowsum(cbind(num=child[[yc]],den=1),child$household_id,reorder=FALSE)
      ix<-match(rownames(nn),h$household_id)
      stopifnot(!anyNA(ix))
      numer[ix]<-nn[,1];denom[ix]<-nn[,2]
    }
    d<-des;d$variables$num_phase2<-numer;d$variables$den_phase2<-denom
    ratio<-svyratio(~num_phase2,~den_phase2,d)
    point<-as.numeric(coef(ratio));vv<-as.numeric(vcov(ratio))
    dp<-abs(point-prepared$direct[i]);dv<-abs(vv-prepared$Taylor_variance[i])
    stopifnot(dp<1e-10,dv<1e-10)
    out[[length(out)+1L]]<-data.frame(year=year,outcome=name,ubigeo=ub,
       full_households=nrow(h),full_psus=length(unique(h$psu_key)),
       full_strata=length(unique(h$stratum)),point=point,survey_variance=vv,
       reconstructed_variance=prepared$Taylor_variance[i],point_abs_difference=dp,
       variance_abs_difference=dv,
       singleton_Taylor_component=prepared$singleton_Taylor_component[i])
  }
}
folder<-file.path(phase,'validacion');dir.create(folder,showWarnings=FALSE)
write.csv(do.call(rbind,out),file.path(folder,paste0('diseno_survey_validation_',year,'.csv')),row.names=FALSE)
capture.output(sessionInfo(),file=file.path(folder,paste0('diseno_survey_session_',year,'.txt')))
cat('PASS',year,'independent survey checks',length(out),'\n')
