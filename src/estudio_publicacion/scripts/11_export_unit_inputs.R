# Private comparator input export. No change to original/frozen ENDES artifacts.
argv<-commandArgs(trailingOnly=FALSE);self<-sub('^--file=','',argv[grepl('^--file=',argv)])
study<-normalizePath(file.path(dirname(self),'..'));out<-file.path(study,'datos/private_competitors')
dir.create(out,recursive=TRUE,showWarnings=FALSE)
.libPaths(c(file.path(study,'.Rlib'),.libPaths()))
num<-function(v)as.numeric(v)
aux<-read.csv(file.path(study,'datos/processed/aux_all_districts.csv'),colClasses=c(ubigeo='character'),check.names=FALSE)
write.csv(aux,file.path(out,'area_aux.csv'),row.names=FALSE,na='')
summary<-list()
for(year in c(2019,2025)){
 a<-readRDS(file.path(study,'datos/raw',paste0('analytic_private_',year,'.rds')))
 hh<-a$household;child<-a$child
 hh<-hh[order(hh$HHID),];hh$household_id<-paste0(year,'_H',sprintf('%05d',seq_len(nrow(hh))))
 idx<-match(child$HHID,hh$HHID);stopifnot(!anyNA(idx))
 child$household_id<-hh$household_id[idx]
 frame<-data.frame(household_id=hh$household_id,year=year,ubigeo=hh$ubigeo,
   dept_code=substr(hh$ubigeo,1,2),prov_code=substr(hh$ubigeo,1,4),weight=hh$weight,
   psu=hh$psu,stratum=hh$strata,psu_key=hh$psu_key,cluster_altitude=hh$altitude,
   interview_year=hh$interview_year,stringsAsFactors=FALSE)
 d<-child[which(child$eligible_6_59),]
 d<-d[order(d$household_id,num(d$HC0)),]
 unit<-data.frame(child_id=paste0(d$household_id,'_C',sprintf('%02d',num(d$HC0))),
   household_id=d$household_id,year=year,ubigeo=d$ubigeo,dept_code=d$dept_code,prov_code=d$prov_code,
   age_months=num(d$HC1),sex=num(d$HC27),weight=d$weight,psu=d$psu,stratum=d$strata,psu_key=d$psu_key,
   cluster_altitude=d$altitude,interview_year=d$interview_year,de_facto=d$de_facto,
   y_anemia_old=d$anemia_old,y_anemia_who2024=d$anemia_who2024,y_stunting=d$stunting,
   stringsAsFactors=FALSE)
 j<-match(unit$ubigeo,aux$ubigeo)
 for(x in c('altitude','latitude','longitude','poverty','electricity','spanish','illiteracy','biomass'))unit[[paste0('aux_',x)]]<-aux[[x]][j]
 unit$auxiliary_match<-!is.na(j)
 stopifnot(!anyDuplicated(unit$child_id),all(unit$age_months>=6&unit$age_months<60),all(unit$de_facto==1),all(unit$weight>0),all(unit$sex%in%c(1,2)))
 write.csv(unit,file.path(out,paste0('children_',year,'.csv')),row.names=FALSE,na='')
 write.csv(frame,file.path(out,paste0('households_',year,'.csv')),row.names=FALSE,na='')
 saveRDS(list(household=frame,child=unit,area_aux=aux),file.path(out,paste0('private_inputs_',year,'.rds')))
 summary[[as.character(year)]]<-data.frame(year=year,households=nrow(frame),psus=nrow(unique(frame[,c('psu_key','stratum')])),strata=length(unique(frame$stratum)),
   children=nrow(unit),anemia_measured=sum(!is.na(unit$y_anemia_who2024)),stunting_measured=sum(!is.na(unit$y_stunting)),
   measured_both=sum(complete.cases(unit[,c('y_anemia_who2024','y_stunting')])),
   auxiliary_unmatched_children=sum(!unit$auxiliary_match),positive_weight_households=sum(frame$weight>0),missing_sex=sum(is.na(unit$sex)))
}
write.csv(do.call(rbind,summary),file.path(out,'export_summary.csv'),row.names=FALSE)
cat('Private export complete; microdata must never be redistributed.\n')
print(do.call(rbind,summary))
