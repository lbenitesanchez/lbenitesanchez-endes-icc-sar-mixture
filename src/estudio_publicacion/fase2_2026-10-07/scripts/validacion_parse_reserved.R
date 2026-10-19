# DATA-STEWARD PROCESS ONLY. Reserved outcomes parsed only after exact frozen
# protocol and completed2019 temporal-prediction lock approval. Test responses
# are physically quarantined and never passed to fitting subprocesses.
args <- commandArgs(trailingOnly=TRUE)
root <- normalizePath(if(length(args))args[1]else'.')
study <- dirname(root)
.libPaths(c(file.path(study,'.Rlib'),.libPaths()))
suppressPackageStartupMessages({library(haven);library(jsonlite);library(digest)})
protocol_path <- file.path(root,'protocol_freeze.json')
protocol <- fromJSON(protocol_path,simplifyVector=FALSE)
protocol_sha <- digest(file=protocol_path,algo='sha256')
access_path <- file.path(root,'validacion/externa_reserved_access_approval.json')
access <- if(file.exists(access_path))fromJSON(access_path,simplifyVector=FALSE)else list()
approved <- isTRUE(protocol$approved_for_reserved_outcome_access) ||
            (isTRUE(access$approved_for_reserved_outcome_access)&&identical(access$protocol_sha256,protocol_sha))
if(!approved)stop('Reserved child response access remains locked')
paths <- file.path(root,'datos/private_external_raw',c('RECH0_2024.SAV','RECH1_2024.SAV','RECH6_2024.SAV'))
h <- read_sav(paths[1]);r <- read_sav(paths[2]);c <- read_sav(paths[3])
stopifnot(!anyDuplicated(h$HHID),!anyDuplicated(r[,c('HHID','HVIDX')]),!anyDuplicated(c[,c('HHID','HC0')]),
          all(as.numeric(h$HV001)==as.numeric(h$HV021)))
hh <- data.frame(HHID=as.character(h$HHID),
  household_id=sprintf('2024_H%05d',seq_len(nrow(h))),year=2024,
  ubigeo=sprintf('%06d',as.integer(h$UBIGEO)),weight=as.numeric(h$HV005)/1e6,
  psu=as.numeric(h$HV001),stratum=as.numeric(h$HV022),
  cluster_altitude=as.numeric(h$HV040),interview_year=as.numeric(h$HV007))
hh$psu_key <- paste(hh$stratum,hh$psu,sep=':')
stopifnot(all(is.finite(hh$weight)&hh$weight>=0))
cc <- as.data.frame(c);cc$HHID <- as.character(cc$HHID)
cc <- merge(cc,hh,by='HHID',all.x=TRUE,sort=FALSE)
de <- data.frame(HHID=as.character(r$HHID),HC0=as.numeric(r$HVIDX),de_facto=as.numeric(r$HV103))
cc <- merge(cc,de,by=c('HHID','HC0'),all.x=TRUE,sort=FALSE)
stopifnot(nrow(cc)==nrow(c),!anyNA(cc$weight),!anyNA(cc$de_facto))
age <- as.numeric(cc$HC1);hb <- as.numeric(cc$HC53);old <- as.numeric(cc$HC56);haz <- as.numeric(cc$HC70)
altitude_adjustment <- ifelse(cc$cluster_altitude<500,0,
                   0.0056384*cc$cluster_altitude+0.0000003*cc$cluster_altitude^2)
anemia <- ifelse(!is.na(hb)&hb>=0&hb<999,as.integer(hb-altitude_adjustment<ifelse(age<24,105,110)),NA)
old_anemia <- ifelse(!is.na(old)&old>=0&old<999,as.integer(old<110),NA)
stunting <- ifelse(!is.na(haz)&haz>=-600&haz<=600,as.integer(haz< -200),NA)
eligible <- which(age>=6&age<60&cc$de_facto==1)
child <- data.frame(child_id=paste0(cc$household_id[eligible],'_C',sprintf('%02d',as.numeric(cc$HC0[eligible]))),
   household_id=cc$household_id[eligible],year=2024,ubigeo=cc$ubigeo[eligible],
   dept_code=substr(cc$ubigeo[eligible],1,2),prov_code=substr(cc$ubigeo[eligible],1,4),
   age_months=age[eligible],sex=as.numeric(cc$HC27[eligible]),weight=cc$weight[eligible],
   psu=cc$psu[eligible],stratum=cc$stratum[eligible],psu_key=cc$psu_key[eligible],
   cluster_altitude=cc$cluster_altitude[eligible],interview_year=cc$interview_year[eligible],de_facto=1,
   y_anemia_old=old_anemia[eligible],y_anemia_who2024=anemia[eligible],y_stunting=stunting[eligible])
stopifnot(!anyDuplicated(child$child_id),all(child$sex%in%c(1,2)))
is_test <- vapply(child$prov_code,function(p){
  key <- digest(paste0('ENDES-phase2-20261007|',p),algo='sha256',serialize=FALSE)
  # Avoid32bit integer overflow: accumulate first8hex digits asdouble.
  digits <- strsplit(substr(key,1,8),'')[[1]]
  values <- match(digits,c(as.character(0:9),letters[1:6]))-1
  sum(values*16^(7:0))%%5==0
},logical(1))
out <- file.path(root,'datos/private_external');dir.create(out,recursive=TRUE,showWarnings=FALSE)
write.csv(hh[,setdiff(names(hh),'HHID')],file.path(out,'households_2024.csv'),row.names=FALSE,na='')
write.csv(child[!is_test,],file.path(out,'children_2024_training.csv'),row.names=FALSE,na='')
write.csv(child[is_test,],file.path(out,'children_2024_test.csv'),row.names=FALSE,na='')
write.csv(child,file.path(out,'children_2024_scoring_only.csv'),row.names=FALSE,na='')
saveRDS(list(household=hh,child=cc),file.path(root,'datos/private_external_raw/analytic_private_2024.rds'))
valid <- age>=6&age<60&cc$de_facto==1&!is.na(anemia)&!is.na(cc$HC57A)
match <- all(anemia[valid]==as.integer(as.numeric(cc$HC57A[valid])<4))
validation <- list(status=if(match)'PASS_RESERVED_OUTCOME_PREPARATION_PRIVATELY_QUARANTINED'else'FAIL_ANEMIA_CLASSIFICATION',
 protocol_sha256=protocol_sha,access_approval_sha256=if(file.exists(access_path))digest(file=access_path,algo='sha256')else NULL,
 child_keys_unique=TRUE,household_keys_unique=TRUE,roster_keys_unique=TRUE,WHO2024_HC57A_classification_match=match,
 official_classification_compared_records=sum(valid),full_household_frame_preserved=TRUE,
 household_frame_n=nrow(hh),eligible6_59_n=nrow(child),training_children=nrow(child[!is_test,]),test_children=nrow(child[is_test,]),
 test_response_files='private_external/children_2024_test.csv and scoring_only; fittingprocess forbidden',
 empirical_outcome_statistics_released=FALSE,model_scores_computed=FALSE,
 raw_member_sha256=setNames(lapply(paths,function(p)digest(file=p,algo='sha256')),basename(paths)),
 generated_private_sha256=setNames(lapply(file.path(out,c('households_2024.csv','children_2024_training.csv','children_2024_test.csv','children_2024_scoring_only.csv')),
                         function(p)digest(file=p,algo='sha256')),c('households','training','test','temporal_scoring')))
write_json(validation,file.path(root,'validacion/externa_reserved_preparation.json'),pretty=TRUE,auto_unbox=TRUE,null='null')
if(!match)stop('Reserved classification disagrees with officialHC57A; do not fit until documented repair')
cat('Reserved inputs prepared privately; no outcome averages or model scores printed.\n')
