# Reconstruct design-based marginal ratios and their full covariance.
# Run from estudio_publicacion or its parent. Original Dropbox sources read only.
# Variance: first-stage with-replacement Taylor linearization, all household
# PSUs retained. Outcome-specific observed denominators retain partial measurements.
args <- commandArgs(trailingOnly=TRUE)
study <- if(dir.exists('datos/raw')) normalizePath('.') else normalizePath('estudio_publicacion')
.libPaths(c(file.path(study,'.Rlib'), .libPaths()))
suppressPackageStartupMessages({library(haven);library(survey);library(readxl)})
options(survey.lonely.psu='adjust')
src <- Sys.getenv('ENDES_AUX_ROOT', unset='PRIVATE_INPUTS')
out <- file.path(study,'datos/processed');dir.create(out,recursive=TRUE,showWarnings=FALSE)
num <- function(x) as.numeric(x)
pad <- function(x) sprintf('%06d',as.integer(x))
writeout <- function(x,name) write.csv(x,file.path(out,name),row.names=FALSE,na='')
outcomes <- c('anemia_old','anemia_who2024','stunting')

# The auxiliary workbook lacks explicit extraction provenance. Keep it transparently
# as an author-supplied compilation; do not claim independently verified census data.
xx <- as.data.frame(read_excel(file.path(src,'consolidado_nuevo_completo_con_directos.xlsx')))
xc <- as.data.frame(read_excel(file.path(src,'consolidado_censo_ubigeo_actualizado.xlsx')))
aux <- data.frame(ubigeo=pad(xx$Ubigeo),department=xx$Departamento,province=xx$Provincia,district=xx$Distrito,
 altitude=xx$Altitud_msnm,latitude=xx$Latitud,longitude=xx$Longitud,poverty=xx$pct_pobreza_total,
 sewer=xx[['BAÑO Red pública de desagüe dentro de la vivienda']]+xx[['BAÑO Red pública de desagüe fuera de la vivienda, pero dentro de la edificación']],electricity=xx$Electricidad,spanish=xx$Castellano,
 illiteracy=xx$Analfabetismo,floor_earth=xx[['PISO Tierra']],
 water_piped=xx[['AGUA Red pública dentro de la vivienda']],
 biomass=xc$Biomasa[match(pad(xx$Ubigeo),pad(xc$Ubigeo))],
 refrigerator=xx$Refrigeradora_congeladora,gas=xx$COCINAR_gasglp+xx$COCINAR_gasnatural)
aux$dept_code<-substr(aux$ubigeo,1,2);aux$prov_code<-substr(aux$ubigeo,1,4)
aux$auxiliary_source_status<-'author_supplied_compilation_unverified_extraction'
stopifnot(!anyDuplicated(aux$ubigeo));writeout(aux,'aux_all_districts.csv')

summaries <- list();validation <- list();flows <- list();hbchecks<-list();unmatched<-list()
for(year in c(2019,2025)) {
 cat('Preparing year',year,'\n')
 if(year==2019){
   h<-read_sav(file.path(study,'datos/raw/ENDES2019/Modulo64/RECH0.SAV'))
   r<-read_sav(file.path(study,'datos/raw/ENDES2019/Modulo64/RECH1.sav'))
   c<-read_sav(file.path(study,'datos/raw/ENDES2019/Modulo74/RECH6.SAV'))
 }else{
   p<-file.path(study,'datos/raw/ENDES2025/Base_Datos')
   h<-read_sav(file.path(p,'RECH0_2025.SAV'));r<-read_sav(file.path(p,'RECH1_2025.SAV'));c<-read_sav(file.path(p,'RECH6_2025.SAV'))
 }
 stopifnot(!anyDuplicated(h$HHID),!anyDuplicated(c[,c('HHID','HC0')]),!anyDuplicated(r[,c('HHID','HVIDX')]))
 hh<-data.frame(HHID=as.character(h$HHID),ubigeo=pad(h$UBIGEO),weight=num(h$HV005)/1e6,
   psu=num(h$HV001),strata=num(h$HV022),altitude=num(h$HV040),interview_year=num(h$HV007))
 stopifnot(all(num(h$HV001)==num(h$HV021)), all(is.finite(hh$weight)&hh$weight>=0))
 hh$psu_key<-paste(hh$strata,hh$psu,sep=':')
 # FULL household survey frame before age/outcome/domain filtering.
 des_full<-svydesign(ids=~psu,strata=~strata,weights=~weight,data=hh,nest=TRUE)
 frame<-unique(hh[,c('psu_key','strata')]);stratum_m<-table(frame$strata)
 child<-as.data.frame(c);child$HHID<-as.character(child$HHID)
 child<-merge(child,hh,by='HHID',all.x=TRUE,sort=FALSE)
 rr<-data.frame(HHID=as.character(r$HHID),HC0=num(r$HVIDX),de_facto=num(r$HV103),de_jure=num(r$HV102))
 child<-merge(child,rr,by=c('HHID','HC0'),all.x=TRUE,sort=FALSE)
 stopifnot(nrow(child)==nrow(c),!anyNA(child$weight),!anyNA(child$de_facto))
 age<-num(child$HC1);rawhb<-num(child$HC53);oldhb<-num(child$HC56);haz<-num(child$HC70)
 # HC53 units are tenths g/dL, numerically identical to g/L. No double adjustment.
 # WHO/INEI 2024: no altitude correction below 500m; use exact altitude formula above.
 adjustment<-ifelse(child$altitude<500,0,0.0056384*child$altitude+0.0000003*child$altitude^2)
 hb_new<-rawhb-adjustment
 child$anemia_old<-ifelse(!is.na(oldhb)&oldhb>=0&oldhb<999,as.integer(oldhb<110),NA)
 child$anemia_who2024<-ifelse(!is.na(rawhb)&rawhb>=0&rawhb<999,as.integer(hb_new<ifelse(age<24,105,110)),NA)
 child$stunting<-ifelse(!is.na(haz)&haz>=-600&haz<=600,as.integer(haz< -200),NA)
 child$dept_code<-substr(child$ubigeo,1,2);child$prov_code<-substr(child$ubigeo,1,4)
 child$eligible_6_59<-age>=6&age<60&child$de_facto==1
 child$eligible_6_35<-age>=6&age<36&child$de_facto==1
 child$eligible_0_59<-age>=0&age<60&child$de_facto==1
 child$eligible_all_children<-age>=6&age<60
 child$eligible_complete<-child$eligible_6_59&complete.cases(child[,outcomes])
 child$eligible_calendar<-child$eligible_6_59&child$interview_year==year
 if(year==2025){
  vv<-child$eligible_6_59&!is.na(child$HC57A)&!is.na(child$anemia_who2024)
  hbchecks[[as.character(year)]]<-data.frame(year=year,n=sum(vv),classification_match=mean(child$anemia_who2024[vv]==(num(child$HC57A[vv])<4)),
   floor_Hb_match=mean(floor(hb_new[vv])==num(child$HC56A[vv])),altitude_threshold_m=500)
  stopifnot(all(child$anemia_who2024[vv]==(num(child$HC57A[vv])<4)))
 }

 # Domain estimator: score sums at PSU, with zeros for every unsampled/empty PSU.
 # Marginal denominators, cross-covariance from same shared full survey frame.
 calc <- function(sel,domain_id,level,universe,validate=FALSE){
  d<-child[which(sel),,drop=FALSE]; w<-d$weight
  p<-den<-nobs<-rep(NA_real_,3); names(p)<-names(den)<-names(nobs)<-outcomes
  score<-matrix(0,nrow(d),3,dimnames=list(NULL,outcomes))
  z<-list(year=year,level=level,domain=domain_id,universe=universe,n_eligible=nrow(d),
   n_psu=length(unique(d$psu_key)),n_strata=length(unique(d$strata)),weight_eligible=sum(w))
  for(j in seq_along(outcomes)){
   y<-d[[outcomes[j]]];ok<-!is.na(y); nobs[j]<-sum(ok);den[j]<-sum(w[ok]);p[j]<-if(den[j]>0)sum(w[ok]*y[ok])/den[j] else NA_real_
   if(den[j]>0) score[ok,j]<-w[ok]*(y[ok]-p[j])/den[j]
   nm<-outcomes[j];z[[paste0('y_',nm)]]<-p[j];z[[paste0('n_',nm)]]<-nobs[j];
   z[[paste0('cases_',nm)]]<-sum(y[ok]);z[[paste0('weighted_den_',nm)]]<-den[j]
   z[[paste0('n_eff_',nm)]]<-if(sum(w[ok]^2)>0)den[j]^2/sum(w[ok]^2) else NA_real_
   z[[paste0('n_psu_',nm)]]<-length(unique(d$psu_key[ok]));z[[paste0('n_strata_',nm)]]<-length(unique(d$strata[ok]))
   z[[paste0('missing_',nm)]]<-nrow(d)-nobs[j]
   z[[paste0('missing_weighted_fraction_',nm)]]<-if(sum(w)>0)sum(w[!ok])/sum(w) else NA_real_
   z[[paste0('missing_bound_lower_',nm)]]<-if(sum(w)>0)sum(w[ok]*y[ok])/sum(w) else NA_real_
   z[[paste0('missing_bound_upper_',nm)]]<-if(sum(w)>0)(sum(w[ok]*y[ok])+sum(w[!ok]))/sum(w) else NA_real_
  }
  if(nrow(d)>0){
   U<-rowsum(score,d$psu_key,reorder=FALSE);ss<-d$strata[match(rownames(U),d$psu_key)]
   V<-matrix(0,3,3)
   for(st in unique(ss)){
    uh<-U[ss==st,,drop=FALSE];m<-as.numeric(stratum_m[as.character(st)])
    if(m>1) V<-V+m/(m-1)*(crossprod(uh)-tcrossprod(colSums(uh))/m)
    else V<-V+crossprod(uh) # full-score grand mean is zero for a ratio
   }
  }else V<-matrix(NA_real_,3,3)
  V[is.na(p),]<-NA;V[,is.na(p)]<-NA
  for(j in 1:3){
   nm<-outcomes[j];z[[paste0('var_',nm)]]<-V[j,j]
   z[[paste0('n_eff_design_',nm)]]<-if(is.finite(V[j,j])&&V[j,j]>0) p[j]*(1-p[j])/V[j,j] else NA_real_
   z[[paste0('zero_variance_',nm)]]<-is.finite(V[j,j])&&V[j,j]<=1e-15
  }
  z$cov_old_stunting<-V[1,3];z$cov_who_stunting<-V[2,3];z$cov_old_who<-V[1,2]
  z$delta_anemia<-p[2]-p[1];z$var_delta_anemia<-V[1,1]+V[2,2]-2*V[1,2]
  pair<-!is.na(d$anemia_old)&!is.na(d$anemia_who2024)
  z$n_pair_rules<-sum(pair);z$weighted_reclassified_to_anemia<-if(sum(w[pair])>0)sum(w[pair]*(d$anemia_old[pair]==0&d$anemia_who2024[pair]==1))/sum(w[pair]) else NA_real_
  z$weighted_reclassified_from_anemia<-if(sum(w[pair])>0)sum(w[pair]*(d$anemia_old[pair]==1&d$anemia_who2024[pair]==0))/sum(w[pair]) else NA_real_
  z$n_pair_old<-sum(complete.cases(d[,c('anemia_old','stunting')]));z$n_pair_who<-sum(complete.cases(d[,c('anemia_who2024','stunting')]))
  z$D_old_singular<-if(all(is.finite(V[c(1,3),c(1,3)])))det(V[c(1,3),c(1,3)])<=1e-14 else TRUE
  z$D_who_singular<-if(all(is.finite(V[c(2,3),c(2,3)])))det(V[c(2,3),c(2,3)])<=1e-14 else TRUE
  z$eligible_bivariate<-nobs[2]>0&&nobs[3]>0
  if(validate&&nrow(d)>0){
   # Independent survey package validation on household totals + delta method.
   vals<-matrix(0,nrow(hh),6);colnames(vals)<-c('num_a','num_w','num_s','den_a','den_w','den_s')
   for(j in 1:3){ok<-!is.na(d[[outcomes[j]]]);a<-rowsum(cbind(num=d[[outcomes[j]]][ok],den=rep(1,sum(ok))),d$HHID[ok]);ix<-match(rownames(a),hh$HHID);vals[ix,j]<-a[,1];vals[ix,j+3]<-a[,2]}
   dh<-des_full;dh$variables<-cbind(dh$variables,as.data.frame(vals));tt<-svytotal(~num_a+num_w+num_s+den_a+den_w+den_s,dh)
   J<-matrix(0,3,6);for(j in 1:3){J[j,j]<-1/den[j];J[j,j+3]<- -p[j]/den[j]}
   Vs<-J%*%vcov(tt)%*%t(J)
   validation[[length(validation)+1]]<<-data.frame(year=year,level=level,domain=domain_id,universe=universe,max_abs_variance_difference=max(abs(V-Vs),na.rm=TRUE),full_psus=nrow(frame),full_strata=length(stratum_m),singleton_full_strata=sum(stratum_m==1))
   stopifnot(max(abs(V-Vs),na.rm=TRUE)<1e-10)
  }
  as.data.frame(z,stringsAsFactors=FALSE)
 }
 # Primary district output contains all observed eligible domains, including unmatched.
 domains<-sort(unique(child$ubigeo[child$eligible_6_59]));rows<-vector('list',length(domains))
 for(i in seq_along(domains))rows[[i]]<-calc(child$eligible_6_59&child$ubigeo==domains[i],domains[i],'district','6_59',validate=i<=3)
 dr<-do.call(rbind,rows);dr$ubigeo<-dr$domain;dr$dept_code<-substr(dr$ubigeo,1,2);dr$prov_code<-substr(dr$ubigeo,1,4)
 dr<-merge(dr,aux,by=c('ubigeo','dept_code','prov_code'),all.x=TRUE,sort=TRUE)
 dr$auxiliary_match<-!is.na(dr$district)
 # Retain 120699 as aggregate, no unsupported disaggregation to current districts.
 dr$district[dr$ubigeo=='120699']<-'MAZAMARI-PANGOA (historical combined area)'
 dr$province[dr$ubigeo=='120699']<-'SATIPO';dr$department[dr$ubigeo=='120699']<-'JUNIN'
 writeout(dr,paste0('areas_',year,'_6_59.csv'))
 unmatched[[as.character(year)]]<-dr[!dr$auxiliary_match,c('year','ubigeo','department','province','district','n_eligible','n_anemia_old','n_stunting')]
 # Aggregate direct benchmarks use exactly same full-frame domain variance.
 ag<-list()
 for(univ in c('6_59','6_35','0_59','all_children','complete','calendar')){
  ee<-child[[paste0('eligible_',univ)]]
  ag[[length(ag)+1]]<-calc(ee,'PE','national',univ,validate=univ=='6_59')
  if(univ%in%c('6_59','6_35','0_59')){
   for(de in sort(unique(child$dept_code[ee])))ag[[length(ag)+1]]<-calc(ee&child$dept_code==de,de,'department',univ,validate=de=='01'&&univ=='6_59')
  }
 }
 aa<-do.call(rbind,ag);writeout(aa,paste0('aggregates_',year,'.csv'));summaries[[as.character(year)]]<-aa[aa$level=='national',]
 flows[[as.character(year)]]<-data.frame(year=year,full_households=nrow(hh),full_psus=nrow(frame),full_strata=length(stratum_m),children_module=nrow(child),eligible_de_facto_6_59=sum(child$eligible_6_59),eligible_all_6_59=sum(child$eligible_all_children),interview_following_year_households=sum(hh$interview_year>year),primary_observed_domains=nrow(dr),matched_aux_domains=sum(dr$auxiliary_match),unmatched_domains=sum(!dr$auxiliary_match))
 # Private intermediate for reproducibility only; do not distribute microdata in reports.
 saveRDS(list(household=hh,child=child),file.path(study,'datos/raw',paste0('analytic_private_',year,'.rds')))
 cat('Wrote areas',year,'domains',nrow(dr),'\n')
}
writeout(do.call(rbind,summaries),'national_summary.csv');writeout(do.call(rbind,flows),'sample_flow.csv')
writeout(do.call(rbind,validation),'variance_survey_validation.csv');writeout(do.call(rbind,hbchecks),'who2024_official_validation.csv')
writeout(do.call(rbind,unmatched),'unmatched_geography.csv')
capture.output(sessionInfo(),file=file.path(out,'R_sessionInfo.txt'))
cat('Finished all data reconstruction\n')
