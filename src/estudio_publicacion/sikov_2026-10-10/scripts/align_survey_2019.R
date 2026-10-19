# Read existing audited private intermediate; never change old data/results.
args <- commandArgs(trailingOnly=TRUE)
root <- normalizePath(args[1]); study <- normalizePath(file.path(root,'..'))
.libPaths(c(file.path(study,'.Rlib'),.libPaths()))
suppressPackageStartupMessages(library(survey))
options(survey.lonely.psu='adjust')
out <- file.path(root,'datos');dir.create(out,recursive=TRUE,showWarnings=FALSE)
source <- file.path(study,'datos/raw/analytic_private_2019.rds')
dd <- readRDS(source);h <- dd$household;c <- dd$child
stopifnot(!anyDuplicated(h$HHID),!anyDuplicated(c[,c('HHID','HC0')]))
n <- function(x)as.numeric(x)
age <- n(c$HC1);hb <- n(c$HC56);rawhb <- n(c$HC53)
haz <- n(c$HC70);hazold <- n(c$HC5)
validhb <- !is.na(hb)&hb>=0&hb<999
validhaz <- !is.na(haz)&haz>=-600&haz<=600
validhazold <- !is.na(hazold)&hazold>=-600&hazold<=600
c$anemia <- ifelse(validhb,as.integer(hb<110),NA)
c$stunting <- ifelse(validhaz,as.integer(haz< -200),NA)
c$growth_hc70_p025 <- ifelse(validhaz,as.integer(haz<qnorm(.025)*100),NA)
c$growth_hc70_p10 <- ifelse(validhaz,as.integer(haz<qnorm(.10)*100),NA)
c$growth_hc5_p10 <- ifelse(validhazold,as.integer(hazold<qnorm(.10)*100),NA)
c$growth_hc5_minus128 <- ifelse(validhazold,as.integer(hazold< -128),NA)
outcomes <- c('anemia','stunting','growth_hc70_p025','growth_hc70_p10','growth_hc5_p10','growth_hc5_minus128')
frame <- unique(h[,c('psu_key','strata')]);m <- table(frame$strata)
des <- svydesign(ids=~psu,strata=~strata,weights=~weight,data=h,nest=TRUE)
qa <- list(); rows <- list()
calc <- function(sel,code,universe,independent=FALSE){
 d<-c[which(sel),,drop=FALSE];w<-d$weight
 score<-matrix(0,nrow(d),length(outcomes));pr<-den<-rep(NA_real_,length(outcomes))
 z<-list(ubigeo=code,year=2019,universe=universe,n_eligible=nrow(d),n_psu=length(unique(d$psu_key)))
 for(j in seq_along(outcomes)){
  ep<-outcomes[j];v<-d[[ep]];ok<-is.finite(v)&is.finite(w)&w>0
  den[j]<-sum(w[ok]);pr[j]<-if(den[j]>0)sum(w[ok]*v[ok])/den[j] else NA_real_
  if(den[j]>0)score[ok,j]<-w[ok]*(v[ok]-pr[j])/den[j]
  z[[paste0('y_',ep)]]<-pr[j];z[[paste0('n_',ep)]]<-sum(ok)
  z[[paste0('weighted_den_',ep)]]<-den[j]
  z[[paste0('n_eff_',ep)]]<-if(sum(w[ok]^2)>0)den[j]^2/sum(w[ok]^2) else NA_real_
 }
 U<-rowsum(score,d$psu_key,reorder=FALSE);st<-d$strata[match(rownames(U),d$psu_key)]
 V<-matrix(0,length(outcomes),length(outcomes))
 for(s in unique(st)){
  uh<-U[st==s,,drop=FALSE];mh<-as.numeric(m[as.character(s)])
  if(mh>1)V<-V+mh/(mh-1)*(crossprod(uh)-tcrossprod(colSums(uh))/mh) else V<-V+crossprod(uh)
 }
 V[!is.finite(pr),]<-NA_real_;V[,!is.finite(pr)]<-NA_real_
 for(j in seq_along(outcomes))z[[paste0('var_',outcomes[j])]]<-V[j,j]
 z$cov_anemia_stunting<-V[1,2]
 for(j in 3:length(outcomes))z[[paste0('cov_anemia_',outcomes[j])]]<-V[1,j]
 if(independent){
  vals<-matrix(0,nrow(h),2*length(outcomes));colnames(vals)<-paste0('v',1:ncol(vals))
  for(j in seq_along(outcomes)){
   ok<-is.finite(d[[outcomes[j]]])&is.finite(w)&w>0
   a<-rowsum(cbind(num=d[[outcomes[j]]][ok],den=1),d$HHID[ok]);ix<-match(rownames(a),h$HHID)
   vals[ix,j]<-a[,1];vals[ix,j+length(outcomes)]<-a[,2]
  }
  dh<-des;dh$variables<-cbind(dh$variables,as.data.frame(vals));tt<-svytotal(as.formula(paste('~',paste(colnames(vals),collapse='+'))),dh)
  J<-matrix(0,length(outcomes),ncol(vals))
  for(j in seq_along(outcomes)){J[j,j]<-1/den[j];J[j,j+length(outcomes)]<- -pr[j]/den[j]}
  Vs<-J%*%vcov(tt)%*%t(J);delta<-max(abs(V-Vs),na.rm=TRUE)
  stopifnot(delta<1e-10)
  qa[[length(qa)+1]]<<-data.frame(ubigeo=code,universe=universe,max_abs_cov_difference=delta,full_psus=nrow(frame),full_strata=length(m))
 }
 as.data.frame(z,stringsAsFactors=FALSE)
}
for(u in c('6_59','0_59')){
 eligible<-age>=if(u=='6_59')6 else 0
 eligible<-eligible&age<60&c$de_facto==1
 codes<-sort(unique(c$ubigeo[eligible]));cat(u,length(codes),'observed districts\n')
 for(i in seq_along(codes))rows[[length(rows)+1]]<-calc(eligible&c$ubigeo==codes[i],codes[i],u,i<=3)
 rows[[length(rows)+1]]<-calc(eligible,'PE',u,TRUE)
}
write.csv(do.call(rbind,rows),file.path(out,'survey2019_harmonized_and_definition_sensitivities.csv'),row.names=FALSE,na='')
write.csv(do.call(rbind,qa),file.path(root,'validacion/align_survey2019_independent_survey_checks.csv'),row.names=FALSE)
capture.output(sessionInfo(),file=file.path(root,'validacion/align_survey2019_R_sessionInfo.txt'))
cat('PASS full-household Taylor ratios and independent survey package covariance\n')
