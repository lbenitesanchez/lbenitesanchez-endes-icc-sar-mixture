study <- if(dir.exists('datos/raw')) normalizePath('.') else normalizePath('estudio_publicacion')
.libPaths(c(file.path(study,'.Rlib'),.libPaths()))
suppressPackageStartupMessages(library(survey))
res<-list()
for(year in c(2019,2025)){
  raw<-readRDS(file.path(study,'datos/raw',paste0('analytic_private_',year,'.rds')))
  hh<-raw$household;d<-raw$child[raw$child$eligible_6_59,,drop=FALSE]
  outcomes<-c('anemia_old','anemia_who2024','stunting')
  vals<-matrix(0,nrow(hh),6);colnames(vals)<-c('num_a','num_w','num_s','den_a','den_w','den_s')
  den<-p<-numeric(3)
  for(j in 1:3){ok<-!is.na(d[[outcomes[j]]]);a<-rowsum(cbind(num=d[[outcomes[j]]][ok],den=rep(1,sum(ok))),d$HHID[ok]);ix<-match(rownames(a),hh$HHID);vals[ix,j]<-a[,1];vals[ix,j+3]<-a[,2];den[j]<-sum(d$weight[ok]);p[j]<-sum(d$weight[ok]*d[[outcomes[j]]][ok])/den[j]}
  hh<-cbind(hh,as.data.frame(vals));des<-svydesign(ids=~psu,strata=~strata,weights=~weight,data=hh,nest=TRUE)
  for(mode in c('adjust','certainty','average')){
    options(survey.lonely.psu=mode)
    tt<-svytotal(~num_a+num_w+num_s+den_a+den_w+den_s,des)
    J<-matrix(0,3,6);for(j in 1:3){J[j,j]<-1/den[j];J[j,j+3]<- -p[j]/den[j]}
    V<-J%*%vcov(tt)%*%t(J)
    for(j in 1:3)res[[length(res)+1]]<-data.frame(year=year,universe='6_59',outcome=outcomes[j],singleton_option=mode,prevalence=p[j],variance=V[j,j],standard_error_pct=100*sqrt(V[j,j]),official_comparator_se_pct=if(year==2025&&j==1).47938626636625425 else NA_real_,interpretation='sensitivity handling; certainty is not evidence of genuine certainty strata')
  }
}
options(survey.lonely.psu='adjust')
ans<-do.call(rbind,res)
write.csv(ans,file.path(study,'datos/processed/variance_singleton_sensitivity.csv'),row.names=FALSE)
print(ans[,c('year','outcome','singleton_option','standard_error_pct','official_comparator_se_pct')])
