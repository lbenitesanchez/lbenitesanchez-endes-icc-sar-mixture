args<-commandArgs(trailingOnly=TRUE);root<-if(length(args))normalizePath(args[1])else normalizePath('fase2_2026-10-07')
.libPaths(c(file.path(dirname(root),'.Rlib'),.libPaths()));suppressPackageStartupMessages(library(readxl))
f<-file.path(root,'fuentes/geografia/poverty2018_anexo.xlsx');x<-as.data.frame(suppressMessages(read_excel(f,sheet='Anexo2',col_names=FALSE,col_types='text')))
ii<-which(grepl('^[0-9]{6}$',x[[1]]) & !grepl('00$',x[[1]]))
d<-data.frame(ubigeo=x[ii,1],district=x[ii,3],district_group_suffix=x[ii,2],population_projected_2020=as.numeric(x[ii,4]),poverty_ci95_lower_pct=as.numeric(x[ii,5]),poverty_ci95_upper_pct=as.numeric(x[ii,6]),poverty_rank=as.numeric(x[ii,7]),source_sheet='Anexo2',source_row=ii)
d$poverty<-(d$poverty_ci95_lower_pct+d$poverty_ci95_upper_pct)/2
d$poverty_definition<-'midpoint_proxy_of_official_published_95pct_CI_not_published_point_estimate';d$poverty_reference_year<-2018;d$poverty_publication_date<-'2020-02-20'
stopifnot(nrow(d)==1874,!anyDuplicated(d$ubigeo),all(d$poverty_ci95_lower_pct<=d$poverty_ci95_upper_pct),all(d$poverty>=0&d$poverty<=100))
write.csv(d[order(d$ubigeo),],file.path(root,'datos/geografia/poverty_official_2018.csv'),row.names=FALSE,na='');cat('Extracted',nrow(d),'poverty CI records; explicit midpoint proxy.\n')
