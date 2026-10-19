# Re-extract the official 2024 municipality directory, without official-contact columns.
args<-commandArgs(trailingOnly=TRUE);root<-if(length(args))normalizePath(args[1])else normalizePath('fase2_2026-10-07')
.libPaths(c(file.path(dirname(root),'.Rlib'),.libPaths()));suppressPackageStartupMessages(library(readxl))
p<-list.dirs(file.path(root,'fuentes/geografia/directorio_2024_excel'),recursive=TRUE,full.names=TRUE);p<-p[grepl('/1.Directorio-MPyMD-2024$',p)];stopifnot(length(p)==1)
ans<-list();metadata<-list()
for(f in sort(list.files(p,pattern="^[0-9]",full.names=TRUE))){
 x<-as.data.frame(suppressMessages(read_excel(f,col_names=FALSE,col_types='text')))
 getcol<-function(pattern){j<-which(vapply(x,function(v)any(grepl(pattern,v),na.rm=TRUE),logical(1)));stopifnot(length(j)==1);j}
 pc<-getcol('Población Total.*Proyectada');ac<-getcol('Altitud');lc<-getcol('Longitud Oeste');tc<-getcol('Latitud')
 idx<-which(grepl('^[0-9]{6}$',x[[1]]));stopifnot(length(idx)>0)
 y<-data.frame(ubigeo=x[idx,1],district=x[idx,2],creation_legal_text=x[idx,4- as.integer(pc==10)],capital=x[idx,pc-6],capital_category=x[idx,pc-5],capital_altitude_m=as.numeric(x[idx,ac]),capital_longitude_dms=x[idx,lc],capital_latitude_dms=x[idx,tc],population_projected_2024=as.numeric(x[idx,pc]),source_workbook=basename(f),source_sheet=excel_sheets(f)[1],source_row=idx,source_edition=2024)
 # Legal text position found by the actual header, rather than an assumption about blank columns.
 legal<-getcol('Dispositivo Legal');y$creation_legal_text<-x[idx,legal]
 ans[[length(ans)+1]]<-y
 metadata[[length(metadata)+1]]<-data.frame(file=basename(f),n_districts=nrow(y),population_col=pc,altitude_col=ac,longitude_col=lc,latitude_col=tc,legal_col=legal)
}
d<-do.call(rbind,ans);d<-d[order(d$ubigeo),];stopifnot(!anyDuplicated(d$ubigeo),all(nchar(d$ubigeo)==6),all(d$population_projected_2024>=0))
write.csv(d,file.path(root,'datos/geografia/directorio_oficial_distritos_2024.csv'),row.names=FALSE,na='')
write.csv(do.call(rbind,metadata),file.path(root,'validacion/geografia_directorio_extract_metadata.csv'),row.names=FALSE)
cat('Extracted',nrow(d),'official district records; projected total',sum(d$population_projected_2024),'\n')
