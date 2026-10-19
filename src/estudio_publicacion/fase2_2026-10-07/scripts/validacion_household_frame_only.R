# Only household frame metadata is read before response unlock. RECH1/RECH6
# child response files are neither extracted nor parsed by this program.
args <- commandArgs(trailingOnly=TRUE)
root <- normalizePath(if(length(args)) args[1] else '.')
study <- dirname(root)
.libPaths(c(file.path(study, '.Rlib'), .libPaths()))
suppressPackageStartupMessages(library(haven))
h <- read_sav(file.path(root,'datos/private_external_raw/RECH0_2024.SAV'))
stopifnot(!anyDuplicated(h$HHID), all(as.numeric(h$HV001)==as.numeric(h$HV021)))
frame <- data.frame(household_id=as.character(h$HHID),
                    ubigeo=sprintf('%06d',as.integer(h$UBIGEO)),
                    weight=as.numeric(h$HV005)/1e6, psu=as.numeric(h$HV001),
                    stratum=as.numeric(h$HV022), cluster_altitude=as.numeric(h$HV040),
                    interview_year=as.numeric(h$HV007))
frame$psu_key <- paste(frame$stratum,frame$psu,sep=':')
stopifnot(all(is.finite(frame$weight)&frame$weight>=0))
write.csv(frame,file.path(root,'datos/private_external_raw/household_frame_2024.csv'),row.names=FALSE,na='')
cat('Household frame only written; child responses not read.\n')
