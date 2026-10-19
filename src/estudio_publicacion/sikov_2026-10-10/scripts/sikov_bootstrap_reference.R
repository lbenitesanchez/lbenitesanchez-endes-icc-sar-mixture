# Identical innovations, point fits and MSE formula independently in CRAN sae.
args <- commandArgs(trailingOnly=TRUE);indir<-args[[1]];outdir<-args[[2]]
spatial <- if(length(args)>2) args[[3]]=="SAR" else TRUE
.libPaths(c(Sys.getenv("SIKOV_SAE_LIBRARY"),.libPaths()))
suppressPackageStartupMessages(library(sae))
d<-read.csv(file.path(indir,"data.csv"));y<-d$y;x<-d$x;D<-d$D;X<-cbind(1,x)
W<-as.matrix(read.csv(file.path(indir,"W.csv"),header=FALSE))
if(!spatial)W[,]<-0
U<-as.matrix(read.csv(file.path(indir,"u.csv"),header=FALSE));E<-as.matrix(read.csv(file.path(indir,"e.csv"),header=FALSE))
par<-read.csv(file.path(indir,"params.csv"));A<-par$A;rho<-par$rho;beta<-c(par$beta0,par$beta1)
I<-diag(nrow(X));Ci<-solve(t(I-rho*W)%*%(I-rho*W));G<-A*Ci;V<-G+diag(D);Vi<-solve(V);Q<-solve(t(X)%*%Vi%*%X)
K<-G%*%Vi;H<-X-K%*%X;M<-H%*%Q%*%t(X)%*%Vi+K
plugin<-diag(G-G%*%Vi%*%G)+rowSums((H%*%Q)*H)
squares<-matrix(0,nrow(U),nrow(X));terms<-squares;preds<-squares
for(b in 1:nrow(U)) {
 truth<-as.vector(X%*%beta+solve(I-rho*W)%*%(sqrt(A)*U[b,]))
 yboot<-truth+sqrt(D)*E[b,]
 fit<-if(spatial) eblupSFH(yboot~x,vardir=D,proxmat=W,method="REML",MAXITER=1000,PRECISION=1e-9) else eblupFH(yboot~x,vardir=D,method="REML",MAXITER=1000,PRECISION=1e-9)
 if(!fit$fit$convergence)stop("reference failed")
 Ab<-fit$fit$refvar;rb<-if(spatial)fit$fit$spatialcorr else 0
 Cb<-solve(t(I-rb*W)%*%(I-rb*W));Gb<-Ab*Cb;Vib<-solve(Gb+diag(D));Qb<-solve(t(X)%*%Vib%*%X);Kb<-Gb%*%Vib;Hb<-X-Kb%*%X
 pluginb<-diag(Gb-Gb%*%Vib%*%Gb)+rowSums((Hb%*%Qb)*Hb)
 fitted<-as.vector(fit$eblup);oracle<-as.vector(M%*%yboot)
 squares[b,]<-(fitted-truth)^2;terms[b,]<-(-pluginb)+(fitted-oracle)^2;preds[b,]<-fitted
}
dir.create(outdir,recursive=TRUE,showWarnings=FALSE)
write.csv(data.frame(naive=colMeans(squares),bc=2*plugin+colMeans(terms),plugin=plugin),file.path(outdir,"mse.csv"),row.names=FALSE)
write.table(preds,file.path(outdir,"preds.csv"),sep=",",row.names=FALSE,col.names=FALSE)
