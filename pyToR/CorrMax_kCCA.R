library(acepack)
library(minerva)
library(kernlab)
library(energy)
library(mvtnorm)
library(AlterCorr)
library(kernlab)

script_path <- normalizePath(sub("--file=", "", commandArgs(trailingOnly = FALSE)[grep("--file=", commandArgs())]))
rep_courant <- dirname(script_path)
rep_fromR = paste0(rep_courant,'/fromR/')
rep_fromPy = paste0(rep_courant,'/fromPy/')
cat("rep_fromR = ", rep_fromR, "\n")
setwd(rep_fromR)


data = read.csv(paste0(rep_fromPy,"data.csv"))


p=ncol(data)


Corr_kCCA = matrix(rep(0,p*p),p,p)
for (i in (1:p)){
  for (j in (1:p)){
    x <- matrix(c(data[,i],data[,i]),2)
    y <- matrix(c(data[,j],data[,j]),2)
    Corr_kCCA[i,j] = kcca(x,y,ncomps=2)@kcor[2]
  }
}


write.csv(Corr_kCCA, file = paste0(rep_fromR,"Corr_kCCA.csv"), row.names = F)
