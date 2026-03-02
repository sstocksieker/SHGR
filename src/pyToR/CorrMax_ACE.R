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


Corr_ACE = matrix(rep(0,p*p),p,p)
for (i in (1:p)){
  for (j in (1:p)){
    Corr_ACE[i,j] = ace(data[,i], data[,j])$rsq
  }
}


write.csv(Corr_ACE, file = paste0(rep_fromR,"Corr_ACE.csv"), row.names = F)



