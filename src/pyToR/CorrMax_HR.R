library(acepack)
library(minerva)
library(kernlab)
library(energy)
library(mvtnorm)
library(AlterCorr)
library(kernlab)
library(HellCor)


script_path <- normalizePath(sub("--file=", "", commandArgs(trailingOnly = FALSE)[grep("--file=", commandArgs())]))
rep_courant <- dirname(script_path)
rep_fromR = paste0(rep_courant,'/fromR/')
rep_fromPy = paste0(rep_courant,'/fromPy/')
cat("rep_fromR = ", rep_fromR, "\n")
setwd(rep_fromR)


data = read.csv(paste0(rep_fromPy,"data.csv"))


p=ncol(data)

Corr_HR = matrix(rep(0,p*p),p,p)
Pval_HR = matrix(rep(0,p*p),p,p)
for (i in (1:p)){
  for (j in (1:p)){
    corr = HellCor(data[,i], data[,j])
    Corr_HR[i,j] = corr$Hcor
    Pval_HR[i,j] = corr$p.value
  }
}


write.csv(Corr_HR, file = paste0(rep_fromR,"Corr_HR.csv"), row.names = F)
write.csv(Pval_HR, file = paste0(rep_fromR,"Pval_HR.csv"), row.names = F)
