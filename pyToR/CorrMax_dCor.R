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

Corr_dCor = matrix(rep(0,p*p),p,p)
Pval_dCor = Corr_dCor
for (i in (1:p)){
  for (j in (1:p)){
    corr = AlterCorr(data[,i], data[,j], type = "dCor",R=1)
    Corr_dCor[i,j] = corr$Correlation
    Pval_dCor[i,j] = corr$pvalue
  }
}

write.csv(Corr_dCor, file = paste0(rep_fromR,"Corr_dCor.csv"), row.names = F)
write.csv(Pval_dCor, file = paste0(rep_fromR,"Pval_dCor.csv"), row.names = F)



