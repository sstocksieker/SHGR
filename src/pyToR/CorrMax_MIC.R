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

Corr_MIC = matrix(rep(0,p*p),p,p)
Pval_MIC = Corr_MIC
for (i in (1:p)){
  for (j in (1:p)){
    corr = AlterCorr(data[,i], data[,j], type = "MIC",R=1)
    Corr_MIC[i,j] = corr$Correlation
    Pval_MIC[i,j] = corr$pvalue
  }
}

write.csv(Corr_MIC, file = paste0(rep_fromR,"Corr_MIC.csv"), row.names = F)
write.csv(Pval_MIC, file = paste0(rep_fromR,"Pval_MIC.csv"), row.names = F)


