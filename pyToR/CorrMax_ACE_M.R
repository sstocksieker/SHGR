# Modules
library(acepack)
library(minerva)
library(kernlab)
library(energy)
library(mvtnorm)
library(AlterCorr)
library(kernlab)


script_path <- normalizePath(sub("--file=", "", commandArgs(trailingOnly = FALSE)[grep("--file=", commandArgs())]))
rep_courant <- dirname(script_path)
# rep_courant <- 'C://Users//samgo//OneDrive//Perso//Thèse//Travaux//KurtHGR_PW_mixed//Code//sansBruit//pyToR' #dirname(script_path)
rep_fromR = paste0(rep_courant,'/fromR/')
rep_fromPy = paste0(rep_courant,'/fromPy/')
cat("rep_fromR = ", rep_fromR, "\n")
setwd(rep_fromR)


data = read.csv(paste0(rep_fromPy,"data_M.csv"))




# Illustration
## Data
p=ncol(data)

# ## CCA
# Corr_CCA = matrix(rep(0,p*p),p,p)
# for (i in (1:p)){
#   for (j in (1:p)){
#     Corr_CCA[i,j] = cancor(data[,i], data[,j])$cor
#   }
# }
# ## kCCA
# Corr_kCCA = matrix(rep(0,p*p),p,p)
# for (i in (1:p)){
#   for (j in (1:p)){
#     x <- matrix(c(data[,i],data[,i]),2)
#     y <- matrix(c(data[,j],data[,j]),2)
#     Corr_kCCA[i,j] = kcca(x,y,ncomps=2)@xcoef[1,2]
#   }
# }
# ## kMMD
# Corr_kMMD = matrix(rep(0,p*p),p,p)
# for (i in (1:p)){
#   for (j in (1:p)){
#     x <- matrix(c(data[,i],data[,i]),2)
#     y <- matrix(c(data[,j],data[,j]),2)
#     Corr_kMMD[i,j] = kmmd(x,y,ncomps=2)@mmdstats[1]
#   }
# }
# ## RDC
# Corr_RDC = matrix(rep(0,p*p),p,p)
# for (i in (1:p)){
#   for (j in (1:p)){
#     Corr_RDC[i,j] = AlterCorr(data[,i], data[,j], type = "RDC")$Correlation
#   }
# }
# ## MIC
# Corr_MIC = matrix(rep(0,p*p),p,p)
# for (i in (1:p)){
#   for (j in (1:p)){
#     Corr_MIC[i,j] = AlterCorr(data[,i], data[,j], type = "MIC",R=1)$Correlation
#   }
# }
# ## dCor
# Corr_dCor = matrix(rep(0,p*p),p,p)
# for (i in (1:p)){
#   for (j in (1:p)){
#     Corr_dCor[i,j] = AlterCorr(data[,i], data[,j], type = "dCor",R=1)$Correlation
#   }
# }
## ACE
Corr_ACE = rep(0,p)
for (i in (1:p)){
    Corr_ACE[i] = ace(data[,-i], data[,i])$rsq
}




# Export 
# ## CCA
# write.csv(Corr_CCA, file = paste0(rep_fromR,"Corr_CCA.csv"), row.names = F)
# ## kCCA
# write.csv(Corr_kCCA, file = paste0(rep_fromR,"Corr_kCCA.csv"), row.names = F)
# ## kMMD
# write.csv(Corr_kMMD, file = paste0(rep_fromR,"Corr_kMMD.csv"), row.names = F)
# ## RDC
# write.csv(Corr_RDC, file = paste0(rep_fromR,"Corr_RDC.csv"), row.names = F)
# ## MIC
# write.csv(Corr_MIC, file = paste0(rep_fromR,"Corr_MIC.csv"), row.names = F)
# ## dCor
# write.csv(Corr_dCor, file = paste0(rep_fromR,"Corr_dCor.csv"), row.names = F)
## ACE
write.csv(Corr_ACE, file = paste0(rep_fromR,"Corr_ACE_M.csv"), row.names = F)



