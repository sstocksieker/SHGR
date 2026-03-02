
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
from scipy.stats.contingency import association
import copy
from tqdm import trange
import torchsort



############################################################################################
##########                             SHGR PARAMETERS                            ##########
############################################################################################

hp_hgr_epochs = 100
hp_hgr_bs=64
hp_hgr_lr=10e-3
hp_hgr_dimHL=[64,32,16,8]
hp_hgr_eps_es = .5
hp_penalRank = 1.
hp_power = 2
hp_hgr_mp = 20
hp_alpha_test = .05





############################################################################################
##########                              SHGR PAIRWISE                             ##########
############################################################################################



########## DEFINITION ##########
class SHGRp(nn.Module):
    def __init__(self, p,
                 dimHL=hp_hgr_dimHL, 
                 penal_rank=hp_penalRank, 
                 power = hp_power):
        
        super().__init__()
        torch.manual_seed(42)
        self.p = p
        self.dimHL = dimHL
        self.penal_rank = penal_rank
        self.power = power

        self.encoders = nn.ModuleList()
        for i in range(p):
            encoder = nn.Sequential(
                nn.Linear(1, dimHL[0]),
                nn.ReLU(),
                nn.Linear(dimHL[0], dimHL[1]),
                nn.Tanh(),
                nn.Linear(dimHL[1], dimHL[2]),
                nn.Tanh(),
                nn.Linear(dimHL[2], dimHL[3]),
                nn.Tanh(),
                nn.Linear(dimHL[3], 1)
            )
            self.encoders.append(encoder)

    def forward(self, x):
        encoded_variables = []
        for i in range(self.p):
            input_variable = x[:, i:i+1]  
            encoded_variable = self.encoders[i](input_variable)
            encoded_variables.append(encoded_variable)
        encoded = torch.cat(encoded_variables, dim=1)  
        return encoded
    
    def correlationLin_HGR(self,inputs):
        inputs=inputs.T
        correlation_matrix = torch.corrcoef(inputs)
        correlation_matrix = torch.sqrt(correlation_matrix**2)
        loss_corr = torch.sum(correlation_matrix**self.power)
        return -loss_corr 

    def correlation_HGR(self,inputs):
        inputs = inputs.T
        rank=torchsort.soft_rank(inputs, regularization_strength=self.penal_rank)
        correlation_matrix = torch.corrcoef(rank)
        correlation_matrix = torch.sqrt(correlation_matrix**2)
        loss_corr = torch.sum(correlation_matrix**self.power)
        return -loss_corr


      
########## TRAINING ##########
def train_SHGRp(inputs,epochs=hp_hgr_epochs, 
                batch_size=hp_hgr_bs,
                lr=hp_hgr_lr,
                dimHL=hp_hgr_dimHL,
                eps_es = hp_hgr_eps_es,
                max_patience = hp_hgr_mp, 
                type_HGR='Spearman',
                penal_rank=hp_penalRank, 
                power=hp_power):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    inputs0=inputs.copy()
    if inputs.shape[0] > 50000:
        idx = torch.randperm(inputs.shape[0])[:50000]
        inputs = inputs[idx]
    p=inputs.shape[1]
    model = SHGRp(p,dimHL=dimHL,penal_rank=penal_rank, power=power).to(device)
    model_opt = copy.deepcopy(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0001)
    if torch.is_tensor(inputs) == False:
        inputs=torch.FloatTensor(inputs)
    inputs=inputs.to(device)
    res = []
    losses = []
    losses_inputs = []
    best_loss = 0
    best_loss_inputs = 0
    best_epoch = 0
    patience = 0
    mseLoss = nn.MSELoss()
    train_loader = torch.utils.data.DataLoader(
        inputs,batch_size=batch_size,shuffle=True,generator= torch.Generator().manual_seed(42))
    stop_execution = False
    for epoch in trange(epochs, desc="Proving P=NP", unit="carrots"):        
        if max_patience is not None:
            if stop_execution:
                break
        for X in train_loader:
            encoded = model(X.to(device))
            if type_HGR == 'Spearman':
                loss = model.correlation_HGR(encoded)
            elif type_HGR == 'Pearson':
                loss = model.correlationLin_HGR(encoded)
            elif type_HGR == 'Spearman_E':
                loss = model.correlation_HGR_E(encoded)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.cpu().detach().numpy()+0)
        encoded_inputs = model(inputs)
        if type_HGR == 'Spearman':
            loss_inputs = model.correlation_HGR(encoded_inputs)
            losses_inputs.append(loss_inputs.cpu().detach().numpy()+0)
        elif type_HGR == 'Pearson':
            loss_inputs = model.correlationLin_HGR(encoded_inputs)
            losses_inputs.append(loss_inputs.cpu().detach().numpy()+0)
        elif type_HGR == 'Spearman_E':
            loss_inputs = model.correlation_HGR_E(encoded_inputs)
            losses_inputs.append(loss_inputs.cpu().detach().numpy()+0)
        if max_patience is not None:
            if best_loss_inputs-loss_inputs  > eps_es:
                patience = 0
            else:
                patience += 1
            if patience >= max_patience:
                print("Early stopping triggered. Training stopped.")
                stop_execution = True
                break
        if loss_inputs < best_loss_inputs:
            model_opt = copy.deepcopy(model)
            best_loss_inputs = loss_inputs
            best_epoch = epoch  
    encoded = model_opt(torch.FloatTensor(inputs0).to(device))
    return model_opt, encoded, losses, losses_inputs, best_epoch
  
  
  
########## FUNCTIONS ##########
def eta_squared(categories, values):
    groups = pd.DataFrame({'cat': categories, 'val': values})
    grand_mean = groups['val'].mean()
    sct = ((groups['val'] - grand_mean) ** 2).sum()
    sce = groups.groupby('cat').apply(
    lambda g: len(g) * (g['val'].mean() - grand_mean) ** 2, 
    include_groups=False).sum()
    return sce / sct

def SHGR_P(data, 
           type_HGR='Spearman',
           epochs=hp_hgr_epochs, 
           test=False, 
           mask_test=True, 
           alpha=hp_alpha_test, 
           type_corr='pearson', 
           encoded=False, 
           **kwargs):
  
    quali = data.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    quanti = data.select_dtypes(include=['number']).columns.tolist()
    if len(quali) > 0:
        data_sort = data[quanti+quali]
        data_num = data_sort[quanti]
        cols = data_sort.columns
        p=len(cols)
        assoc = pd.DataFrame(np.zeros((p, p)), columns=cols, index=cols)
        for i in range(p):
            for j in range(i,p):
                if i == j:
                    assoc.iloc[i,j]=1.0
                else:
                    var_i = data_sort.iloc[:,i]
                    var_j = data_sort.iloc[:,j]
                    if cols[i] in quali and cols[j] in quali:
                        assoc.iloc[i,j]=assoc.iloc[j,i]=association(pd.crosstab(var_i, var_j).values, method="cramer")
                    elif (cols[i] in quanti and cols[j] in quali):
                        assoc.iloc[i,j]=assoc.iloc[j,i]=eta_squared(var_j, var_i)
                    elif (cols[i] in quali and cols[j] in quanti):
                        assoc.iloc[i,j]=assoc.iloc[j,i]=eta_squared(var_i, var_j)
    else:
        data_sort = data[quanti+quali]
        data_num = data_sort[quanti]
        cols = data_sort.columns
        p=len(cols)
        assoc = pd.DataFrame(np.zeros((p, p)), columns=cols, index=cols)
    data_num = np.array(data_num) 
    _, encoded_HGR, _, _, _ = train_SHGRp(data_num, type_HGR=type_HGR,epochs=epochs, **kwargs)
    encoded_HGR = pd.DataFrame(encoded_HGR.detach().cpu().numpy())
    encoded_HGR=(encoded_HGR - encoded_HGR.mean(axis=0))/encoded_HGR.std(axis=0)   
    p = data_num.shape[1]
    assoc.iloc[:p, :p] = np.abs(encoded_HGR.corr(method=type_corr))    

    pvalues = np.ones((p,p))
    if type_corr == 'spearman':
        for i in np.arange(p):
            for j in np.arange(p):
                pvalues[i,j] = spearmanr(encoded_HGR.iloc[:,i],encoded_HGR.iloc[:,j]).pvalue  
    elif type_corr == 'pearson':
        for i in np.arange(p):
            for j in np.arange(p):
                pvalues[i,j] = spearmanr(encoded_HGR.iloc[:,i],encoded_HGR.iloc[:,j]).pvalue  
    if mask_test:
        assoc.iloc[:p, :p] = assoc.iloc[:p, :p] * (pvalues<alpha)
    assoc=assoc.loc[data.columns, data.columns]
    if test:
        return assoc, pd.DataFrame(np.round(pvalues,2))
    elif encoded:
        return assoc, encoded_HGR
    else:
        return assoc
      
      
      
      

############################################################################################
##########                            SHGR MULTIVARIATE                           ##########
############################################################################################



########## DEFINITION ##########
class SHGRm(nn.Module):
    def __init__(self, p,
                 dimHL=hp_hgr_dimHL,
                 penal_rank=hp_penalRank, 
                 power = hp_power, 
                 target=None):
      
        super().__init__()
        torch.manual_seed(42)
        if target is None:
            self.p = p
        else:
            self.p = p-1
        self.dimHL = dimHL
        self.penal_rank = penal_rank
        self.power = power
        self.target = target

        self.encoders = nn.ModuleList()
        for i in range(self.p):
            encoder = nn.Sequential(
                nn.Linear(1, dimHL[0]),
                nn.ReLU(),
                nn.Linear(dimHL[0], dimHL[1]),
                nn.Tanh(),
                nn.Linear(dimHL[1], dimHL[2]),
                nn.Tanh(),
                nn.Linear(dimHL[2], dimHL[3]),
                nn.Tanh(),
                nn.Linear(dimHL[3], 1)
            )
            self.encoders.append(encoder)
            encoder = nn.Sequential(
                nn.Linear(self.p-1, dimHL[0]),
                nn.ReLU(),
                nn.Linear(dimHL[0], dimHL[1]),
                nn.Tanh(),
                nn.Linear(dimHL[1], dimHL[1]),
                nn.Tanh(),
                nn.Linear(dimHL[1], dimHL[2]),
                nn.Tanh(),
                nn.Linear(dimHL[2], dimHL[3]),
                nn.Linear(dimHL[3], 1)
            )
            self.encoders.append(encoder)

    def forward(self, x):
        encoded_variables = []
        j = 0
        if self.target is None:
            for i in range(self.p):
                input_variable = x[:, i:i+1]  
                encoded_variable = self.encoders[j](input_variable)
                encoded_variables.append(encoded_variable)
                j+=1
                input_variable = torch.cat((x[:,:i], x[:,i+1:]),dim=1)  
                encoded_variable = self.encoders[j](input_variable)
                encoded_variables.append(encoded_variable)
                j+=1
        else:
            y=x[:, self.target:self.target+1]
            x = torch.cat((x[:, :self.target], x[:, self.target+1:]), dim=1)
            for i in range(self.p):
                input_variable = y 
                encoded_variable = self.encoders[j](input_variable)
                encoded_variables.append(encoded_variable)
                j+=1
                input_variable = torch.cat((x[:,:i], x[:,i+1:]),dim=1) 
                encoded_variable = self.encoders[j](input_variable)
                encoded_variables.append(encoded_variable)
                j+=1
        encoded = torch.cat(encoded_variables, dim=1)  
        return encoded

    def correlationLin_HGR(self,inputs, eps=1e-4):
        inputs = inputs.T
        corr_matrix = torch.corrcoef(inputs)
        corr_matrix = torch.sqrt(corr_matrix**2)
        indices = torch.arange(0, corr_matrix.shape[1], 2)
        succ_corr = corr_matrix[indices, indices + 1]
        loss_corr = torch.sum(succ_corr**self.power)       
        if torch.isnan(loss_corr):
            print("loss_corr : ",loss_corr)
            print("inputs : ",inputs.isnan().sum(1))
            stop
        return -loss_corr

    def correlationLinEsp_HGR(self,inputs):
        mean = inputs.mean(dim=0, keepdim=True)
        std = inputs.std(dim=0, keepdim=True)
        inputs_n = (inputs - mean) / (std)
        correlation_matrix = torch.corrcoef(inputs_n.T)
        correlation_matrix = torch.sqrt(correlation_matrix**2)
        indices = torch.arange(0, correlation_matrix.shape[1], 2)
        succ_corr = correlation_matrix[indices, indices + 1]
        loss_corr = torch.sum(succ_corr**self.power) 
        return -loss_corr
    
    def correlation_HGR(self,inputs, eps=1e-4):
        inputs=inputs.T
        rank=torchsort.soft_rank(inputs, regularization_strength=self.penal_rank)
        corr_matrix = torch.corrcoef(rank)
        corr_matrix = torch.sqrt(corr_matrix**2)
        indices = torch.arange(0, corr_matrix.shape[1], 2)
        succ_corr = corr_matrix[indices, indices + 1]
        loss = (torch.sum(succ_corr**2))*self.power
        return -loss
      
      
########## TRAINING ##########
def train_SHGRm(inputs,
                epochs=hp_hgr_epochs,
                batch_size=hp_hgr_bs,
                lr=hp_hgr_lr,
                dimHL=hp_hgr_dimHL,
                eps_es = hp_hgr_eps_es,
                max_patience = hp_hgr_mp, 
                type_HGR='Spearman', 
                target=None,
                penal_rank=hp_penalRank, 
                power=hp_power):
  
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    p=inputs.shape[1]
    model = SHGRm(p,dimHL=dimHL,penal_rank=penal_rank, power=power, target=target).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0001)
    if torch.is_tensor(inputs) == False:
        inputs=torch.FloatTensor(inputs)
    inputs=inputs.to(device)
    res = []
    losses = []
    losses_inputs = []
    best_loss = 0
    best_loss_inputs = 0
    best_epoch = 0
    patience = 0
    mseLoss = nn.MSELoss()
    train_loader = torch.utils.data.DataLoader(
        inputs,batch_size=batch_size,shuffle=True,generator= torch.Generator().manual_seed(42))
    stop_execution = False
    for epoch in trange(epochs, desc="Proving P=NP", unit="carrots"):
        if max_patience is not None:
            if stop_execution:
                break
        for X in train_loader:
            encoded = model(X.to(device))
            if type_HGR == 'Spearman':
                loss = model.correlation_HGR(encoded)
            elif type_HGR == 'Pearson':
                loss = model.correlationLin_HGR(encoded)
            elif type_HGR == 'NN':
                loss = model.correlationLinEsp_HGR(encoded)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.cpu().detach().numpy()+0)
        # Best and early stopping
        encoded_inputs = model(inputs)
        if type_HGR == 'Spearman':
            loss_inputs = model.correlation_HGR(encoded_inputs)
            losses_inputs.append(loss_inputs.cpu().detach().numpy()+0)
        elif type_HGR == 'Pearson':
            loss_inputs = model.correlationLin_HGR(encoded_inputs)
            losses_inputs.append(loss_inputs.cpu().detach().numpy()+0)
        elif type_HGR == 'NN':
            loss_inputs = model.correlationLinEsp_HGR(encoded_inputs)
            losses_inputs.append(loss_inputs.cpu().detach().numpy()+0)
        if max_patience is not None:
            if best_loss_inputs-loss_inputs  > eps_es:
                patience = 0
            else:
                patience += 1
            if patience >= max_patience:
                print("Early stopping triggered. Training stopped.")
                stop_execution = True
                break
        if loss_inputs < best_loss_inputs:
            model_opt = copy.deepcopy(model)
            best_loss_inputs = loss_inputs
            best_epoch = epoch  
    encoded = model_opt(inputs)
    return model_opt, encoded, losses, losses_inputs, best_epoch

  
  
########## FUNCTIONS ##########
def corr_multi(df):
    p = df.shape[1]
    if p % 2 != 0:
        raise ValueError("The number of columns must be even")
    corrs = []
    pvals = []
    for i in range(0, p, 2):
        col1, col2 = df.columns[i], df.columns[i + 1]
        r, pval = pearsonr(df[col1], df[col2])
        corrs.append(abs(r))      
        pvals.append(pval)        
    return np.array(corrs), np.array(pvals)

def SHGR_M(data, 
           type_HGR='Spearman',
           epochs=100, 
           test=False, 
           mask_test=True, 
           alpha=hp_alpha_test, 
           **kwargs):   
    data_num = np.array(data) 
    _, encoded_HGR, _, _, _ = train_SHGRm(data_num, type_HGR=type_HGR,epochs=epochs, **kwargs)
    encoded_HGR = pd.DataFrame(encoded_HGR.detach().cpu().numpy())
    encoded_HGR=(encoded_HGR - encoded_HGR.mean(axis=0))/encoded_HGR.std(axis=0)  
    p = data_num.shape[1] 
    corrM, pvalues = corr_multi(encoded_HGR)
    assoc = np.abs(corrM)    
    if mask_test:
        assoc = assoc * (pvalues<alpha)
    if test:
        return assoc, pd.DataFrame(np.round(pvalues,2))
    else:
        return assoc
      
      
      
      
      
############################################################################################
##########                                SHGR FULL                               ##########
############################################################################################



########## DEFINITION ##########
class SHGRf(nn.Module):
    def __init__(self, p, q, 
                 dimHL=hp_hgr_dimHL,
                 penal_rank=hp_penalRank, 
                 power = hp_power):
      
        super().__init__()
        torch.manual_seed(42)
        self.p = p
        self.q = q
        self.dimHL = dimHL
        self.penal_rank = penal_rank
        self.power = power

        self.encoders = nn.ModuleList()
        for d in [p,q]:
            encoder = nn.Sequential(
                nn.Linear(d, dimHL[0]),
                nn.ReLU(),
                nn.Linear(dimHL[0], dimHL[1]),
                nn.Tanh(),
                nn.Linear(dimHL[1], dimHL[2]),
                nn.Tanh(),
                nn.Linear(dimHL[2], dimHL[3]),
                nn.Tanh(),
                nn.Linear(dimHL[3], 1)
            )
            self.encoders.append(encoder)

    def forward(self, x, y):
        encoded_variables = []
        encoded_variable = self.encoders[0](x)
        encoded_variables.append(encoded_variable)
        encoded_variable = self.encoders[1](y)
        encoded_variables.append(encoded_variable)
        encoded = torch.cat(encoded_variables, dim=1)
        return encoded
    
    def correlationLin_HGR(self,inputs):
        inputs=inputs.T
        correlation_matrix = torch.corrcoef(inputs)
        correlation_matrix = torch.sqrt(correlation_matrix**2)
        loss_corr = torch.sum(correlation_matrix**self.power)
        return -loss_corr 

    def correlation_HGR(self,inputs):
        inputs = inputs.T      
        rank=torchsort.soft_rank(inputs, regularization_strength=self.penal_rank)
        correlation_matrix = torch.corrcoef(rank)
        correlation_matrix = torch.sqrt(correlation_matrix**2)
        loss = (torch.sum(correlation_matrix) - torch.trace(correlation_matrix))**self.power
        return -loss

      
      
########## TRAINING ##########  
def train_SHGRf(inputsX, inputsY,
                epochs=hp_hgr_epochs,
                batch_size=hp_hgr_bs,
                lr=hp_hgr_lr,
                dimHL=hp_hgr_dimHL,
                eps_es = hp_hgr_eps_es,
                max_patience = hp_hgr_mp, 
                type_HGR='Spearman',
                penal_rank=hp_penalRank, 
                power=hp_power):
  
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    p=inputsX.shape[1]
    q=inputsY.shape[1]
    model = SHGRf(p,q,dimHL=dimHL,penal_rank=penal_rank, power=power).to(device)
    model_opt = copy.deepcopy(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0001)
    if torch.is_tensor(inputsX) == False:
        inputsX=torch.FloatTensor(inputsX)
    if torch.is_tensor(inputsY) == False:
        inputsY=torch.FloatTensor(inputsY)
    inputsX=inputsX.to(device)
    inputsY=inputsY.to(device)
    res = []
    losses = []
    losses100 = []
    losses_inputs = []
    encod100 = []
    best_loss = 0
    best_loss_inputs = 0
    best_epoch = 0
    patience = 0
    epoch_prec = 0
    mseLoss = nn.MSELoss()
    loaderX = torch.utils.data.DataLoader(inputsX, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(42))
    loaderY = torch.utils.data.DataLoader(inputsY, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(42))
    stop_execution = False
    for epoch in trange(epochs, desc="Proving P=NP", unit="carrots"):
        for X, Y in zip(loaderX, loaderY):
            encoded = model(X.to(device), Y.to(device))
            if type_HGR == 'Spearman':
                loss = model.correlation_HGR(encoded)
            elif type_HGR == 'Pearson':
                loss = model.correlationLin_HGR(encoded)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.cpu().detach().numpy()+0)
        # Best and early stopping
        encoded_inputs = model(inputsX, inputsY)
        if type_HGR == 'Spearman':
            loss_inputs = model.correlation_HGR(encoded_inputs)
            losses_inputs.append(loss_inputs.cpu().detach().numpy()+0)
        elif type_HGR == 'Pearson':
            loss_inputs = model.correlationLin_HGR(encoded_inputs)
            losses_inputs.append(loss_inputs.cpu().detach().numpy()+0)
        if max_patience is not None:
            if best_loss_inputs-loss_inputs  > eps_es:
                patience = 0
            else:
                patience += 1
            if patience >= max_patience:
                print("Early stopping triggered. Training stopped.")
                stop_execution = True
                break
        if loss_inputs < best_loss_inputs:
            model_opt = copy.deepcopy(model)
            best_loss_inputs = loss_inputs
            best_epoch = epoch  
    encoded = model_opt(inputsX, inputsY)
    return model_opt, encoded, losses, losses_inputs, best_epoch
  
  
  
########## FUNCTIONS ##########
def SHGR_F(dataX, dataY, 
           type_HGR='Spearman',
           epochs=hp_hgr_epochs, 
           test=False, 
           mask_test=True, 
           alpha=hp_alpha_test, 
           **kwargs): 
    dataX = np.array(dataX) 
    dataY = np.array(dataY)
    n_samples = min(len(dataX), len(dataY))
    dataX = dataX[:n_samples,:]
    dataY = dataY[:n_samples,:]
    _, encoded_HGR, _,_, _ = train_SHGRf(dataX, dataY, type_HGR=type_HGR,epochs=epochs, **kwargs)
    encoded_HGR = pd.DataFrame(encoded_HGR.detach().cpu().numpy())
    np.abs(encoded_HGR.corr(method="pearson"))    
    assoc, pvalues = spearmanr(encoded_HGR.iloc[:,0],encoded_HGR.iloc[:,1])
    if mask_test:
        assoc = assoc * (pvalues<alpha)
    if test:
        return np.abs(assoc), pd.DataFrame(np.round(pvalues,2))
    else:
        return np.abs(assoc)
      
      
      
def Contrib_M(inputs,
              target, 
              type_HGR='Spearman',
              epochs=hp_hgr_epochs, 
              test=False, 
              mask_test=True, 
              alpha=hp_alpha_test, 
              **kwargs):   
    ref = SHGR_M(inputs)[0]
    return ref - SHGR_M(inputs, target=target,**kwargs)
    
  
def Contrib_MF(X, Y, 
               type_HGR='Spearman',
               epochs=hp_hgr_epochs, 
               test=False, 
               mask_test=True, 
               alpha=hp_alpha_test, 
               **kwargs):   
    Ctr = np.repeat(0.,X.shape[1])
    ref = SHGR_F(np.array(X),np.array(Y), **kwargs)
    i=0
    for col in X.columns:
        hgr_i=SHGR_F(np.array(X.drop(columns=col)),np.array(Y), **kwargs)
        Ctr[i] = np.abs(ref - hgr_i)
        i += 1
    return Ctr