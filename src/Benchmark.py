import numpy as np
from scipy.stats import gamma
from math import pi, sqrt
from maxcorr import indicator
import torch.nn as nn
import pandas as pd
import torch
import copy
from tqdm import trange
from scipy.stats import pearsonr
import warnings





############################################################################################
##########                             HGR PARAMETERS                             ##########
############################################################################################

hp_hgr_epochs = 100
hp_hgr_bs=64
hp_hgr_lr=10e-3
hp_hgr_dimHL=[64,32,16,8]
hp_penalRank = 1.
hp_power = 2
hp_hgr_eps_es = .5
hp_hgr_mp = 20





############################################################################################
##########          MaxCorr : hgr_nn, hgr_kde, GEDI (dk, sk), lat                 ##########
############################################################################################
########## Definition

def MaxCorr_P(data, algo='dk', **kwargs): 
    p=data.shape[1]
    assoc = pd.DataFrame(np.zeros((p, p)), columns=data.columns, index=data.columns)
    ind = indicator(semantics='hgr', algorithm=algo, backend='numpy')
    for i in range(p):
        for j in range(i,p):
            if i == j:
                assoc.iloc[i,j]=1.0
            else:
                assoc.iloc[i,j]=assoc.iloc[j,i]=ind.compute(data.iloc[:,i], data.iloc[:,j])
    return assoc
  
  
def MaxCorr_M(data, algo='dk', **kwargs): 
    p=data.shape[1]
    assoc = pd.DataFrame(np.zeros(p))
    ind = indicator(semantics='hgr', algorithm=algo, backend='numpy')
    for i in range(p):
        assoc.iloc[i]=ind.compute(data.iloc[:,i], data.drop(data.columns[i], axis=1))
    return assoc
  
  
  
  
  
############################################################################################
##########                              HGR_NN-Multi                              ##########
############################################################################################

########## Definition
class HGRnn_m(nn.Module):
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
            encoder = nn.Sequential(
                nn.Linear(p-1, dimHL[0]),
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
        for i in range(self.p):
            input_variable = x[:, i:i+1] 
            encoded_variable = self.encoders[j](input_variable)
            encoded_variables.append(encoded_variable)
            j+=1
            input_variable = torch.cat((x[:,:i], x[:,i+1:]),dim=1)  
            encoded_variable = self.encoders[j](input_variable)
            encoded_variables.append(encoded_variable)
            j+=1
        encoded = torch.cat(encoded_variables, dim=1)  
        return encoded
    
    def correlationLin_HGR(self,inputs):
        mean = inputs.mean(dim=0, keepdim=True)
        std = inputs.std(dim=0, keepdim=True)
        inputs_n = (inputs - mean) / (std)
        correlation_matrix = torch.corrcoef(inputs_n.T)
        indices = torch.arange(0, correlation_matrix.shape[1], 2)
        succ_corr = correlation_matrix[indices, indices + 1]
        loss_corr = torch.sum(succ_corr**self.power) 
        return -loss_corr
      
      
########## Training
def train_HGRnn_m(inputs,
                  epochs=hp_hgr_epochs,
                  batch_size=hp_hgr_bs,
                  lr=hp_hgr_lr,
                  dimHL=hp_hgr_dimHL,
                  eps_es = hp_hgr_eps_es,
                  max_patience = hp_hgr_mp, 
                  type_HGR='Pearson',
                  penal_rank=hp_penalRank, 
                  power=hp_power):
  
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    p=inputs.shape[1]
    model = HGRnn_m(p,dimHL=dimHL,penal_rank=penal_rank, power=power).to(device)
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
    epoch_prec = 0
    mseLoss = nn.MSELoss()
    train_loader = torch.utils.data.DataLoader(
        inputs,batch_size=batch_size,shuffle=True,generator= torch.Generator().manual_seed(42))
    stop_execution = False
    for epoch in trange(epochs, desc="Proving P=NP", unit="carrots"):
        if max_patience is not None:
            if stop_execution:
                break
        # Mini batch learning
        for X in train_loader:
            encoded = model(X.to(device))
            if type_HGR == 'Spearman':
                loss = model.correlation_HGR(encoded)
            elif type_HGR == 'Pearson':
                loss = model.correlationLin_HGR(encoded)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.cpu().detach().numpy()+0)
        # Best and early stopping
        encoded_inputs = model(inputs)
        if type_HGR == 'Spearman':
            loss_inputs = model.correlation_HGR(encoded_inputs)
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
    encoded = model_opt(inputs)
    return model_opt, encoded, losses, losses_inputs, best_epoch
  
  
  
########## Functions
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

def HGR_NN_M(data, type_HGR='Pearson',epochs=100, test=False, mask_test=True, alpha=0.05, **kwargs): 
    data_num = np.array(data) 
    _, encoded_HGR, _, _, _ = train_HGRnn_m(data_num, type_HGR=type_HGR,epochs=epochs, **kwargs)
    encoded_HGR = pd.DataFrame(encoded_HGR.detach().cpu().numpy())  
    corrM, pvalues = corr_multi(encoded_HGR)
    assoc = np.abs(corrM)    
    return assoc
  
  
  

############################################################################################
##########                                  HSIC                                  ##########
############################################################################################

########## Definition
"""
python implementation of Hilbert Schmidt Independence Criterion
hsic_gam implements the HSIC test using a Gamma approximation
Python 2.7.12

Gretton, A., Fukumizu, K., Teo, C. H., Song, L., Scholkopf, B., 
& Smola, A. J. (2007). A kernel statistical test of independence. 
In Advances in neural information processing systems (pp. 585-592).

Shoubo (shoubo.sub AT gmail.com)
09/11/2016

Inputs:
X 		n by dim_x matrix
Y 		n by dim_y matrix
alph 		level of test

Outputs:
testStat	test statistics
thresh		test threshold for level alpha test
"""
def rbf_dot(pattern1, pattern2, deg):
	size1 = pattern1.shape
	size2 = pattern2.shape

	G = np.sum(pattern1*pattern1, 1).reshape(size1[0],1)
	H = np.sum(pattern2*pattern2, 1).reshape(size2[0],1)

	Q = np.tile(G, (1, size2[0]))
	R = np.tile(H.T, (size1[0], 1))

	H = Q + R - 2* np.dot(pattern1, pattern2.T)

	H = np.exp(-H/2/(deg**2))

	return H


def hsic_gam(X, Y, alph = 0.5):
	"""
	X, Y are numpy vectors with row - sample, col - dim
	alph is the significance level
	auto choose median to be the kernel width
	"""
	n = X.shape[0]

	# ----- width of X -----
	Xmed = X

	G = np.sum(Xmed*Xmed, 1).reshape(n,1)
	Q = np.tile(G, (1, n) )
	R = np.tile(G.T, (n, 1) )

	dists = Q + R - 2* np.dot(Xmed, Xmed.T)
	dists = dists - np.tril(dists)
	dists = dists.reshape(n**2, 1)

	width_x = np.sqrt( 0.5 * np.median(dists[dists>0]) )
	# ----- -----

	# ----- width of X -----
	Ymed = Y

	G = np.sum(Ymed*Ymed, 1).reshape(n,1)
	Q = np.tile(G, (1, n) )
	R = np.tile(G.T, (n, 1) )

	dists = Q + R - 2* np.dot(Ymed, Ymed.T)
	dists = dists - np.tril(dists)
	dists = dists.reshape(n**2, 1)

	width_y = np.sqrt( 0.5 * np.median(dists[dists>0]) )
	# ----- -----

	bone = np.ones((n, 1), dtype = float)
	H = np.identity(n) - np.ones((n,n), dtype = float) / n

	K = rbf_dot(X, X, width_x)
	L = rbf_dot(Y, Y, width_y)

	Kc = np.dot(np.dot(H, K), H)
	Lc = np.dot(np.dot(H, L), H)

	testStat = np.sum(Kc.T * Lc) / n

	varHSIC = (Kc * Lc / 6)**2

	varHSIC = ( np.sum(varHSIC) - np.trace(varHSIC) ) / n / (n-1)

	varHSIC = varHSIC * 72 * (n-4) * (n-5) / n / (n-1) / (n-2) / (n-3)

	K = K - np.diag(np.diag(K))
	L = L - np.diag(np.diag(L))

	muX = np.dot(np.dot(bone.T, K), bone) / n / (n-1)
	muY = np.dot(np.dot(bone.T, L), bone) / n / (n-1)

	mHSIC = (1 + muX * muY - muX - muY) / n

	al = mHSIC**2 / varHSIC
	bet = varHSIC*n / mHSIC

	thresh = gamma.ppf(1-alph, al, scale=bet)[0][0]

	return (testStat, thresh)
  

  
########## Functions
def nHSIC(X,Y,alpha=0.05):
    hsic_x=hsic_gam(X.values.reshape(-1, 1), X.values.reshape(-1, 1), alph = alpha)
    hsic_y=hsic_gam(Y.values.reshape(-1, 1), Y.values.reshape(-1, 1), alph = alpha)
    hsic_data=hsic_gam(X.values.reshape(-1, 1), Y.values.reshape(-1, 1), alph = alpha)
    return hsic_data[0]/np.sqrt(hsic_x[0]*hsic_y[0])
  
def NHSIC(data):
    p = data.shape[1]
    HGR_nHSIC = np.ones((p,p))
    for i in np.arange(p):
        for j in np.arange(p):
            HGR_nHSIC[i,j]=nHSIC(data.iloc[:,i], data.iloc[:,j])
    return HGR_nHSIC
  
def nHSIC_M(X,Y,alpha=0.05):
    hsic_x=hsic_gam(X.values.reshape(-1, 1), X.values.reshape(-1, 1), alph = alpha)
    hsic_y=hsic_gam(Y.values, Y.values, alph = alpha)
    hsic_data=hsic_gam(X.values.reshape(-1, 1), Y.values, alph = alpha)
    return hsic_data[0]/np.sqrt(hsic_x[0]*hsic_y[0])
def NHSIC_M(data):
    p = data.shape[1]
    HGR_nHSIC = np.ones(p)
    for i in np.arange(p):
        HGR_nHSIC[i]=nHSIC_M(data.iloc[:,i], data.drop(columns=[data.columns[i]]))
    return HGR_nHSIC
  
def NHSIC_M_y(data):
    p = data.shape[1]
    y=data.iloc[:,p-1]
    X=data.drop(columns=[data.columns[p-1]])
    ref_corr = nHSIC_M(y, X)
    HGR_nHSIC = np.ones(p-1)
    for i in np.arange(p-1):
        HGR_nHSIC[i]=ref_corr - nHSIC_M(y, X.drop(columns=[X.columns[i]]))
    return HGR_nHSIC
  
  
  
  

############################################################################################
##########                                HGR-KDE                                 ##########
############################################################################################

########## Definition
class kde:
    """
    A Gaussian KDE implemented in pytorch for the gradients to flow in pytorch optimization.
    Keep in mind that KDE are not scaling well with the number of dimensions and this implementation is not really
    optimized...
    """
    def __init__(self, x_train):
        n, d = x_train.shape

        self.n = n
        self.d = d

        self.bandwidth = (n * (d + 2) / 4.) ** (-1. / (d + 4))
        self.std = self.bandwidth

        self.train_x = x_train

    def pdf(self, x):
        s = x.shape
        d = s[-1]
        s = s[:-1]
        assert d == self.d

        data = x.unsqueeze(-2)

        train_x = _unsqueeze_multiple_times(self.train_x, 0, len(s))

        pdf_values = (
                         torch.exp(-((data - train_x).norm(dim=-1) ** 2 / (self.bandwidth ** 2) / 2))
                     ).mean(dim=-1) / np.sqrt(2 * pi) / self.bandwidth

        return pdf_values


def _unsqueeze_multiple_times(input, axis, times):
    """
    Utils function to unsqueeze tensor to avoid cumbersome code
    :param input: A pytorch Tensor of dimensions (D_1,..., D_k)
    :param axis: the axis to unsqueeze repeatedly
    :param times: the number of repetitions of the unsqueeze
    :return: the unsqueezed tensor. ex: dimensions (D_1,... D_i, 0,0,0, D_{i+1}, ... D_k) for unsqueezing 3x axis i.
    """
    output = input
    for i in range(times):
        output = output.unsqueeze(axis)
    return output
  
# Independence of 2 variables
def _joint_2(X, Y, density, damping=1e-10):
    X = (X - X.mean()) / X.std()
    Y = (Y - Y.mean()) / Y.std()
    data = torch.cat([X.unsqueeze(-1), Y.unsqueeze(-1)], -1)
    joint_density = density(data)

    nbins = int(min(50, 5. / joint_density.std))
    #nbins = np.sqrt( Y.size/5 )
    x_centers = torch.linspace(-2.5, 2.5, nbins)
    y_centers = torch.linspace(-2.5, 2.5, nbins)

    xx, yy = torch.meshgrid([x_centers, y_centers])
    grid = torch.cat([xx.unsqueeze(-1), yy.unsqueeze(-1)], -1)
    h2d = joint_density.pdf(grid) + damping
    h2d /= h2d.sum()
    return h2d
  
def hgr(X, Y, density, damping = 1e-10):
    """
    An estimator of the Hirschfeld-Gebelein-Renyi maximum correlation coefficient using Witsenhausen’s Characterization:
    HGR(x,y) is the second highest eigenvalue of the joint density on (x,y). We compute here the second eigenvalue on
    an empirical and discretized density estimated from the input data.
    :param X: A torch 1-D Tensor
    :param Y: A torch 1-D Tensor
    :param density: so far only kde is supported
    :return: numerical value between 0 and 1 (0: independent, 1:linked by a deterministic equation)
    """
    h2d = _joint_2(X, Y, density, damping=damping)
    marginal_x = h2d.sum(dim=1).unsqueeze(1)
    marginal_y = h2d.sum(dim=0).unsqueeze(0)
    Q = h2d / (torch.sqrt(marginal_x) * torch.sqrt(marginal_y))
    return torch.svd(Q)[1][1]
  
def HGR_KDE(data):
    p = data.shape[1]
    HGR_KDE = np.zeros((p, p))
    for i in np.arange(p):
        for j in np.arange(p):
            x_i = data.iloc[:, i].values.astype(np.float32)
            x_j = data.iloc[:, j].values.astype(np.float32)
            HGR_KDE[i, j] = hgr(torch.from_numpy(x_i), torch.from_numpy(x_j), kde).item()
    return HGR_KDE


  

############################################################################################
##########                         ACE (Python) : Unused                          ##########
############################################################################################
import numpy as np
import seaborn as sns
import contextlib
import io
from ace import model, ace

def ace_correlation_matrix(data, n_permutations=0, max_outers=5, verbose=False, test=False):
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    ace.MAX_OUTERS = max_outers  # limite globale des itérations
    n_features = data.shape[1]
    corr_matrix = pd.DataFrame(np.nan, index=data.columns, columns=data.columns)
    pval_matrix = pd.DataFrame(np.nan, index=data.columns, columns=data.columns) 

    for i in range(n_features):
        target_name = data.columns[i]
        y = data[target_name].values
        data_reduced = data.drop(columns=target_name)
        x_names = data_reduced.columns
        x = [data_reduced[col].values for col in x_names]
        try:
            myace = model.Model()
            with contextlib.redirect_stdout(io.StringIO()):
                myace.build_model_from_xy(x, y)
            x_transforms = myace.ace.x_transforms
            for j, name in enumerate(x_names):
                corr = pearsonr(x_transforms[j], y)[0]
                pval = pearsonr(x_transforms[j], y)[1]
                corr_matrix.loc[target_name, name] = corr
                pval_matrix.loc[target_name, name] = pval

        except Exception as e:
            print(f"[Erreur] Variable cible {target_name} : {e}")
    np.fill_diagonal(corr_matrix.values, 1.0)
    np.fill_diagonal(pval_matrix.values, 1.0)
    if test:
        return corr_matrix, pval_matrix
    else:
        return corr_matrix