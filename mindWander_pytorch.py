import torch
from torch import optim

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error as mse

import math
import matplotlib.pyplot as plt
from numpy import linalg as LA
plt.close('all')

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

# define vectorized sigmoid                                                                                                                
sigmoid_v = np.vectorize(sigmoid)

Rshape = [5, 291, 5000] # repeatition , time and voxels                                                                                   
Repeat = Rshape[0]; Time = Rshape[1]; Voxels = Rshape[2]

#For synthatic data in 3D
from scipy.stats import beta

#generate gamma with beta distribution with a = 5, b = 1
gamma_org = np.array([[beta.rvs(1, 0.5) for i in range(0,Time)] for j in range(0,Repeat)]); print(gamma_org.shape)
#generate Rideal with Gaussian distribution with mean 0 and var 1 
Ridl_org = np.random.normal(0, 1, (Time, Voxels))

#generate synthetic Robs 
#Robs_org = np.transpose(Ridl_org.T * gamma_org)
Robs_3d = []
for i in range(0,Repeat):
    temp = np.transpose(Ridl_org.T*gamma_org[i,:])
    Robs_3d.append(temp) 
Robs_3d = np.array(Robs_3d)
Robs_3d_noisy = []
for i in range(0,Repeat):
    temp = Robs_3d[i]
    Robs_3d_noisy.append(temp + np.random.normal(0,1, (Time, Voxels))) # noise with 0.5 std
R_obs_3d_syn = np.array(Robs_3d_noisy)
print("syn data", R_obs_3d_syn.shape)
#Real data  
R_obs_ten_gamma_1k_voxels  = np.load("R_top_5k_voxels.dat") #R_3d_1k_mat.dat")
R_obs_3d_real = R_obs_ten_gamma_1k_voxels[0:Repeat,0:Time,0:Voxels]
print(R_obs_3d_real.shape)

class DenoisingBraindata(torch.nn.Module):
    def __init__(self, n_repeat, n_time, n_voxels):
        super().__init__()
        self.r_ideal_transpose = torch.nn.Embedding(n_voxels, n_time) # voxels x time for transpose matrix stored in embedding
        self.gamma   = torch.nn.Embedding(n_repeat, n_time) # gamma is of shape repeat x time
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self,repeat,rideal_transpose_indices):
        gamma = self.sigmoid(self.gamma(repeat))
        rideal = self.r_ideal_transpose(rideal_transpose_indices)
        pred =rideal*gamma
        return (pred)

gamma_tr_init = 1*np.random.rand(Rshape[0], Rshape[1]) # repeats x time , init (0,1)                                                      
R_idl_init = 2*np.random.rand(Rshape[1], Rshape[2]) - 1  #time x voxels, init (-1,1), for now taking 120 voxels                            
model = DenoisingBraindata(Repeat, Time, Voxels)
#model.gamma.weight.data.copy_(torch.from_numpy(gamma_tr_init))
#model.r_ideal_transpose.weight.data.copy_(torch.from_numpy(R_idl_init.T))
#print(model.parameters())
loss_func = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2) # learning rate
n_epoch = 600
list_loss = []
model.train()
for epoch in range(0,n_epoch):
    loss = torch.tensor([0],  dtype=torch.float)
    optimizer.zero_grad()
    for r in range(0,Repeat):
        Robs_r = torch.from_numpy(R_obs_3d_real[r].T).float()
        predicted_Robs = model( torch.tensor([r], dtype=torch.long), torch.from_numpy(np.arange(Voxels)).long())
        error = loss_func(predicted_Robs, Robs_r)
        loss += error
    # Backpropagate
    loss.backward()    
    # Update the parameters
    optimizer.step()
    list_loss.append(loss.item())
#print(list_loss)
plt.plot(list_loss)
plt.show()
model.eval()
# For experiments with real Robs 3d - 5 repetitions and 120 voxels
Ridl_pred_tensor =  model.r_ideal_transpose(torch.from_numpy(np.arange(Voxels)).long())
gamma_pred_tensor = model.gamma(torch.from_numpy(np.arange(Repeat)).long())
#gamma_pred_bias = model.gamma_bias(torch.from_numpy(np.arange(Repeat)).long())
Ridl_pred_transpose = Ridl_pred_tensor.detach().numpy()
gamma_pred = gamma_pred_tensor.detach().numpy()  
Ridl_pred = Ridl_pred_transpose.T
"""
for v in range(0,1):#voxels count
    for r in range(0,5):#Repeat count:
        plt.plot(R_obs_3d_real[r][:,v],'k', label='Robs noisy')
        plt.plot(Ridl_pred[:,v],'b',label='Rideal predicted')
        #plt.plot(Ridl_org[:,v],'r',label='Rideal original')
        plt.legend()
        plt.show()

for r in range (0, Repeat):
    plt.plot(sigmoid_v(gamma_pred[r]),label='predicted gamma')
    #plt.plot((gamma_org[r]), label='original gamma')
    plt.legend()
    plt.show()
"""
#correlation between Rideal and Robs
for new_repeat in range(0,10):
    R_obs_new_repeat = R_obs_ten_gamma_1k_voxels[new_repeat,0:Time,0:Voxels]
    R_obs_mean = np.mean(R_obs_3d_real, axis=0) 
    #R_obs_mean = np.mean(R_obs_ten_gamma_1k_voxels[0:Repeat,0:Time,0:Voxels], axis=0)
    voxcorrs = np.zeros([Voxels])
    for v in range(0,Voxels):
        a = np.corrcoef(R_obs_new_repeat[:,v], Ridl_pred[:,v])[0,1]
        b = np.corrcoef(R_obs_new_repeat[:,v], R_obs_mean[:,v])[0,1]
        voxcorrs[v] = abs(a)-abs(b)
    print("Correlation with the repeat",new_repeat, np.mean(voxcorrs))

