import argparse
import torch
import math
import h5py
#export HDF5_USE_FILE_LOCKING='FALSE'
from torch import optim
device_number = 1
cuda = torch.cuda.set_device(device_number)

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error as mse
from module import *
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy import linalg as LA
plt.close('all')

def sigmoid(x):
  return (1 / (1 + math.exp(-x)))

sigmoid_v = np.vectorize(sigmoid)

def denoise_data(train_data, alpha, beta, lambda_param, n_epoch, lr):

  Repeat = train_data.shape[0]; Voxels = train_data.shape[2]
  model = DenoisingBraindata(Repeat,  train_data.shape[1], Voxels)
  model.cuda(); model.train()

  loss_func = torch.nn.MSELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr) 
  
  with torch.cuda.device(device_number):
    for epoch in range(0,n_epoch):
      loss = torch.tensor([0],  dtype=torch.float).cuda()
      optimizer.zero_grad()
      for r in range(0,Repeat):
        Robs_r = torch.from_numpy(train_data[r]).float().cuda()
        predicted_Robs = model( torch.tensor([r], dtype=torch.long).cuda(), torch.from_numpy(np.arange(Voxels)).long().cuda())
        
        gamma = model.gamma(torch.tensor([r], dtype=torch.long).cuda())
        x = (alpha-1)*torch.log(model.sigmoid(gamma))
        y = (beta-1)*torch.log(torch.from_numpy(np.ones(gamma.shape[1])).float().cuda() - model.sigmoid(gamma))
        gamma_reg_term = lambda_param*torch.sum(x+y)
        
        error = loss_func(predicted_Robs, Robs_r)  - gamma_reg_term
    
        loss += error
      # Backpropagate
      loss.backward()    
      # Update the parameters
      optimizer.step()
      
  return model

def eval_denoising(model, Robs_test_repeat, train_data, name):
  model.eval()

  Repeat = train_data.shape[0]; Voxels = train_data.shape[2]
  Ridl_pred_tensor    = model.r_ideal_transpose(torch.from_numpy(np.arange(Voxels)).long().cuda())
  Ridl_pred_transpose = Ridl_pred_tensor.detach().cpu().numpy()
  gamma_pred_tensor   = model.gamma(torch.from_numpy(np.arange(Repeat)).long().cuda())
  
  gamma_pred = gamma_pred_tensor.detach().cpu().numpy()  
  Ridl_pred = Ridl_pred_transpose.T

  #Save the learnt Ridl and gamma by model over training dataset 
  Ridl_pred.dump('Ridl_'+ name + '.dat')
  gamma_pred.dump('Gamma_'+ name + '.dat')
  
  #Correlation of Robs of new repeat with Rideal and Robs_mean
  R_obs_train_mean  = np.mean(train_data, axis=0)
  voxcorrs     = np.zeros([Voxels])
  voxcorrs_abs = np.zeros([Voxels])

  for v in range(0,Voxels):
    a = np.corrcoef(R_obs_test_repeat[:,v], Ridl_pred[:,v])[0,1]
    b = np.corrcoef(R_obs_test_repeat[:,v], R_obs_train_mean[:,v])[0,1]
    #voxcorrs_abs[v] = abs(a)-abs(b)
    voxcorrs[v] = a - b
    
  print('The corr difference for',name)
  print("abs %0.4f, orig %0.4f" % (np.mean(voxcorrs_abs),np.mean(voxcorrs)) )
  voxcorrs_abs.dump('corr_'+ name + 'abs.dat')
  voxcorrs.dump('corr_'+ name + '.dat')
  
  return 0

if __name__ == '__main__':

  f = h5py.File('validation_notavg.hf5', 'r') 
  print(f.keys())
  dataset = f['data']
  #datset = f.get('data')
  
  data1 = np.nan_to_num(np.reshape(dataset, (90,10,32*100*100)))
  data2 = data1[:,0,:] # repitition is in middle index

  #print(np.var(data2,axis=1))
  #f = h5py.File('validation_notavg.hf5', 'r')
  #print(f)
  #data = np.load('validation_notavg.hf5')
  #data = hf.get('dataset_name').value

"""
  for sub in ['AA']:

    name = 'fullmatrix_R_'+sub+'.dat.npy'    
    R_obs  = np.load(name)
    R_anchor_pred = np.load(sub+'_pred.npy')
    
    indices = [0,1,2,3,4,5,6,7,8,9] # all data thats why
    lambda_param = 0.0001
    n_epoch =  700
    Voxels = R_obs.shape[2]
    train_data = R_obs[indices,:,0:Voxels]
    R_obs_test_repeat= R_anchor_pred[:,0:Voxels]

    for alpha in [8,20,5,3,50]:                                                                     
      for beta in [0.1,0.3,0.7,1,2]:                                                                        
        name = sub +'_'+'all'+'_'+str(alpha)+ '_'+str(beta)+str(lambda_param)                                         
        model = denoise_data(train_data, alpha, beta, lambda_param, n_epoch, lr=1e-2)
        eval_denoising(model,R_obs_test_repeat, train_data,name)                                                            
"""
"""
    for i in range(10): 
      indices = [0,1,2,3,4,5,6,7,8,9]
      del indices[i] 
      train_data(indices)
      R_obs_test_repeat(i)
      for alpha in [8,20,16,50,5,30,300]:#,50,500]: #np.arange(1,1,1):
        for beta in [0.3, 0.7, 1]: # np.arange(0,1,0.1):
          name = 'SJs_norm_wreg_0.00001_' + str(Repeat)+'_'+str(i)+'_'+str(alpha)+ '_'+str(beta)
          model = denoise_data(R_obs_full,start, Repeat, Voxels, alpha, beta,indices,name,t=0)
          eval_denoising(model, R_obs_full, start, Repeat, Voxels, name,indices)
"""
