import cottoncandy as cc
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt
from numpy import linalg as LA
plt.close('all')
import math
import json
access_key = 'SSE14CR7P0AEZLPC7X0R'
secret_key = 'K0MmeXiXotrGIiTeRwEKizkkhR4qFV8tr8cIXprI'
endpoint_url = 'http://c3-dtn02.corral.tacc.utexas.edu:9002/'
cci = cc.get_interface('story-mri-data', ACCESS_KEY=access_key, SECRET_KEY=secret_key,endpoint_url=endpoint_url)

XFM_file = cci.download_json('subject_xfms') 
print(type(XFM_file))

print(XFM_file)
print(XFM_file['AHfs'])

#with open(XFM_file, 'w') as outfile:
#    json.dump(data, outfile)
#    json.dump(XFM_file, 'xfm_file')

"""
R_obs = cci.download_raw_array('AHfs/wheretheressmoke-10sessions')
print (R_obs.shape)

Robs_oneti = R_obs[:,:,90000:91000]
Robs_oneti.dump("R_3d_1k_mat.dat")

name = ["R_3d_1_10k_mat.dat","R_3d_2_10k_mat.dat","R_3d_3_10k_mat.dat","R_3d_4_10k_mat.dat","R_3d_5_10k_mat.dat","R_3d_6_10k_mat.dat","R_3d_7_10k_mat.dat","R_3d_8_10k_mat.dat","R_3d_9_10k_mat.dat"]
for i in range(0,9):
    Robs_oneti = R_obs[:,:,10000*i:10000*(i+1)]
    Robs_oneti.dump(name[i])
"""
"""
R_obs_onegamma  = np.load("R_matrix.dat")
Rshape = [10, 100, 200] # repeatition , time and voxels 
T = Rshape[1]
Voxels_count = Rshape[2]
np.random.seed(100)
gamma_tr = np.random.rand(Rshape[0], Rshape[1]) # repeats x time                                                            
R_idl = 2*np.random.rand(Rshape[1], Rshape[2]) - 1  #time x voxels, for now taking 1k voxels                                        
gamma_t_one = gamma_tr[1:2][:] # taking gamma for first repeatition of the story for all time slots

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

# define vectorized sigmoid
sigmoid_v = np.vectorize(sigmoid)

#initial guess
guess = np.concatenate((gamma_t_one.T, R_idl), axis=1).transpose()
#bnds = [(0, 1) for _ in range(0,T)] + [(-np.inf, np.inf) for _ in range(0 , T*Voxels_count)]

def error_unbound_gamma(x, R,Time,Voxels):
    x1 = np.reshape(x,(Voxels+1,Time))
    gamma_sig  = sigmoid_v(x1[0,:])
    mean_error = mse(R.T,x1[1:,:]*gamma_sig)  + 0.0005*(LA.norm(x1[1:,:]) + LA.norm(x1[0,:])) 
    return mean_error

res= minimize(error_unbound_gamma,guess,args=(R_obs_onegamma[0:T,0:Voxels_count],T,Voxels_count), method='SLSQP')  

#res_unbounded_gamma_Reg_Ridl= minimize(error_unbound_gamma,guess,args=(R_obs_onegamma[0:T,0:Voxels_count],T,Voxels_count), method='SLSQP')
#, options={'disp': True, 'maxiter':200} )

#results on Error 2
res = res #_unbounded_gamma_Reg_Ridl
print("Result for unbounded sigmoid gamma")
print(res)
pred_x = np.reshape(res.x,(Voxels_count+1,T))
gamma_pred = pred_x[0,:]
gamma_pred.dump("gamma_pred.dat")
print("Robs elements",R_obs_onegamma[0:2,0:2])
print("Rideal elements", pred_x[0:2,0:2])
print("gamma vector is",sigmoid_v(gamma_pred))
plt.plot(sigmoid_v(gamma_pred))
plt.show()





"""
