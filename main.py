import sympy as sp
import numpy as np
from utils import generate_gaussian_term,generate_penalty_term,generate_taylor_expansion,generate_tanh
import matplotlib.pyplot as plt
from functools import reduce
import torch
import torch.nn as nn
import torch.nn.functional as F
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plotting
plt.style.use('seaborn-v0_8')

def g(f,lb):
    return f-lb

def phi(g):
    return torch.exp(g)

def z_rbf(x,centers,sigma):
    diff = x.T[:,None]-centers[None,:]
    diff_sqrd = (diff)**2
    z = diff_sqrd *(- sigma[None,:])
    return z
def z_x_rbf(x,centers,sigma):
    z_x = -(x.T[:,None]-centers[None,:])*2*sigma[None,:]
    return z_x

def z_xx_rbf(x,centers,sigma):
    z_xx = -2*sigma[None,:]
    return z_xx


def rbf_features(x,centers,sigma):
    z = z_rbf(x,centers,sigma)
    return torch.exp(z)

def rbf_features_dh(x,centers,sigma):
    H = rbf_features(x,centers,sigma)
    z_x = z_x_rbf(x,centers,sigma)     
    return H*z_x

def rbf_features_ddh(x,centers,sigma):
    H = rbf_features(x,centers,sigma)
    dH = rbf_features_dh(x,centers,sigma)
    z_x = z_x_rbf(x,centers,sigma)
    z_xx = z_xx_rbf(x,centers,sigma)
    
    return z_xx*H+z_x*dH

def dtanh(z,eta,A):
    return (torch.ones_like(z)-torch.tanh(eta*z/A)**2)

num_hidden = 100
A = 2
eta = 0.0005
x_min = 0
x_max = 100
x = torch.arange(x_min,x_max,0.1).float().unsqueeze(1).T
num_centers = 20

z = 2*((x+x.min())/(x.max()-x.min()))-1

random_W = torch.randn(num_hidden,1).float()
random_b = torch.randn(num_hidden,1).float()

centers = torch.linspace(x_min,x_max,num_hidden).unsqueeze(1).float()
centers = centers.repeat(1,num_centers)
offset_range = 5  # controls spread around base center
offsets = torch.linspace(-offset_range, offset_range, num_centers).unsqueeze(0)  # (1, R)
displaced_centers = centers + offsets

sigma = torch.abs(torch.randn(num_hidden,1)) * 0.01  # broader, safer
sigma = sigma.repeat(1,num_centers)

# H = rbf_features(x,displaced_centers,sigma)
# dH_x = rbf_features_dh(x,displaced_centers,sigma)
# dH_xx = rbf_features_ddh(x,displaced_centers,sigma) 

H = torch.tanh(z.T@random_W.T + random_b.T)

dH_t = (1-H**2)*random_W.T

d2H_t2 = -2*H*dH_t*random_W.T

fig,axs = plt.subplots(2,2,figsize=(12,15),sharex=True,sharey=True)
fig.suptitle("Time Derivatives Constraining Over Randomly Sampled Basis Functions at Different Step Sizes",fontsize=16,fontweight='bold')
for i in range(4):
    A = 5
    eta = 0.001
    
    # Define the vertices of the rhomboid (diamond shape)
    vertices = np.array([
        [0, 3],   # top
        [3, 0],   # right
        [0, -3],  # bottom
        [-3, 0],  # left
    ])

    # Close the polygon by repeating the first point
    vertices = np.vstack([vertices, vertices[0]])

    # Plot
    if i == 0:
        axs[0,0].plot(vertices[:, 0], vertices[:, 1], 'r-', linewidth=2)
        axs[0,0].fill(vertices[:, 0], vertices[:, 1], alpha=0.2, color='yellow', label='Feasible Region')
    elif i == 1:
        axs[0,1].plot(vertices[:, 0], vertices[:, 1], 'r-', linewidth=2)
        axs[0,1].fill(vertices[:, 0], vertices[:, 1], alpha=0.2, color='yellow', label='Feasible Region')
    elif i == 2:
        axs[1,0].plot(vertices[:, 0], vertices[:, 1], 'r-', linewidth=2)
        axs[1,0].fill(vertices[:, 0], vertices[:, 1], alpha=0.2, color='yellow', label='Feasible Region')
    else:
        axs[1,1].plot(vertices[:, 0], vertices[:, 1], 'r-', linewidth=2)
        axs[1,1].fill(vertices[:, 0], vertices[:, 1], alpha=0.2, color='yellow', label='Feasible Region')

    random_beta_x = torch.rand(num_hidden,1)*3-1
    random_beta_y = torch.rand(num_hidden,1)*3-1
    a = torch.tensor([3],dtype=torch.float)
    
    x_0 = H@random_beta_x
    dx_0_dt = dH_t@random_beta_x
    d2x_0_d2t = d2H_t2@random_beta_x
    
    y_0 = H@random_beta_y
    dy_0_dt = dH_t@random_beta_y
    d2y_0_d2t = d2H_t2@random_beta_y

    g_1_1 = g(dx_0_dt-dy_0_dt,a)
    dg_1_1_dt = d2x_0_d2t-d2y_0_d2t
    phi_1_1 = torch.where(torch.exp(g_1_1)>3,torch.exp(g_1_1),torch.zeros_like(torch.exp(g_1_1))).sum()
    
    dg_1_1_dbeta_y = -dH_t
    dg_1_1_dbeta_y_dt = -d2H_t2
    dg_1_1_dbeta_x = dH_t
    dg_1_1_dbeta_x_dt = d2H_t2
    
    dphi_1_1_dbeta_y = ((dg_1_1_dbeta_y.T.sum(dim=1))*phi_1_1).unsqueeze(1)
    dphi_1_1_dbeta_x = ((dg_1_1_dbeta_x.T.sum(dim=1))*phi_1_1).unsqueeze(1)
    d2phi_1_1_d2beta_x = (torch.matmul(dg_1_1_dbeta_x.T,dg_1_1_dbeta_x))*phi_1_1
    d2phi_1_1_dbeta_x_dbeta_y = dg_1_1_dbeta_x.T@torch.diag(g_1_1.squeeze(1))@dg_1_1_dbeta_y
    d2phi_1_1_d2beta_y = (torch.matmul(dg_1_1_dbeta_y.T,dg_1_1_dbeta_y))*phi_1_1
    d2phi_1_1_dbeta_y_dbeta_x = dg_1_1_dbeta_y.T@torch.diag(g_1_1.squeeze(1))@dg_1_1_dbeta_x
    # d2phi_1_1_dbeta_y_dbeta_x =

    g_2_1 = g(dx_0_dt+dy_0_dt,a)
    dg_2_1_dt = d2x_0_d2t+d2y_0_d2t
    phi_2_1 = torch.where(torch.exp(g_2_1)>3,torch.exp(g_2_1),torch.zeros_like(torch.exp(g_2_1))).sum()
    
    dg_2_1_dbeta_x = dH_t
    dg_2_1_dbeta_x_dt = d2H_t2
    dg_2_1_dbeta_y = dH_t
    dg_2_1_dbeta_y_dt = d2H_t2

    dphi_2_1_dbeta_y = ((dg_2_1_dbeta_y.T.sum(dim=1))*phi_2_1).unsqueeze(1)
    dphi_2_1_dbeta_x = ((dg_2_1_dbeta_x.T.sum(dim=1))*phi_2_1).unsqueeze(1)
    d2phi_2_1_d2beta_x = (torch.matmul(dg_2_1_dbeta_x.T,dg_2_1_dbeta_x))*phi_2_1
    d2phi_2_1_dbeta_x_dbeta_y = dg_2_1_dbeta_x.T@torch.diag(g_1_1.squeeze(1))@dg_2_1_dbeta_y
    d2phi_2_1_d2beta_y = (torch.matmul(dg_2_1_dbeta_y.T,dg_2_1_dbeta_y))*phi_2_1
    d2phi_2_1_dbeta_y_dbeta_x = dg_2_1_dbeta_y.T@torch.diag(g_1_1.squeeze(1))@dg_2_1_dbeta_x


    g_3_1 = g(-dx_0_dt+dy_0_dt,a)
    dg_3_1_dt = -d2x_0_d2t+d2y_0_d2t
    phi_3_1 = torch.where(torch.exp(g_3_1)>3,torch.exp(g_3_1),torch.zeros_like(torch.exp(g_3_1))).sum()
    
    dg_3_1_dbeta_x = -dH_t
    dg_3_1_dbeta_x_dt = -d2H_t2
    dg_3_1_dbeta_y = dH_t
    dg_3_1_dbeta_y_dt = d2H_t2

    dphi_3_1_dbeta_y = ((dg_3_1_dbeta_y.T.sum(dim=1))*phi_3_1).unsqueeze(1)
    dphi_3_1_dbeta_x = ((dg_3_1_dbeta_x.T.sum(dim=1))*phi_3_1).unsqueeze(1)
    d2phi_3_1_d2beta_x = (torch.matmul(dg_3_1_dbeta_x.T,dg_3_1_dbeta_x))*phi_3_1
    d2phi_3_1_dbeta_x_dbeta_y = dg_3_1_dbeta_x.T@torch.diag(g_1_1.squeeze(1))@dg_3_1_dbeta_y
    d2phi_3_1_d2beta_y = (torch.matmul(dg_3_1_dbeta_y.T,dg_3_1_dbeta_y))*phi_3_1
    d2phi_3_1_dbeta_y_dbeta_x = dg_3_1_dbeta_y.T@torch.diag(g_1_1.squeeze(1))@dg_3_1_dbeta_x


    g_4_1 = g(-dx_0_dt-dy_0_dt,a)
    dg_4_1_dt = -d2x_0_d2t-d2y_0_d2t
    phi_4_1 = torch.where(torch.exp(g_4_1)>3,torch.exp(g_4_1),torch.zeros_like(torch.exp(g_4_1))).sum()
    
    dg_4_1_dbeta_x = -dH_t
    dg_4_1_dbeta_x_dt = -d2H_t2
    dg_4_1_dbeta_y = -dH_t
    dg_4_1_dbeta_y_dt = -d2H_t2

    dphi_4_1_dbeta_y = ((dg_4_1_dbeta_y.T.sum(dim=1))*phi_4_1).unsqueeze(1)
    dphi_4_1_dbeta_x = ((dg_4_1_dbeta_x.T.sum(dim=1))*phi_4_1).unsqueeze(1)
    d2phi_4_1_d2beta_x = (torch.matmul(dg_4_1_dbeta_x.T,dg_4_1_dbeta_x))*phi_4_1
    d2phi_4_1_dbeta_x_dbeta_y = dg_4_1_dbeta_x.T@torch.diag(g_1_1.squeeze(1))@dg_4_1_dbeta_y
    d2phi_4_1_d2beta_y = (torch.matmul(dg_4_1_dbeta_y.T,dg_4_1_dbeta_y))*phi_4_1
    d2phi_4_1_dbeta_y_dbeta_x = dg_4_1_dbeta_y.T@torch.diag(g_1_1.squeeze(1))@dg_4_1_dbeta_x

    for j in range(3):
        
        tanhdphi_1_1_dbeta_x = torch.tanh(eta*dphi_1_1_dbeta_x/A)
        dtanhdphi_1_1_dbeta_x = dtanh(dphi_1_1_dbeta_x,eta,A)
        dphi_1_1_dbeta_x_dt = (dg_1_1_dbeta_x_dt+dg_1_1_dbeta_x*(dg_1_1_dt))*torch.exp(g_1_1)
        # ddphi_1_1_dbeta_x_dt_dbeta_x = 
        dtanhdphi_1_1_dbeta_x_dt = eta*dphi_1_1_dbeta_x_dt.T/A*dtanhdphi_1_1_dbeta_x*(eta/A)
        # ddtanhdphi_1_1_dbeta_x_dt_dbeta_x = eta*ddphi_1_1_dbeta_x_dt_dbeta_x.T/A*dtanhdphi_1_1_dbeta_x*(eta/A)
        dtanhdphi_1_1_dbeta_x_dbeta_x = torch.diag(dtanhdphi_1_1_dbeta_x.squeeze(1))*(eta/A)*d2phi_1_1_d2beta_x*(eta/A)

        tanhdphi_1_1_dbeta_y = torch.tanh(eta*dphi_1_1_dbeta_y/A)
        dtanhdphi_1_1_dbeta_y = dtanh(dphi_1_1_dbeta_y,eta,A)
        dphi_1_1_dbeta_y_dt = (dg_1_1_dbeta_y_dt+dg_1_1_dbeta_y*(dg_1_1_dt))*torch.exp(g_1_1)
        dtanhdphi_1_1_dbeta_y_dt = eta*dphi_1_1_dbeta_y_dt.T/A*dtanhdphi_1_1_dbeta_y*(eta/A)
        dtanhdphi_1_1_dbeta_y_dbeta_x = torch.diag(dtanhdphi_1_1_dbeta_y.squeeze(1))*(eta/A)*d2phi_1_1_dbeta_y_dbeta_x*(eta/A)

        tanhdphi_2_1_dbeta_x = torch.tanh(eta*dphi_2_1_dbeta_x/A)
        dtanhdphi_2_1_dbeta_x = dtanh(dphi_2_1_dbeta_x,eta,A)
        dphi_2_1_dbeta_x_dt = (dg_2_1_dbeta_x_dt+dg_2_1_dbeta_x*(dg_2_1_dt))*torch.exp(g_2_1)
        dtanhdphi_2_1_dbeta_x_dt = eta*dphi_2_1_dbeta_x_dt.T/A*dtanhdphi_2_1_dbeta_x*(eta/A)
        dtanhdphi_2_1_dbeta_x_dbeta_x = torch.diag(dtanhdphi_2_1_dbeta_x.squeeze(1))*(eta/A)*d2phi_2_1_d2beta_x*(eta/A)
        # dupdate_2_1_beta_x_dbeta_x =

        tanhdphi_2_1_dbeta_y = torch.tanh(eta*dphi_2_1_dbeta_y/A)
        dtanhdphi_2_1_dbeta_y = dtanh(dphi_2_1_dbeta_y,eta,A)
        dphi_2_1_dbeta_y_dt = (dg_2_1_dbeta_y_dt+dg_2_1_dbeta_y*(dg_2_1_dt))*torch.exp(g_2_1)
        dtanhdphi_2_1_dbeta_y_dt = eta*dphi_2_1_dbeta_y_dt.T/A*dtanhdphi_2_1_dbeta_y*(eta/A)
        dtanhdphi_2_1_dbeta_y_dbeta_x = torch.diag(dtanhdphi_2_1_dbeta_y.squeeze(1))*(eta/A)*d2phi_2_1_dbeta_y_dbeta_x*(eta/A)
        # dupdate_2_1_beta_y_dbeta_y =

        tanhdphi_3_1_dbeta_x = torch.tanh(eta*dphi_3_1_dbeta_x/A)
        dtanhdphi_3_1_dbeta_x = dtanh(dphi_3_1_dbeta_x,eta,A) 
        dphi_3_1_dbeta_x_dt = (dg_3_1_dbeta_x_dt+dg_3_1_dbeta_x*(dg_3_1_dt))*torch.exp(g_3_1)
        dtanhdphi_3_1_dbeta_x_dt =  eta*dphi_3_1_dbeta_x_dt.T/A*dtanhdphi_3_1_dbeta_x*(eta/A)
        dtanhdphi_3_1_dbeta_x_dbeta_x = torch.diag(dtanhdphi_3_1_dbeta_x.squeeze(1))*(eta/A)*d2phi_3_1_d2beta_x*(eta/A)
        

        tanhdphi_3_1_dbeta_y = torch.tanh(eta*dphi_3_1_dbeta_y/A)
        dtanhdphi_3_1_dbeta_y = dtanh(dphi_3_1_dbeta_y,eta,A)
        dphi_3_1_dbeta_y_dt = (dg_3_1_dbeta_y_dt+dg_3_1_dbeta_y*(dg_3_1_dt))*torch.exp(g_3_1)
        dtanhdphi_3_1_dbeta_y_dt = eta*dphi_3_1_dbeta_y_dt.T/A*dtanhdphi_3_1_dbeta_y*(eta/A)
        dtanhdphi_3_1_dbeta_y_dbeta_x = torch.diag(dtanhdphi_3_1_dbeta_y.squeeze(1))*(eta/A)*d2phi_3_1_dbeta_y_dbeta_x*(eta/A)
        
        tanhdphi_4_1_dbeta_x = torch.tanh(eta*dphi_4_1_dbeta_x/A)
        dtanhdphi_4_1_dbeta_x = dtanh(dphi_4_1_dbeta_x,eta,A) 
        dphi_4_1_dbeta_x_dt = (dg_4_1_dbeta_x_dt+dg_4_1_dbeta_x*(dg_4_1_dt))*torch.exp(g_4_1)
        dtanhdphi_4_1_dbeta_x_dt = eta*dphi_4_1_dbeta_x_dt.T/A*dtanhdphi_4_1_dbeta_x*(eta/A)
        dtanhdphi_4_1_dbeta_x_dbeta_x = torch.diag(dtanhdphi_4_1_dbeta_x.squeeze(1))*(eta/A)*d2phi_4_1_d2beta_x*(eta/A)
        # dupdate_4_1_beta_x_dbeta_x =

        tanhdphi_4_1_dbeta_y = torch.tanh(eta*dphi_4_1_dbeta_y/A)
        dtanhdphi_4_1_dbeta_y = dtanh(dphi_4_1_dbeta_y,eta,A) 
        dphi_4_1_dbeta_y_dt = (dg_4_1_dbeta_y_dt+dg_4_1_dbeta_y*(dg_4_1_dt))*torch.exp(g_4_1)
        dtanhdphi_4_1_dbeta_y_dt =  eta*dphi_4_1_dbeta_y_dt.T/A*dtanhdphi_4_1_dbeta_y*(eta/A)
        dtanhdphi_4_1_dbeta_y_dbeta_x = torch.diag(dtanhdphi_4_1_dbeta_y.squeeze(1))*(eta/A)*d2phi_4_1_dbeta_y_dbeta_x*(eta/A)
        # dupdate_4_1_beta_y_dbeta_y =

        tanhdphi_1_dbeta_x = tanhdphi_1_1_dbeta_x+tanhdphi_2_1_dbeta_x+tanhdphi_3_1_dbeta_x+tanhdphi_4_1_dbeta_x
        dtanhdphi_1_dbeta_x_dbeta_x = dtanhdphi_1_1_dbeta_x_dbeta_x+dtanhdphi_2_1_dbeta_x_dbeta_x+dtanhdphi_3_1_dbeta_x_dbeta_x+dtanhdphi_4_1_dbeta_x_dbeta_x
        print(tanhdphi_1_dbeta_x.shape)
        print(dtanhdphi_1_dbeta_x_dbeta_x.shape)
        input('hipi')
        # dupdate_1_beta_x_dbeta_x = dupdate_1_1_beta_x_dbeta_x+dupdate_2_1_beta_x_dbeta_x+dupdate_3_1_beta_x_dbeta_x+dupdate_4_1_beta_x_dbeta_x
        
        tanhdphi_1_dbeta_y = tanhdphi_1_1_dbeta_y+tanhdphi_2_1_dbeta_y+tanhdphi_3_1_dbeta_y+tanhdphi_4_1_dbeta_y
        # dupdate_1_beta_y_dbeta_y = dupdate_1_1_beta_y_dbeta_y+dupdate_2_1_beta_y_dbeta_y+dupdate_3_1_beta_y_dbeta_y+dupdate_4_1_beta_y_dbeta_y        
        
        dtanhdphi_1_dbeta_x_dt = dtanhdphi_1_1_dbeta_x_dt+dtanhdphi_2_1_dbeta_x_dt+dtanhdphi_3_1_dbeta_x_dt+dtanhdphi_4_1_dbeta_x_dt
        # ddtanhdphi_1_dbeta_x_dt_dbeta_x = ddtanhdphi_1_1_dbeta_x_dt_dbeta_x+ddtanhdphi_2_1_dbeta_x_dt_dbeta_x+ddtanhdphi_3_1_dbeta_x_dt_dbeta_x+ddtanhdphi_4_1_dbeta_x_dt_dbeta_x


        dtanhdphi_1_dbeta_y_dt = dtanhdphi_1_1_dbeta_y_dt+dtanhdphi_2_1_dbeta_y_dt+dtanhdphi_3_1_dbeta_y_dt+dtanhdphi_4_1_dbeta_y_dt
        # ddtanhdphi_1_dbeta_y_dt_dbeta_y = ddtanhdphi_1_1_dbeta_y_dt_dbeta_y+ddtanhdphi_2_1_dbeta_y_dt_dbeta_y+ddtanhdphi_3_1_dbeta_y_dt_dbeta_y+ddtanhdphi_4_1_dbeta_y_dt_dbeta_y
        
        x_1 = H@(random_beta_x-eta*A*tanhdphi_1_dbeta_x)
        dx_1_dt = dH_t@(random_beta_x-eta*A*tanhdphi_1_dbeta_x)+H@(-eta*A*dtanhdphi_1_dbeta_x_dt/A*eta).diag().unsqueeze(1)
        # dx_1_dt_dbeta_x = dH_t@(torch.ones_like(dtanhdphi_1_dbeta_x_dbeta_x)-eta*A*dtanhdphi_1_dbeta_x_dbeta_x)+\
        #                     H@(-eta*A*ddtanhdphi_1_dbeta_x_dt_dbeta_x/A*eta).diag().unsqueeze(1)

        y_1 = H@(random_beta_y-eta*A*tanhdphi_1_dbeta_y)
        dy_1_dt = dH_t@(random_beta_y-eta*A*tanhdphi_1_dbeta_y)+H@(-eta*A*dtanhdphi_1_dbeta_y_dt/A*eta).diag().unsqueeze(1)
        # dy_1_dt_dbeta_y =

        g_1_2 = g(dx_1_dt-dy_1_dt,a)
        phi_1_2 = torch.where(torch.exp(g_1_2)>3,torch.exp(g_1_2),torch.zeros_like(torch.exp(g_1_2))).sum()
        
        # dg_1_2_dbeta_x = dx_1_dt_dbeta_x-dy_1_dt_dbeta_x
        # dphi_1_2_dbeta_x = ((dg_1_2_dbeta_x.T.sum(dim=1))*phi_1_2).unsqueeze(1)

        # dx_1_dt_dbeta_x = dH_t@(torch.ones_like(random_beta_x)-eta*A*update_1_beta_x)
        # dy_1_dt_dbeta_x =
        # dg_1_2_dbeta_x = dx_1_dt_dbeta_x - dy_1_dt_dbeta_x
        g_2_2 = g(dx_1_dt+dy_1_dt,a)
        
        g_3_2 = g(-dx_1_dt+dy_1_dt,a)
        g_4_2 = g(-dx_1_dt-dy_1_dt,a)
        
        phi_2_2 = torch.where(torch.exp(g_2_2)>3,torch.exp(g_2_2),torch.zeros_like(torch.exp(g_2_2)))
        phi_3_2 = torch.where(torch.exp(g_3_2)>3,torch.exp(g_3_2),torch.zeros_like(torch.exp(g_3_2)))
        phi_4_2 = torch.where(torch.exp(g_4_2)>3,torch.exp(g_4_2),torch.zeros_like(torch.exp(g_4_2)))
        
        # dphi_1_2_dbeta_x = (dg_1_2_dbeta_x_dt.T@phi_1_2+dg_1_2_dbeta_x.T@(dg_1_2_dt*phi_1_2))
        
        # dphi_2_2_beta_x_dt = (dg_2_2_dbeta_x_dt.T@phi_2_2+dg_2_2_dbeta_x.T@(dg_2_2_dt*phi_1_2))
        # dphi_3_2_beta_x_dt = (dg_3_2_dbeta_x_dt.T@phi_3_2+dg_3_2_dbeta_x.T@(dg_3_2_dt*phi_1_2))
        # dphi_4_2_beta_x_dt = (dg_4_2_dbeta_x_dt.T@phi_4_2+dg_4_2_dbeta_x.T@(dg_4_2_dt*phi_1_2))

        # d2x_1_d2t = torch.diff(dy_1_dt,dim=0)/torch.diff(x.T,dim=0)
        # d2y_1_d2t = torch.diff(dx_1_dt,dim=0)/torch.diff(x.T,dim=0)
        
        # axs[1].plot(x.T.detach().numpy(),dx_0_dt.detach().numpy(),label='dx_0_dt'+str(i),color='g',alpha=0.5+0.1*i)
        # axs[1].plot(x.T.detach().numpy(),dx_1_dt.detach().numpy(),label='dx_1_dt'+str(i),color='b',alpha=0.5+0.1*i)
        # axs[1].legend()
        # axs[2].plot(x.T.detach().numpy(),dy_0_dt.detach().numpy(),label='dy_0_dt'+str(i),color='g',alpha=0.5+0.1*i)
        # axs[2].plot(x.T.detach().numpy(),dy_1_dt.detach().numpy(),label='dy_1_dt'+str(i),color='b',alpha=0.5+0.1*i)
        # axs[2].legend()
        # axs[3].plot(x.T.detach().numpy(),(torch.exp(g_1_1)+torch.exp(g_2_1)+torch.exp(g_3_1)+torch.exp(g_4_1)).detach().numpy(),color='g',label='penalty_0'+str(i))
        # axs[3].plot(x.T.detach().numpy(),(phi_1_2+phi_2_2+phi_3_2+phi_4_2).detach().numpy(),label='penalty_1'+str(i),color='b',alpha=0.5+0.1*i)
        # axs[3].legend()
        if i == 0:
            axs[0,0].plot(dx_0_dt.detach().numpy(),dy_0_dt.detach().numpy(),label='dx_0_dt vs dy_0_dt'+str(i),color='g',alpha=0.5+0.1*j)
            axs[0,0].plot(dx_1_dt.detach().numpy(),dy_1_dt.detach().numpy(),label='dx_1_dt vs dy_1_dt'+str(i),color='b',alpha=0.5+0.1*j)
            axs[0,0].set_xlabel(r"$\frac{dx}{dt}$",fontsize=14)
            axs[0,0].set_ylabel(r"$\frac{dy}{dt}$",fontsize=14)
        elif i == 1:
            axs[0,1].plot(dx_0_dt.detach().numpy(),dy_0_dt.detach().numpy(),label='dx_0_dt vs dy_0_dt'+str(i),color='g',alpha=0.5+0.1*j)
            axs[0,1].plot(dx_1_dt.detach().numpy(),dy_1_dt.detach().numpy(),label='dx_1_dt vs dy_1_dt'+str(i),color='b',alpha=0.5+0.1*j)
            axs[0,1].set_xlabel(r"$\frac{dx}{dt}$",fontsize=14)
            axs[0,1].set_ylabel(r"$\frac{dy}{dt}$",fontsize=14)
        elif i == 2:
            axs[1,0].plot(dx_0_dt.detach().numpy(),dy_0_dt.detach().numpy(),label='dx_0_dt vs dy_0_dt'+str(i),color='g',alpha=0.5+0.1*j)
            axs[1,0].plot(dx_1_dt.detach().numpy(),dy_1_dt.detach().numpy(),label='dx_1_dt vs dy_1_dt'+str(i),color='b',alpha=0.5+0.1*j)
            axs[1,0].set_xlabel(r"$\frac{dx}{dt}$",fontsize=14)
            axs[1,0].set_ylabel(r"$\frac{dy}{dt}$",fontsize=14)
        else:
            if j == 2:
                axs[1,1].plot(dx_0_dt.detach().numpy(),dy_0_dt.detach().numpy(),label='dx_0_dt vs dy_0_dt',color='g',alpha=0.5+0.1*j)
                axs[1,1].plot(dx_1_dt.detach().numpy(),dy_1_dt.detach().numpy(),label='dx_1_dt vs dy_1_dt',color='b',alpha=0.5+0.1*j)
            else:
                axs[1,1].plot(dx_0_dt.detach().numpy(),dy_0_dt.detach().numpy(),color='g',alpha=0.5+0.1*j)
                axs[1,1].plot(dx_1_dt.detach().numpy(),dy_1_dt.detach().numpy(),color='b',alpha=0.5+0.1*j)
            axs[1,1].set_xlabel(r"$\frac{dx}{dt}$",fontsize=14)
            axs[1,1].set_ylabel(r"$\frac{dy}{dt}$",fontsize=14)
            axs[1,1].legend()
        eta*=2

fig.savefig('phase_over_iterations.png',dpi=300,bbox_inches='tight')
    # # axs[2].set_xlabel('x')
    # print(np.trapz(y=(torch.exp(dy_x_1-a)+torch.exp(-dy_x_1-a)).squeeze(1).detach().numpy(),x=x.T.squeeze(1).detach().numpy())-np.trapz(y=(torch.exp(dy_x_0-a)+torch.exp(-dy_x_0-a)).squeeze(1).detach().numpy(),x=x.T.squeeze(1).detach().numpy()))
    # input('hipi')
    
    # fig,axs = plt.subplots(3,1,sharex=True)
    # axs[0].plot(x.T.detach().numpy(),dy_x_0.detach().numpy(),label='dy_x_0')
    # axs[0].plot(x.T.detach().numpy(),dy_x_1.detach().numpy(),label='dy_x_1')
    # # axs[0].plot(x.T.detach().numpy(),dy_x_1_relu.detach().numpy(),label='dy_x_1_relu')
    # axs[0].legend()
    # axs[1].plot(x.T.detach().numpy(),torch.exp(dy_x_0-a).detach().numpy(),label='exp(dy_x_0-3)')
    # axs[1].plot(x.T.detach().numpy(),torch.exp(dy_x_1-a).detach().numpy(),label='exp(dy_x_1-3)')
    # axs[2].plot(x.T.detach().numpy(),torch.exp(-dy_x_0-a).detach().numpy(),label='exp(-dy_x_0-3)')
    # axs[2].plot(x.T.detach().numpy(),torch.exp(-dy_x_1-a).detach().numpy(),label='exp(-dy_x_1-3)')
    
    # axs[2].set_xlabel('x')
    # plt.show()


    # i = torch.arange(num_hidden).unsqueeze(1)
    # j = torch.arange(num_hidden).unsqueeze(0)
    # bandwidth = 1.5

    # mask = torch.exp(-((i - j)**2) / (2 * bandwidth**2))  # (10, 10)
    # B = mask.unsqueeze(0) * torch.randn(1, num_hidden, num_hidden)*10

    # random_beta = B
    
    # a = torch.tensor([3],dtype=torch.float)
    
    # y_0 = torch.einsum('ijk,ijk->i',H,B).unsqueeze(1)
    # dy_x_0 = torch.einsum('ijk,ijk->i',dH_x,B).unsqueeze(1)
    # dy_xx_0 = torch.einsum('ijk,ijk->i',dH_xx,B).unsqueeze(1)
    # exp_y_0 = torch.exp(dy_x_0-a)+torch.exp(-dy_x_0-a)

    # g_lb_0 = g(dy_x_0,a)
    # phi_lb_0 = torch.where(torch.exp(g_lb_0)>1,torch.exp(g_lb_0),torch.zeros_like(torch.exp(g_lb_0)))
    # dg_0 = a
    
    # update_lb_0 = (dH_x.T)@phi_lb_0
    
    # g_ub_0 = g(-dy_x_0,a)
    # phi_ub_0 = torch.where(torch.exp(g_ub_0)>1,torch.exp(g_ub_0),torch.zeros_like(torch.exp(g_ub_0)))
    # dg_0 = a
    # update_ub_0 = (dH_x.T)@phi_ub_0
    
    # y_1 = torch.einsum('ijk,ijk->i',H,B-eta*A*torch.tanh(eta*update_ub_0.T/A)+eta*A*torch.tanh(eta*update_lb_0.T/A)).unsqueeze(1)
    
    # # dy_x_1_relu = torch.einsum('ijk,ijk->i',dH_x,B-eta*A*torch.tanh(eta*update_ub_0.T/A)-eta*A*torch.tanh(eta*update_lb_0.T/A)).unsqueeze(1)\
    # # -torch.relu(-(torch.einsum('ijk,ijk->i',H,-eta*A*(torch.ones_like(update_ub_0.T)-torch.tanh(eta*update_ub_0.T/A)**2)).unsqueeze(1)*\
    # #                ((eta/A)*(dy_xx_0*phi_ub_0)*(dy_x_0*dy_x_0))))\
    # # +torch.relu((torch.einsum('ijk,ijk->i',H,+eta*A*(torch.ones_like(update_lb_0.T)-torch.tanh(eta*update_lb_0.T/A)**2)).unsqueeze(1)*\
    # #                ((eta/A)*(dy_xx_0*phi_lb_0)*(dy_x_0*dy_x_0))))
    
    # dy_x_1 = torch.einsum('ijk,ijk->i',dH_x,B-eta*A*torch.tanh(eta*update_ub_0.T/A)-eta*A*torch.tanh(eta*update_lb_0.T/A)).unsqueeze(1)\
    # +torch.einsum('ijk,ijk->i',H,-eta*A*(torch.ones_like(update_ub_0.T)-torch.tanh(eta*update_ub_0.T/A)**2)).unsqueeze(1)*\
    #                ((eta/A)*(dy_xx_0*phi_ub_0)*(dy_x_0*dy_x_0))\
    # +torch.einsum('ijk,ijk->i',H,+eta*A*(torch.ones_like(update_lb_0.T)-torch.tanh(eta*update_lb_0.T/A)**2)).unsqueeze(1)*\
    #                ((eta/A)*(dy_xx_0*phi_lb_0)*(dy_x_0*dy_x_0))
    

    # # fig,axs = plt.subplots(3,1,sharex=True)
    # # axs[0].plot(x.T.detach().numpy(),y_0.detach().numpy(),label='y_0')
    # # axs[0].plot(x.T.detach().numpy(),y_1.detach().numpy(),label='y_1')
    # # axs[0].legend()
    # # axs[1].plot(x.T.detach().numpy(),torch.exp(dy_x_0-a).detach().numpy())
    # # axs[2].plot(x.T.detach().numpy(),torch.exp(-dy_x_0-a).detach().numpy())
    # # plt.show()
    # # axs[2].set_xlabel('x')
    # print(np.trapz(y=(torch.exp(dy_x_1-a)+torch.exp(-dy_x_1-a)).squeeze(1).detach().numpy(),x=x.T.squeeze(1).detach().numpy())-np.trapz(y=(torch.exp(dy_x_0-a)+torch.exp(-dy_x_0-a)).squeeze(1).detach().numpy(),x=x.T.squeeze(1).detach().numpy()))
    # input('hipi')
    
    # fig,axs = plt.subplots(3,1,sharex=True)
    # axs[0].plot(x.T.detach().numpy(),dy_x_0.detach().numpy(),label='dy_x_0')
    # axs[0].plot(x.T.detach().numpy(),dy_x_1.detach().numpy(),label='dy_x_1')
    # # axs[0].plot(x.T.detach().numpy(),dy_x_1_relu.detach().numpy(),label='dy_x_1_relu')
    # axs[0].legend()
    # axs[1].plot(x.T.detach().numpy(),torch.exp(dy_x_0-a).detach().numpy(),label='exp(dy_x_0-3)')
    # axs[1].plot(x.T.detach().numpy(),torch.exp(dy_x_1-a).detach().numpy(),label='exp(dy_x_1-3)')
    # axs[2].plot(x.T.detach().numpy(),torch.exp(-dy_x_0-a).detach().numpy(),label='exp(-dy_x_0-3)')
    # axs[2].plot(x.T.detach().numpy(),torch.exp(-dy_x_1-a).detach().numpy(),label='exp(-dy_x_1-3)')
    
    # axs[2].set_xlabel('x')
    # plt.show()

    # correction_1 = torch.einsum('ijk,ijk->i',H_rbf,eta*A*torch.tanh(eta*update_ub_0.T/A)-eta*A*torch.tanh(eta*update_lb_0.T/A)).unsqueeze(1)
    # exp_y_1 = torch.exp(y_1-a)+torch.exp(-y_1-a)

    # g_ub_1 = g(y_1,a)
    # phi_ub_1 = torch.where(torch.exp(g_ub_1)>1,torch.exp(g_ub_1),torch.zeros_like(torch.exp(g_ub_1)))

    # dg_1 = a
    # # dy_1 = torch.einsum('bkl,ijkl-> bij',H,(torch.ones_like(random_beta)[None,:]-(((torch.ones_like(random_beta)-torch.tanh(eta*update_ub_0.T/A))**2)[None,:])\
    # #              *torch.einsum('bij,bkl->ijkl',(dy_0*phi_ub_0[:,None]),dy_0)+(((torch.ones_like(random_beta)-torch.tanh(eta*update_lb_0.T/A))**2)[None,:])\
    # #              *torch.einsum('bij,bkl->ijkl',(dy_0*phi_lb_0[:,None]),dy_0)))
    # dy_1 = torch.einsum('bkl,ijkl-> bij',H,(torch.ones_like(random_beta)[None,:]-(((torch.ones_like(random_beta)-torch.tanh(eta*update_ub_0.T/A))**2)[None,:])\
    #              *torch.einsum('bij,bkl->ijkl',(dy_0*phi_ub_0[:,None]),dy_0)+(((torch.ones_like(random_beta)-torch.tanh(eta*update_lb_0.T/A))**2)[None,:])\
    #              *torch.einsum('bij,bkl->ijkl',(dy_0*phi_lb_0[:,None]),dy_0)))
    
    # update_ub_1 = ((dy_1.T)@phi_ub_1)
   
    # g_lb_1 = g(-y_1,a)
    # phi_lb_1 = torch.where(torch.exp(g_lb_1)>1,torch.exp(g_lb_1),torch.zeros_like(torch.exp(g_lb_1)))

    # dg_1 = a    
    # update_lb_1 = ((dy_1.T)@phi_lb_1)
    

    # y_2 = torch.einsum('ijk,ijk->i',H_rbf,B+A*eta*torch.tanh(eta*update_ub_0.T/A)+A*eta*torch.tanh(eta*update_ub_1.T/A)\
    #                    -A*eta*torch.tanh(eta*update_lb_0.T/A)-A*eta*torch.tanh(eta*update_lb_1.T/A)).unsqueeze(1)
    
    # # correction_2 = torch.einsum('ijk,ijk->i',H_rbf,-eta*A*torch.tanh(eta*update_1.T/A)).unsqueeze(1)
    # exp_y_2= torch.exp(y_2-a)+torch.exp(-y_2-a)
    # # y_2 = H.T@(random_beta-A*torch.tanh(eta*update_0/A)-A*torch.tanh(eta*update_1/A))

    # fig,axs = plt.subplots(2,1,figsize=(15,6))
    # axs[0].plot(x.T.detach().numpy(),y_0.detach().numpy(),label='y_0')
    # axs[0].plot(x.T.detach().numpy(),y_1.detach().numpy(),label='y_1')
    # axs[0].plot(x.T.detach().numpy(),y_2.detach().numpy(),label='y_2')
    # axs[0].plot([x.min().item(),x.max().item()],[3,3],linestyle='--',color='black')
    # axs[0].plot([x.min().item(),x.max().item()],[-3,-3],linestyle='--',color='black')
    # axs[0].legend()
    # max_exp_1 = torch.max(exp_y_1).item()
    # max_exp_0 = torch.max(exp_y_0).item()
    # max_exp_2 = torch.max(exp_y_2).item()
    
    # max_exp = max([max_exp_0,max_exp_1,max_exp_2])
    
    # axs[1].plot(x.T.detach().numpy(),(exp_y_0/max_exp).detach().numpy(),label='Exponential Loss_0')
    # axs[1].plot(x.T.detach().numpy(),(exp_y_1/max_exp).detach().numpy(),label='Exponential Loss_1')
    # axs[1].plot(x.T.detach().numpy(),(exp_y_2/max_exp).detach().numpy(),label='Exponential Loss_2')
    # axs[1].legend()
    # axs[1].set_ylim(ymin=0)
    # # axs[1].set_yscale('log')
    
    # plt.show()



    # random_beta = torch.randn(num_hidden,1).float()
    # H = torch.tanh(random_W@z + random_b)
    # dH_x = ((1-torch.tanh(random_W@z + random_b))**2)*random_W
    # dH_beta =  H
    # a = torch.tensor([3],dtype=torch.float)
    
    # y_0 = H.T@random_beta

    # g_0 = g(y_0,a)
    # phi_0 = torch.where(torch.exp(g_0)>2,torch.exp(g_0),torch.zeros_like(torch.exp(g_0)))
    # dg_0 = a
    # dy_0 = dH_beta 
    # update_0 = (dy_0)@phi_0

    # y_1 = H.T@(random_beta-A*torch.tanh(eta*update_0/A))

    # g_1 = g(y_1,a)
    # phi_1 = torch.where(torch.exp(g_1)>2,torch.exp(g_1),torch.zeros_like(torch.exp(g_1)))
    
    # dg_1 = a
    # dy_1 = (H.T@(torch.ones(size=random_beta.shape)-eta*(torch.ones(size=random_beta.shape)-torch.tanh(eta*update_0/A))**2*\
    #             ((dy_0*phi_0.T)@dy_0.T))).T
    
    # print(((dy_0*phi_0.T)@dy_0.T).shape)
    # print((eta*(torch.ones(size=random_beta.shape)-torch.tanh(eta*update_0/A))**2).shape)
    # print((torch.ones(size=random_beta.shape)-eta*(torch.ones(size=random_beta.shape)-torch.tanh(eta*update_0/A))**2*\
    #             ((dy_0*phi_0.T)@dy_0.T)).shape)
    # print(dy_1.shape)
    
    # update_1 = (dy_1)@phi_1
    # print(update_1.shape)
    # input('a yipi')

    # y_2 = H.T@(random_beta-A*torch.tanh(eta*update_0/A)-A*torch.tanh(eta*update_1/A))

#     plt.plot(x.T.detach().numpy(),y_0.detach().numpy())
#     plt.plot(x.T.detach().numpy(),y_1.detach().numpy())
# #     plt.plot(x.T.detach().numpy(),y_2.detach().numpy())
#     # plt.ylim(-5,5)
#     plt.show()

# y_1 = H.T@(random_beta - A*torch.tanh(eta*/A))
# t = sp.Symbol('t')
# theta = sp.symbols('theta0:3')
# elem = [t]
# elem.extend((list(theta)))

# f1 = sp.Function('f1')(*elem)

# new = [t]
# new.extend([f1.diff(theta[i]) for i in range(len(theta))])
# print(new)
# input('hipi')
# f2 = sp.Function('f1')(*new)

# theta = sp.Matrix([sp.Symbol('theta0'),sp.Symbol('theta1'),sp.Symbol('theta2')])
# f1 = sp.Function('f1')(t,theta)
# f2 = sp.Function('f2')(t,theta)
# print(f1.diff(theta[0]))
# input('yipi')
# new = [f1.diff(theta[i]) for i in range(len(theta))]
# print(new)
# input('hipi')
# f3= sp.Function('f3')(t,sp.Matrix(new))
# print(f3)
# f1_0 = sp.Symbol('f1_0')
# f2_0 = sp.Symbol('f2_0')
# f_0 = sp.Matrix([f1_0,f2_0])
# f = sp.Matrix([f1,f2])
# g = sp.Function('g')(f1,f2)
# g_expr = sp.Matrix([-f[0]**2 - f[1]**2 + 1])
# grad_g = g_expr.jacobian(f)
# f_center = sp.Matrix([0,0])

# penalty_term = generate_penalty_term(generate_taylor_expansion(g_expr,grad_g,f,f_0))
# gaussian_amplitude = generate_gaussian_term(f,f_center,sp.Matrix([[1,0],[0,1]]))[0,0] 
# gaussian_term = generate_gaussian_term(f,f_0,sp.Matrix([[1,0],[0,1]]))[0,0]

# step_size = 0.0001

# total_penalty = penalty_term*gaussian_term
# total_penalty_modulated = penalty_term*gaussian_term*gaussian_amplitude
# penalty_term_grad = total_penalty.jacobian(f)
# penalty_term_grad_modulated = total_penalty_modulated.jacobian(f)

# num_angle = 12
# num_radius = 20
# max_radius = 1
# angle = np.linspace(-np.pi,np.pi,num_angle)
# radius = np.linspace(0.01,1,num_radius)

# penalties = []
# penalties_modulated = []
# pairs = [(angle[i],radius[j])for i in range(num_angle) for j in range(num_radius)]
# for pair in pairs:
#     x = np.cos(pair[0])*pair[1].round(2)
#     y = np.sin(pair[0])*pair[1].round(2)
#     f0_vals = np.array([x,y])
#     sub_penalty=penalty_term_grad.subs(dict(zip(f_0,f0_vals)))
#     sub_penalty_modulated=penalty_term_grad_modulated.subs(dict(zip(f_0,f0_vals)))
#     penalties_modulated.append(sub_penalty_modulated)
#     penalties.append(sub_penalty)

# penalties_total = reduce(lambda a, b: a + b, penalties).T
# penalties_modulated_total = reduce(lambda a, b: a + b, penalties_modulated).T
# f_corrected = f-step_size*penalties_total

# penalties_eval = sp.lambdify(list(f),penalties_total,modules=['numpy'])
# penalties_modulated_eval = sp.lambdify(list(f),penalties_modulated_total,modules=['numpy'])
# f_corrected_eval = sp.lambdify(list(f),f_corrected,modules=['numpy'])

# x,y = np.meshgrid(np.linspace(-3,3,100),np.linspace(-3,3,100))
# f_grid = np.vstack([x.ravel(),y.ravel()])


# # t = np.linspace(-2*np.pi,2*np.pi,1000)
# # plt.plot(np.cos(t),np.sin(t))   
# # plt.scatter(f_grid[0,:],f_grid[1,:],c='red',alpha=0.2)
# # f_eval = f_corrected_eval(f_grid[0],f_grid[1])
# # # plt.scatter(f_eval[0,0,:],f_eval[1,0,:],c='blue',alpha=0.2)
# # for i in range (30):
# #     f_eval = f_corrected_eval(f_eval[0,0,:],f_eval[1,0,:])
# #     # plt.scatter(f_eval[0,0,:],f_eval[1,0,:],c='green',alpha=0.2)
# # f_eval = f_corrected_eval(f_eval[0,0,:],f_eval[1,0,:])
# # plt.scatter(f_eval[0,0,:],f_eval[1,0,:],c='orange',alpha=0.7)

# # plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
# # tick_positions = np.arange(-3, 3, 0.5)  # from -2 to 2, step 0.5
# # plt.xticks(tick_positions)
# # plt.yticks(tick_positions)
# # plt.show()
# # print(f_eval.shape)
# # input('yipi')
# z = penalties_eval(f_grid[0],f_grid[1]).squeeze(1)*(-step_size)

# fig, axes = plt.subplots(2, 2, figsize=(14, 6))
# fig.suptitle(r'3D Visualization of Penalty function Components', fontsize=14,fontweight='bold')
# contour_00=axes[0,0].contour(x,y,z[0].reshape(x.shape),levels=7,cmap='viridis')
# cbar_00 = fig.colorbar(contour_00, ax=axes[0, 0])
# cbar_00.set_label('Penalty')
# axes[0,0].set_title(r'$\mathbf{-\eta \cdot \nabla_f P\left( T(g(f),f_0)\right) \cdot G(f, f_0, \Sigma)_1}$', fontsize=12)
# axes[0,0].set_xlabel('x')
# axes[0,0].set_ylabel('y')
# axes[0,0].grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5,color='black')
# axes[0,0].set_facecolor('white')

# contour_01=axes[0,1].contour(x,y,z[1].reshape(x.shape),levels=7,cmap='viridis')
# cbar_01 = fig.colorbar(contour_01, ax=axes[0, 1])
# cbar_01.set_label('Penalty')
# axes[0,1].set_title(r'$\mathbf{-\eta \cdot \nabla_f P\left( T(g(f),f_0)\right) \cdot G(f, f_0, \Sigma)_2}$', fontsize=12)
# axes[0,1].set_xlabel('x')
# axes[0,1].set_ylabel('y')
# axes[0,1].grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5,color='black')
# axes[0,1].set_facecolor('white')
# z = penalties_modulated_eval(f_grid[0],f_grid[1]).squeeze(1)*(-step_size)
# contour_10=axes[1,0].contour(x,y,z[0].reshape(x.shape),levels=7,cmap='viridis')
# cbar_10 = fig.colorbar(contour_10, ax=axes[1, 0])
# cbar_10.set_label('Penalty')
# axes[1,0].set_title(r'$\mathbf{-\eta \cdot \nabla_f P\left( T(g(f),f_0)\right) \cdot G(f, f_0, \Sigma)\cdot G(f, f_{center}, \Sigma)_1}$', fontsize=12)
# axes[1,0].set_xlabel('x')
# axes[1,0].set_ylabel('y')
# axes[1,0].grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5,color='black')
# axes[1,0].set_facecolor('white')

# contour_11=axes[1,1].contour(x,y,z[1].reshape(x.shape),levels=7,cmap='viridis')
# cbar_11 = fig.colorbar(contour_11, ax=axes[1, 1])
# cbar_11.set_label('Penalty')
# axes[1,1].set_title(r'$\mathbf{-\eta \cdot \nabla_f P\left( T(g(f),f_0)\right) \cdot G(f, f_0, \Sigma)\cdot G(f, f_{center}, \Sigma)_2}$', fontsize=12)
# axes[1,1].set_xlabel('x')
# axes[1,1].set_ylabel('y')
# axes[1,1].grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5,color='black')
# axes[1,1].set_facecolor('white')

# contour = plt.contour(x, y, z[0].reshape(x.shape)**2+z[1].reshape(y.shape)**2, cmap='viridis')
# stream = plt.streamplot(x, y, z[0].reshape(x.shape), z[1].reshape(y.shape),
                        #  color=np.sqrt(z[0].reshape(x.shape)**2 + z[1].reshape(y.shape)**2), cmap='viridis', linewidth=1.5)

# Add color bar to represent vector magnitude
# plt.colorbar(stream.lines, label='Vector Magnitude')

# plt.colorbar(contour,label='Penalty')
# cbar = plt.colorbar(stream.lines)
# cbar.set_label('Vector Magnitude')
plt.show()
# f_expr_eval = sp.lambdify(t,f_expr,modules=['numpy'])
# time = np.linspace(-2*np.pi,2*np.pi,1000)
# f_1_0_num = f_1_0_eval(time)
# f_expr_num = f_expr_eval(time)

# plt.plot(f_1_0_num[0,0,:],f_1_0_num[0,1,:])
# plt.scatter(f_eval[0,0,:],f_eval[1,0,:],c='blue',alpha=0.2)
# f_new_eval = f_corrected_eval(f_eval[0,0,:],f_eval[1,0,:])
# plt.scatter(f_new_eval[0,0,:],f_new_eval[1,0,:],c='green',alpha=0.2)
# f_new_eval = f_corrected_eval(f_new_eval[0,0,:],f_new_eval[1,0,:])
# plt.scatter(f_new_eval[0,0,:],f_new_eval[1,0,:],c='orange',alpha=0.2)
# plt.scatter(x=[1,0,-1,0,np.sqrt(2)/2,-np.sqrt(2)/2,np.sqrt(2)/2,-np.sqrt(2)/2],
#             y=[0,1,0,-1,np.sqrt(2)/2,np.sqrt(2)/2,-np.sqrt(2)/2,-np.sqrt(2)/2],marker='x',color='red')
# plt.show()