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
