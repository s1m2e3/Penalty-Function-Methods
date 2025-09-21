import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D  # for legend proxy
import seaborn as sns
sns.set_theme(style="darkgrid")
def functional_gradient(f_t,a,eta,lb=1.0,_lambda=1.0):
    
    return - (eta*_lambda**2*torch.sigmoid((f_t@a.unsqueeze(1)-1)*_lambda)@a.unsqueeze(0))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
a_1=torch.tensor([1.0,1.0]).float().to(device)
a_2=torch.tensor([1.0,0.0]).float().to(device)
a_3=torch.tensor([0.0,1.0]).float().to(device)
# a_4 = torch.tensor([-1.0,1.0]).float()

num_grad_steps = 100
num_functional_grad_steps = 1000
num_trajectory_steps = 10
num_sampled_weights = 3
eta_gradient = 1e-3
num_samples = 5
eta = 0.1

fig,axs = plt.subplots(3,1,figsize=(12,15),sharex=True)
fig.suptitle("Constrained Output Trajectories Across Vector-Space Optimization with Different Random Weights",fontsize=16,fontweight='bold')

t_title_function = "Trajectory Timestep"
x_title_function = r"$f_{\theta}^{1}-f^1_{target}$"
y_title_function = r"$f_{\theta}^{2}-f^2_{target}$"
xy_title_function = r"$f_{\theta}^{1}+f_{\theta}^{2}$"
lambda_ = 1
lines = {}
for i in range(num_sampled_weights):
    total_sampled_h_0 = torch.rand([num_samples,2]).float().to(device)
    h_0 = -5+10*total_sampled_h_0
    W_h = (torch.rand(2,2)*1-2).requires_grad_(True).to(device)
    b_h = (torch.randn(2)*1-2).requires_grad_(True).to(device)
    W_x_1 = ((torch.randn(10,10)*1-2)*0.1).requires_grad_(True).to(device)
    W_x_2 = ((torch.randn(10,10)*1-2)*0.1).requires_grad_(True).to(device)
    b_x_1 = ((torch.randn(10)*1-2)*0.1).requires_grad_(True).to(device)
    b_x_2 = ((torch.randn(10)*1-2)*0.1).requires_grad_(True).to(device)

    W_h_1 = ((torch.randn(2,10)*1-2)*0.1).requires_grad_(True).to(device)
    W_h_2 = ((torch.randn(10,2)*1-2)*0.1).requires_grad_(True).to(device)
    b_h_1 = ((torch.randn(10)*1-2)*0.1).requires_grad_(True).to(device)
    b_h_2 = ((torch.randn(2)*1-2)*0.1).requires_grad_(True).to(device)
    W_x = ((torch.randn(10,2)*1-2)*0.1).requires_grad_(True).to(device)
    b_x = ((torch.randn(2)*1-2)*0.1).requires_grad_(True).to(device)
    lines[i]={}

    for grad_step in range(num_grad_steps):
        loss = 0
        for t in range(num_trajectory_steps):
            eta = 0.1
            x = -5+10*torch.rand([num_samples,10]).float().to(device)
            x = torch.relu(x@W_x_1+b_x_1)@W_x_2+b_x_2
            h_0 = torch.relu(h_0@W_h_1+b_h_1)@W_h_2+b_h_2
            z = h_0@W_h + x@W_x + b_h + b_x
            h_1 = torch.relu(z)
            lambda_ = 1
            for j in range(num_functional_grad_steps):
                    
                penalty_1 = torch.nn.functional.softplus(h_1@a_1.unsqueeze(1)-1,beta=lambda_)
                h_1 = torch.where(penalty_1>0.6,h_1+functional_gradient(h_1,a_1,eta,lambda_),h_1)

                penalty_2 = torch.nn.functional.softplus(-h_1@a_2.unsqueeze(1),beta=lambda_)
                h_1 = torch.where(penalty_2>0.6,h_1-functional_gradient(-h_1,a_2,eta,lambda_),h_1)

                penalty_3 = torch.nn.functional.softplus(-h_1@a_3.unsqueeze(1),beta=lambda_)
                h_1 = torch.where(penalty_3>0.6,h_1-functional_gradient(-h_1,a_3,eta,lambda_),h_1)
                
                if torch.all(penalty_1<0.6).item() and torch.all(penalty_2<0.6).item() and torch.all(penalty_3<0.6).item():
                    print('break functional gradient loop')
                    break
                # penalty = torch.nn.functional.softplus(h_1@a_4.unsqueeze(1)-1,beta=lambda_)
                # h_1 = torch.where(penalty>0.13,h_1+functional_gradient(h_1,a_4,eta,lambda_),h_1)
                eta = eta*0.999
                lambda_ = lambda_*1.1
            loss = torch.nn.functional.mse_loss(h_1,(torch.ones_like(h_1)*0.4).to(device)) + loss
        
            if grad_step == num_grad_steps-1:
                lines[i][t] = h_1
            h_0 = h_1
            # loss = torch.nn.functional.mse_loss(h_1,(torch.ones_like(h_1)*0.4).to(device)) + loss
        # loss = torch.nn.functional.mse_loss(h_1,(torch.ones_like(h_1)*0.4).to(device))
        grad_W_x = torch.autograd.grad(loss,W_x,retain_graph=True)
        W_x = (W_x - eta_gradient*grad_W_x[0].clip(-1,1)).clip(-3,3)
        grad_W_h = torch.autograd.grad(loss,W_h,retain_graph=True)
        W_h = (W_h - eta_gradient*grad_W_h[0].clip(-1,1)).clip(-3,3)
        grad_b_x = torch.autograd.grad(loss,b_x,retain_graph=True)
        b_x = (b_x - eta_gradient*grad_b_x[0].clip(-1,1)).clip(-3,3)
        grad_b_h = torch.autograd.grad(loss,b_h,retain_graph=True)
        b_h = (b_h - eta_gradient*grad_b_h[0].clip(-1,1)).clip(-3,3)
        grad_W_x_1 = torch.autograd.grad(loss,W_x_1,retain_graph=True)
        W_x_1 = (W_x_1 - eta_gradient*grad_W_x_1[0].clip(-1,1)).clip(-3,3)
        grad_W_x_2 = torch.autograd.grad(loss,W_x_2,retain_graph=True)
        W_x_2 = (W_x_2 - eta_gradient*grad_W_x_2[0].clip(-1,1)).clip(-3,3)
        grad_b_x_1 = torch.autograd.grad(loss,b_x_1,retain_graph=True)
        b_x_1 = (b_x_1 - eta_gradient*grad_b_x_1[0].clip(-1,1)).clip(-3,3)
        grad_b_x_2 = torch.autograd.grad(loss,b_x_2,retain_graph=True)
        b_x_2 = (b_x_2 - eta_gradient*grad_b_x_2[0].clip(-1,1)).clip(-3,3)
        grad_W_h_1 = torch.autograd.grad(loss,W_h_1,retain_graph=True)
        W_h_1 = (W_h_1 - eta_gradient*grad_W_h_1[0].clip(-1,1)).clip(-3,3)
        grad_W_h_2 = torch.autograd.grad(loss,W_h_2,retain_graph=True)
        W_h_2 = (W_h_2 - eta_gradient*grad_W_h_2[0].clip(-1,1)).clip(-3,3)
        grad_b_h_1 = torch.autograd.grad(loss,b_h_1,retain_graph=True)
        b_h_1 = (b_h_1 - eta_gradient*grad_b_h_1[0].clip(-1,1)).clip(-3,3)
        grad_b_h_2 = torch.autograd.grad(loss,b_h_2)
        b_h_2 = (b_h_2 - eta_gradient*grad_b_h_2[0].clip(-1,1)).clip(-3,3)
        print("in vector-space gradient descent iteration",grad_step,"loss is",loss)
        
# Define the vertices of the rhomboid (diamond shape)
vertices = np.array([
        [1, 0],   # top
        [0, 1],   # right
        [0, 0],  # bottom
    ])

# Close the polygon by repeating the first point
vertices = np.vstack([vertices, vertices[0]])

# plt.plot(vertices[:, 0], vertices[:, 1], 'r-', linewidth=2)
# plt.fill(vertices[:, 0], vertices[:, 1], alpha=0.2, color='yellow', label='Feasible Region')
colors = ['r','b','g']

for i in range(len(lines)):
    resulting_data=torch.stack([lines[i][j] for j in range(len(lines[i]))])
    for j in range(resulting_data.shape[1]):
        axs[0].plot(np.arange(resulting_data.shape[0]),((resulting_data[:,j,0]-0.4)).cpu().detach().numpy(),color=colors[i-1])
        axs[0].hlines(1, xmin=0, xmax=resulting_data.shape[0]-1, colors='black', linestyles='dashed')
        axs[1].plot(np.arange(resulting_data.shape[0]),((resulting_data[:,j,1]-0.4)).cpu().detach().numpy(),color=colors[i-1])
        axs[1].hlines(1, xmin=0, xmax=resulting_data.shape[0]-1, colors='black', linestyles='dashed')
        axs[2].plot(np.arange(resulting_data.shape[0]),(resulting_data[:,j,0]+resulting_data[:,j,1]).cpu().detach().numpy(),color=colors[i-1])
        axs[2].hlines(1, xmin=0, xmax=resulting_data.shape[0]-1, colors='black', linestyles='dashed')

axs[0].set_ylabel(x_title_function,fontsize=14)
axs[0].set_ylim([-0.2,0.2])
axs[1].set_ylabel(y_title_function,fontsize=14)
axs[1].set_ylim([-0.2,0.2])
axs[2].set_ylabel(xy_title_function,fontsize=14)
axs[2].set_xlabel(t_title_function,fontsize=14)

legend_elements = [
    Line2D([0], [0], color='r', label='First Batch of Sampled Weights',lw='2'),
    Line2D([0], [0], color='b', label='Second Batch of Sampled Weights',lw='2'),
    Line2D([0], [0], color='g', label='Third Batch of Sampled Weights',lw='2')
]
fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.95, 0.95))
plt.show()
# fig.savefig("constrained_trajectories.png",dpi=300)
