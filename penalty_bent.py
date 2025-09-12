import sympy as sp
import numpy as np
from utils import generate_gaussian_term,generate_penalty_term,generate_taylor_expansion,generate_tanh
import matplotlib.pyplot as plt
from functools import reduce
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plotting

plt.style.use('seaborn-v0_8')
theta = sp.symbols('theta0:2')
theta_2 = sp.symbols('theta_0:2')
max_bound = 2
min_bound = -2
A = max_bound-min_bound/2
C = max_bound+min_bound/2

const = 5*theta[0]+8*theta[1]
eta = 0.5
f1 = sp.Function('f1')(*theta)
f1_exp = generate_penalty_term(f1)
eval_f1 =  f1_exp.subs(f1,const)
const_1 = f1.subs(f1,const)

new = [theta[i]-eta*A*sp.tanh(eval_f1.diff(theta[i])/A) for i in range(len(theta))]
subs_dict = dict(zip(theta,theta_2))
intermediate_subs = const.subs(subs_dict)
subs_dict = dict(zip(theta_2,new))
final_subs = intermediate_subs.subs(subs_dict)
const_2 = final_subs

eval_f2 = generate_penalty_term(const_2)

new=[new[i]-eta*A*sp.tanh(eval_f2.diff(theta[i])/A) for i in range(len(theta))]

subs_dict = dict(zip(theta,theta_2))
intermediate_subs = const.subs(subs_dict)
subs_dict = dict(zip(theta_2,new))
final_subs = intermediate_subs.subs(subs_dict)
const_3 = final_subs
eval_f3 = generate_penalty_term(const_3)

norm_const_1 = sp.Pow(sp.Pow(const_1.diff(theta[0]),2)+sp.Pow(const_1.diff(theta[1]),2),1/2)
norm_const_2 = sp.Pow(sp.Pow(const_2.diff(theta[0]),2)+sp.Pow(const_2.diff(theta[1]),2),1/2)
norm_const_3 = sp.Pow(sp.Pow(const_3.diff(theta[0]),2)+sp.Pow(const_3.diff(theta[1]),2),1/2)

const_f1_lam = sp.lambdify([theta[0],theta[1]],const_1,modules=['numpy'])
const_f2_lam = sp.lambdify([theta[0],theta[1]],const_2,modules=['numpy'])
const_f3_lam = sp.lambdify([theta[0],theta[1]],const_3,modules=['numpy'])

norm_f1_lam = np.vectorize(sp.lambdify([theta[0],theta[1]],norm_const_1,modules=['numpy']))
norm_f2_lam = np.vectorize(sp.lambdify([theta[0],theta[1]],norm_const_2,modules=['numpy']))
norm_f3_lam = np.vectorize(sp.lambdify([theta[0],theta[1]],norm_const_3,modules=['numpy']))

theta0,theta1 = np.meshgrid(np.linspace(-5,5,100),np.linspace(-8,8,320))
eval_f1_grid = const_f1_lam(theta0,theta1)
eval_f2_grid = const_f2_lam(theta0,theta1)
eval_f3_grid = const_f3_lam(theta0,theta1)

eval_f1_norm_grid = norm_f1_lam(theta0,theta1)
eval_f2_norm_grid = norm_f2_lam(theta0,theta1)
eval_f3_norm_grid = norm_f3_lam(theta0,theta1)


vmin = min(np.min(eval_f1_norm_grid), np.min(eval_f2_norm_grid), np.min(eval_f3_norm_grid))
vmax = max(np.max(eval_f1_norm_grid), np.max(eval_f2_norm_grid), np.max(eval_f3_norm_grid))

fig, axes = plt.subplots(1,3,figsize=(15,12))
im1 = axes[0].imshow(eval_f1_norm_grid, cmap='Reds', extent=(theta0.min(), theta0.max(), theta1.min(), theta1.max()),vmin=vmin,vmax=vmax)
im2 = axes[1].imshow(eval_f2_norm_grid, cmap='Reds', extent=(theta0.min(), theta0.max(), theta1.min(), theta1.max()),vmin=vmin,vmax=vmax)
im3 = axes[2].imshow(eval_f3_norm_grid, cmap='Reds', extent=(theta0.min(), theta0.max(), theta1.min(), theta1.max()),vmin=vmin,vmax=vmax)
axes[0].set_title('No Updates')
axes[1].set_title('First Update')
axes[2].set_title('Second Update')
cbar1 = fig.colorbar(im1, ax=axes[0])
cbar2 = fig.colorbar(im2, ax=axes[1])
cbar3 = fig.colorbar(im3, ax=axes[2])
plt.show()

vmin = min(np.min(eval_f1_grid), np.min(eval_f2_grid), np.min(eval_f3_grid))
vmax = max(np.max(eval_f1_grid), np.max(eval_f2_grid), np.max(eval_f3_grid))

fig, axes = plt.subplots(1,3,figsize=(15,12))
im1 = axes[0].imshow(np.where(eval_f1_grid<0,0,eval_f1_grid), cmap='Reds', extent=(theta0.min(), theta0.max(), theta1.min(), theta1.max()),vmin=0,vmax=5)
im2 = axes[1].imshow(np.where(eval_f2_grid<0,0,eval_f2_grid), cmap='Reds', extent=(theta0.min(), theta0.max(), theta1.min(), theta1.max()),vmin=0,vmax=5)
im3 = axes[2].imshow(np.where(eval_f3_grid<0,0,eval_f3_grid), cmap='Reds', extent=(theta0.min(), theta0.max(), theta1.min(), theta1.max()),vmin=0,vmax=5)
axes[0].set_title('No Updates')
axes[1].set_title('First Update')
axes[2].set_title('Second Update')
cbar1 = fig.colorbar(im1, ax=axes[0])
cbar2 = fig.colorbar(im2, ax=axes[1])
cbar3 = fig.colorbar(im3, ax=axes[2])
plt.show()

fig = plt.figure(figsize=(15,12))
ax1 = fig.add_subplot(131, projection='3d')
ax2 = fig.add_subplot(132, projection='3d')
ax3 = fig.add_subplot(133, projection='3d')

ax1.plot_surface(theta0, theta1, eval_f1_grid, cmap='Reds', edgecolor='none')
ax1.view_init(elev=5, azim=-20)
ax1.dist = 3
ax2.plot_surface(theta0, theta1, eval_f2_grid, cmap='Reds', edgecolor='none')
ax2.view_init(elev=5, azim=-20)
ax2.dist = 3
ax3.plot_surface(theta0, theta1, eval_f3_grid, cmap='Reds', edgecolor='none')
ax3.view_init(elev=5, azim=-20)
ax3.dist = 3
ax1.set_title('No Updates')
ax2.set_title('First Update')
ax3.set_title('Second Update')

plt.show()

