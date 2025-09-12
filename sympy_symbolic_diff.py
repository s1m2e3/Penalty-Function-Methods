from sympy import symbols, Matrix, tanh, exp, diff
import sympy as sp
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial
# Derivatives
def diff_over_time_vector(f, t_vars):
    partials = []
    for i in range(len(t_vars)):
        print('differentiated with respect to t'+str(i))
        partials.append(diff(f[i,:], t_vars[i]))
    return Matrix.vstack(*partials)
    
def diff_over_beta_vector(f, beta_vars):
    
    partials = []
    for i in range(len(beta_vars)):
        print('differentiated with respect to beta'+str(i))
        partials.append(diff(f, beta_vars[i]))
    
    return Matrix(partials)
    
def bounded_tanh(f_expr, A_val, eta_val):
    
    for i in range(f_expr.shape[0]):
        for j in range(f_expr.shape[1]):
            f_expr[i,j]= eta*A_val*tanh(f_expr[i,j]/A_val*eta_val)
    return f_expr

def less_than_inequality(f_expr, a_val):
    return f_expr - np.ones(shape=f_expr.shape)*a_val

def smooth_penalty(f_expr):
    return f_expr.applyfunc(exp)

def differentiate_wrt_ti(expr,ti):
    return expr.diff(ti)

# Parameters
n_weights = 5
n_samples = 5
a = -5
b = 5
A = 1
eta = 1e-3

# Symbolic variables
t = Matrix(list(symbols(f't0:{n_samples}', real=True)))
beta_x = Matrix(symbols(f'beta_x0:{n_weights}', real=True))
beta_y = Matrix(symbols(f'beta_y0:{n_weights}', real=True))

# # Random weights and biases

W_random = (a + (b - a) * np.random.randn(n_weights)).reshape(beta_x.shape)
b_random = (a + (b - a) * np.random.randn(n_weights)).reshape(beta_x.shape)

# # H: tanh(t * W + b)
H = t*W_random.T + np.tile(b_random.T,(n_samples,1))
H = H.applyfunc(tanh)

dH_dt = diff_over_time_vector(H, t)
d2H_d2t = diff_over_time_vector(dH_dt, t)

## Initial states
x0 = H*beta_x
y0 = H*beta_y
## Differentiate with respect to time
dx0_dt = diff_over_time_vector(x0,t)
dy0_dt = diff_over_time_vector(y0,t)

## Constraints
c1_1_lhs = dx0_dt - dy0_dt
c2_1_lhs = dx0_dt + dy0_dt
c3_1_lhs = -dx0_dt + dy0_dt
c4_1_lhs = -dx0_dt - dy0_dt

## Penalties 1
phi_1_1 = sum(smooth_penalty(less_than_inequality(c1_1_lhs, a)))/c1_1_lhs.shape[0]
phi_1_2 = sum(smooth_penalty(less_than_inequality(c2_1_lhs, a)))/c2_1_lhs.shape[0]
phi_1_3 = sum(smooth_penalty(less_than_inequality(c3_1_lhs, a)))/c3_1_lhs.shape[0]
phi_1_4 = sum(smooth_penalty(less_than_inequality(c4_1_lhs, a)))/c4_1_lhs.shape[0]
### Gradients
dphi_1_1_d_beta_x = diff_over_beta_vector(phi_1_1, beta_x)
dphi_1_2_d_beta_x = diff_over_beta_vector(phi_1_2, beta_x)
dphi_1_3_d_beta_x = diff_over_beta_vector(phi_1_3, beta_x)
dphi_1_4_d_beta_x = diff_over_beta_vector(phi_1_4, beta_x)

dphi_1_1_d_beta_y = diff_over_beta_vector(phi_1_1, beta_y)
dphi_1_2_d_beta_y = diff_over_beta_vector(phi_1_2, beta_y)
dphi_1_3_d_beta_y = diff_over_beta_vector(phi_1_3, beta_y)
dphi_1_4_d_beta_y = diff_over_beta_vector(phi_1_4, beta_y)

# # Updates
update_phi_1_1_x = bounded_tanh(dphi_1_1_d_beta_x, A, eta)
update_phi_1_2_x = bounded_tanh(dphi_1_2_d_beta_x, A, eta)
update_phi_1_3_x = bounded_tanh(dphi_1_3_d_beta_x, A, eta)
update_phi_1_4_x = bounded_tanh(dphi_1_4_d_beta_x, A, eta)

update_phi_1_1_y = bounded_tanh(dphi_1_1_d_beta_y, A, eta)
update_phi_1_2_y = bounded_tanh(dphi_1_2_d_beta_y, A, eta)
update_phi_1_3_y = bounded_tanh(dphi_1_3_d_beta_y, A, eta)
update_phi_1_4_y = bounded_tanh(dphi_1_4_d_beta_y, A, eta)

## Total update
update_phi_1_x = update_phi_1_1_x + update_phi_1_2_x + update_phi_1_3_x + update_phi_1_4_x
update_phi_1_y = update_phi_1_1_y + update_phi_1_2_y + update_phi_1_3_y + update_phi_1_4_y

## Next states
x1 = H*(beta_x-update_phi_1_x)
y1 = H*(beta_y-update_phi_1_y)

## Differentiate with respect to time
dx1_dt = diff_over_time_vector(x1,t)
dy1_dt = diff_over_time_vector(y1,t)

## Constraints
c1_2_lhs = dx1_dt - dy1_dt
c2_2_lhs = dx1_dt + dy1_dt
c3_2_lhs = -dx1_dt + dy1_dt
c4_2_lhs = -dx1_dt - dy1_dt

## Penalties 1
phi_2_1 = sum(smooth_penalty(less_than_inequality(c1_2_lhs, a)))/c1_2_lhs.shape[0]
phi_2_2 = sum(smooth_penalty(less_than_inequality(c2_2_lhs, a)))/c2_2_lhs.shape[0]
phi_2_3 = sum(smooth_penalty(less_than_inequality(c3_2_lhs, a)))/c3_2_lhs.shape[0]
phi_2_4 = sum(smooth_penalty(less_than_inequality(c4_2_lhs, a)))/c4_2_lhs.shape[0]
### Gradients
dphi_2_1_d_beta_x = diff_over_beta_vector(phi_2_1, beta_x)
dphi_2_2_d_beta_x = diff_over_beta_vector(phi_2_2, beta_x)
dphi_2_3_d_beta_x = diff_over_beta_vector(phi_2_3, beta_x)
dphi_2_4_d_beta_x = diff_over_beta_vector(phi_2_4, beta_x)

dphi_2_1_d_beta_y = diff_over_beta_vector(phi_2_1, beta_y)
dphi_2_2_d_beta_y = diff_over_beta_vector(phi_2_2, beta_y)
dphi_2_3_d_beta_y = diff_over_beta_vector(phi_2_3, beta_y)
dphi_2_4_d_beta_y = diff_over_beta_vector(phi_2_4, beta_y)

## Updates
update_phi_2_1_x = bounded_tanh(dphi_2_1_d_beta_x, A, eta)
update_phi_2_2_x = bounded_tanh(dphi_2_2_d_beta_x, A, eta)
update_phi_2_3_x = bounded_tanh(dphi_2_3_d_beta_x, A, eta)
update_phi_2_4_x = bounded_tanh(dphi_2_4_d_beta_x, A, eta)

update_phi_2_1_y = bounded_tanh(dphi_2_1_d_beta_y, A, eta)
update_phi_2_2_y = bounded_tanh(dphi_2_2_d_beta_y, A, eta)
update_phi_2_3_y = bounded_tanh(dphi_2_3_d_beta_y, A, eta)
update_phi_2_4_y = bounded_tanh(dphi_2_4_d_beta_y, A, eta)

## Total update
update_phi_2_x = update_phi_2_1_x + update_phi_2_2_x + update_phi_2_3_x + update_phi_2_4_x
update_phi_2_y = update_phi_2_1_y + update_phi_2_2_y + update_phi_2_3_y + update_phi_2_4_y


## Next states
x2 = H*(beta_x-update_phi_1_x-update_phi_2_x)
y2 = H*(beta_y-update_phi_1_y-update_phi_2_y)

## Differentiate with respect to time
dx2_dt = diff_over_time_vector(x2,t)
dy2_dt = diff_over_time_vector(y2,t)

print('dx2_dt',dx2_dt)