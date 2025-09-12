import sympy as sp
import numpy as np


def generate_gaussian_term(f,f_0,Sigma):
    return sp.exp(-1/2*sp.Transpose(f-f_0)*Sigma*(f-f_0))

def generate_penalty_term(g):
    return sp.exp(g)
def generate_tanh(g):
    return sp.tanh(g)

def generate_taylor_expansion(g,g_grad,f,f_0):
    g_at_f_0 = g.subs(dict(zip(f, f_0)))
    g_grad_at_f_0 = g_grad.subs(dict(zip(f, f_0)))
    delta_f = f - f_0
    return g_at_f_0 + (g_grad_at_f_0*delta_f)
