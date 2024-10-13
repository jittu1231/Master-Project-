# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 19:14:21 2024

@author: User
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.sparse import diags, identity
from scipy.sparse.linalg import spsolve
from scipy.linalg import svd

hbar = 1.0
m = 1.0
a = 1.0
b = 5.0
t_final = 10.0 
dt = 0.01

x = np.linspace(-50, 50, 500) 
dx = x[1] - x[0]
N = len(x)

Psi1 = np.exp(-a * np.outer(x**2, np.ones_like(x)))
Psi2 = np.exp(-b * np.outer(np.ones_like(x), x**2))
Psi = Psi1 * Psi2

U, S, VT = svd(Psi)

plt.figure(figsize=(10, 8))
plt.pcolor(x, x, Psi, shading='auto')
plt.colorbar(label='Ψ(x1, x2)')
plt.title('Initial Ψ(x1, x2)')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

lambda_1 = S[0] 
psi_1 = U[:, 0] 
phi_1 = VT[0, :]

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(x, psi_1, label='ψ₁(x₁)')
plt.title('ψ₁(x₁)')
plt.xlabel('x1')
plt.ylabel('Amplitude')
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(x, phi_1, label='φ₁(x₂)', color='orange')
plt.title('φ₁(x₂)')
plt.xlabel('x2')
plt.ylabel('Amplitude')
plt.grid()

plt.tight_layout()
plt.show()

V1 = np.exp(-a * x**2) * 2

T = - (hbar**2 / (2 * m * dx**2)) * (diags([1, -2, 1], [-1, 0, 1], shape=(N, N)))

H1 = T + diags(V1, 0)

def crank_nicolson_step(H, psi, dt):
    I = identity(N)
    A = I - 1j * H * dt / (2 * hbar)
    B = I + 1j * H * dt / (2 * hbar)
    psi_new = spsolve(A, B @ psi)
    return psi_new

def evolve_in_time_crank_nicolson(H, psi_0, dt, t_final):
    time_steps = int(t_final / dt)
    psi_t = psi_0
    for _ in range(time_steps):
        psi_t = crank_nicolson_step(H, psi_t, dt)
    return psi_t

psi_1_final = np.abs(evolve_in_time_crank_nicolson(H1, psi_1, dt, t_final))
phi_1_final = np.abs(evolve_in_time_crank_nicolson(H1, phi_1, dt, t_final))

psi_1_final /= np.sqrt(np.sum(np.abs(psi_1_final)**2) * dx)
phi_1_final /= np.sqrt(np.sum(np.abs(phi_1_final)**2) * dx)

Psi_final = lambda_1 * np.outer(psi_1_final, phi_1_final)

plt.figure(figsize=(10, 8))
plt.plot(x, Psi1[0], label="Initial ψ₁(x₁)")
plt.plot(x, psi_1_final, label="Final ψ₁(x₁) after time evolution")
plt.plot(x, V1, label="Potential V1")
plt.xlabel('x1')
plt.ylabel('wavefunction ψ')
plt.title("ψ₁(x₁) vs x1 after time evolution")
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(10, 8))
plt.pcolor(x, x, Psi_final, shading='auto')
plt.colorbar(label='Ψ(x1, x2) after time evolution')
plt.title('Final Ψ(x1, x2) after Time Evolution using Crank-Nicolson')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

plt.figure(figsize=(10, 8))
plt.pcolor(x, x, Psi - Psi_final, shading='auto')
plt.colorbar(label='Difference Ψ_initial - Ψ_final')
plt.title('Difference between Initial and Time-Evolved Ψ(x1, x2)')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()


