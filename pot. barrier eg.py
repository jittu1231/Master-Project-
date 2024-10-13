# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 11:35:01 2024

@author: User
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, identity
from scipy.sparse.linalg import spsolve
import matplotlib.animation as animation

hbar = 1.0 
m = 1.0   
x_min, x_max = -10.0, 10.0 
N = 1000  
dx = (x_max - x_min) / N 
x = np.linspace(x_min, x_max, N) 

t_final = 2.0 
dt = 0.001  

x0 = -5.0 
k0 = 5.0 
sigma = 0.5 
Psi0 = np.exp(-(x - x0)**2 / (2 * sigma**2)) * np.exp(1j * k0 * x)
Psi0 /= np.sqrt(np.sum(np.abs(Psi0)**2) * dx)  

V0 = 1.0 
barrier_width = 2.0 
V = np.zeros_like(x)
V[(x > -barrier_width / 2) & (x < barrier_width / 2)] = V0 

T_coeff = -hbar**2 / (2 * m * dx**2)
T = T_coeff * (diags([1, -2, 1], [-1, 0, 1], shape=(N, N)))
V_diag = diags(V, 0)
H = T + V_diag

def crank_nicolson_step(H, psi, dt):
    I = identity(N)
    A = I - 1j * H * dt / (2 * hbar)
    B = I + 1j * H * dt / (2 * hbar)
    psi_new = spsolve(A, B @ psi)
    return psi_new

def evolve_in_time(psi_0, H, dt, t_final):
    time_steps = int(t_final / dt)
    psi_t = psi_0.copy()
    psi_t_list = [psi_t]
    
    for _ in range(time_steps):
        psi_t = crank_nicolson_step(H, psi_t, dt)
        psi_t_list.append(psi_t)
        
    return psi_t_list

Psi_t = evolve_in_time(Psi0, H, dt, t_final)


plt.figure(figsize=(10, 8))
plt.plot(x, np.abs(Psi0)**2, label="Initial |Ψ(x, 0)|²", color='blue')
plt.plot(x, V / V0, label="Potential barrier V(x)", color='red', linestyle='--')

plt.plot(x, np.abs(Psi_t[-1])**2, label=f"Final |Ψ(x, t={t_final})|²", color='green')
plt.xlabel('x')
plt.ylabel('|Ψ(x, t)|²')
plt.legend()
plt.grid(True)
plt.title("Quantum Tunneling of Gaussian Wavepacket")
plt.show()

fig, ax = plt.subplots()
ax.set_xlim(x_min, x_max)
ax.set_ylim(0, np.max(np.abs(Psi0)**2) * 1.2)
ax.set_xlabel('x')
ax.set_ylabel('|Ψ(x, t)|²')

line, = ax.plot([], [], lw=2)
potential_line, = ax.plot(x, V / V0, color='red', linestyle='--', label="Potential barrier")

def init():
    line.set_data([], [])
    return line,

def animate(i):
    line.set_data(x, np.abs(Psi_t[i])**2)
    return line,

ani = animation.FuncAnimation(fig, animate, init_func=init, frames=len(Psi_t), interval=30, blit=True)

plt.legend()
plt.show()
