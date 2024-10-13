# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 19:06:06 2024

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd

def gaussian(x, mu=0, sigma=1):
    return np.exp(- (x - mu)**2 / (2 * sigma**2))

x1 = np.linspace(-5, 5, 100)
x2 = np.linspace(-5, 5, 100)
X1, X2 = np.meshgrid(x1, x2)

phi_x1 = gaussian(X1)
phi_x2 = gaussian(X2)
psi = phi_x1 * phi_x2

plt.figure(figsize=(10, 8))
plt.pcolor(x1, x2, psi, shading='auto')
plt.colorbar(label='psi(x1, x2)')
plt.title('2D Gaussian Function psi(x1, x2) = phi(x1) * phi(x2)')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

U, S, VT = svd(psi)

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(x1, U[:, 0], label='First singular vector (U[:, 0])')
plt.title('First Singular Vector - phi(x1)')
plt.xlabel('x1')
plt.ylabel('Amplitude')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(x2, VT[0, :], label='First singular vector (VT[0, :])')
plt.title('First Singular Vector - phi(x2)')
plt.xlabel('x2')
plt.ylabel('Amplitude')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(S, 'o-')
plt.title('Singular Values of psi')
plt.xlabel('Index')
plt.ylabel('Singular Value')
plt.yscale('log') 
plt.show()
