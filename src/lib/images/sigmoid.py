#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# src/lib/images/sigmoid.py

# Author : Irreq


# Externa bibliotek
import matplotlib.pyplot as plt
import numpy as np

# LaTeX:
# $$  { \sigma } (x) =  \frac{\mathrm{1} }{\mathrm{1} + e^{-x}}  $$



def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

start = -10
end = 10
steps = 100
y = np.linspace(start, end, steps)

plt.plot(y, sigmoid(y), 'b', label=f'numpy.linspace({start},{end},{steps})')

# Rita rutmönster
plt.grid()

# Titel
plt.title('Sigmoidfunktion')

# Legenden längst ned
plt.legend(loc='lower right')

# Rita grafen
plt.show()
