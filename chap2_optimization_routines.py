## Benjamin MacLellan 2020

#%%
from configstyle import ConfigStyle
style = ConfigStyle()

# %%
import sys
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import networkx as nx
import itertools
import random
import os
import copy
import time
import autograd.numpy as np
from autograd import elementwise_grad, grad, jacobian, hessian

if True:
    np.random.seed(0)

plt.close('all')
c = ConfigStyle()

def func(x):
    return np.sum(x**2 + 0.1*x**3 + 0.13 * x**4 +  1.0 * np.sin(1.4*x))

grad = elementwise_grad(func)
hess = hessian(func)
results = minimize(func, [0.0, 0.0])

n = 100
x1, x2 = np.linspace(-1, 1, n), np.linspace(-1, 1, n)
f = np.zeros([n,n])
for m in range(n):
    for j in range(n):
        f[m, j] = -func(np.array([x1[m], x2[j]]))


fig, axs = plt.subplots(3, 3, figsize=[c.width_page, c.height_page/2], gridspec_kw={'wspace':0.2, 'hspace':0.2})
pmesh_style = {'cmap': c.cmap, 'rasterized': True}
marker_style = {'marker':'o', 'color':c.cp[0], 'zorder':4, 'alpha': 0.8}
xmin_style = {'marker':'x', 'color':c.cp[3], 'zorder':4, 'alpha': 1.0}
arrow_style = {'color': c.cp[4], 'width':0.001, 'head_width':0.055, 'length_includes_head':False, 'shape':'full'}
line_style = { 'ls':'-', 'lw':1, 'color':c.cp[2], 'alpha':0.6}
contour_style = {'colors': 'grey', 'alpha': 0.3}
for ax in axs.flatten():
    ax.set(xticklabels=[], yticklabels=[])
    ax.grid(False)

# #%% brute force
# m_grid = 7
# x = [(-1 + 2/(m_grid+1), -1 + 2*j/(m_grid+1)) for j in range(m_grid, 0, -1)]
# for m, (ax, xm) in enumerate(zip(axs[:,0], x)):
#     for j in range(1, m_grid+1):
#         ax.plot([-1, 1], 2*[-1 + 2*j/(m_grid+1)], **{'ls':'--', 'lw':0.5, 'color':'black', 'alpha':0.5})
#     for j in range(1, m_grid+1):
#         ax.plot(2*[-1 + 2*j/(m_grid+1)], [-1, 1], **line_style)
#     ax.pcolormesh(x1, x2, f, **pmesh_style)
#     ax.contour(x1, x2, f, levels=5, **contour_style)
#     ax.scatter(xm[0], xm[1], **marker_style)

#%% evolutionary methods
n_pop = 10
g, fg = None, np.inf
w, phi_p, phi_g = 0.2, 1, 2

pop = []
for m in range(n_pop):
    xm = np.random.uniform(-1, 1, 2)
    if func(xm) < fg:
        g, fg = xm, func(xm)
    pm = xm
    vm = np.random.uniform(-2, 2, 2)
    pop.append({'xm':xm, 'pm':pm, 'vm':vm, 'f':func(xm)})

gens = [copy.deepcopy(pop)]
saves = (3, 6)
for gen in range(1, max(saves)+1):
    for m, particle in enumerate(pop):
        rp, rg = np.random.uniform(0,1,2), np.random.uniform(0,1,2)
        particle['vm'] = w * particle['vm'] + phi_p * rp * (particle['pm'] - particle['xm']) + phi_g * rg * (g - particle['xm'])
        particle['xm'] = particle['xm'] + particle['vm']
        particle['f'] = func(particle['xm'])

        if func(particle['xm']) < func(particle['pm']):
            particle['pm'] = particle['xm']
        if func(particle['xm']) < func(g):
            g = copy.deepcopy(particle['xm'])
        pop[m] = particle

    if gen in saves:
        gens.append(copy.deepcopy(pop))

for i, (ax, pop) in enumerate(zip(axs[:, 0], gens)):
    ax.pcolormesh(x1, x2, f, **pmesh_style)
    ax.contour(x1, x2, f, levels=5, **contour_style)
    ax.scatter(results.x[0], results.x[1], **xmin_style)
    for m, particle in enumerate(pop):
        ax.scatter(particle['xm'][0], particle['xm'][1], **marker_style)
        # if i  < 2:
        #     particle_next = gens[i+1][m]
        #     ax.arrow(particle['xm'][0], particle['xm'][1],
        #              particle_next['xm'][0] - particle['xm'][0], particle_next['xm'][1] - particle['xm'][1],  **arrow_style)
    ax.set(xlim=[-1,1], ylim=[-1,1])

#%% gradient descent
x0 = np.array([-0.8, 0.8])
x, g, alpha = [x0], [grad(x0)], 0.19
for m in range(1,3):
    g.append(grad(x[m-1]))
    x.append(x[m-1] - alpha * grad(x[m-1]))

for m, (ax, xm) in enumerate(zip(axs[:,1], x)):
    update = -alpha*grad(xm)
    ax.pcolormesh(x1, x2, f, **pmesh_style)
    ax.contour(x1, x2, f, levels=5, **contour_style)
    ax.scatter(results.x[0], results.x[1], **xmin_style)

    ax.scatter(xm[0], xm[1], **marker_style)
    ax.arrow(xm[0], xm[1], update[0], update[1], **arrow_style)
    if m > 0:
        for n in range(1,m+1):
            ax.plot([x[n-1][0], x[n][0]], [x[n-1][1], x[n][1]], **line_style)

#%% newton's method
x0 = np.array([-0.8, 0.8])
x, g, alpha = [x0], [np.dot(grad(x0), np.linalg.inv(hess(x0)))], 0.6
for m in range(1,3):
    x.append(x[m-1] - alpha * np.dot(grad(x[m-1]), np.linalg.inv(hess(x[m-1]))))
for m, (ax, xm) in enumerate(zip(axs[:,2], x)):
    update = - alpha * np.dot(grad(xm), np.linalg.inv(hess(xm)))
    ax.pcolormesh(x1, x2, f, **pmesh_style)
    ax.contour(x1, x2, f, levels=5, **contour_style)
    ax.scatter(results.x[0], results.x[1], **xmin_style)

    ax.scatter(xm[0], xm[1], **marker_style)
    ax.arrow(xm[0], xm[1], update[0], update[1], **arrow_style)
    if m > 0:
        for n in range(1,m+1):
            ax.plot([x[n-1][0], x[n][0]], [x[n-1][1], x[n][1]], **line_style)

#%% aesthetic changes to axes
for ax in axs.flatten():
    ax.set_aspect('equal')

methods = [r"\textbf{(a)} Evolutionary algorithm", r"\textbf{(b)} Gradient descent", r"\textbf{(c)} Newton's method"]
for (ax, method) in zip(axs[0,:], methods):
    ax.set(title=method)

for ax in axs.flatten():
    ax.set(xlabel=r"${x}_1$", ylabel=r"${x}_2$")
    ax.xaxis.labelpad = -5
    ax.yaxis.labelpad = -5

ax = axs[1,0]
ax.annotate(r"\textit{Algorithm progression}", xy=(-0.5, -0.4), xycoords='axes fraction', xytext=(-0.4, 0.5), ha='center', va='center', rotation=90)# arrowprops=dict(arrowstyle="->", color='k'))
ax.annotate('', xy=(-0.3, 2.0), xycoords='axes fraction', xytext=(-0.3, -1.0), arrowprops=dict(arrowstyle="<-", color='k'))

#%% save plot
c.save_figure(fig=fig, filename=None)