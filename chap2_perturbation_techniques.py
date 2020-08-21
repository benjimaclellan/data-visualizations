
# %% add ASOPE packages
import sys
sys.path.append("C:/Users/benjamin/Documents/INRS - Projects/asope")

#%%
from configstyle import ConfigStyle
style = ConfigStyle()

# %%
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
import lhsmdu
import sobol_seq

# if True:
#     np.random.seed(0)

plt.close('all')
c = ConfigStyle()

def func(x):
    return (1-0.7*np.exp(-0.5 * (1.0 * (0.2 + x[0])**2 + 10.0 * (0.1 - x[1])**2)) - 0.3* (np.cos(2*x[0]) * np.cos(x[1])) )
    # return 1-np.exp(-0.5 * (1.0 * (0.2 + x[0])**2 + 10.0 * (0.1 - x[1])**2))

grad = elementwise_grad(func)
hess = hessian(func)

n = 100
x1, x2 = np.linspace(-1, 1, n), np.linspace(-1, 1, n)
f = np.zeros([n,n])
for m in range(n):
    for j in range(n):
        f[m, j] = -func(np.array([x1[m], x2[j]]))
f = f.T
maxf, minf = None, None

fig, axs = plt.subplots(4, 3, figsize=[c.width_page, c.height_page/1.5], gridspec_kw={'wspace':0.6, 'hspace':0.5})
pmesh_style = {'cmap': c.cmap, 'rasterized': True}
marker_style = {'marker':'.',  'zorder':4, 'alpha': 0.3}
arrow_style = {'color': c.cp[4], 'width':0.019, 'head_width':0.055, 'length_includes_head':False, 'shape':'full'}
line_style = { 'ls':'-', 'lw':1, 'color':c.cp[2], 'alpha':0.6}
contour_style = {'colors': 'grey', 'alpha': 0.3}
bar_style = {'color':c.cp[0:2], 'alpha':0.6}
for ax in axs.flatten():
    ax.set(xticklabels=[], yticklabels=[])
    ax.grid(False)

#%% one-at-a-time MC
n_samples = 1000
n_plot_samples = 250
samples_x1 = [np.array([np.random.normal(0.0, 0.4), 0.0]) for i in range(n_samples//2)]
samples_x2 = [np.array([0.0, np.random.normal(0.0, 0.4)]) for i in range(n_samples//2)]
scores_x1 = list(map(func, samples_x1))
scores_x2 = list(map(func, samples_x2))

# x1 vs x2
axs[0, 0].pcolormesh(x1, x2, f, **pmesh_style)
axs[0, 0].contour(x1, x2, f, levels=5, **contour_style)
for (xm, score) in zip(samples_x1, scores_x1):
    if np.random.uniform() < n_plot_samples/n_samples:
        axs[0, 0].scatter(xm[0], xm[1], color=c.cp[0], **marker_style)
for (xm, score) in zip(samples_x2, scores_x2):
    if np.random.uniform() < n_plot_samples/n_samples:
        axs[0, 0].scatter(xm[0], xm[1], color=c.cp[1], **marker_style)
axs[0, 0].set(xlabel=r"$x_1$", ylabel=r"$x_2$",
              xlim=[-1, 1], ylim=[-1, 1], aspect='equal')

# x1 axis
for (xm, score) in zip(samples_x1, scores_x1):
    if np.random.uniform() < n_plot_samples / n_samples:
        axs[1,0].scatter(xm[0], score, color=c.cp[0], **marker_style)
axs[1,0].set(xlim=[-1,1], ylim=[minf, maxf], xlabel=r"$x_1$", ylabel=r"$F(x_1, x_2=0)$")

# x2 axis
for (xm, score) in zip(samples_x2, scores_x2):
    if np.random.uniform() < n_plot_samples / n_samples:
        axs[2,0].scatter(xm[1], score, color=c.cp[1], **marker_style)
axs[2,0].set(xlim=[-1,1], ylim=[minf, maxf], xlabel=r"$x_2$", ylabel=r"$F(x_1=0, x_2)$")

# relative sensitivity
axs[3,0].bar([0,1], [np.var(scores_x1), np.var(scores_x2)],  **bar_style)
axs[3,0].set(xticks=[0,1], xticklabels=[r'$x_1$', r'$x_2$'], ylabel=r"Sensitivity: $\mathrm{Var}[F]$")


#%% full MC: first order sensitivity coefficient
#% method taken from http://www.andreasaltelli.eu/file/repository/PUBLISHED_PAPER.pdf
k = 2
N = 100
def sobol(k, N):
    return 2 * sobol_seq.i4_sobol_generate(k, N) - 1
samp = sobol(2*k, N)

A = samp[:,0:2]
B = samp[:,2:4]
VX = np.array([None, None])

stack = np.vstack([A, B])
VY = np.var([func(stack[i,:]) for i in range(2*N)])

Ey = np.mean([func(stack[i,:]) for i in range(2*N)])
f0 = Ey

for i in range(k):
    AiB = copy.deepcopy(A)
    AiB[:,i] = B[:,i]
    BiA = copy.deepcopy(B)
    BiA[:,i] = A[:,i]

    VXij = 0
    for j in range(N):
        VXij += func(A[j,:]) * func(BiA[j,:])
    VXi = VXij/N - f0**2
    VX[i] = VXi
Si = VX/VY # sensitivity coefficient

n_samples = 60
samples = sobol(k, N).T

# plot x1 vs x2 with F(x) as color
axs[0, 1].pcolormesh(x1, x2, f, **pmesh_style)
axs[0, 1].contour(x1, x2, f, levels=5, **contour_style)
for i in range(n_samples):
    xm = samples[:,i]
    axs[0, 1].scatter(xm[0], xm[1], color=c.cp[3], **marker_style)
axs[0, 1].set(xlabel=r"$x_1$", ylabel=r"$x_2$",
              xlim=[-1, 1], ylim=[-1, 1], aspect='equal')

# x1 axis
for i in range(n_samples):
    xm = samples[:,i]
    score = func(xm)
    axs[1,1].scatter(xm[0], score, color=c.cp[0], **marker_style)
axs[1,1].set(xlim=[-1,1], ylim=[minf, maxf], xlabel=r"$x_1$", ylabel=r"$F(x_1, x_2\sim \mathcal{U}[-1,1])$")

# x2 axis
for i in range(n_samples):
    xm = samples[:,i]
    score = func(xm)
    if np.random.uniform() < n_plot_samples / n_samples:
        axs[2,1].scatter(xm[1], score, color=c.cp[1], **marker_style)
axs[2,1].set(xlim=[-1,1], ylim=[minf, maxf], xlabel=r"$x_2$", ylabel=r"$F(x_1 \sim \mathcal{U}[-1,1], x_2)$")

# relative sensitivity
axs[3,1].bar([0,1], Si, **bar_style)
axs[3,1].set(xticks=[0,1], xticklabels=[r'$x_1$', r'$x_2$'], ylabel=r"Sensitivity: $S_i$")


#%% gradient
x0 = np.array([0.0, 0.0])

axs[0, 2].pcolormesh(x1, x2, f, **pmesh_style)
axs[0, 2].contour(x1, x2, f, levels=5, **contour_style)

axs[0, 2].scatter(x0[0], x0[1], **marker_style)
axs[0, 2].arrow(x0[0], x0[1], grad(x0)[0], 0.0, **arrow_style)
axs[0, 2].arrow(x0[0], x0[1], 0.0, grad(x0)[1], **arrow_style)
axs[0, 2].set(xlabel=r"$x_1$", ylabel=r"$x_2$",
              xlim=[-1, 1], ylim=[-1, 1], aspect='equal')
# x1 axis
diff = 0.15
x1_tmp = np.linspace(-1, 1, 200)
axs[1,2].plot(x1_tmp, list(map(func, [np.array([tmp, 0.0]) for tmp in x1_tmp])), **line_style)
axs[1,2,].plot([x0[0]- diff, x0[0]+diff], [func(x0) - diff * grad(x0)[0], func(x0) + diff * grad(x0)[0]], color=c.cp[0])
axs[1,2].set(xlim=[-1,1], ylim=[minf, maxf], xlabel=r"$x_1$", ylabel=r"$F(x_1, x_2=0)$")

# x2 axis
x2_tmp = np.linspace(-1, 1, 100)
axs[2,2].plot(x2_tmp, list(map(func,  [np.array([0.0, tmp]) for tmp in x2_tmp])), **line_style)
axs[2,2,].plot([x0[1]- diff, x0[1]+diff], [func(x0) - diff * grad(x0)[1], func(x0) + diff * grad(x0)[1]], color=c.cp[1])
axs[2,2].set(xlim=[-1,1], ylim=[minf, maxf], xlabel=r"$x_2$", ylabel=r"$F(x_1=0, x_2)$")
# axs[2,2].annotate(r"$ \frac{\partial F}{\partial x_2} $", xy=[0, func(x0)])

# relative sensitivity
axs[3,2].bar([0,1], np.abs(grad(x0)), **bar_style)
axs[3,2].set(xticks=[0,1], xticklabels=[r'$x_1$', r'$x_2$'], ylabel=r"Sensitivity: $\left\vert \displaystyle\frac{\partial F}{\partial x_j} \right\vert$")


#%% aesthetics
methods = [r"\textbf{(a)} Individual parameter", r"\textbf{(b)} Variance-based", r"\textbf{(c)} Gradient based"]
for (ax, method) in zip(axs[0,:], methods):
    ax.set(title=method)
for ax in axs.flatten():
    ax.xaxis.labelpad = -5
    ax.yaxis.labelpad = -5
for ax in axs[1:3,:].flatten():
    ax.set(xlim=[-1,1], ylim=[-0.1,1.1])
# for ax in axs[:, 0:2].flatten():
#     ax.set(xticks=[-1,1])
ax = axs[1,0]
ax.annotate(r"\textit{Steps in sensitivity analysis technique}", xy=(-0.5, -0.9), xycoords='axes fraction', xytext=(-0.4, 0.0), ha='center', va='center', rotation=90)# arrowprops=dict(arrowstyle="->", color='k'))
ax.annotate('', xy=(-0.3, 2.5), xycoords='axes fraction', xytext=(-0.3, -2.9), arrowprops=dict(arrowstyle="<-", color='k'))


#%% save plot
c.save_figure(fig=fig, filename=None)
