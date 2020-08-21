
# %% add ASOPE packages
import sys
sys.path.append("C:/Users/benjamin/Documents/INRS - Projects/asope")

#%%
from configstyle import ConfigStyle

# %%
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrow, Ellipse
import matplotlib.gridspec as gridspec
import seaborn as sns
import networkx as nx
import itertools
import random
import os
import copy
import time
import autograd.numpy as np
from numpy.linalg import eig
import autograd.scipy as sp
from autograd import elementwise_grad, grad, jacobian, hessian

plt.close('all')
c = ConfigStyle()
save = True

fig, axs = plt.subplots(2, 2, figsize=[c.width_page, c.height_page/2], gridspec_kw={'wspace':0.2, 'hspace':0.6})
pmesh_style = {'cmap': c.cmap, 'rasterized': True}
xmin_style = {'marker':'o', 'color':c.cp[3], 's':15, 'zorder':4, 'alpha': 1.0}
arrow_style = {'color': c.cp[4], 'width':0.1, 'head_width':0.3, 'length_includes_head':False, 'shape':'full', 'zorder':6}
contour_style = {'linestyles':'-', 'colors': 'grey', 'alpha': 0.3, 'levels':8}
ARROW_SCALE = 3

def booth(x):
    return (x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2  #booth
def matyas(x):
    return 0.26*(x[0]**2 + x[1]**2) - 0.48*x[0]*x[1]  #matyas
def rosenbrock(x):
    return 100*(x[1] - x[0]**2)**2 + (1-x[0])**2  #rosenbrock
def himmelblaus(x):
    return ((x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2)  #himmelblaus
def ackley(x):
    return -20*np.exp( -0.2 * np.sqrt(0.5*(x[0]**2 + x[1]**2))) - np.exp( 0.5*(np.cos(2*np.pi*x[0]) + np.cos(2*np.pi*x[1]))) + np.exp(1) + 20

#**************************************************************
# example function 1
func = matyas
grad = elementwise_grad(func)
hess = hessian(func)

p1 = (0.0, 0.0)
ps = (p1,)

n = 100
x1, x2 = np.linspace(-5, 5, n), np.linspace(-5, 5, n)
f = np.zeros([n,n])
for i in range(n):
    for j in range(n):
        f[i,j] = np.log10(func(np.array([x1[i], x2[j]])))
f = f.T

#%%
ax = axs[0,0]
ax.set(xlabel=r"$x_1$", ylabel=r"$x_2$", aspect='equal')

ax.pcolormesh(x1, x2, f, **pmesh_style)
ax.contour(x1, x2, f, **contour_style)
for p in ps:
    H = hess(np.array(p))
    H = H/np.sum(np.diag(H))
    ax.scatter(p[0], p[1], **xmin_style)
    ax.arrow(p[0], p[1], ARROW_SCALE*H[0,0], 0, **arrow_style)
    ax.arrow(p[0], p[1], 0, ARROW_SCALE*H[1,1], **arrow_style)

#%%
ax = axs[1,0]
ax.set(xlabel=r"$x_1$", ylabel=r"$x_2$", aspect='equal')

ax.pcolormesh(x1, x2, f, **pmesh_style)
ax.contour(x1, x2, f, **contour_style)
for p in ps:
    H = hess(np.array(p))
    H = H/np.sum(np.diag(H))
    ax.scatter(p[0], p[1], **xmin_style)
    eigvals, eigvecs = eig(H)
    for i in range(2):
        eigval, eigvec = eigvals[i], eigvecs[:,i]
        ax.arrow(p[0], p[1], ARROW_SCALE*eigval*eigvec[0], ARROW_SCALE*eigval*eigvec[1], **arrow_style)

#********************************************************
# second example function
func = rosenbrock
grad = elementwise_grad(func)
hess = hessian(func)

p1 = (1.0, 1.0)
ps = (p1,)

n = 100
x1, x2 = np.linspace(-5, 5, n), np.linspace(-5, 5, n)
f = np.zeros([n,n])
for m in range(n):
    for j in range(n):
        f[m,j] = np.log10(func(np.array([x1[m], x2[j]])))
f = f.T

#%%
ax = axs[0,1]
ax.set(xlabel=r"$x_1$", ylabel=r"$x_2$", aspect='equal')

ax.pcolormesh(x1, x2, f, **pmesh_style)
ax.contour(x1, x2, f, **contour_style)
for p in ps:
    H = hess(np.array(p))
    H = H/np.sum(np.diag(H))
    ax.scatter(p[0], p[1], **xmin_style)
    ax.arrow(p[0], p[1], ARROW_SCALE*H[0,0], 0, **arrow_style)
    ax.arrow(p[0], p[1], 0, ARROW_SCALE*H[1,1], **arrow_style)

#%%
ax = axs[1,1]
ax.set(xlabel=r"$x_1$", ylabel=r"$x_2$", aspect='equal')

ax.pcolormesh(x1, x2, f, **pmesh_style)
ax.contour(x1, x2, f, **contour_style)
for p in ps:
    H = hess(np.array(p))
    H = H/np.sum(np.diag(H))
    ax.scatter(p[0], p[1], **xmin_style)
    eigvals, eigvecs = eig(H)
    for i in range(2):
        eigval, eigvec = eigvals[i], eigvecs[:,i]
        ax.arrow(p[0], p[1], ARROW_SCALE*eigval*eigvec[0], ARROW_SCALE*eigval*eigvec[1], **arrow_style)

#%% aesthetics
for ax in axs.flatten():
    ax.grid(False)

labels = [r"\textbf{(a)} "+"Matya function\nParameter curvatures",
          r"\textbf{(c)} "+"Rosenbrock function\nParameter curvatures",
          r"\textbf{(b)} "+"Matya function\nPrinciple curvatures",
          r"\textbf{(d)} "+"Rosenbrock function\nPrinciple curvatures", ]
for (ax, label) in zip(axs.flatten(), labels):
    ax.locator_params(axis='y', nbins=2)
    ax.locator_params(axis='x', nbins=2)
    ax.set(title=label)

#%% save plot
if save: c.save_figure(fig=fig, filename='twodimensional')

#-----------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------

#%%

fig, axs = plt.subplots(1, 3, figsize=[c.width_page, c.height_page/3], gridspec_kw={'wspace':0.4, 'hspace':0.0})
arrow_style = {'color': c.cp[4], 'width':0.1, 'head_width':0.3, 'length_includes_head':False, 'shape':'full', 'zorder':6}
stem_style = {'linefmt':'-', 'markerfmt':'D', 'use_line_collection':True}
cmap = plt.get_cmap("coolwarm")

m = 5
def rosenbrock_m(x):
    f = 0
    for i in range(0, m-1):
        f += 100*(x[i+1] - x[i]**2)**2 + (1-x[i])**2
    return f

def rastrigin_m(x):
    f = 10*m
    for i in range(0, m):
        f += x[i]**2 - 10*np.cos(2*np.pi*x[i])
    return f

func = rosenbrock_m
p = np.ones(m)

grad = elementwise_grad(func)
hess = hessian(func)

H = hess(np.array(p))
eigvals, eigvecs = eig(H)

#%% plot Hessian as heatmap
ax = axs[0]
sns.heatmap(ax=ax, data=H, vmin=np.min(H), vmax=np.max(H),
            cmap=cmap, square=True, annot=False, fmt="1.1f", linewidths=0.5,
            cbar_kws = dict(use_gridspec=False,location="bottom",ticks=[np.min(H),np.max(H)]))
ax.set_xticklabels([r"$\partial x_{"+str(i)+r"}$" for i in range(m)], ha='center', va='center')
ax.set_yticklabels([r"$\partial x_{"+str(i)+r"}$" for i in range(m)], ha='center', va='center')

# plot parameter curvature (Hessian diagonal)
ax = axs[1]
markerline, stemlines, baseline = ax.stem(np.diag(H), **stem_style)
plt.setp(stemlines, 'color', cmap(0.0), 'linestyle', '-')
plt.setp(markerline, 'markerfacecolor', cmap(0.0), 'linestyle', '')
plt.setp(baseline, 'linestyle', '--', 'color',c.cp[4])

ax.set(xticks=[i for i in range(m)],
       xticklabels=[r"$H_{"+"{},{}".format(i,i)+r"}$" for i in range(m)],
       xlabel='Hessian diagonal values')

# plot principle directions
ax = axs[2]
color_scale = lambda eigval: cmap(eigval/(np.max(eigvals)-np.min(eigvals)))
for i in range(m):
    eigval, eigvec = eigvals[i], eigvecs[:,i]
    baseline = 2*(m-1)-2*i
    markerline, stemlines, baseline = ax.stem(eigvec + baseline, bottom=baseline, **stem_style)
    plt.setp(stemlines, 'color', color_scale(eigval), 'linestyle', '-')
    plt.setp(markerline, 'markerfacecolor', color_scale(eigval), 'linestyle', '')
    plt.setp(baseline, 'linestyle', '--', 'color', c.cp[4])

    lam = r"$\lambda_\mathrm{"+"{}".format(i)+r"}$ = "+"{:1.0f}".format(eigval)
    ax.annotate(lam, xy=(1.05, (2*m-2*i-1)/(2*m)), xycoords='axes fraction', fontsize=10, ha="left", va="center")

ax.set(xticks=[i for i in range(m)], xticklabels=[r"$x_{"+"{}".format(i)+r"}$" for i in range(m)],
       yticks=[2*i for i in range(m)], yticklabels=[r"$\mathrm{\mathbf{v}}_\mathrm{"+"{}".format(m-i-1)+r"}$" for i in range(m)],
       ylim=[-1, -1+2*m],
       xlabel="Principle direction, $\mathrm{\mathbf{v}}_\mathrm{k}$")

#aesthetics
labels = [r"\textbf{(a)} Hessian matrix", r"\textbf{(b)} Parameter curvatures", r"\textbf{(c)} Principle directions/curvatures"]
for (ax, label) in zip(axs, labels):
    ax.set(title=label)
for ax in axs[1:]:
    ax.grid(False)

#%% save plot
plt.show()
if save: c.save_figure(fig, "multiparameters")

