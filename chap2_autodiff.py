
# %% add ASOPE packages
import sys
sys.path.append("C:/Users/benjamin/Documents/INRS - Projects/asope")

#%%
from configstyle import ConfigStyle
style = ConfigStyle()

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
import autograd.scipy as sp
from autograd import elementwise_grad, grad, jacobian, hessian

plt.close('all')
c = ConfigStyle()

def func(x):
    return np.power(np.exp(np.cos(x)), 2)

n = 1000
x = np.linspace(-10, 10, n)

fig, axs = plt.subplots(4, 1, figsize=[c.width_page, c.height_page/1.5])
arrow_style = {'color': c.cp[4], 'width':0.01, 'head_width':0.05, 'length_includes_head':True, 'head_length': 0.01, 'shape':'full'}
frarrow_style = {'color': c.cp[3], 'width':0.01, 'head_width':0.05, 'length_includes_head':True, 'head_length': 0.01}
mode_style = {'color': c.cp[3], 'width':0.012, 'head_width':0.07, 'length_includes_head':True, 'head_length': 0.02, 'shape':'full'}

line_style = {'lw':1}

ax = axs[3]
ax.plot(x, func(x), color=c.cp[0], label=r"$F(x)$", alpha=0.9, ls='-', **line_style)
ax.plot(x, elementwise_grad(func)(x), color=c.cp[1], alpha=0.6, ls='--', label=r"$\displaystyle\frac{\mathrm{d}F(x)}{\mathrm{d}x}$", **line_style)
# ax.plot(x, elementwise_grad(elementwise_grad(func))(x), color=c.cp[3], alpha=0.6, ls='--', label=r"$F''(x)$", **line_style)

# to double check
w0, w0d = x, np.ones_like(x)
w1, w1d = np.cos(w0), -np.sin(w0) * w0d
w2, w2d = np.exp(w1), np.exp(w1) * w1d
w3, w3d = np.power(w2,2), 2 * w2 * w2d
Fd = w3d
Fd_analytic = -2*np.sin(x) * np.exp(2*np.cos(x))
# ax.plot(x, Fd_analytic, color=c.cp[3], alpha=0.6, ls='--', label=r"$F''(x)$", **line_style)


ax = axs[3]
ax.set(xlabel=r"$x$")
ax.legend()


#%% computational graph
ax = axs[0]
ax.axis('equal')
transform = ax.transAxes
ax.set(xlim=[0,1])
radius = 0.04
text_diff = 0.3

def add_node(ax, xy, text, color_index=5):
    circle = Circle(xy, radius=radius, alpha=0.9, color=c.cp[color_index], zorder=3)
    ax.add_patch(circle)
    ax.annotate(text, xy=xy, xycoords='axes fraction', fontsize=10, ha="center", va="center")
    return
def add_edge(ax, xy1, xy2, text, updown=+1):
    arrow = FancyArrow(xy1[0]+radius, xy1[1], xy2[0]-xy1[0]-2*radius, xy2[1]-xy1[1], transform=transform, **arrow_style)
    ax.add_patch(arrow)
    ax.annotate(text, xy=((xy2[0]+xy1[0])/2, 0.5 + updown * text_diff), xycoords='axes fraction', fontsize=9, ha="center", va="center", rotation=0)
    return
add_node(ax=ax, xy=(0.1,0.5), text=r"$\texttt{x}$", color_index=1)
add_edge(ax=ax, xy1=(0.1,0.5), xy2=(0.3,0.5), text=r"$w_0 = x$", updown=+1)

add_node(ax=ax, xy=(0.3,0.5), text=r"$\texttt{cos}$")
add_edge(ax=ax, xy1=(0.3,0.5), xy2=(0.5,0.5), text=r"$w_1 = \mathrm{cos}(x)$", updown=-1)

add_node(ax=ax, xy=(0.5,0.5), text=r"$\texttt{exp}$")
add_edge(ax=ax, xy1=(0.5,0.5), xy2=(0.7,0.5), text=r"$w_2 = \exp(\cos(x))$", updown=+1)

add_node(ax=ax, xy=(0.7,0.5), text=r"$^\texttt{**}\texttt{2}$")
add_edge(ax=ax, xy1=(0.7,0.5), xy2=(0.9,0.5), text=r"$w_3 = \exp(\cos(x))^2$", updown=-1)

add_node(ax=ax, xy=(0.9,0.5), text=r"$\texttt{F(x)}$", color_index=1)

ax.set(xticks=[], yticks=[], ylabel=r"Computational graph",  facecolor='lightgrey')


#%% forward mode
ax = axs[1]
ax.axis('equal')
transform = ax.transAxes
ax.set(xlim=[0,1])
radius = 0.035
text_diff = 0.33

def add_node(ax, xy, text, color_index=5):
    circle = Circle(xy, radius=radius, alpha=0.9, color=c.cp[color_index], zorder=3)
    ax.add_patch(circle)
    ax.annotate(text, xy=xy, xycoords='axes fraction', fontsize=10, ha="center", va="center")
    return
def add_edge(ax, xy1, xy2, text, updown=+1):
    arrow = FancyArrow(xy1[0]+radius, xy1[1], xy2[0]-xy1[0]-2*radius, xy2[1]-xy1[1], transform=transform, **arrow_style)
    ax.add_patch(arrow)

    shape = 'left' if updown == -1 else 'right'
    arrow = FancyArrow(xy1[0]+radius, xy1[1] + updown * 0.1, xy2[0]-xy1[0]-2*radius, 0.0, transform=transform, shape=shape, **frarrow_style)
    ax.add_patch(arrow)
    ax.annotate(text, xy=((xy2[0]+xy1[0])/2, 0.5 + updown * text_diff), xycoords='axes fraction', fontsize=9, ha="center", va="center", rotation=0, color=c.cp[3])
    return
add_node(ax=ax, xy=(0.1,0.5), text=r"$\texttt{x}$", color_index=1)
add_edge(ax=ax, xy1=(0.1,0.5), xy2=(0.3,0.5), text=r"$\displaystyle\frac{\mathrm{d}w_0}{\mathrm{d}x}=1$", updown=+1)

add_node(ax=ax, xy=(0.3,0.5), text=r"$\texttt{cos}$")
add_edge(ax=ax, xy1=(0.3,0.5), xy2=(0.5,0.5), text=r"$\displaystyle\frac{\mathrm{d}w_1}{\mathrm{d}w_0} = -\mathrm{sin}(w_0)\displaystyle\frac{\mathrm{d}w_0}{\mathrm{d}x}$", updown=-1)

add_node(ax=ax, xy=(0.5,0.5), text=r"$\texttt{exp}$")
add_edge(ax=ax, xy1=(0.5,0.5), xy2=(0.7,0.5), text=r"$\displaystyle\frac{\mathrm{d}w_2}{\mathrm{d}w_1}=\mathrm{exp}(w_1) \displaystyle\frac{\mathrm{d}w_1}{\mathrm{d}w_0}$", updown=+1)

add_node(ax=ax, xy=(0.7,0.5), text=r"$^\texttt{**}\texttt{2}$")
add_edge(ax=ax, xy1=(0.7,0.5), xy2=(0.9,0.5), text=r"$\displaystyle\frac{\mathrm{d}w_3}{\mathrm{d}w_2}= 2 w_2 \displaystyle\frac{\mathrm{d}w_2}{\mathrm{d}w_1}$", updown=-1)

add_node(ax=ax, xy=(0.9,0.5), text=r"$\texttt{F(x)}$", color_index=1)

arrow = FancyArrow(0.05, 0.05, 0.9, 0.0, transform=transform, **mode_style)
ax.add_patch(arrow)

ax.set(xticks=[], yticks=[], ylabel=r"Forward mode")

#%% reverse mode
ax = axs[2]
ax.axis('equal')
transform = ax.transAxes
ax.set(xlim=[0,1])
def add_node(ax, xy, text, color_index=5):
    circle = Circle(xy, radius=radius, alpha=0.9, color=c.cp[color_index], zorder=3)
    ax.add_patch(circle)
    ax.annotate(text, xy=xy, xycoords='axes fraction', fontsize=10, ha="center", va="center")
    return
def add_edge(ax, xy1, xy2, text, updown=+1):
    arrow = FancyArrow(xy1[0]+radius, xy1[1], xy2[0]-xy1[0]-2*radius, xy2[1]-xy1[1], transform=transform, **arrow_style)
    ax.add_patch(arrow)

    shape = 'left' if updown == +1 else 'right'
    arrow = FancyArrow(xy2[0]-radius, xy1[1] + updown * 0.1, -(xy2[0]-xy1[0]-2*radius), 0.0, transform=transform, shape=shape, **frarrow_style)
    ax.add_patch(arrow)
    ax.annotate(text, xy=((xy2[0]+xy1[0])/2, 0.5 + updown * text_diff), xycoords='axes fraction', fontsize=9, ha="center", va="center", rotation=0, color=c.cp[3])
    return
add_node(ax=ax, xy=(0.1,0.5), text=r"$\texttt{x}$", color_index=1)
add_edge(ax=ax, xy1=(0.1,0.5), xy2=(0.3,0.5), text=r"$\displaystyle\frac{\mathrm{d}w_1}{\mathrm{d}w_0}= - \displaystyle\frac{\mathrm{d}w_2}{\mathrm{d}w_1} \mathrm{sin}(w_0)$", updown=+1)

add_node(ax=ax, xy=(0.3,0.5), text=r"$\texttt{cos}$")
add_edge(ax=ax, xy1=(0.3,0.5), xy2=(0.5,0.5), text=r"$ \displaystyle\frac{\mathrm{d}w_2}{\mathrm{d}w_1} = \displaystyle\frac{\mathrm{d}w_3}{\mathrm{d}w_2} \mathrm{exp}(w_1)$", updown=-1)

add_node(ax=ax, xy=(0.5,0.5), text=r"$\texttt{exp}$")
add_edge(ax=ax, xy1=(0.5,0.5), xy2=(0.7,0.5), text=r"$\displaystyle\frac{\mathrm{d}w_3}{\mathrm{d}w_2}= 2 w_2 \displaystyle\frac{\mathrm{d}F(x)}{\mathrm{d}w_3}$", updown=+1)

add_node(ax=ax, xy=(0.7,0.5), text=r"$^\texttt{**}\texttt{2}$")
add_edge(ax=ax, xy1=(0.7,0.5), xy2=(0.9,0.5), text=r"$\displaystyle\frac{\mathrm{d}F(x)}{\mathrm{d}w_3}= 1$", updown=-1)

add_node(ax=ax, xy=(0.9,0.5), text=r"$\texttt{F(x)}$", color_index=1)

arrow = FancyArrow(0.95, 0.05, -0.9, 0.0, transform=transform, **mode_style)
ax.add_patch(arrow)

ax.set(xticks=[], yticks=[], ylabel=r"Reverse mode")



#%% aesthetics
for ax in axs:
    ax.grid(False)

labels = [r"\textbf{(a)}", r"\textbf{(b)}", r"\textbf{(c)}", r"\textbf{(d)}"]
for (ax, label) in zip(axs, labels):
    ax.annotate(label, xy=(0.0, 0.5), xytext=(-0.07, 0.5), xycoords='axes fraction', fontsize=10,
                ha="center", va="center", zorder=10)

#%% save plot
c.save_figure(fig=fig, filename=None)
