
# %% add ASOPE packages
import sys
sys.path.append("C:/Users/benjamin/Documents/INRS - Projects/asope")

#%%
from configstyle import ConfigStyle
style = ConfigStyle()

# %%
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrow, Ellipse, Polygon
import matplotlib.gridspec as gridspec
import seaborn as sns
import networkx as nx
import itertools
import random
import os
import copy
import numpy as np
from scipy.special import binom

plt.close('all')
c = ConfigStyle()
plt.rc('text.latex', preamble=r'\usepackage{amsmath} \renewcommand{\vec}[1]{\mathbf{#1}}')

save_plots = True
fig, axs = plt.subplots(1, 2, figsize=[c.width_page, c.height_page/3])

arrow_style = {'color': c.cp[4], 'width':0.005, 'head_width':0.015, 'length_includes_head':True, 'head_length': 0.01, 'shape':'full'}
node_style = {'radius': 0.065}
TEXT_XSEP, TEXT_YSEP = (0.06, 0.00)
TRI_ROT = 3 * np.pi / 2
TRI_LEN = 0.03
COLOR_DEFAULT = 'lightgrey' #c.cp[4]

#%% define plotting functions
bernstein = lambda n, k, t: binom(n,k)* t**k * (1.-t)**(n-k)

def bezier(points, num=200):
    N = len(points)
    t = np.linspace(0, 1, num=num)
    curve = np.zeros((num, 2))
    for i in range(N):
        curve += np.outer(bernstein(N - 1, i, t), points[i])
    return curve

def make_node(id, params, func, pos, color=None, alpha=1.0):
    return {'id': id, 'params': params, 'func': func, 'pos': pos, 'color': color, 'alpha':alpha}

def plot_node(ax, node):
    color = node['color'] if node['color'] is not None else COLOR_DEFAULT
    circle = Circle(node['pos'], color=color, alpha=node['alpha'], zorder=3, **node_style)
    ax.add_patch(circle)
    text_str = node['params'] + "\n" + node['func']
    ax.annotate(node['id'], xy=node['pos'], xycoords='data', fontsize=10, ha="center", va="center")
    ax.annotate(text_str, xy=(node['pos'][0] + TEXT_XSEP, node['pos'][1] + TEXT_YSEP), xycoords='data', fontsize=10, ha="left", va="center")
    return

def plot_nodes(ax, nodes):
    for id, node in nodes.items():
        plot_node(ax, node)
    return

def plot_arrow(ax, pos1, pos2):
    x1, y1 = pos1
    x2, y2 = pos2
    arrow = FancyArrow(x1, y1, x2-x1, y2-y1, **arrow_style)
    ax.add_patch(arrow)
    return

def plot_bezier(ax, pos1, pos2):
    x1, y1 = pos1
    x2, y2 = pos2
    xmid, ymid = (x2 + x1)/2, (y2 + y1)/2
    nodes1 = np.array([[x1, y1], [x1, ymid],
                       [(3*x1 + x2)/4, ymid],
                       [xmid, ymid],
                       [(x1 + 3*x2) / 4, ymid],
                       [x2, ymid], [x2, y2]])
    curve1 = bezier(nodes1, num=256)
    ax.plot(curve1[:, 0], curve1[:, 1], ls='-', color=c.cp[4])
    return

def plot_triangle(ax, pos):
    rc = np.sqrt(3) / 3 * TRI_LEN  # radius of circle touching the tri points
    points = np.array([[pos[0] + rc * np.cos(TRI_ROT + 0 * np.pi / 3), pos[1] + rc * np.sin(TRI_ROT + 0 * np.pi / 3)],
                       [pos[0] + rc * np.cos(TRI_ROT + 2 * np.pi / 3), pos[1] + rc * np.sin(TRI_ROT + 2 * np.pi / 3)],
                       [pos[0] + rc * np.cos(TRI_ROT + 4 * np.pi / 3), pos[1] + rc * np.sin(TRI_ROT + 4 * np.pi / 3)]])
    poly = Polygon(points, color=c.cp[4], zorder=6)
    ax.add_patch(poly)
    return

def plot_edges(ax, nodes, connections):

    for connection in connections:
        xy1, xy2 = nodes[connection[0]]['pos'], nodes[connection[1]]['pos']
        radius = node_style['radius']
        angle = np.arctan2(xy2[1] - xy1[1], xy2[0] - xy1[0])

        # x1, y1 = xy1[0] + radius * np.cos(angle), xy1[1] + radius * np.sin(angle)
        # x2, y2 = xy2[0] - radius * np.cos(angle), xy2[1] - radius * np.sin(angle)
        # plot_arrow(ax, (x1, y1), (x2, y2))

        x1, y1 = xy1[0], xy1[1] + radius * np.sign(xy2[1]-xy1[1])
        x2, y2 = xy2[0], xy2[1] - radius * np.sign(xy2[1]-xy1[1])
        plot_bezier(ax, (x1, y1), (x2, y2))
        plot_triangle(ax, (x2,  y2))

    return

#%%
nodes = {}
nodes[1] = make_node(id=r"CW", params=r"$\vec{x}_\mathrm{CW}$", func=r"$f_\mathrm{CW} (\Psi, \vec{x}_\mathrm{CW})$", pos=(0.2, 0.8))
nodes[2] = make_node(id=r"PM", params=r"$\vec{x}_\mathrm{PM}$", func=r"$f_\mathrm{PM} (\Psi, \vec{x}_\mathrm{PM})$", pos=(0.2, 0.5))
nodes[3] = make_node(id=r"PD", params=r"$\vec{x}_\mathrm{PD}$", func=r"$f_\mathrm{PD} (\Psi, \vec{x}_\mathrm{PD})$", pos=(0.2, 0.2))

connections = ((1, 2), (2, 3))

ax = axs[0]
plot_nodes(ax, nodes)
plot_edges(ax, nodes, connections)

#%%
nodes = {}
nodes[1] = make_node(id=r"CW", params=r"$\vec{x}_\mathrm{CW}$", func=r"$f_\mathrm{CW}$", pos=(0.2, 0.9))
nodes[2] = make_node(id=r"PM", params=r"$\vec{x}_\mathrm{PM}$", func=r"$f_\mathrm{PM}$", pos=(0.2, 0.5))
nodes[3] = make_node(id=r"PD", params=r"$\vec{x}_\mathrm{PD}$", func=r"$f_\mathrm{PD}$", pos=(0.2, 0.1))
nodes[4] = make_node(id=r"WS", params=r"$\vec{x}_\mathrm{WS}$", func=r"$f_\mathrm{WS}$", pos=(0.6, 0.5))
nodes[5] = make_node(id=r"BS", params=r"$\vec{x}_\mathrm{BS}$", func=r"$f_\mathrm{BS}$", pos=(0.2, 0.7))
nodes[6] = make_node(id=r"BS", params=r"$\vec{x}_\mathrm{BS}$", func=r"$f_\mathrm{BS}$", pos=(0.2, 0.3))

connections = ((1, 5), (5, 2), (2, 6), (5, 4), (4,6), (6,3))

ax = axs[1]
plot_nodes(ax, nodes)
plot_edges(ax, nodes, connections)

#%% aesthetics
for ax in axs:
    ax.set(xlim=[0, 1], ylim=[0, 1], xticks=[], yticks=[])
    ax.axis('equal')

    ax.grid(False)
    ax.axis('off')

subfigures = [r"\textbf{(a)}", r"\textbf{(b)}"]
for (ax, subfigure) in zip(axs, subfigures):
    ax.set(title=subfigure)

#%% save plot
if save_plots: c.save_figure(fig=fig, filename='example_graph')
# plt.close(fig)

##--------------------------------------------------------------------------------------------------
# evolutionary operators figure
##--------------------------------------------------------------------------------------------------
#%%
fig, axs = plt.subplots(1, 4, figsize=[c.width_page, c.height_page/3])

#%% add node
nodes = {}
nodes[1] = make_node(id=r"CW", params=r"$\vec{x}_\mathrm{CW}$", func=r"$f_\mathrm{CW} (\Psi, \vec{x}_\mathrm{CW})$", pos=(0.4, 0.8))
nodes[2] = make_node(id=r"PM", params=r"$\vec{x}_\mathrm{PM}$", func=r"$f_\mathrm{PM} (\Psi, \vec{x}_\mathrm{PM})$", pos=(0.4, 0.6))
nodes[3] = make_node(id=r"EDFA", params=r"$\vec{x}_\mathrm{WS}$", func=r"$f_\mathrm{WS} (\Psi, \vec{x}_\mathrm{WS})$", pos=(0.4, 0.4))
nodes[5] = make_node(id=r"WS", params=r"$\vec{x}_\mathrm{WS}$", func=r"$f_\mathrm{WS} (\Psi, \vec{x}_\mathrm{WS})$", pos=(0.4, 0.4))
nodes[4] = make_node(id=r"PD", params=r"$\vec{x}_\mathrm{PD}$", func=r"$f_\mathrm{PD} (\Psi, \vec{x}_\mathrm{PD})$", pos=(0.4, 0.2))

COLOR_HL = color=c.cp[3]
for id,node in nodes.items():
    node['params'] = r"\phantom{0}"
    node['func'] = r"\phantom{0}"

# starting
ax = axs[0]
nodes_tmp = {id:node for (id, node) in nodes.items() if id not in [3,5]}
plot_nodes(ax, nodes_tmp)
plot_edges(ax, nodes, ((1, 2), (2, 4)))

# add node
ax = axs[1]
nodes_tmp = {id:node for (id, node) in nodes.items() if id not in [5]}
nodes_tmp[3]['color'] = COLOR_HL
plot_nodes(ax, nodes_tmp)
plot_edges(ax, nodes, ((1, 2), (2, 3), (3, 4)))

# swap node
ax = axs[2]
nodes_tmp = {id:node for (id, node) in nodes.items() if id not in [3]}
nodes_tmp[5]['color'] = COLOR_HL
plot_nodes(ax, nodes_tmp)
plot_edges(ax, nodes, ((1, 2), (2,3), (3, 4)))

# remove node
ax = axs[3]
nodes_tmp = {id:node for (id, node) in nodes.items() if id not in [3]}
nodes_tmp[5]['color'] = COLOR_DEFAULT
nodes_tmp[2]['color'] = COLOR_HL
nodes_tmp[2]['alpha'] = 0.2
nodes_tmp[2]['id'] = ""
plot_nodes(ax, nodes_tmp)
plot_edges(ax, nodes, ((1, 3), (3, 4)))

#%% aesthetics
for ax in axs:
    ax.set(xlim=[0, 1], ylim=[0, 1], xticks=[], yticks=[])
    ax.axis('equal')

    ax.grid(False)
    ax.axis('off')

subfigures = [r"Initial graph", r"\textbf{(a)} Add node", r"\textbf{(b)} Swap node", r"\textbf{(c)} Remove node"]
for (ax, subfigure) in zip(axs, subfigures):
    ax.set(title=subfigure)

#%% save plot
if save_plots: c.save_figure(fig=fig, filename='evolution_ops1')

plt.show()