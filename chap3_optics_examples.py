
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
node_style = {'radius': 0.05}

Y_CENTER = 0.5
HEIGHT = 0.3

#%% lens design
parameters = {'r1': 1, 'r2': -1, 'r3': 1, 'x1': 1, 'x2': 1.3}


#%% aesthetics
for ax in axs:
    ax.set(xlim=[0, 1], ylim=[0, 1], xticks=[], yticks=[])
    ax.axis('equal')
    ax.grid(False)
    # ax.axis('off')

subfigures = [r"\textbf{(a) Lens system design}", r"\textbf{(b) Nanophotonic device}", ]
for (ax, subfigure) in zip(axs, subfigures):
    ax.set(title=subfigure)

#%% save plot
# if save_plots: c.save_figure(fig=fig, filename='evolution_ops1')

plt.show()