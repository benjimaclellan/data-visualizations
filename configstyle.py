import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cycler
from matplotlib import cm
import seaborn as sns
import os
import sys

class ConfigStyle(object):
    
    def __init__(self):
        plt.style.use(r"thesis-style.mplstyle")

        self.cp = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]

        self.cmap = cm.get_cmap('summer')

        # these are in inches
        self.width_page = 8.5 - 0.8 * 2
        self.height_page = 11.5 - 0.8 * 2
        
        self.dpi = 300

        self.bbox_inches = None

        self.fig_exts = ['.png', '.pdf']
        
        self.save_dir = r"C:\Users\benjamin\Documents\Communication - Papers\thesis\figs"

        return

    def get_filename(self):
        return os.path.join(self.save_dir, os.path.basename(sys.argv[0]).split('.')[0])

    def save_figure(self, fig, filename=None):
        kwargs = {'dpi': self.dpi, 'transparent': False}
        for ext in self.fig_exts:
            if filename is None:
                fig.savefig(os.path.join(self.save_dir, os.path.basename(sys.argv[0]).split('.')[0] + ext), **kwargs)
            elif type(filename) is str:
                fig.savefig(os.path.join(self.save_dir, os.path.basename(sys.argv[0]).split('.')[0] + "__" + filename + ext), **kwargs)
        return 
    
    def remove_edge_labels(self, ax, bases=['x', 'y']):
        if 'x' in bases:
            xticklabels = ax.get_xticklabels()
            xticklabels[0], xticklabels[-1] = '', ''
            ax.set_xticklabels(xticklabels)
        return