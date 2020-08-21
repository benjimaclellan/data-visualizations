
#%%
# see https://schemdraw.readthedocs.io/en/latest/elements/flow.html for API details
import schemdraw
from schemdraw import flow
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('Agg')

#%%
from configstyle import ConfigStyle
c = ConfigStyle()

#%% parameter optimization
d = schemdraw.Drawing(unit=1, inches_per_unit=0.3, fontsize=9)
UNITS = 0.75 * d.unit
W = 7 * d.unit
H = 1.5 * d.unit

label = r"Create graph $G$"
d.add(flow.Start(w=W, h=H, label=label))
d.add(flow.Arrow('down', l=UNITS))

label = r"Compile function $F(\mathrm{\mathbf{x}})$ from $G$"
d.add(flow.Box(w=W, h=H, label=label))
d.add(flow.Arrow('down', l=UNITS))

label = r"Compile derivatives" + "\n"+ r"$\nabla F(\mathrm{\mathbf{x}})$ and $\mathrm{\mathbf{H}}(\mathrm{\mathbf{x}})$" + "\n" + "from automatic differentiation"
d.add(flow.Box(w=W, h=H, label=label))
d.add(flow.Arrow('down', l=UNITS))


label = "Randomly select initial population\n" + "$\{\mathrm{\mathbf{x}}\} \sim \mathcal{U}$"
d.add(flow.Box(w=W, h=H, label=label))
d.add(flow.Arrow('down', l=UNITS))

label = "Calculate "+r"$F(\mathrm{\mathbf{x}}_i)$"+" for each\nsolution in the population"
b1 = d.add(flow.Data(w=W, h=H, label=label))
d.add(flow.Arrow('down', l=UNITS))

label = "Number of\ngenerations reached?"
b2 = d.add(flow.Decision(w=1.5*W, h=1.5*H, E='No', S="Yes", label=label))
d.push()
d.add(flow.Line('right', xy=b2.E, l=1.5*UNITS))

label = "Apply mutation, crossover,\nand selection operators"
b3 = d.add(flow.Box(w=W, h=H, label=label, anchor='W'))
d.add(flow.Line('up', xy=b3.N, toy=b1.E))
d.add(flow.Arrow('left', tox=b1.E))

d.pop()
d.add(flow.Arrow('down', l=UNITS))
label = "Local optimization with L-BFGS\nwith best solution "+r"$\mathrm{\mathbf{x}}_\mathrm{min}$"+" as intial guess"
d.add(flow.Subroutine(w=1.3*W, h=H, label=label))

d.add(flow.Arrow('down', l=UNITS))
label = "Calculate sensitivity analysis on\nbest solution, "+r"$\mathrm{\mathbf{H}}(\mathrm{\mathbf{x}}_\mathrm{min})$"
d.add(flow.Data(w=W, h=H, label=label))

fig = d.draw().fig
c.save_figure(fig, "parameter")


# ********************************************************************************************************************
#%% topology optimization
del(d)
d = schemdraw.Drawing(unit=1, inches_per_unit=0.3, fontsize=9)
UNITS = 0.75 * d.unit
W = 7 * d.unit
H = 1.5 * d.unit

label = "Randomly create population\nof graphs "+r"$\{G\}$"
d.add(flow.Start(w=W, h=1.5*H, label=label))
d.add(flow.Arrow('down', l=UNITS))

label = "For each graph "+r"$G_i$"+"\noptimize parameters for "+r"$\mathrm{\mathbf{x}}_\mathrm{min}$"
o1 = d.add(flow.Subroutine(w=1.3*W, h=H, label=label))
d.add(flow.Arrow('down', l=UNITS))

label = "Perform sensitivity (Hessian) analysis\non each graph "+r"$G_i$ at $\mathrm{\mathbf{x}}_\mathrm{min}$"
d.add(flow.Subroutine(w=1.3*W, h=H, label=label))
d.add(flow.Arrow('down', l=UNITS))

label = "Number of\ngenerations reached?"
b2 = d.add(flow.Decision(w=1.5*W, h=1.5*H, E='No', S="Yes", label=label))
d.push()
d.add(flow.Line('right', xy=b2.E, l=2.5*UNITS))

label = "Check which graph evolution\noperations are valid"
b3 = d.add(flow.Box(w=W, h=H, label=label, anchor='W'))
d.add(flow.Arrow('up', xy=b3.N, toy=o1.E-UNITS))

label = "Mutate graphs with\nvalid evolution operations"
b4 = d.add(flow.Box(w=W, h=H, label=label, anchor='S'))
d.add(flow.Arrow('left', xy=b4.W, tox=o1.E))

d.pop()
d.add(flow.Arrow('down', l=UNITS))
label = "Perform noise analysis"
d.add(flow.Subroutine(w=1.3*W, h=H, label=label))

d.add(flow.Arrow('down', l=UNITS))
label = "Return best performing system"
d.add(flow.Start(w=W, h=H, label=label))

fig = d.draw().fig
c.save_figure(fig, "topology")