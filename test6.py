from pyvista import examples
import pyvista as pv
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pyvista as pv
sphere = pv.Sphere()
sphere['Data'] = sphere.points[:, 2]
plotter = pv.Plotter()
_ = plotter.add_mesh(sphere, show_scalar_bar=False)
_ = plotter.add_scalar_bar('Data', interactive=True, vertical=False,
                           outline=True, fmt='%10.5f')
plotter.show(cpos="xy")