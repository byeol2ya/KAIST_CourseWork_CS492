import numpy as np
import pyvista as pv
# light = pv.Light(position=(-1, 1, 1), color='white')
# light.positional = True
# light.light_type = 2

def create_mesh(value):
    res = int(value)
    sphere = pv.Sphere(phi_resolution=res, theta_resolution=res)
    p.add_mesh(sphere, name='sphere', show_edges=True)
    return


#https://docs.pyvista.org/examples/00-load/create-poly.html#sphx-glr-download-examples-00-load-create-poly-py
vertices = np.array([[0, 0, 0],
                     [1, 0, 0],
                     [1, 1, 0],
                     [0, 1, 0],
                     [0.5, 0.5, -1]])

# mesh faces
faces = np.hstack([[4, 0, 1, 2, 3],  # square
                   [3, 0, 1, 4],     # triangle
                   [3, 1, 2, 4]])    # triangle

surf = pv.PolyData(vertices, faces)

vertices = np.array([[0, 0, 0.5],
                     [1, 0, 0.5],
                     [1, 1, 0.5],
                     [0, 1, 0.5],
                     [0.5, 0.5, -0.5]])

# mesh faces
faces = np.array([[3, 0, 1, 2],  # triangle
                    [3, 0, 2, 3],  # triangle
                    [3, 0, 1, 4],     # triangle
                    [3, 1, 2, 4]])    # triangle

print(type(vertices), type(faces))
surf2 = pv.PolyData(vertices, faces)

# plot each face with a different color
#surf.plot(scalars=np.arange(3), cpos=[-1, 1, 0.5],opacity=0.5)
p = pv.Plotter()
p.add_mesh(surf, show_edges=True, color="blue",opacity=0.5)
p.add_mesh(surf2, show_edges=True, color="red",opacity=0.5)
# p.add_light(light)
p.add_slider_widget(create_mesh, [5, 100], title='Resolution')
p.show(cpos="xy")
