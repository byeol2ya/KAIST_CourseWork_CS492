import numpy as np
import pyvista as pv
import math
from star.ch.star import STAR
import chumpy as ch
import sys
import os

#https://brownbears.tistory.com/296
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import conditional_vae_star as cvs

from scipy.stats import truncnorm
#https://docs.pyvista.org/examples/03-widgets/slider-bar-widget.html

GUI = pv.Plotter()
TARGET_INDEX = 0
NUM_BETAS = None
BETAS = None

def ith_reset(value=None):
    global BETAS, NUM_BETAS
    NUM_BETAS = 300
    create_mesh(0)
    print('\n' * 80)
    print("reset i^th beta")


def reset(value=None):
    global BETAS, NUM_BETAS
    NUM_BETAS = 300
    BETAS = np.zeros(NUM_BETAS)
    create_mesh(0)
    #https://teamtreehouse.com/community/clear-screen-for-pycharm-as-if-it-were-on-console-or-cmd
    print('\n' * 80)
    print("reset all")
    return

def change_beta_idx(value):
    global TARGET_INDEX
    TARGET_INDEX = value
    return

def create_mesh(value):
    global BETAS, NUM_BETAS, TARGET_INDEX, GUI
    gender = 'female'
    BETAS[math.floor(TARGET_INDEX)] = value

    v, f = load_star_obj(gender=gender, betas=BETAS, num_betas=NUM_BETAS)

    person = pv.PolyData(v, f)
    mout = person.clean()
    GUI.add_mesh(mout, name='star', show_edges=True,opacity=0.5)
    print(BETAS)
    return

def load_star_obj(gender,betas, num_betas = 300, num_pose = 24 * 3):

    model = STAR(gender=gender, num_betas=num_betas, betas= ch.array(betas), pose=ch.array(np.zeros(num_pose)))
    # model.shape = ch.array(betas)  # Pose
    # model.pose = ch.array(np.zeros(num_pose))  # Pose

    v = np.array(model).astype('float32')
    f = model.f
    f_anno = np.full((f.shape[0], 1), 3)
    f = np.hstack((f_anno, f))
    return v,f

def cvae_wrapper():
    cvs.setup_trained_model()

def main():
    reset()

    GUI.add_slider_widget(change_beta_idx, [0, 299], title='Beta Index', pointa=(.01, .93), pointb=(.99, .93), value=0)
    GUI.add_slider_widget(create_mesh, [-100, 100], title='i^th Beta Value', pointa=(.01, .1), pointb=(.99, .1))
    GUI.add_slider_widget(reset, [0, 0], title='Reset', pointa=(.01, .8), pointb=(.11, .8), value=0)
    GUI.add_slider_widget(ith_reset, [0, 0], title='i^th B Rst', pointa=(.01, .7), pointb=(.11, .7), value=0)
    GUI.show(cpos="xy")


if __name__ == "__main__":
    main()