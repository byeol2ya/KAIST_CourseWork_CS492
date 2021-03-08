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
import torch
import load_chumpy as lc

from scipy.stats import truncnorm
#https://docs.pyvista.org/examples/03-widgets/slider-bar-widget.html

LEFT = 0
RIGHT_TOP = 1
RIGHT_MIDDLE = 2
RIGHT_BOT = 3
WIDTH = 1900
HEIGHT = 1000
GUI = pv.Plotter(window_size=[WIDTH, HEIGHT], shape='1|3')

class CacheController:
    def __init__(self):
        self.cache = {'data':None, 'num_data':None, 'target_index':0}

    def __call__(self,cache_key):
        return self.cache[cache_key]

Beta, LatentZ = CacheController(), CacheController()

create_using_specific_beta_list = []
create_using_specific_latentZ_list = []


def ith_beta_reset(value=None):
    Beta.cache['num_beta'] = cvs.INPUT_DIM
    print("reset i^th beta")
    create_using_target_beta(0)
    #print('\n' * 20)


def ith_latentZ_reset(value=None):
    LatentZ.cache['num_beta'] = cvs.LATENT_DIM
    print("reset i^th beta")
    cvae_wrapper()
    #print('\n' * 20)


def reset(value=None):
    global Beta, LatentZ
    Beta.cache['num_data'] = cvs.INPUT_DIM
    Beta.cache['data'] = np.zeros(Beta.cache['num_data'])

    LatentZ.cache['num_data'] = cvs.LATENT_DIM
    LatentZ.cache['data'] = np.zeros(LatentZ.cache['num_data'])
    print("reset all")
    create_using_target_beta(0)
    #https://teamtreehouse.com/community/clear-screen-for-pycharm-as-if-it-were-on-console-or-cmd
    #print('\n' * 20)
    return


def change_beta_idx(value):
    Beta.cache['target_index'] = value
    return

def change_latentZ_idx(value):
    LatentZ.cache['target_index'] = value
    return

def create_using_target_beta(value):
    Beta.cache['data'][math.floor(Beta.cache['target_index'])] = value
    LatentZ.cache['data'] = np.zeros(LatentZ.cache['num_data'])
    create_mesh()


def create_using_target_latentZ(value):
    LatentZ.cache['data'][math.floor(LatentZ.cache['target_index'])] = value
    cvae_wrapper()


#cannot change beta inside create_mesh
def create_mesh():
    global GUI
    gender = 'female'

    v, f = load_star_obj(gender=gender, betas=Beta.cache['data'], num_betas=Beta.cache['num_data'])

    person = pv.PolyData(v, f)
    mout = person.clean()
    GUI.subplot(LEFT)
    GUI.add_mesh(mout, name='star', show_edges=True,opacity=0.5)

    #print('\n' * 20)
    #print(Beta.cache['data'][:10])
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
    print('run cave_wrapper')
    #npy_file_path = 'C:/Users/TheOtherMotion/Documents/GitHub/STAR-Private/demo/saved_bonelength_validation.npy'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # model, _, test_iterator, _ = cvs.setup_trained_model(trained_time="2021_03_08_08_49_35")
    model, _, test_iterator, _ = cvs.setup_trained_model()
    # for param in model.parameters():
    #     print(param.cpu().detach().numpy())
    model.eval()

    #data = np.array([np.load(npy_file_path)[0].astype(np.float32)])
    data = lc.make_data(np.array([Beta.cache['data']]))#######################################################################
    delta = LatentZ.cache['data']


    #push
    previous_beta = Beta.cache['data']

    Beta.cache['data'] = np.squeeze(cvs.get_star(model,beta=data,delta=delta).cpu().numpy())
    temp = Beta.cache['data'].copy()

    print('\n' * 20)
    Beta.cache['data'] = (Beta.cache['data'] - 0.5) * 10
    #https://rfriend.tistory.com/426
    Beta.cache['data'] = np.where(Beta.cache['data'] < -5, -5, np.where(Beta.cache['data'] > 5, 5, Beta.cache['data']))
    #make no change at beta
    create_mesh()
    print(f"input: {previous_beta[:20]}")
    print(f"Z: {LatentZ.cache['data']}")
    print(f"normalized output: {temp[:20]}")
    print(f"output: {Beta.cache['data'][:20]}")
    #pop
    Beta.cache['data'] = previous_beta



def create_using_specific_beta0(value):
    Beta.cache['data'][0] = value
    LatentZ.cache['data'] = np.zeros(LatentZ.cache['num_data'])
    create_mesh()
def create_using_specific_beta1(value):
    Beta.cache['data'][1] = value
    LatentZ.cache['data'] = np.zeros(LatentZ.cache['num_data'])
    create_mesh()
def create_using_specific_beta2(value):
    Beta.cache['data'][2] = value
    LatentZ.cache['data'] = np.zeros(LatentZ.cache['num_data'])
    create_mesh()
def create_using_specific_beta3(value):
    Beta.cache['data'][3] = value
    LatentZ.cache['data'] = np.zeros(LatentZ.cache['num_data'])
    create_mesh()
def create_using_specific_beta4(value):
    Beta.cache['data'][4] = value
    LatentZ.cache['data'] = np.zeros(LatentZ.cache['num_data'])
    create_mesh()
def create_using_specific_beta5(value):
    Beta.cache['data'][5] = value
    LatentZ.cache['data'] = np.zeros(LatentZ.cache['num_data'])
    create_mesh()
def create_using_specific_beta6(value):
    Beta.cache['data'][6] = value
    LatentZ.cache['data'] = np.zeros(LatentZ.cache['num_data'])
    create_mesh()
def create_using_specific_beta7(value):
    Beta.cache['data'][7] = value
    LatentZ.cache['data'] = np.zeros(LatentZ.cache['num_data'])
    create_mesh()
def create_using_specific_beta8(value):
    Beta.cache['data'][8] = value
    LatentZ.cache['data'] = np.zeros(LatentZ.cache['num_data'])
    create_mesh()
def create_using_specific_beta9(value):
    Beta.cache['data'][9] = value
    LatentZ.cache['data'] = np.zeros(LatentZ.cache['num_data'])
    create_mesh()

def create_using_specific_latentZ0(value):
    LatentZ.cache['data'][0] = value
    cvae_wrapper()
def create_using_specific_latentZ1(value):
    LatentZ.cache['data'][1] = value
    cvae_wrapper()
def create_using_specific_latentZ2(value):
    LatentZ.cache['data'][2] = value
    cvae_wrapper()
def create_using_specific_latentZ3(value):
    LatentZ.cache['data'][3] = value
    cvae_wrapper()
def create_using_specific_latentZ4(value):
    LatentZ.cache['data'][4] = value
    cvae_wrapper()
def create_using_specific_latentZ5(value):
    LatentZ.cache['data'][5] = value
    cvae_wrapper()
def create_using_specific_latentZ6(value):
    LatentZ.cache['data'][6] = value
    cvae_wrapper()
def create_using_specific_latentZ7(value):
    LatentZ.cache['data'][7] = value
    cvae_wrapper()
def create_using_specific_latentZ8(value):
    LatentZ.cache['data'][8] = value
    cvae_wrapper()
def create_using_specific_latentZ9(value):
    LatentZ.cache['data'][9] = value
    cvae_wrapper()

def init_slider_beta():
    global GUI, create_using_specific_beta_list

    create_using_specific_beta_list += [create_using_specific_beta0,
                                        create_using_specific_beta1,
                                        create_using_specific_beta2,
                                        create_using_specific_beta3,
                                        create_using_specific_beta4,
                                        create_using_specific_beta5,
                                        create_using_specific_beta6,
                                        create_using_specific_beta7,
                                        create_using_specific_beta8,
                                        create_using_specific_beta9]

    for i in range(10):
        GUI.subplot(RIGHT_TOP)
        GUI.add_slider_widget(create_using_specific_beta_list[i], [-5, 5], title='Beta ' + str(i) +' Value', pointa=(.05+float(i//5)*.25, .90-float(i%5)*.15), pointb=(.25+float(i//5)*.25, .90-float(i%5)*.15), value=0)

def init_slider_latentZ():
    global GUI, create_using_specific_latentZ_list

    create_using_specific_latentZ_list += [create_using_specific_latentZ0,
                                        create_using_specific_latentZ1,
                                        create_using_specific_latentZ2,
                                        create_using_specific_latentZ3,
                                        create_using_specific_latentZ4,
                                        create_using_specific_latentZ5,
                                        create_using_specific_latentZ6,
                                        create_using_specific_latentZ7,
                                        create_using_specific_latentZ8,
                                        create_using_specific_latentZ9]

    for i in range(10):
        GUI.subplot(RIGHT_MIDDLE)
        GUI.add_slider_widget(create_using_specific_latentZ_list[i], [-5, 5], title='latentZ ' + str(i) +' Value', pointa=(.05+float(i//5)*.25, .90-float(i%5)*.15), pointb=(.25+float(i//5)*.25, .90-float(i%5)*.15), value=0)


def init_sliders():
    global GUI
    #beta
    init_slider_beta()
    GUI.subplot(RIGHT_TOP)
    GUI.add_slider_widget(change_beta_idx, [0, Beta.cache['num_data']-1], title='Beta Index', pointa=(.55, .90), pointb=(.95, .90), value=0)
    GUI.subplot(RIGHT_TOP)
    GUI.add_slider_widget(create_using_target_beta, [-5, 5], title='i^th Beta Value', pointa=(.55, .75), pointb=(.95, .75))
    GUI.subplot(RIGHT_TOP)
    GUI.add_slider_widget(reset, [1, 1.1], title='Reset', pointa=(.85, .1), pointb=(.95, .1), value=1.05)
    GUI.subplot(RIGHT_TOP)
    GUI.add_slider_widget(ith_beta_reset, [1, 1.1], title='i^th B Rst', pointa=(.85, .25), pointb=(.95, .25), value=1.05)
    #v
    init_slider_latentZ()
    GUI.subplot(RIGHT_MIDDLE)
    GUI.add_slider_widget(change_latentZ_idx, [0, LatentZ.cache['num_data']-1], title='LatentZ Index', pointa=(.55, .90), pointb=(.95, .90), value=0)
    GUI.subplot(RIGHT_MIDDLE)
    GUI.add_slider_widget(create_using_target_latentZ, [-100, 100], title='i^th LatentZ Value', pointa=(.55, .75), pointb=(.95, .75))
    GUI.subplot(RIGHT_MIDDLE)
    GUI.add_slider_widget(reset, [1, 1.1], title='Reset', pointa=(.85, .1), pointb=(.95, .1), value=1.05)
    GUI.subplot(RIGHT_MIDDLE)
    GUI.add_slider_widget(ith_latentZ_reset, [1, 1.1], title='i^th Z Rst', pointa=(.85, .25), pointb=(.95, .25), value=1.05)

def main():
    reset()
    init_sliders()
    GUI.show(cpos="xy")
#plotter.add_text("Airplane Example\n", font_size=30)


if __name__ == "__main__":
    main()

