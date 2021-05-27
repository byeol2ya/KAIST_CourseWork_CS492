import numpy as np
import pyvista as pv
import math
from star.ch.star import STAR
import chumpy as ch
import sys
import os

#https://brownbears.tistory.com/296
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from model import cvs_shortcut_v5 as cvs
import torch
import load_chumpy as lc

#https://docs.pyvista.org/examples/03-widgets/slider-bar-widget.html

LATENT_DIM = 10
BONE_DIM = 23

LEFT = 0
RIGHT_TOP = 1
RIGHT_MIDDLE = 2
RIGHT_BOT = 3
WIDTH = 1900
HEIGHT = 1000
COUNT = -39
root = "../outputs/weight/"

cvs.TIMEPATH = '2021_05_22_16_59_22_99'
cvs.TIMEPATH = os.path.join(root, 'cvae_' + cvs.TIMEPATH)

GUI = pv.Plotter(window_size=[WIDTH, HEIGHT], shape='1|3')

class CacheController:
    def __init__(self):
        self.cache = {'data':None, 'num_data':None, 'target_index':0}

    def __call__(self,cache_key):
        return self.cache[cache_key]

Beta, LatentZ, Bone = CacheController(), CacheController(), CacheController()

create_using_specific_beta_list = []
create_using_specific_latentZ_list = []
create_using_specific_bone_list = []


def ith_beta_reset(value=None):
    Beta.cache['num_beta'] = cvs.DATAPATH.get_beta_size()
    print("reset i^th beta")
    create_using_target_beta(0)
    #print('\n' * 20)


def ith_latentZ_reset(value=None):
    LatentZ.cache['num_beta'] = LATENT_DIM
    print("reset i^th beta")
    cvae_wrapper()
    #print('\n' * 20)


def ith_bone_reset(value=None):
    Bone.cache['num_beta'] = BONE_DIM
    print("reset i^th beta")
    cvae_wrapper()
    #print('\n' * 20)



def reset(value=None):
    global Beta, LatentZ
    Beta.cache['num_data'] = cvs.DATAPATH.get_beta_size()
    Beta.cache['data'] = np.zeros(Beta.cache['num_data'])

    LatentZ.cache['num_data'] = LATENT_DIM
    LatentZ.cache['data'] = np.zeros(LatentZ.cache['num_data'])


    Bone.cache['num_data'] = BONE_DIM
    Bone.cache['data'] = np.zeros(Bone.cache['num_data'])
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

def change_bone_idx(value):
    Bone.cache['target_index'] = value
    return

def create_using_target_beta(value):
    Beta.cache['data'][math.floor(Beta.cache['target_index'])] = value
    LatentZ.cache['data'] = np.zeros(LatentZ.cache['num_data'])
    Bone.cache['data'] = np.zeros(Bone.cache['num_data'])
    create_mesh(IsNewBeta=True)

def create_using_target_latentZ(value):
    LatentZ.cache['data'][math.floor(LatentZ.cache['target_index'])] = value
    cvae_wrapper()

def create_using_target_bone(value):
    Bone.cache['data'][math.floor(Bone.cache['target_index'])] = value
    cvae_wrapper()



#cannot change beta inside create_mesh
def create_mesh(IsNewBeta=False):
    global GUI, COUNT
    if COUNT < 0:
        COUNT += 1
        return
    if COUNT == 0:
        IsNewBeta = True
        COUNT += 1

    if IsNewBeta is True:
        print("start with new beta")
    else:
        print("start with preserved beta")

    gender = 'female'

    original_mesh, generated_mesh, original_joint, generated_joint, f= load_obj(gender=gender, betas=Beta.cache['data'],  _z=LatentZ.cache['data'], _bone=Bone.cache['data'], IsNewBeta=IsNewBeta)

    original = pv.PolyData(original_mesh, f)
    generated = pv.PolyData(generated_mesh, f)
    original_out = original.clean()
    generated_out = generated.clean()

    # GUI.add_lines(lines=np.array(original_joint), color='#00005F')
    # GUI.add_lines(lines=np.array(generated_joint), color='#0000FF')
    GUI.subplot(LEFT)
    GUI.add_mesh(original_out, name='original', show_edges=True,opacity=0.7,color='#FFFFFF')
    GUI.add_mesh(generated_out, name='generated', show_edges=True,opacity=0.5,color='#FF0000')

    #print('\n' * 20)
    #print(Beta.cache['data'][:10])
    return


def load_obj(gender,betas, _z, _bone,IsNewBeta):
    test_loss, test_losses, original_mesh, original_joint, generated_mesh, generated_joint  = cvs.setup_trained_model(betas, _z, _bone, IsNewBeta)

    print(Beta.cache['data'])
    print(LatentZ.cache['data'])
    print(Bone.cache['data'])
    return original_mesh, generated_mesh, original_joint, generated_joint, f()


def cvae_wrapper():
    print('run cave_wrapper')
    # model, _, test_iterator, _ = cvs.setup_trained_model(_beta=Beta.cache['data'],  _z=LatentZ.cache['data'], _bone=Bone.cache['data'])
    # model.eval()

    #push
    previous_beta = Beta.cache['data']

    # Beta.cache['data'] = np.squeeze(beta).cpu().numpy())
    # temp = Beta.cache['data'].copy()
    #
    # print('\n' * 20)
    # Beta.cache['data'] = (Beta.cache['data'] - 0.5) * 10
    # #https://rfriend.tistory.com/426
    # Beta.cache['data'] = np.where(Beta.cache['data'] < -5, -5, np.where(Beta.cache['data'] > 5, 5, Beta.cache['data']))
    # #make no change at beta
    create_mesh()
    # print(f"input: {previous_beta[:20]}")
    # print(f"Z: {LatentZ.cache['data']}")
    # print(f"normalized output: {temp[:20]}")
    # print(f"output: {Beta.cache['data'][:20]}")
    #pop
    Beta.cache['data'] = previous_beta



def create_using_specific_beta0(value):
    Beta.cache['data'][0] = value
    # LatentZ.cache['data'] = np.zeros(LatentZ.cache['num_data'])
    # Bone.cache['data'] = np.zeros(Bone.cache['num_data']) - 100.0
    create_mesh(IsNewBeta=True)
def create_using_specific_beta1(value):
    Beta.cache['data'][1] = value
    # LatentZ.cache['data'] = np.zeros(LatentZ.cache['num_data'])
    # Bone.cache['data'] = np.zeros(Bone.cache['num_data']) - 100.0
    create_mesh(IsNewBeta=True)
def create_using_specific_beta2(value):
    Beta.cache['data'][2] = value
    # LatentZ.cache['data'] = np.zeros(LatentZ.cache['num_data'])
    # Bone.cache['data'] = np.zeros(Bone.cache['num_data']) - 100.0
    create_mesh(IsNewBeta=True)
def create_using_specific_beta3(value):
    Beta.cache['data'][3] = value
    # LatentZ.cache['data'] = np.zeros(LatentZ.cache['num_data'])
    # Bone.cache['data'] = np.zeros(Bone.cache['num_data']) - 100.0
    create_mesh(IsNewBeta=True)
def create_using_specific_beta4(value):
    Beta.cache['data'][4] = value
    # LatentZ.cache['data'] = np.zeros(LatentZ.cache['num_data']) - 100.0
    # Bone.cache['data'] = np.zeros(Bone.cache['num_data']) - 100.0
    create_mesh(IsNewBeta=True)
def create_using_specific_beta5(value):
    Beta.cache['data'][5] = value
    # LatentZ.cache['data'] = np.zeros(LatentZ.cache['num_data']) - 100.0
    # Bone.cache['data'] = np.zeros(Bone.cache['num_data']) - 100.0
    create_mesh(IsNewBeta=True)
def create_using_specific_beta6(value):
    Beta.cache['data'][6] = value
    # LatentZ.cache['data'] = np.zeros(LatentZ.cache['num_data']) - 100.0
    # Bone.cache['data'] = np.zeros(Bone.cache['num_data']) - 100.0
    create_mesh(IsNewBeta=True)
def create_using_specific_beta7(value):
    Beta.cache['data'][7] = value
    # LatentZ.cache['data'] = np.zeros(LatentZ.cache['num_data']) - 100.0
    # Bone.cache['data'] = np.zeros(Bone.cache['num_data']) - 100.0
    create_mesh()
def create_using_specific_beta8(value):
    Beta.cache['data'][8] = value
    # LatentZ.cache['data'] = np.zeros(LatentZ.cache['num_data']) - 100.0
    # Bone.cache['data'] = np.zeros(Bone.cache['num_data']) - 100.0
    create_mesh(IsNewBeta=True)
def create_using_specific_beta9(value):
    Beta.cache['data'][9] = value
    # # Beta.cache['data'][:]
    # LatentZ.cache['data'] = np.zeros(LatentZ.cache['num_data']) - 100.0
    # Bone.cache['data'] = np.zeros(Bone.cache['num_data']) - 100.0
    create_mesh(IsNewBeta=True)

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

def create_using_specific_bone0(value):
    Bone.cache['data'][0] = value
    cvae_wrapper()
def create_using_specific_bone1(value):
    Bone.cache['data'][1] = value
    cvae_wrapper()
def create_using_specific_bone2(value):
    Bone.cache['data'][2] = value
    cvae_wrapper()
def create_using_specific_bone3(value):
    Bone.cache['data'][3] = value
    cvae_wrapper()
def create_using_specific_bone4(value):
    Bone.cache['data'][4] = value
    cvae_wrapper()
def create_using_specific_bone5(value):
    Bone.cache['data'][5] = value
    cvae_wrapper()
def create_using_specific_bone6(value):
    Bone.cache['data'][6] = value
    cvae_wrapper()
def create_using_specific_bone7(value):
    Bone.cache['data'][7] = value
    cvae_wrapper()
def create_using_specific_bone8(value):
    Bone.cache['data'][8] = value
    cvae_wrapper()
def create_using_specific_bone9(value):
    Bone.cache['data'][9] = value
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
        GUI.add_slider_widget(create_using_specific_beta_list[i], [-1, 1], title='Beta ' + str(i) +' Value', pointa=(.05+float(i//5)*.25, .90-float(i%5)*.15), pointb=(.25+float(i//5)*.25, .90-float(i%5)*.15), value=0)

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
        GUI.add_slider_widget(create_using_specific_latentZ_list[i], [-20, 20], title='latentZ ' + str(i) +' Value', pointa=(.05+float(i//5)*.25, .90-float(i%5)*.15), pointb=(.25+float(i//5)*.25, .90-float(i%5)*.15), value=0)

def init_slider_bone():
    global GUI, create_using_specific_bone_list

    create_using_specific_bone_list += [create_using_specific_bone0,
                                        create_using_specific_bone1,
                                        create_using_specific_bone2,
                                        create_using_specific_bone3,
                                        create_using_specific_bone4,
                                        create_using_specific_bone5,
                                        create_using_specific_bone6,
                                        create_using_specific_bone7,
                                        create_using_specific_bone8,
                                        create_using_specific_bone9]

    for i in range(10):
        GUI.subplot(RIGHT_BOT)
        GUI.add_slider_widget(create_using_specific_bone_list[i], [-5, 5], title='bone ' + str(i) +' Value', pointa=(.05+float(i//5)*.25, .90-float(i%5)*.15), pointb=(.25+float(i//5)*.25, .90-float(i%5)*.15), value=0)


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
    #bone
    init_slider_bone()
    GUI.subplot(RIGHT_BOT)
    GUI.add_slider_widget(change_bone_idx, [0, Bone.cache['num_data']-1], title='Bone Index', pointa=(.55, .90), pointb=(.95, .90), value=0)
    GUI.subplot(RIGHT_BOT)
    GUI.add_slider_widget(create_using_target_bone, [-100, 100], title='i^th Bone Value', pointa=(.55, .75), pointb=(.95, .75))
    GUI.subplot(RIGHT_BOT)
    GUI.add_slider_widget(reset, [1, 1.1], title='Reset', pointa=(.85, .1), pointb=(.95, .1), value=1.05)
    GUI.subplot(RIGHT_BOT)
    GUI.add_slider_widget(ith_bone_reset, [1, 1.1], title='i^th Z Rst', pointa=(.85, .25), pointb=(.95, .25), value=1.05)

def main():
    reset()
    init_sliders()
    GUI.show(cpos="xy")
#plotter.add_text("Airplane Example\n", font_size=30)

def f():
    ret = np.loadtxt('./f.txt',dtype=np.int,delimiter=' ') - 1
    # ret = np.hstack(np.array([ret]) - 1)
    return ret

if __name__ == "__main__":
    main()

