# -*- coding: utf-8 -*-
#
# Copyright (C) 2020 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG),
# acting on behalf of its Max Planck Institute for Intelligent Systems and the
# Max Planck Institute for Biological Cybernetics. All rights reserved.
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights
# on this computer program. You can only use this computer program if you have closed a license agreement
# with MPG or you get the right to use the computer program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and liable to prosecution.
# Contact: ps-license@tuebingen.mpg.de
#
#
# If you use this code in a research publication please consider citing the following:
#
# STAR: Sparse Trained  Articulated Human Body Regressor <https://arxiv.org/pdf/2008.08535.pdf>
#
#
# Code Developed by:
# Ahmed A. A. Osman

from star.ch.star import STAR
import chumpy as ch
import numpy as np

from scipy.stats import truncnorm

import cal_bone_length as bl

def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

def save_as_obj(model,save_path, name):
    f_str = (model.f + 1).astype('str')
    f_anno = np.full((f_str.shape[0], 1), 'f')

    v_str = np.array(model).astype('str')
    v_anno = np.full((v_str.shape[0], 1), 'v')
    v = np.hstack((v_anno, v_str))
    f = np.hstack((f_anno, f_str))
    output = np.vstack((v, f))

    np.savetxt(save_path + name + ".obj", output, delimiter=" ", fmt="%s")


def extract_obj():

    num_pose = 24 * 3
    num_betas = 300
    model = STAR(gender='female',num_betas=10)
    # ## Assign random pose and shape parameters
    # for j in range(0,10):
    #     model.betas[:] = 0.0  #Each loop all PC components are set to 0.
    #     for i in np.linspace(-3,3,10): #Varying the jth component +/- 3 standard deviations
    #         model.betas[j] = i
    #         model.betas = ch.array(np.zeros(num_betas)) #Betas
    model.shape = ch.array(np.zeros(num_betas))  # Pose
    model.pose = ch.array(np.zeros(num_pose))  # Pose

    save_as_obj(model, "./", name="output_2_0")


def make_data(base_data=np.zeros(0), save_path=None):
    num_pose = 24 * 3
    num_additional = 14

    if base_data.size == 0:
        num_betas = 300
        num_data = 30000
    else:
        num_data, num_betas = base_data.shape
    pose = ch.array(np.zeros(num_pose))  # Pose

    ret = np.zeros((num_data, num_betas + num_additional))



    for i in range(num_data):
        #if i%100 == 0:
        print (i)
        if base_data.size == 0:
            X = get_truncated_normal(mean=0, sd=3, low=-3, upp=3)
            betas_numpy = X.rvs(num_betas)
        else:
            betas_numpy = base_data[i,:num_betas]
        betas = ch.array(betas_numpy)
        model = STAR(gender='female', num_betas=num_betas, pose=pose, betas=betas)
        # [beta0 ... beta n v0_x v0_y v0_z v1_x v1_y v1_z ...]

        pos_vertices_vector = np.ravel(np.array(model.J_transformed))
        pos_J = bl.cal_bone(pos_vertices_vector)
        contents = np.hstack((betas_numpy, pos_J))
        ret[i,:] = contents[:]
        #print(contents[:])
#https://rfriend.tistory.com/358
    if save_path is not None:
        np.save(save_path, ret)
    return ret

def main():
    extract_obj()
    # ret_back = np.load('./saved_300betas_base.npy')
    # make_data(ret_back, './saved_210217.npy')
    # x_save_load = np.load('./saved.npy')
    # for i in range(0,10):
    #    print(x_save_load[i,:])


if __name__ == "__main__":
    main()