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

def save_as_obj(model,save_path, name):
    f_str = (model.f + 1).astype('str')
    f_anno = np.full((f_str.shape[0], 1), 'f')

    v_str = np.array(model).astype('str')
    v_anno = np.full((v_str.shape[0], 1), 'v')
    v = np.hstack((v_anno, v_str))
    f = np.hstack((f_anno, f_str))
    output = np.vstack((v, f))

    np.savetxt(save_path + name + ".obj", output, delimiter=" ", fmt="%s")


# model = STAR(gender='female',num_betas=10)
# ## Assign random pose and shape parameters
# model.pose[:] = np.random.rand(model.pose.size) * .2
# model.betas[:] = np.random.rand(model.betas.size) * .03
#
# for j in range(0,10):
#     model.betas[:] = 0.0  #Each loop all PC components are set to 0.
#     for i in np.linspace(-3,3,10): #Varying the jth component +/- 3 standard deviations
#         model.betas[j] = i

num_pose = 24*3
num_betas = 10
# betas = ch.array(np.zeros(num_betas)) #Betas
# pose = ch.array(np.zeros(num_pose)) #Pose
pose = ch.array((np.random.rand(num_pose)) - 0.5) * 1
print(pose)
# betas = ch.array(
#             np.array([ 2.25176191, -3.7883464, 0.46747496, 3.89178988,
#                       2.20098416, 0.26102114, -3.07428093, 0.55708514,
#                       -3.94442258, -2.88552087])) * .03
betas = ch.array(
            np.array([ 1.0, 0.0, 0.0, 0.0,
                      0.0, 0.0, 0.0, 0.0,
                      0.0, 0.0]))

# for j in range(0,10):
#     model.betas[:] = 0.0  #Each loop all PC components are set to 0.
#     for i in np.linspace(-3,3,10): #Varying the jth component +/- 3 standard deviations
#         model.betas[j] = i

model = STAR(gender='female', num_betas=num_betas, pose=pose, betas=betas)

save_as_obj(model, "./", name="output_2_0")
# np.savetxt("./output.obj", output, delimiter=" ", fmt="%s")
