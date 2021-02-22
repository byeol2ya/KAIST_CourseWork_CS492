from star.ch.star import STAR
import chumpy as ch
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from star.ch.verts import verts_decorated_quat
from star.config import cfg

import cal_bone_length

#f
#J_regressor
#kintree_table
#posedirs
#shapedirs
#v_template
#weights

def save_as_obj(v,f,save_path, name):
    f_str = (f + 1).astype('str')
    f_anno = np.full((f_str.shape[0], 1), 'f')

    v_str = v.astype('str')
    v_anno = np.full((v_str.shape[0], 1), 'v')

    v = np.hstack((v_anno, v_str))
    f = np.hstack((f_anno, f_str))
    output = np.vstack((v, f))

    np.savetxt(save_path + name + ".obj", output, delimiter=" ", fmt="%s")

def get_gender_model(gender):
    if gender == 'male':
        fname = cfg.path_male_star
    elif gender == 'female':
        fname = cfg.path_female_star
    else:
        fname = cfg.path_neutral_star

    if not os.path.exists(fname):
        raise RuntimeError('Path does not exist %s' % (fname))
    model_dict = np.load(fname, allow_pickle=True)

    return model_dict

def extractor_template(gender, save_path, name):
    model_dict = get_gender_model(gender)
    v_tempalate_np = np.array(ch.array(model_dict['v_template']))
    f_np = np.array(model_dict['f'])
    local_name = name
    save_as_obj(v_tempalate_np, f_np, save_path, local_name)

#https://stackoverflow.com/questions/27786868/python3-numpy-appending-to-a-file-using-numpy-savetxt
def extractor_weight_and_joint(gender, save_path, name, isWeight = True):
    file = open(save_path + name + ".txt", 'w+')
    model_dict = get_gender_model(gender)

    if isWeight:
        weights = np.array(ch.array(model_dict['weights'])).astype('float')
        file.write(f'{weights.shape[1]} {weights.shape[0]} \n')
        np.savetxt(file, weights, delimiter=" ", fmt="%.9f") # need to check, is it round or round up or round down
    else:
        joints_base_vertex_weight = np.array(ch.array(model_dict['J_regressor'])).astype('float')
        file.write(f'{joints_base_vertex_weight.shape[0]}\n{joints_base_vertex_weight.shape[1]}\n')
        np.savetxt(file, joints_base_vertex_weight, delimiter=" ", fmt="%.9f") # need to check, is it round or round up or round down

    file.close()

def extractor(gender, save_path, name, type, total=-1, zfillnum = 0):
    model_dict = get_gender_model(gender)

    type_dimension = len(np.array(ch.array(model_dict[type])).shape)
    if type_dimension == 3:
        if total == -1:
            total = np.array(ch.array(model_dict[type])).shape[2]
        for i in range(total):
            v_tempalate_np = np.array(ch.array(model_dict['v_template']))
            specificdir_np = np.array(ch.array(model_dict[type][:, :, i]))  # Shape Corrective Blend shapes
            v_np = v_tempalate_np + specificdir_np
            f_np = np.array(model_dict['f'])
            local_name = name + str(i).zfill(zfillnum)
            save_as_obj(v_np, f_np, save_path, local_name)

def save_shape_dif():
    gender = "female"
    name = "shape"
    save_path = "E:/STAR/f_star/f_blendshape/"
    extractor(gender=gender, name=name, save_path=save_path, type="shapedirs")

    name = "Pose"
    save_path = "E:/STAR/f_star/f_pose_blendshapes/"
    extractor(gender=gender, name=name, save_path=save_path, type="posedirs", zfillnum=3)

    name = "f_shapeAv"
    save_path = "E:/STAR/f_star/"
    extractor_template(gender, save_path, name)

    name = "f_weight"
    save_path = "E:/STAR/f_star/"
    extractor_weight_and_joint(gender=gender, save_path=save_path, name=name)

    name = "f_joints_mat"
    save_path = "E:/STAR/f_star/"
    extractor_weight_and_joint(gender=gender, save_path=save_path, name=name, isWeight=False)

    ##############################################################################################################################################################
    gender = "male"
    name = "shape"
    save_path = "E:/STAR/m_star/m_blendshape/"
    extractor(gender=gender, name=name, save_path=save_path, type="shapedirs")

    name = "Pose"
    save_path = "E:/STAR/m_star/m_pose_blendshapes/"
    extractor(gender=gender, name=name, save_path=save_path, type="posedirs", zfillnum=3)

    name = "m_shapeAv"
    save_path = "E:/STAR/m_star/"
    extractor_template(gender, save_path, name)

    name = "m_weight"
    save_path = "E:/STAR/m_star/"
    extractor_weight_and_joint(gender=gender, save_path=save_path, name=name)

    name = "m_joints_mat"
    save_path = "E:/STAR/m_star/"
    extractor_weight_and_joint(gender=gender, save_path=save_path, name=name, isWeight=False)

def get_joints_base_vertex_weight(gender):
    model_dict = get_gender_model(gender)
    joints_base_vertex_weight = np.array(ch.array(model_dict['J_regressor'])).astype('float')

    joint_base_vertex_list = []
    for i in range(0, joints_base_vertex_weight.shape[0]):
        joint_base_vertex_list += [[]]
        for k in range(0, joints_base_vertex_weight.shape[1]):
            if joints_base_vertex_weight[i][k] > 0.0:
                joint_base_vertex_list[i] += [k]

def main():
    gender = 'female'
    joint_base_vertex_list = get_joints_base_vertex_weight(gender)
    model_dict = get_gender_model(gender)
    shapedirs = np.array(ch.array(model_dict['shapedirs']))  # Shape Corrective Blend shapes


def get_bone_length(gender = 'female', betas = np.zeros(0)):
    num_pose = 24 * 3

    if betas.size == 0:
        num_betas = 300
        betas = np.zeros(num_betas)
    else:
        num_betas = betas.size
    model = STAR(gender=gender, num_betas=num_betas, betas=ch.array(betas), pose=ch.array(np.zeros(num_pose)))

    joints = np.array(ch.array(model.J_transformed)).astype('float').reshape(num_pose)
    left_idx_list = [3,1,4,7,3,2,5,8,3,6,9,13,16,18,20,9,14,17,19,21,9,12]
    right_idx_list = [1,4,7,10,2,5,8,11,6,9,13,16,18,20,22,14,17,19,21,23,12,15]
    bone_length = cal_bone_length.cal_bone(joints,left_idx_list,right_idx_list) #left and right are not symmetric

    return bone_length

def get_bone_length_dif(betas,gender = 'female'):
    ret = get_bone_length(gender=gender,betas=betas) - get_bone_length(gender=gender)
    return ret

def load_betas_bone_length_dif(gender='female', save_path_betas_bone_length_dif_0p1='./saved_bonelength_0p1.npy',
                              save_path_betas_bone_length_dif_1='./saved_bonelength_1.npy'):
    if (not os.path.isfile(save_path_betas_bone_length_dif_0p1)) or (not os.path.isfile(save_path_betas_bone_length_dif_1)):
        save_betas_bone_length_dif(gender=gender,
                                   save_path_betas_bone_length_dif_0p1=save_path_betas_bone_length_dif_0p1,
                                   save_path_betas_bone_length_dif_1=save_path_betas_bone_length_dif_1)

    betas_bone_length_dif_0p1 = np.load(save_path_betas_bone_length_dif_0p1)
    betas_bone_length_dif_1 = np.load(save_path_betas_bone_length_dif_1)

    #https://rfriend.tistory.com/357
    betas_bone_length_dif_0p1_idx = np.argsort(betas_bone_length_dif_0p1)
    betas_bone_length_dif_1_idx = np.argsort(betas_bone_length_dif_1)

    return betas_bone_length_dif_0p1, betas_bone_length_dif_1

#12,13,14,15 #left
#17,18,19,20 #right
def order_beta_bone_dif_by_bone_idx(gender='female'):
    betas_bone_length_dif_0p1, betas_bone_length_dif_1 = load_betas_bone_length_dif(gender=gender)

    for bone_idx in range(12,16,1):
        local_bone = betas_bone_length_dif_0p1[:,bone_idx-1]
        local_bone_sort_idx = np.argsort(local_bone)
        local_bone_sort = local_bone[local_bone_sort_idx]
        print(f'bone_idx:\n{bone_idx}\nmax beta idx: {local_bone_sort_idx[-1]}\nmin beta idx: {local_bone_sort_idx[0]}')
        print(f'local_bone_sort:\n{local_bone_sort}\nlocal_bone_sort_idx:\n{local_bone_sort_idx}')




    for bone_idx in range(17,21,1):
        local_bone = betas_bone_length_dif_0p1[:,bone_idx-1]
        local_bone_sort_idx = np.argsort(local_bone)
        local_bone_sort = local_bone[local_bone_sort_idx]
        print(f'bone_idx:\n{bone_idx}\nmax beta idx: {local_bone_sort_idx[-1]}\nmin beta idx: {local_bone_sort_idx[0]}')
        print(f'local_bone_sort:\n{local_bone_sort}\nlocal_bone_sort_idx:\n{local_bone_sort_idx}')

    xmax = np.amax(betas_bone_length_dif_0p1, axis=0)
    ymax = np.amax(betas_bone_length_dif_0p1, axis=1)
    print('abc')


def save_betas_bone_length_dif(gender='female', save_path_betas_bone_length_dif_0p1='./saved_bonelength_0p1.npy',
                              save_path_betas_bone_length_dif_1='./saved_bonelength_1.npy'):
    num_betas = 300
    num_bone  = 22
    betas_bone_length_dif_0p1 = np.zeros((num_betas,num_bone))
    betas_bone_length_dif_1 = np.zeros((num_betas,num_bone))

    for i in range(num_betas):
        print(i)
        local_betas = np.zeros(num_betas)
        local_betas[i] = 0.1
        betas_bone_length_dif_0p1[i] = get_bone_length_dif(betas=local_betas,gender=gender)
        local_betas[i] = 1
        betas_bone_length_dif_1[i] = get_bone_length_dif(betas=local_betas,gender=gender)
    np.save(save_path_betas_bone_length_dif_0p1, betas_bone_length_dif_0p1)
    np.save(save_path_betas_bone_length_dif_1, betas_bone_length_dif_1)



if __name__ == "__main__":
    # save_shape_dif()
    #main()
    order_beta_bone_dif_by_bone_idx()