from star.ch.star import STAR
import chumpy as ch
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from star.ch.verts import verts_decorated_quat
from star.config import cfg

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
        joints = np.array(ch.array(model_dict['J_regressor'])).astype('float')
        file.write(f'{joints.shape[0]}\n{joints.shape[1]}\n')
        np.savetxt(file, joints, delimiter=" ", fmt="%.9f") # need to check, is it round or round up or round down

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

if __name__ == "__main__":
    gender = "female"
    name = "shape"
    save_path = "E:/STAR/f_star/f_blendshape/"
    extractor(gender=gender,name = name, save_path=save_path, type="shapedirs")

    name = "Pose"
    save_path = "E:/STAR/f_star/f_pose_blendshapes/"
    extractor(gender=gender,name=name,save_path=save_path, type="posedirs", zfillnum=3)

    name = "f_shapeAv"
    save_path = "E:/STAR/f_star/"
    extractor_template(gender, save_path, name)

    name = "f_weight"
    save_path = "E:/STAR/f_star/"
    extractor_weight_and_joint(gender=gender,save_path=save_path,name=name)

    name = "f_joints_mat"
    save_path = "E:/STAR/f_star/"
    extractor_weight_and_joint(gender=gender,save_path=save_path,name=name,isWeight=False)


    ##############################################################################################################################################################
    gender = "male"
    name = "shape"
    save_path = "E:/STAR/m_star/m_blendshape/"
    extractor(gender=gender,name = name, save_path=save_path, type="shapedirs")

    name = "Pose"
    save_path = "E:/STAR/m_star/m_pose_blendshapes/"
    extractor(gender=gender,name=name,save_path=save_path, type="posedirs", zfillnum=3)

    name = "m_shapeAv"
    save_path = "E:/STAR/m_star/"
    extractor_template(gender, save_path, name)

    name = "m_weight"
    save_path = "E:/STAR/m_star/"
    extractor_weight_and_joint(gender=gender,save_path=save_path,name=name)

    name = "m_joints_mat"
    save_path = "E:/STAR/m_star/"
    extractor_weight_and_joint(gender=gender,save_path=save_path,name=name,isWeight=False)