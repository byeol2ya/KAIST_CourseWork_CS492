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

# based on ~/demo/cal_bone_length.py ~/demo/load_chumpy.py

from star.ch.star import STAR
from star.config import cfg
import os
import chumpy as ch
import numpy as np
import pickle

from scipy.stats import truncnorm

def euclidean_distance(cache, first_idx, second_idx, cache_offset=3):
    first = np.array(cache[cache_offset * first_idx:cache_offset * (first_idx + 1)])
    second = np.array(cache[cache_offset * second_idx:cache_offset * (second_idx + 1)])
    dist = np.linalg.norm(first - second)

    return dist

def load_data_npy(load_path, IsNeedDefault):
    saved_npz = np.load(load_path, allow_pickle=True)
    local_data = {}

    if IsNeedDefault:
        #local_data.deltashape = saved_npz['deltashape']
        #local_data.shape = saved_npz['shape']
        local_data['beta'] = saved_npz['beta']
        local_data['mesh_shape'] = saved_npz['mesh_shape']
        local_data['mesh_shape_pos'] = saved_npz['mesh_shape_pos']
        local_data['joint'] = saved_npz['joint']
        local_data['bonelength'] = saved_npz['bonelength']
    else:
        local_data['beta'] = saved_npz['beta']
        local_data['joint'] = saved_npz['joint']
        local_data['bonelength'] = saved_npz['bonelength']

    saved_npz.close()

    return local_data

def save_data_npy(save_path, local_data, IsNeedDefault):
    if IsNeedDefault:
        np.savez_compressed(save_path,
                 beta=local_data['beta'],
                 mesh_shape=local_data['mesh_shape'],
                 mesh_shape_pos=local_data['mesh_shape_pos'],
                 joint=local_data['joint'],
                 bonelength=local_data['bonelength'])
    else:
        np.savez_compressed(save_path,
                 beta=local_data['beta'],
                 joint=local_data['joint'],
                 bonelength=local_data['bonelength'])


#TODO: how to add avg -> use at beginning
class StarData:
    def __init__(self, num_beta):
        self.num_vertex = 6890
        self.num_axis = 3
        self.num_beta = num_beta

        #self.deltashape = np.zeros((self.num_vertex, self.num_axis, self.num_beta))
        #self.shape = np.zeros((self.num_vertex, self.num_axis, self.num_beta))
        self.beta = np.zeros(self.num_beta)
        self.mesh_shape = None
        self.mesh_shape_pos = None
        self.joint = None
        self.bonelength = None

class DataGenerator:
    def __init__(self, gender, data_path=None, beta=None, num_data=10000, num_beta=300, STATUS=0):
        """

        Note:


        Args:
            STATUS (int): 0 is default, 1 for simple data, 2 for save constant

        """
        self.gender = gender
        self.model_dict = self.load_model(gender=self.gender)
        self.num_data = num_data
        self.max_bonelength = 0
        self.data_idx = 0
        self.STATUS = STATUS

        self.data = [None] * self.num_data
        if data_path is None:
            if beta is not None:
                num_beta = beta.shape[0]

            for i in range(num_data):
                self.data[i] = StarData(num_beta=num_beta)
                # if i % 50 == 0:
                #     print(f'build data {i}')

        #TODO: read all DataGenerator
        else:
            self.data[0] = StarData(num_beta=num_beta)
            self.load_data_npy(data_path)


        #Usage: ~/star/ch/verts.py and ~/star/ch/star.py
        self.template = np.array(ch.array(self.model_dict['v_template'])).astype('float')
        self.shapeblendshape = np.array(ch.array(self.model_dict['shapedirs'][:,:,:num_beta])).astype('float')
        self.jointregessor_matrix = np.array(ch.array(self.model_dict['J_regressor'])).astype('float')

    def load_model(self,gender):
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

    def cal_bonelength_both(self, pos_xyz, offset=3):
        first_idx_list = [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
        second_idx_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

        return self.cal_bonelength(pos_xyz=pos_xyz, first_idx_list=first_idx_list, second_idx_list=second_idx_list, offset=offset)

    def cal_bonelength(self, pos_xyz,
                        first_idx_list=[0, 1, 4, 7, 0, 3, 6, 9, 13, 16, 18, 20, 9, 12],
                        second_idx_list=[1, 4, 7, 10, 3, 6, 9, 13, 16, 18, 20, 22, 12, 15],
                        offset=3):
        local_size = len(first_idx_list)
        bonelength_list = np.zeros(local_size)

        for i in range(local_size):
            bonelength_list[i] = euclidean_distance(cache=pos_xyz,
                                                     first_idx=first_idx_list[i],
                                                     second_idx=second_idx_list[i],
                                                     cache_offset=offset)

            if self.max_bonelength < bonelength_list[i]:
                self.max_bonelength = bonelength_list[i]

        return bonelength_list

    #TODO: check does beta represent coeff. of std.
    def generate_random_beta(self, std_range_list=[(-5,5)], style='uniform'):
        local_data = self.data[self.data_idx]
        if len(std_range_list) == 1:
            std_range_list += std_range_list * local_data.num_beta

        for beta_idx in range(local_data.num_beta):
            std_range = std_range_list[beta_idx]
            if style=='uniform':
                local_data.beta[beta_idx] = np.random.uniform(low=std_range[0], high=std_range[1])
            elif style=='gaussian':
                local_data.beta[beta_idx] = np.random.normal(loc=0, scale=1)

        else:
            assert(True, "Support styles are 'uniform' or 'gaussian'")


    def generate_star(self):
        local_data = self.data[self.data_idx]

        local_cache = np.zeros(self.shapeblendshape.shape)
        #local_data.shape = np.zeros(self.shapeblendshape.shape)

        for i in range(local_data.num_beta):
            local_cache[:,:,i] = self.shapeblendshape[:,:,i] * local_data.beta[i]
            #at each shape
            #local_data.shape[:,:,i] = self.template + local_cache[:,:,i]
        #local_data.deltashape = local_cache

        #TODO: need to change(mesh shape is default shape without pose
        local_data.mesh_shape = np.zeros((local_cache.shape[0],local_cache.shape[1]))
        #for i in range(local_data.num_beta):
            #local_data.mesh_shape += local_data.deltashape[:,:,i]
        local_data.mesh_shape += self.template

    def generate_bone_length(self):
        local_data = self.data[self.data_idx]

        #local_model.v_shape and local_model.v_posed are different.
        #v_posed = v_shape + 0 degree pose blendshape
        #Thus, local_data.mesh_shape == local_model.v_shape
        local_model = STAR(gender=self.gender, num_betas=local_data.num_beta, pose='', betas=ch.array(local_data.beta))
        local_joint = np.array(local_model.J_transformed)
        local_data.joint = local_joint
        local_data.mesh_shape_pos = np.array(local_model)

        local_data.bonelength = self.cal_bonelength_both(np.ravel(local_data.joint))

    def generate(self, std_range_list=[(-5,5)]):
        for self.data_idx in range(self.num_data):
            self.generate_random_beta(std_range_list=std_range_list)
            self.generate_star()
            self.generate_bone_length()

            # if self.data_idx % 50 == 0:
            #     print(self.data_idx)

    # def save_data(self, save_path):
    #     IsAlreadyGeneratedData = (self.data_idx + 1 == self.num_data)
    #     if not IsAlreadyGeneratedData:
    #         self.generate()
    #
    #     with open(save_path, 'wb') as output:
    #         pickle.dump(self.data, output, pickle.HIGHEST_PROTOCOL)

    def save_data_npy(self, save_path):
        local_data = self.data[self.data_idx]
        IsAlreadyGeneratedData = (local_data.mesh_shape_pos is not None)
        if not IsAlreadyGeneratedData:
            self.generate()

        np.savez_compressed(save_path,
                 #deltashape=local_data.deltashape,
                 #shape=local_data.shape,
                 beta=local_data.beta,
                 mesh_shape=local_data.mesh_shape,
                 mesh_shape_pos=local_data.mesh_shape_pos,
                 joint=local_data.joint,
                 bonelength=local_data.bonelength)

    # def load_data(self, load_path):
    #     with open(load_path, 'rb') as input:
    #         self.data = pickle.load(input)


    #TODO: Make more than adding just one data
    #https://jangjy.tistory.com/330
    def load_data_npy(self, load_path):
        print(self.data_idx)
        local_data = self.data[self.data_idx]
        saved_npz = np.load(load_path, allow_pickle=True)

        if self.STATUS == 0:
            # local_data.deltashape = saved_npz['deltashape']
            # local_data.shape = saved_npz['shape']
            local_data.beta = saved_npz['beta']
            local_data.mesh_shape = saved_npz['mesh_shape']
            local_data.mesh_shape_pos = saved_npz['mesh_shape_pos']
            local_data.joint = saved_npz['joint']
            local_data.bonelength = saved_npz['bonelength']
        elif self.STATUS == 1:
            local_data.beta = saved_npz['beta']
            local_data.joint = saved_npz['joint']
            local_data.bonelength = saved_npz['bonelength']
        elif self.STATUS == 2:
            local_data.mesh_shape = saved_npz['mesh_shape']
            local_data.mesh_shape_pos = saved_npz['mesh_shape_pos']


        saved_npz.close()



#pram.
#https://www.geeksforgeeks.org/python-initialize-a-dictionary-with-only-keys-from-a-list/
class DataMerger:
    def __init__(self, root, sample, idx_list, name='final_data'):
        """

        Note:


        Args:
            root: data saved root
            sample (:obj:): one data sample
            idx_list (:list:`int`): idx_list[0] is the first index of data and idx_list[-1] is the end index.

        """
        self.root = root
        self.data_range = [idx_list[0], idx_list[-1]]
        self.data_size = idx_list[-1] - idx_list[0]
        # self.mesh_shape = sample['mesh_shape']
        # self.mesh_shape_pos = sample['mesh_shape_pos']
        # DEBUG
        # for key in sample:
        #     print(([self.data_size] + list(sample[key].shape)))
        self.data = {key: np.zeros(([self.data_size] + list(sample[key].shape))) for key in sample}
        self.name = name

    def merge(self, name='train'):
        for i in range(self.data_range[0],self.data_range[1],1):
            local_name = self.name + "_" + str(i)
            local_path = os.path.join(self.root, local_name + '.npz')
            local_data = load_data_npy(load_path=local_path, IsNeedDefault=False)

            for key in self.data:
                self.data[key][i-self.data_range[0]] += local_data[key]
            print(i)


        save_data_npy(save_path=os.path.join(self.root,name+'.npz'),local_data=self.data,IsNeedDefault=False)

def make():
    for i in range(131072):
        save_path = '../data/final_data_'+str(i)+'.npz'
        #print(save_path)
        np.random.seed(i)
        theStarData = DataGenerator(gender='female', num_data=1)
        theStarData.save_data_npy(save_path)
        #print('finish')


def merge():
    sample_path = '../data/final_data_0.npz'
    sample = load_data_npy(sample_path, IsNeedDefault=False)
    mergedDataTrain = DataMerger(root='../data',sample=sample,idx_list=[0,32768])
    mergedDataTrain.merge()

    mergedDataTest = DataMerger(root='../data',sample=sample,idx_list=[32768,32768+8192])
    mergedDataTest.merge(name='test')

    mergedDataValidation = DataMerger(root='../data',sample=sample,idx_list=[32768+8192,32768+8192+8192])
    mergedDataValidation.merge(name='validation')


def save_reference():
    local_data_path = '../data/final_data_0.npz'
    localData = DataGenerator(gender='female',data_path=local_data_path,beta=None,num_data=1,num_beta=300,STATUS=2)

    save_path = '../data/reference.npz'

    #DEBUG
    _max = -1
    _min = 1
    for index, val in np.ndenumerate(localData.shapeblendshape):
        if val > _max:
            _max = val
        if val < _min:
            _min = val
    print(_max,_min)
    # np.savez_compressed(save_path,
    #                     mesh_shape=localData.data[0].mesh_shape,
    #                     mesh_shape_pos=localData.data[0].mesh_shape_pos,
    #                     shapeblendshape=localData.shapeblendshape,
    #                     jointregressor_matrix=localData.jointregessor_matrix)

if __name__ == "__main__":
    # make()
    # merge()
    save_reference()

#-4를 넣었을떄 x,y,z가 어디까지 확장되는지 4를 넣었을떄 x,y,z가 어디까지 확장되는지
#