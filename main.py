import sys
import os
sys.path.append('C:/Users/TheOtherMotion/Documents/GitHub/STAR-Private/demo/')

import demo.extract_shape_dif as esd
import demo.cal_bone_length as cbl
import numpy as np
import chumpy as ch

from pymoo.algorithms.nsga2 import NSGA2
from pymoo.model.problem import Problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

#https://pymoo.org/getting_started.html#5.-Optimize
class MyProblem(Problem):
    NUM_BETA = 300
    NUM_BONE = 23
    NUM_JOINT = 24

    def __init__(self, target_bone_idx_list= [20-1,21-1]):
        super().__init__(n_var=self.NUM_BETA,
                         n_obj=self.NUM_JOINT,
                         n_constr=self.NUM_BETA,
                         xl=np.ones([self.NUM_BETA])*-5.0,
                         xu=np.ones([self.NUM_BETA])*5.0,
                         elementwise_evaluation=True)

        self.left_idx_list = [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
        self.right_idx_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
        self.NUM_OFFSET = 3
        self.del_joints = np.zeros((self.NUM_BETA,self.NUM_JOINT,self.NUM_OFFSET))
        self.avg_joints = np.zeros((self.NUM_JOINT,3))
        self.gender = 'female'
        self.preprocessing()
        # self.target_bone_idx_list = [20-1,21-1]
        self.target_bone_idx_list = target_bone_idx_list

    def _evaluate(self, x, out, *args, **kwargs):
        local_del_joints = np.zeros((self.NUM_BETA,self.NUM_JOINT,self.NUM_OFFSET))
        offset = 3
        f_list = []

        for beta_idx in range(self.NUM_BETA):
            local_del_joints[beta_idx,:,:] =self.del_joints[beta_idx,:,:]*x[beta_idx]
        for bone_idx in range(self.NUM_BONE):
            local_left_idx = self.left_idx_list[bone_idx]
            local_right_idx = self.right_idx_list[bone_idx]
            temp = local_del_joints[:,local_left_idx,:]
            ttt = np.sum(local_del_joints[:,local_left_idx,:],axis=0)

            local_left_joint = self.avg_joints[local_left_idx,:] + np.sum(local_del_joints[:,local_left_idx,:],axis=0)
            local_right_joint = self.avg_joints[local_right_idx,:] + np.sum(local_del_joints[:,local_right_idx,:],axis=0)

            if bone_idx not in self.target_bone_idx_list:
                pass
                # f_list += [np.absolute(np.linalg.norm(local_left_joint - local_right_joint) - self.bone_length[bone_idx])]
            else:
                # f_list += [(np.linalg.norm(local_left_joint - local_right_joint) - self.bone_length[bone_idx]) * -1]
                f_list += [np.linalg.norm(local_left_joint - local_right_joint) * -1]


        # f1 = x[0] ** 2 + x[1] ** 2
        # f2 = (x[0] - 1) ** 2 + x[1] ** 2
        #
        # g1 = 2 * (x[0] - 0.1) * (x[0] - 0.9) / 0.18
        # g2 = - 20 * (x[0] - 0.4) * (x[0] - 0.6) / 4.8

        out["F"] = f_list
        #out["G"] = [g1, g2]
        out["G"] = []

    def preprocessing(self):
        name = "f_weight"
        self.joint_weight = esd.extractor_weight_and_joint(gender=self.gender, save_path=None, name=name, IsWeight=False,
                                                      IsSave=False)

        model_dict = esd.get_gender_model(self.gender)
        del_shapeblendshape = np.array(ch.array(model_dict['shapedirs']))

        for i in range(300):
            local_del_shapeblendshape = del_shapeblendshape[:, :, i]
            self.del_joints[i] = np.matmul(self.joint_weight, local_del_shapeblendshape)
        print("ha")

        self.get_avg_bone()

    def get_avg_bone(self):
        model_dict = esd.get_gender_model(self.gender)
        v_avg = np.array(ch.array(model_dict['v_template']))
        self.avg_joints = np.matmul(self.joint_weight, v_avg)
        bone_xyz = v_avg.flatten()
        self.bone_length = cbl.cal_bone(bone_xyz,left_idx_list=self.left_idx_list,right_idx_list=self.right_idx_list,offset=3) #23
        print("ha")
    

F ={}

for i in range(0,23):
    problem = MyProblem(target_bone_idx_list=[i])
    print(f'{i} start')

    algorithm = NSGA2(pop_size=100)

    res = minimize(problem,
                   algorithm,
                   ("n_gen", 100),
                   verbose=True,
                   seed=1)
    F[i] = res.F[-1]
    #print(res.X[-1])
    # plot = Scatter()
    # plot.add(res.F, color="red")
    # plot.show()

for key in F:
    print(f'{key}: {F[key]}')