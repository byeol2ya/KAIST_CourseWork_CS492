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
    NUM_BONE = 24

    def __init__(self):
        super().__init__(n_var=self.NUM_BETA,
                         n_obj=self.NUM_JOINT,
                         n_constr=self.NUM_BETA,
                         xl=np.ones([self.NUM_BETA])*-3.0,
                         xu=np.ones([self.NUM_BETA])*3.0,
                         elementwise_evaluation=True)

        self.left_idx_list = [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
        self.right_idx_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

        self.del_joints = np.zeros((300,24,3))
        self.avg_joints = np.zeros((24,3))
        self.gender = 'female'
        self.preprocessing()
        self.target_bone_idx = [20-1,21-1]

    def _evaluate(self, x, out, *args, **kwargs):
        local_size = len(self.left_idx_list)
        offset = 3
        f_list = []
        for k in range(self.NUM_BONE):
            f_list += []
            if k in self.target_bone_idx:
                for i in range(local_size):
                    left_idx = self.left_idx_list[i]
                    right_idx = self.right_idx_list[i]
                    a = np.array([offset * left_idx:offset * (left_idx + 1)])
                    b = np.array(pos_xyz[offset * right_idx:offset * (right_idx + 1)])
                    dist = np.linalg.norm(a - b)
            else:
                for l in range(self.NUM_BETA):
                    xs = np.multiply(x,self.del_joints[:,:,0])
                    xs = np.multiply(xs * xs)
                    ys = np.multiply(x,self.del_joints[:,:,1])
                    ys = np.multiply(ys * ys)
                    zs = np.multiply(x,self.del_joints[:,:,2])
                    zs = np.multiply(zs * zs)


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
        bone_xyz = np.array(ch.array(model_dict['v_template'])).flatten()
        self.bone_length = cbl.cal_bone(bone_xyz,left_idx_list=self.left_idx_list,right_idx_list=self.right_idx_list,offset=3) #23
        print("ha")

problem = MyProblem()

algorithm = NSGA2(pop_size=100)

res = minimize(problem,
               algorithm,
               ("n_gen", 100),
               verbose=True,
               seed=1)

plot = Scatter()
plot.add(res.F, color="red")
plot.show()