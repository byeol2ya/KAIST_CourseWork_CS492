import numpy as np
import random
#from sklearn.preprocessing import minmax_scale

dataN = 30000
num_rest_of_betas = 290
front_ten_betas_ref = np.array([-1.16296, -1.41088, -0.698131, 0.160221, 0.569289, 0.290525, 0.689072, 0.363366, 0.696619, 0.497472, -3.0991, -1.48215, 0.298001, 0.526291, 0.9737, 0.241297, 0.410001, 0.207632, 1.12449, 0.808739, -3.19644, -1.39761, -0.214087, 1.08834, 0.09887, 1.06944, -0.00800285, 0.803048, 1.00178, 0.759757, -4.25677, -0.771968, 1.74032, -0.144907, 2.55413, -0.135255, 0.820355, 0.0371761, -0.517931, 0.622173, -1.12517, -0.942627, -0.101471, -0.811923, 0.72969, -0.0277587, 0.818959, 0.883574, 0.0491776, 0.508118])
#front_ten_betas_scaled = minmax_scale(front_ten_betas_ref, axis=0, copy=True) # axis 맞는지 확인 해야함
front_ten_betas_ref = front_ten_betas_ref.reshape(5, 10)

# 열의 최솟값, 최댓값 구해서 넣음
beta_min = np.min(front_ten_betas_ref, axis=0)
beta_max = np.max(front_ten_betas_ref, axis=0)

final_beta = np.zeros((dataN, 300))

for i in range(dataN):
    betas = np.array([])
    for j in range(10):
        betas = np.append(betas, random.uniform(beta_min[j], beta_max[j]))

    rest_betas = np.array(np.random.rand(num_rest_of_betas))
    beta_array = np.append(betas, rest_betas)
    final_beta[i] = beta_array

np.save('./saved_300betas.npy', final_beta)
