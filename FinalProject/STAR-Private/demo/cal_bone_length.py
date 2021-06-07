import numpy as np

max_val = 0


def cal_bone(pos_xyz, left_idx_list = [0,1,4,7,0,3,6,9,13,16,18,20,9,12], right_idx_list = [1,4,7,10,3,6,9,13,16,18,20,22,12,15], offset=3):
    global max_val
    local_size = len(left_idx_list)
    ret = np.zeros(local_size)

    for i in range(local_size):
        left_idx = left_idx_list[i]
        right_idx = right_idx_list[i]
        a = np.array(pos_xyz[offset*left_idx:offset*(left_idx+1)])
        b = np.array(pos_xyz[offset*right_idx:offset*(right_idx+1)])
        dist = np.linalg.norm(a - b)

        ret[i] = dist

        if max_val < ret[i]:
            max_val = ret[i]
    return ret


def cal_bone_both(pos_xyz,offset=3):
    left_idx_list = [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
    right_idx_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    cal_bone(pos_xyz=pos_xyz,left_idx_list=left_idx_list,right_idx_list=right_idx_list,offset=offset)


def main():
    num_data = 30000
    ret = np.zeros((num_data, 314))

    left_idx_list = [0,1,4,7,0,3,6,9,13,16,18,20,9,12]
    right_idx_list = [1,4,7,10,3,6,9,13,16,18,20,22,12,15]
    x_save_load = np.load('./saved.npy')
    for i in range(0,num_data):
       ret[i,:300] = x_save_load[i,:300]
       ret[i,300:] = cal_bone(x_save_load[i,300:],left_idx_list,right_idx_list)
       #print(x_save_load[i,:])

    print(f'max bone length: {max_val}')
    np.save('./saved_bonelength.npy', ret)

def cut():
    ret = np.load('./saved_210217.npy')
    np.save('./saved_bonelength_train.npy', ret[:20000,:])
    np.save('./saved_bonelength_test.npy', ret[20000:25000,:])
    np.save('./saved_bonelength_validation.npy', ret[25000:30000,:])

if __name__ == "__main__":
    #main()
    cut()