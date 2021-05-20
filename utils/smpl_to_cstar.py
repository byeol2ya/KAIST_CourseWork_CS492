import numpy as np
from numpy.lib.financial import npv

DATAROOT = "C:/Project/Contents/Resources/f_smpl/"

class Data:
    def __init__(self) -> None:
        self.jointregressor_matrix = np.loadtxt('/Data/MGY/STAR-Private/smpl/v_f_joints_mat.txt')
        self.template = np.loadtxt('/Data/MGY/STAR-Private/smpl/v_f_shapeAv.txt')
        self.load_shapeblendshape()

    def load_shapeblendshape(self):
        self.shapeblendshape = np.zeros((6890,3,10),dtype=np.float)
        for i in range(10):
            temp = np.loadtxt('/Data/MGY/STAR-Private/smpl/f_blendshape/v_shape'+str(i)+'.txt')
            self.shapeblendshape[:,:,i] = temp - self.template


def main():
    smplData = Data()
    print('hello')

if __name__ == "__main__":
    main()