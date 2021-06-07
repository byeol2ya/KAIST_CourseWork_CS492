import numpy as np
from numpy.lib.financial import npv

DATAROOT = "C:/Users/Choi Byeoli/STAR-Private/Resources/f_smpl/"

class Data:
    def __init__(self) -> None:
        self.jointregressor_matrix = np.loadtxt('C:/Users/Choi Byeoli/STAR-Private/Resources/f_smpl/f_joints_mat.txt')
        self.template = np.loadtxt('C:/Users/Choi Byeoli/STAR-Private/Resources/f_smpl/f_shapeAv.txt')
        self.load_shapeblendshape()

    def load_shapeblendshape(self):
        self.shapeblendshape = np.zeros((6890,3,10),dtype=np.float)
        for i in range(10):
            temp = np.loadtxt('C:/Users/Choi Byeoli/STAR-Private/Resources/f_smpl/f_blendshape/shape'+str(i)+'.txt')
            self.shapeblendshape[:,:,i] = temp - self.template


def main():
    smplData = Data()
    print('hello')

if __name__ == "__main__":
    main()