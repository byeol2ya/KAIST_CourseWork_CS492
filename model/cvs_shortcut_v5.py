''' This code contains the implementation of conditional VAE
https://github.com/graviraja/pytorch-sample-codes/blob/master/conditional_vae.py
'''

# https://github.com/pytorch/pytorch/issues/9158
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
import time
import datetime
import wandb
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim.lr_scheduler
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
# https://greeksharifa.github.io/references/2020/06/10/wandb-usage/
from torch.nn.parallel.data_parallel import DataParallel

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.datasets_v5 import StarBetaBoneLengthDataset, Normalize
import model.model_shortcut_v5 as md
from demo.load_chumpy import extract_obj, save_as_obj


class PathControllTower:
    def __init__(self, root, sort=None) -> None:
        self.root = root + '/'
        self.sort = sort

    def get_train_data(self):
        return self.root + 'train.npz'

    def get_test_data(self):
        return self.root + 'test.npz'

    def get_validation_data(self):
        return self.root + 'validation.npz'

    def get_reference_data(self):
        return self.root + 'reference.npz'

    def get_sort(self):
        return self.sort

    def get_beta_size(self):
        if self.sort == 'SMPL':
            return 10
        elif self.sort == 'STAR':
            return 300
        else:
            return -1


####################################################################
# Set up your default hyperparameters before wandb.init
# so they get properly set in the sweep
# last channel is always (hidden_dim, target_dim) not follow config.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
e = 0
task = 'train'
####################################################################
# ROOT = (os.path.dirname(os.path.realpath(__file__)))
STARPATH = PathControllTower(root='../data/', sort='STAR')
SMPLPATH = PathControllTower(root='../datasmpl/', sort='SMPL')
DATAPATH = SMPLPATH
GPU_VISIBLE_NUM = os.environ["CUDA_VISIBLE_DEVICES"].count(',') + 1

TIMEPATH = None
# TIMEPATH = '2021_05_22_16_59_22_99'
tm = time.localtime()
TRAINED_TIME = time.strftime('%Y_%m_%d_%H_%M_%S', tm)
####################################################################
# TRAIN_SIZE = 64
# TEST_SIZE = 64
TRAIN_SIZE = 2048
TEST_SIZE = 64
BATCH_SIZE = 32  # number of data points in each batch
N_EPOCHS = 100  # times to run the model on complete data
lr = 1e-2  # learning rate
####################################################################

loss_prev = 0.1
loss_current = 0.0


####################################################################
def _save(func, args, path):
    temp = path.split(os.path.sep)
    dirpath = path[:-(len(temp[-1]) + 1)]
    os.makedirs(dirpath, exist_ok=True)
    func(*args)


def _getfloat(value, attr='item'):
    if hasattr(value, attr):
        return value.item()
    else:
        return value


def weight_interpolate(factor, target_network, network):
    for target_param, param in zip(target_network.parameters(), network.parameters()):
        target_param.data.copy_(param.data * factor + target_param.data * (1.0 - factor))


def weight_add(percent, target_network, network):
    for target_param, param in zip(target_network.parameters(), network.parameters()):
        target_param.data.copy_(param.data * percent + target_param.data)


def weight_self_divide(percent, target_network):
    for target_param in target_network.parameters():
        target_param.data.copy_(target_param.data * percent)


def test(model, model_jacobian, test_iterator, _beta = None, _z = None, _bone = None):
    global generated_beta_list, generated_bonelength_list, task
    task = 'test'
    # set the evaluation mode
    model.eval()

    # test loss for the data
    test_loss = 0
    losses = {'KLD': 0.0, 'RCL_bone': 0.0, 'RCL_x': 0.0, 'Cov': 0.0, 'Jac': 0.0, 'RCL_x_euclidean_distance': 0.0}
    training_bias = (TEST_SIZE - 1) / (BATCH_SIZE - 1)

    # we don't need to track the gradients, since we are not updating the parameters during evaluation / testing
    with torch.no_grad():
        # weight_interpolate(1.0, model_jacobian, model)

        _beta = torch.tensor(_beta, dtype=torch.float32, device=device)
        _z = torch.tensor(_z, dtype=torch.float32, device=device)
        _bone = torch.tensor(_bone, dtype=torch.float32, device=device)
        for i, (beta, shapeblendshape, mesh_shape_pos, jointregressor_matrix) in enumerate(test_iterator):
            beta = beta.to(device)
            shapeblendshape = shapeblendshape.to(device)
            mesh_shape_pos = mesh_shape_pos.to(device)
            jointregressor_matrix = jointregressor_matrix.to(device)
            input_beta = torch.zeros(beta.shape,dtype=torch.float32,device=device)
            input_z = torch.zeros((32,10),dtype=torch.float32,device=device)
            input_bone = torch.zeros((32,23),dtype=torch.float32,device=device)
            for i in range(BATCH_SIZE):
                input_beta[i,:] = _beta
                input_z[i,:] = _z
                input_bone[i,:] = _bone

            output = model(beta=None,
                           shapeblendshape=shapeblendshape,
                           mesh_shape_pos=mesh_shape_pos,
                           jointregressor_matrix=jointregressor_matrix,
                           _select_beta= input_beta,
                           _select_z = input_z,
                           _select_bone = input_bone)

            generated_beta, generated_real_bone_length, mu_Style, std_Style, bonelength_reduced, mesh, generated_mesh, z, real_bone_length, joints, generated_joints = output

            loss_func_basic = md.loss_func_basic(model=model,
                                                 X=mesh, X_hat=generated_mesh,
                                                 bonelength=real_bone_length, generated_bonelength=generated_real_bone_length,
                                                 mu_S=mu_Style, std_S=std_Style,
                                                 mu_B=bonelength_reduced, beta=beta,
                                                 generated_beta=generated_beta,
                                                 training_bias=training_bias,
                                                 BATCH_SIZE=BATCH_SIZE, device=device)

            loss_a, sublosses, L_covarpen = loss_func_basic

            test_loss += _getfloat(loss_a)

            losses['KLD'] += _getfloat(sublosses['KLD'])
            losses['RCL_bone'] += _getfloat(sublosses['RCL_bone'])
            losses['RCL_x'] += _getfloat(sublosses['RCL_x'])
            losses['RCL_x_euclidean_distance'] += _getfloat(sublosses['RCL_x_euclidean_distance'])
            losses['Cov'] += _getfloat(L_covarpen)

            ############################################################################################
            """
            shapeblendshape = shapeblendshape.to(device)
            mesh_shape_pos = mesh_shape_pos.to(device)
            jointregressor_matrix = jointregressor_matrix.to(device)
            z_Style = z_Style.to(device)
            bonelength_reduced = bonelength_reduced.to(device)
            output = model_jacobian(beta=None,
                                    shapeblendshape=shapeblendshape,
                                    mesh_shape_pos=mesh_shape_pos,
                                    jointregressor_matrix=jointregressor_matrix,
                                    mu_S=z_Style, mu_B=bonelength_reduced,
                                    element_idx=-1)
            mu_S, mu_B = output

            for i in range(10):
                shapeblendshape = shapeblendshape.to(device)
                mesh_shape_pos = mesh_shape_pos.to(device)
                jointregressor_matrix = jointregressor_matrix.to(device)
                z_Style = z_Style.to(device)
                bonelength_reduced = bonelength_reduced.to(device)
                output = model_jacobian(beta=None,
                                        shapeblendshape=shapeblendshape,
                                        mesh_shape_pos=mesh_shape_pos,
                                        jointregressor_matrix=jointregressor_matrix,
                                        mu_S=z_Style, mu_B=bonelength_reduced,
                                        element_idx=i)

                mu_hat_S, mu_hat_B, eps_S, eps_B = output
                loss_func_jacobian = md.loss_func_jacobian(mu_S, mu_B, mu_hat_S, mu_hat_B, eps_S, eps_B, BATCH_SIZE)
                loss_jacobian = loss_func_jacobian
                test_loss += _getfloat(loss_jacobian)
                losses['Jac'] += _getfloat(loss_jacobian)
            """

            ############################################################################################

            # save_as_obj(model=mesh[0].cpu().data.numpy(), save_path="/Data/MGY/STAR-Private/outputs/",
            #             name=TRAINED_TIME + "_" + str(0 + BATCH_SIZE * i).zfill(4) + "_before")
            # save_as_obj(model=generated_mesh[0].cpu().data.numpy(), save_path="/Data/MGY/STAR-Private/outputs/",
            #             name=TRAINED_TIME + "_" + str(0 + BATCH_SIZE * i).zfill(4) + "_after")

            break
    test_loss = test_loss
    return test_loss, losses, mesh.cpu().data.numpy()[0], joints.cpu().data.numpy()[0], generated_mesh.cpu().data.numpy()[0], generated_joints.cpu().data.numpy()[0], real_bone_length.cpu().data.numpy()[0], generated_real_bone_length.cpu().data.numpy()[0]


def load_trained_model(model, model_jacobian, _dataset, _iterator, _beta, _z, _bone):
    checkpoint = torch.load(TIMEPATH + '.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    num_batch = 1

    e = epoch

    test_loss, test_losses, mesh, joints, generated_mesh, generated_joints, real_bone_length, generated_real_bone_length= test(model, model_jacobian, _iterator, _beta = _beta, _z = _z, _bone = _bone)

    test_loss /= num_batch

    metrics = {'loss': test_loss,
               'Jac': test_losses['Jac'] / num_batch,
               'KLD': test_losses['KLD'] / num_batch,
               'RCL_bone': test_losses['RCL_bone'] / num_batch,
               'RCL_x': test_losses['RCL_x'] / num_batch,
               'RCL_x_euclidean_distance': test_losses['RCL_x_euclidean_distance'] / num_batch,
               'Cov': test_losses['Cov'] / num_batch,
               }

    # for key in metrics:
    #     print(f'{key}: {metrics[key]}')
    print(f'real_bone_length: {real_bone_length}')
    print(f'generated_real_bone_length: {generated_real_bone_length}')
    print(f'generated - real_bone_length: {generated_real_bone_length - real_bone_length}')
    print(f'Epoch {e}, The Loss: {test_loss:.4f}')

    return test_loss, test_losses, mesh, joints, generated_mesh, generated_joints, real_bone_length, generated_real_bone_length

def transformation():
    mean = 0.0
    std = 5.0

    transform = transforms.Compose([
        Normalize(mean=mean, std=std),
        # ToTensor, #왜 안써야하지?
    ])

    return transform


def setup_trained_model(_beta, _z, _bone, IsNewBeta):
    global TIMEPATH, generated_beta_list, config, TRAINED_TIME

    transform = transformation()

    validation_dataset = StarBetaBoneLengthDataset(
        path_data=DATAPATH.get_validation_data(),
        path_reference=DATAPATH.get_reference_data(),
        transform=transform,
        debug=TEST_SIZE
    )

    validation_iterator = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = md.CVAE(BATCH_SIZE=BATCH_SIZE, IsNewBeta=IsNewBeta)
    # model_jacobian = md.CVAE(BATCH_SIZE=BATCH_SIZE, IsSupporter=True)

    assert (GPU_VISIBLE_NUM > 1, "You need at least 2 GPUs")
    print("Let's use", GPU_VISIBLE_NUM, "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    model = DataParallel(model)
    # model_jacobian = DataParallel(model_jacobian)
    # model_temp = DataParallel(model_jacobian)

    model.to(device)
    # model_jacobian.to(device)
    # model_temp.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    # optimizer_jacobian = optim.Adam(model.parameters(), lr=lr)
    # http://www.gisdeveloper.co.kr/?p=8443
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    # scheduler_jacobian = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    test_loss, test_losses, mesh, joint, generated_mesh, generated_joint, real_bone_length, generated_real_bone_length = load_trained_model(model, None, validation_dataset, validation_iterator, _beta, _z, _bone)

    return test_loss, test_losses, mesh, joint, generated_mesh, generated_joint


# 정확히는 마지막 loss 지나고 나서 구한것
def get_obj(model, save_path, name, _beta, _iterator, IsOriginal=False):
    global BATCH_SIZE
    beta = torch.zeros((BATCH_SIZE, _beta.shape[-1]), dtype=torch.float32, device=device)
    for i in range(BATCH_SIZE):
        beta[i, :] = _beta

    for i, (_, shapeblendshape, mesh_shape_pos, jointregressor_matrix) in enumerate(_iterator):
        beta = _beta.to(device)
        shapeblendshape = shapeblendshape.to(device)
        mesh_shape_pos = mesh_shape_pos.to(device)
        jointregressor_matrix = jointregressor_matrix.to(device)

        output = model(beta=beta,
                       shapeblendshape=shapeblendshape,
                       mesh_shape_pos=mesh_shape_pos,
                       jointregressor_matrix=jointregressor_matrix)

        generated_beta, generated_bonelength, mu_Style, std_Style, bonelength_reduced, mesh, generated_mesh, z_Style, bonelength = output

        if IsOriginal:
            save_as_obj(model=mesh[0].cpu().data.numpy(), save_path=save_path, name=name)
        else:
            save_as_obj(model=generated_mesh[0].cpu().data.numpy(), save_path=save_path, name=name)
        break