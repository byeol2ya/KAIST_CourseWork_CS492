''' This code contains the implementation of conditional VAE
https://github.com/graviraja/pytorch-sample-codes/blob/master/conditional_vae.py
'''

#https://github.com/pytorch/pytorch/issues/9158
# GPU_NUM = 0,1 # 원하는 GPU 번호 입력
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"
GPU_VISIBLE_NUM = 2

import sys
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim.lr_scheduler
import datetime

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.datasets_v2 import StarBetaBoneLengthDataset, Normalize
import modelprob2 as md

#https://greeksharifa.github.io/references/2020/06/10/wandb-usage/
import wandb

from demo.load_chumpy import extract_obj

tm = time.localtime()
stm = time.strftime('%Y_%m_%d_%H_%M_%S', tm)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
# print(f'device: {device}')
# BATCH_SIZE = 64         # number of data points in each batch
# TRAIN_SIZE = 64
# TEST_SIZE = 64
TRAIN_SIZE = 4096
TEST_SIZE = 1024
BATCH_SIZE = 32         # number of data points in each batch
N_EPOCHS = 100           # times to run the model on complete data
INPUT_DIM_MESH = 6890*3     # size of each input
DIM_BONELENGTH = 23     # size of each input
HIDDEN_DIM = 256        # hidden dimension
LATENT_DIM = 50         # latent vector dimension
N_CLASSES = 14          # number of classes in the data
DATA_SIZE = -1
lr = 1e-2               # learning rate
FACTOR = 0.5
TRAINED_TIME = None
# TRAINED_TIME = '2021_05_06_16_35_01'

PATH = None

e = 0
task = 'train'

# Set up your default hyperparameters before wandb.init
# so they get properly set in the sweep
# last channel is always (hidden_dim, target_dim) not follow config.
hyperparameter_defaults = dict(
    name = 'defaults',
    data_size = DATA_SIZE,
    encoder_channel_size = 6,
    encoder_channels_0 = 4,
    encoder_channels_1 = 2,
    encoder_channels_2 = 2,
    encoder_channels_3 = 2,
    encoder_channels_4 = 2,
    # encoder_channels_5 = 1,
    decoder_channel_size = 5,
    decoder_channels_0 = 2,
    decoder_channels_1 = 2,
    decoder_channels_2 = 2,
    decoder_channels_3 = 2,
    #decoder_channels_4 = 1,
    latent_dim = LATENT_DIM,
    batch_size = BATCH_SIZE,
    learning_rate = lr,
    epochs = N_EPOCHS,
    input_dim_mesh = INPUT_DIM_MESH,
    dim_bonelength = DIM_BONELENGTH,
    )

#https://discuss.pytorch.org/t/pixelwise-weights-for-mseloss/1254/2
def weighted_mse_loss(input,target,weights):
    out = (input-target)**2
    weights = torch.ones(out.shape, dtype=torch.float32, device=device)
    for i in range(1, 11):
        weights[:, i-1] += 3.0/float(i)
    # out = out * weights.expand_as(out)
    #print(f'{out.shape}\n{weights.shape}')
    out = out * weights
    loss = torch.sum(out) # or sum over whatever dimensions
    return loss




def calculate_3D_loss(x, reconstructed_x):
    local_loss = pow(x - reconstructed_x, 2)
    local_loss = torch.sum(torch.sqrt(torch.sum(local_loss, -1)))

    return local_loss

def calculate_base_loss(x, reconstructed_x, mean, log_var):
    # reconstruction loss

    RCL = calculate_3D_loss(x, reconstructed_x)/6890.0*float(DIM_BONELENGTH)
    # RCL = weighted_mse_loss(reconstructed_x,x,None)
    # RCL = F.binary_cross_entropy(reconstructed_x, x, size_average=False)
    # kl divergence loss
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    # if RCL < 0:
    #     print(f'RCL: {RCL}, KLD: {KLD}, SUM: {RCL + KLD}')
    return RCL + KLD

def calculate_bonelength_loss(bonelength, reconstructed_bonelength, mean, log_var):
    # reconstruction loss
    RCL = F.mse_loss(reconstructed_bonelength, bonelength) * 10.0
    # RCL = F.binary_cross_entropy(reconstructed_bonelength, bonelength, size_average=False)

    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    # print(RCL)
    return RCL + KLD
    # return KLD
    # return RCL


# def loss_wrapper(x, reconstructed_x, z_mu, z_var, bonelength, reconstructed_bonelength, beta, reconstructed_beta):
#     # loss = calculate_base_loss(x, reconstructed_x, z_mu, z_var)
#     b_beta = 1
#     mean, log_var = z_mu, z_var
#     KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
#     RCL_x = F.mse_loss(x, reconstructed_x) * b_beta
#     RCL_bone = F.mse_loss(reconstructed_bonelength, bonelength) * b_beta
#     RCL_beta = F.mse_loss(reconstructed_beta, beta)

#     # loss = KLD + RCL_beta
#     #loss = KLD + RCL_bone + RCL_beta
#     # loss = KLD + RCL_bone + RCL_x
#     loss = KLD + RCL_x + RCL_bone + RCL_beta
#     # if task == 'train':
#     #     print(KLD.item(), RCL_x.item(), RCL_bone.item())
#     return loss

def weight_interpolate(factor, target_network, network):
    for target_param, param in zip(target_network.parameters(), network.parameters()):
        # print(f'target/origin: {factor} {target_param.data.sum()} {param.data.sum()}')
        # print(f'before: {factor} {(param.data - target_param.data).sum()}')
        target_param.data.copy_(param.data * factor + target_param.data*(1.0 - factor))
        # print(f'target/origin: {factor} {target_param.data.sum()} {param.data.sum()}')
        # print(f'after : {factor} {(param.data - target_param.data).sum()}')

def weight_add(percent, target_network, network):
    for target_param, param in zip(target_network.parameters(), network.parameters()):
        # print(f'target/origin: {factor} {target_param.data.sum()} {param.data.sum()}')
        # print(f'before: {factor} {(param.data - target_param.data).sum()}')
        target_param.data.copy_(param.data * percent + target_param.data)
        # print(f'target/origin: {factor} {target_param.data.sum()} {param.data.sum()}')
        # print(f'after : {factor} {(param.data - target_param.data).sum()}')

def weight_self_divide(percent, target_network):
    for target_param in target_network.parameters():
        # print(f'target/origin: {factor} {target_param.data.sum()} {param.data.sum()}')
        # print(f'before: {factor} {(param.data - target_param.data).sum()}')
        target_param.data.copy_(target_param.data * percent)

def train(model,model_jacobian,model_temp,train_iterator,optimizer,optimizer_jacobian):
    global generated_beta_list, generated_bonelength_list, task
    task = 'train'
    # set the train mode
    model.train()
    model_jacobian.train()

    # loss of the epoch
    train_loss = 0
    losses = {'KLD':0.0,'RCL_bone':0.0,'RCL_beta':0.0,'Cov':0.0,'Jac':0.0}
    training_bias = (TRAIN_SIZE - 1) / (BATCH_SIZE - 1)
    
    optimizer.zero_grad()
    optimizer_jacobian.zero_grad()

    for i, (beta, bonelength) in enumerate(train_iterator):
        # print(i)
        beta = beta.to(device)
        bonelength = bonelength.to(device)
        # reshape the data into [batch_size, 784]
        # print(x.shape)
        # x = x.view(-1, INPUT_DIM)
        # beta = beta.to(device)

        # y = y.view(-1, N_CLASSES)
        # bonelength = bonelength.to(device)

        weight_interpolate(1.0, model_temp, model)
        # forward pass
        # loss
        generated_beta, generated_bonelength, mu_Style, std_Style, bonelength_reduced, mesh, generated_mesh = model(beta, bonelength)
        if e == N_EPOCHS-1 and i == 0:
            generated_beta_list += [[beta, generated_beta]]
            generated_bonelength_list += [[bonelength, generated_bonelength]]
            
        loss_func_output = md.loss_function(model=model,
                                X=mesh, X_hat=generated_mesh,
                                bonelength=bonelength, generated_bonelength=generated_bonelength,
                                mu_S=mu_Style, std_S=std_Style,
                                mu_B=bonelength_reduced, beta=beta,
                                generated_beta=generated_beta,
                                training_bias=training_bias,
                                BATCH_SIZE=BATCH_SIZE, device=device)

        loss_a, sublosses, L_covarpen = loss_func_output
        
        # backward pass
        loss_a.backward()
        train_loss += loss_a.item()

        losses['KLD'] += sublosses['KLD'].item()
        losses['RCL_bone'] += sublosses['RCL_bone'].item()
        losses['RCL_x'] += sublosses['RCL_x'].item()
        losses['Cov'] += L_covarpen.item()

        # update the weights
        optimizer.step()
        optimizer_jacobian.step()

        
        # update the gradients to zero
        optimizer.zero_grad()


        for i in range(10):
            weight_interpolate(1.0, model_jacobian, model_temp)
            loss_jacobian = model_jacobian(beta=None, bonelength=None, mu_S=mu_Style, mu_B=bonelength_reduced, element_idx=i)
            loss_jacobian.backward()
            train_loss += loss_jacobian.item()
            losses['Jac'] += loss_jacobian.item()
            optimizer_jacobian.zero_grad()
            weight_add(1.0, model, model_jacobian)
        
        weight_add(-10.0, model, model_temp)

    
    train_loss =  train_loss
    return train_loss, losses


def test(model,model_jacobian,model_temp,test_iterator,IsNeedSave=False):
    global generated_beta_list, generated_bonelength_list, task
    task = 'test'
    # set the evaluation mode
    model.eval()

    # test loss for the data
    test_loss = 0
    losses = {'KLD':0.0,'RCL_bone':0.0,'RCL_beta':0.0,'Cov':0.0,'Jac':0.0}
    training_bias = (TEST_SIZE - 1) / (BATCH_SIZE - 1)

    # we don't need to track the gradients, since we are not updating the parameters during evaluation / testing
    with torch.no_grad():
        for i, (beta, bonelength) in enumerate(test_iterator):
            # reshape the data
            # x = x.view(-1, INPUT_DIM)
            beta = beta.to(device)

            # y = y.view(-1, N_CLASSES)
            bonelength = bonelength.to(device)

            generated_beta, generated_bonelength, mu_Style, std_Style, bonelength_reduced, mesh, generated_mesh = model(beta, bonelength)

            if i == 0 and IsNeedSave:
                generated_beta_list += [[beta, generated_beta]]
                generated_bonelength_list += [[bonelength, generated_bonelength]]

            loss_func_output = md.loss_function(model=model,
                                    X=mesh, X_hat=generated_mesh,
                                    bonelength=bonelength, generated_bonelength=generated_bonelength,
                                    mu_S=mu_Style, std_S=std_Style,
                                    mu_B=bonelength_reduced, beta=beta,
                                    generated_beta=generated_beta,
                                    training_bias=training_bias,
                                    BATCH_SIZE=BATCH_SIZE, device=device)

            
            loss_a, sublosses, L_covarpen = loss_func_output
            
            test_loss += loss_a.item()

            losses['KLD'] += sublosses['KLD'].item()
            losses['RCL_bone'] += sublosses['RCL_bone'].item()
            losses['RCL_x'] += sublosses['RCL_x'].item()
            losses['Cov'] += L_covarpen.item()

            for i in range(10):
                loss_jacobian = model_jacobian(beta=None, bonelength=None, mu_S=mu_Style, mu_B=bonelength_reduced)
                test_loss += loss_jacobian.item()
                losses['Jac'] += loss_jacobian.item()

            
    test_loss =  test_loss
    return test_loss, losses

"""
#pram: delta should be same size with LATENT_DIM or 1
def get_star(model,beta=None,delta=None):
    if delta is not None:
        assert delta.size != 1 or delta.size != LATENT_DIM, 'delta should be same size with LATENT_DIM or 1'

    #TODO: find optimal transform standardation or normalization
    //transforms=transformation()
    dataset = StarBetaBoneLengthDataset(
        npy_file=None,
        transform=transforms,
        length=DATA_SIZE,
        value=beta
    )
    iterator = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model.eval()

    with torch.no_grad():
        for i, (x, y) in enumerate(iterator):
            x = x.to(device)
            y = y.to(device)

            # forward pass
            reconstructed_x, z_mu, z_var = model(x, y, delta=torch.tensor(delta,device=device))

            # loss
            loss = calculate_loss(x, reconstructed_x, z_mu, z_var)
            print(f'loss: {loss.item()}')

    return reconstructed_x
"""


def load_trained_model(model, model_jacobian, _dataset, _iterator):
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']

    e = epoch

    test_loss,test_losses = test(model, model_jacobian, _iterator, IsNeedSave=True)

    test_loss /= len(_dataset)

    print(f'Epoch {e}, The Loss: {test_loss:.4f}')
    # # create a random latent vector
    # z = torch.randn(1, LATENT_DIM).to(device)
    #
    # # pick randomly 1 class, for which we want to generate the data
    # y = torch.randint(0, N_CLASSES, (1, 1)).to(dtype=torch.long)
    # print(f'Generating a {y.item()}')
    #
    # y = idx2onehot(y).to(device, dtype=z.dtype)
    # z = torch.cat((z, y), dim=1)
    #
    # reconstructed_img = model.decoder(z)
    # img = reconstructed_img.view(28, 28).data
    #
    # plt.figure()
    # plt.imshow(img.cpu(), cmap='gray')
    # plt.show()


def save_trained_model(model, model_jacobian, model_temp,train_dataset, test_dataset, train_iterator, test_iterator, optimizer, optimizer_jacobian, scheduler, scheduler_jacobian):
    global e
    e = 0
    best_test_loss = float('inf')
    example_images = []

    for e in range(N_EPOCHS):
        train_loss, train_losses = train(model, model_jacobian, model_temp, train_iterator, optimizer, optimizer_jacobian)
        test_loss,test_losses = test(model, model_jacobian, model_temp, test_iterator)

        train_loss /= len(train_dataset)
        test_loss /= len(test_dataset)

        metrics = {'train_loss': train_loss, 
                    'test_loss': test_loss,
                    'train_Jac': train_losses['Jac']/len(train_dataset),
                    'test_Jac': test_losses['Jac']/len(test_dataset),
                    'train_KLD': train_losses['KLD']/len(train_dataset),
                    'train_RCL_bone': train_losses['RCL_bone']/len(train_dataset),
                    'train_RCL_beta': train_losses['RCL_x']/len(train_dataset),
                    'train_Cov': train_losses['Cov']/len(train_dataset),
                    'test_KLD': test_losses['KLD']/len(test_dataset),
                    'test_RCL_bone': test_losses['RCL_bone']/len(test_dataset),
                    'test_RCL_beta': test_losses['RCL_x']/len(test_dataset),
                    'test_Cov': test_losses['Cov']/len(test_dataset)
                    }
        wandb.log(metrics)
        # wandb.log({
        #     "Examples": example_images,
        #     "Train Loss":train_loss,
        #     "Test Loss": test_loss})


        print(f'Epoch {e}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

        # if best_test_loss > test_loss:
        #     best_test_loss = test_loss
        #     patience_counter = 1
        # else:
        #     patience_counter += 1
        #
        # if patience_counter > 10:
        #     break
            #pass
        scheduler.step()

    # https://tutorials.pytorch.kr/beginner/saving_loading_models.html
    torch.save({
        'epoch': N_EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, PATH)


def wrapper():
    setup_trained_model()


def transformation():
    mean = 0.0
    std = 5.0

    transform = transforms.Compose([
        Normalize(mean=mean, std=std),
        #ToTensor, #왜 안써야하지?
    ])

    return transform

#https://sanghyu.tistory.com/19
def load_reference(path='./data/reference.npz') -> dict:
    """Example function with PEP 484 type annotations.

    Args:
        ret['shapeblendshape]: shape is (6890,3,300)

    Returns:
        The return value. True for success, False otherwise.

    """
    sample = np.load(path, allow_pickle=True)
    ret = {key: torch.tensor(sample[key],dtype=torch.float32) for key in sample}
    return ret

def setup_trained_model():
    global PATH, generated_beta_list, config, TRAINED_TIME

    # wandb.init(config=hyperparameter_defaults, project="STARVAE-7-with-Cov-Jac")
    # config = wandb.config

    if TRAINED_TIME is None:
        TRAINED_TIME = stm
        PATH = '/Data/MGY/STAR_Private/resources/cvae_' + TRAINED_TIME + '.pt'
    else:
        PATH = '/Data/MGY/STAR_Private/resources/cvae_' + TRAINED_TIME+ '.pt'

    #wandb.config.update()
    transform = transformation()

    train_dataset = StarBetaBoneLengthDataset(
        path='./data/train.npz',
        transform=transform,
        device=None,
        debug=TRAIN_SIZE
    )

    test_dataset = StarBetaBoneLengthDataset(
        path='./data/test.npz',
        transform=transform,
        device=None,
        debug=TEST_SIZE
    )

    validation_dataset = StarBetaBoneLengthDataset(
        path='./data/validation.npz',
        transform=transform,
        device=None,
        debug=TEST_SIZE
    )

    train_iterator = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_iterator = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    validation_dataset = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


    reference = load_reference()
    # model = md.CVAE(reference=reference, BATCH_SIZE=BATCH_SIZE, device=device).to(device)
    # model_jacobian = md.CVAE(reference=reference, BATCH_SIZE=BATCH_SIZE, device=device, IsSupporter=True).to(device)
    model = md.CVAE(reference=reference, BATCH_SIZE=BATCH_SIZE, device=device)
    model_jacobian = md.CVAE(reference=reference, BATCH_SIZE=BATCH_SIZE, device=device, IsSupporter=True)
    model_temp = md.CVAE(reference=reference, BATCH_SIZE=BATCH_SIZE, device=device, IsSupporter=True)

    # if GPU_VISIBLE_NUM > 1:
    #     print("Let's use", GPU_VISIBLE_NUM, "GPUs!")
    #     # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    #     model = nn.DataParallel(model)
    #     model_jacobian = nn.DataParallel(model_jacobian)


    model.to(device)
    model_jacobian.to(device)
    model_temp.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer_jacobian = optim.Adam(model.parameters(), lr=lr)
    #http://www.gisdeveloper.co.kr/?p=8443
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    scheduler_jacobian = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    if not os.path.isfile(PATH):
        #sort, dnn style, load data, input/loss, version
        wandb.init(project="STARVAE-7-with-Cov-Jac")
        wandb.watch(model)
        save_trained_model(model, model_jacobian, model_temp, train_dataset, test_dataset, train_iterator, test_iterator, optimizer, optimizer_jacobian, scheduler, scheduler_jacobian)
    else:
        load_trained_model(model, model_jacobian, train_dataset, train_iterator)

    return model, train_iterator, test_iterator, validation_dataset

def main():
    global generated_beta_list, generated_bonelength_list
    generated_beta_list = []
    generated_bonelength_list = []
    #for i in range(2):
    wrapper()
    # print(hyperparameter_defaults)
    #https://stackoverflow.com/questions/54268029/how-to-convert-a-pytorch-tensor-into-a-numpy-array
    #torch.set_printoptions(precision=2)

    # for beta_pair in generated_beta_list:
    #     original_val = beta_pair[0].detach().cpu().numpy()[0, :]
    #     new_val = beta_pair[1].detach().cpu().numpy()[0, :]
    #     print(f'\n\n\n\n*--------------------beta------------------*')
    #     print(f'original:\n{original_val}')
    #     print(f'new:\n{new_val}')
    #     print(f'divide:\n{original_val - new_val}')

    np.set_printoptions(precision=3, suppress=True)
    # for bone_pair in generated_bonelength_list:
    bone_pair = generated_bonelength_list[-1]
    original_val = bone_pair[0].detach().cpu().numpy()[0, :]
    new_val = bone_pair[1].detach().cpu().numpy()[0, :]
    print(f'\n\n\n\n*--------------------bone------------------*')
    print(f'original:\n{original_val}')
    print(f'new:\n{new_val}')
    print(f'divide:\n{(abs(original_val) - abs(new_val))}')
    print(f'percent:\n{(abs(original_val) - abs(new_val))/abs(original_val) * 100.0}')

    beta_pair = generated_beta_list[-1]
    original_val = beta_pair[0].detach().cpu().numpy()[0,:]
    new_val = beta_pair[1].detach().cpu().numpy()[0,:]
    print(f'\n\n\n\n*--------------------beta------------------*')
    print(f'original:\n{original_val}')
    print(f'new:\n{new_val}')
    print(f'divide:\n{(abs(original_val) - abs(new_val))}')
    print(f'percent:\n{(abs(original_val) - abs(new_val))/abs(original_val) * 100.0}')
    extract_obj(save_path="/Data/MGY/STAR_Private/outputs/",name=TRAINED_TIME+"_original",betas=original_val)
    extract_obj(save_path="/Data/MGY/STAR_Private/outputs/",name=TRAINED_TIME+"_new",betas=new_val)

    # for beta_pair in generated_beta_list:
    #         original_val = beta_pair[0].detach().cpu().numpy()[0,:]
    #         new_val = beta_pair[1].detach().cpu().numpy()[0,:]
    #         print(f'\n\n\n\n*--------------------beta------------------*')
    #         print(f'original:\n{original_val}')
    #         print(f'new:\n{new_val}')
    #         print(f'divide:\n{(abs(original_val) - abs(new_val))}')
    #         print(f'percent:\n{(abs(original_val) - abs(new_val))/abs(original_val) * 100.0}')
    #         extract_obj(save_path="/Data/MGY/STAR_Private/",name="stm"+"_original",betas=original_val)
    #         extract_obj(save_path="C:/Users/TheOtherMotion/Documents/GitHub/STAR_Private/",name="stm"+"_new",betas=new_val)


if __name__ == "__main__":
    #https://www.studytonight.com/post/calculate-time-taken-by-a-program-to-execute-in-python
    start = time.time()
    main()
    end = time.time()
    print(f"Runtime of the program is {str(datetime.timedelta(seconds=end-start))}")
