''' This code contains the implementation of conditional VAE
https://github.com/graviraja/pytorch-sample-codes/blob/master/conditional_vae.py
'''
import os
import sys
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim.lr_scheduler

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
from utils.datasets_v2 import StarBetaBoneLengthDataset, Normalize
#https://greeksharifa.github.io/references/2020/06/10/wandb-usage/
import wandb

from demo.load_chumpy import extract_obj

tm = time.localtime()
stm = time.strftime('%Y_%m_%d_%H_%M_%S', tm)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# print(f'device: {device}')
# BATCH_SIZE = 64         # number of data points in each batch
BATCH_SIZE = 32         # number of data points in each batch
N_EPOCHS = 100           # times to run the model on complete data
INPUT_DIM_MESH = 6890*3     # size of each input
DIM_BONELENGTH = 23     # size of each input
HIDDEN_DIM = 256        # hidden dimension
LATENT_DIM = 32         # latent vector dimension
N_CLASSES = 14          # number of classes in the data
DATA_SIZE = -1
lr = 1e-1               # learning rate

PATH = None

e = 0

# Set up your default hyperparameters before wandb.init
# so they get properly set in the sweep
# last channel is always (hidden_dim, target_dim) not follow config.
hyperparameter_defaults = dict(
    data_size = DATA_SIZE,
    num_channel = 3,
    encoder_channel_size = 5,
    encoder_channels_0 = 16,
    encoder_channels_1 = 4,
    encoder_channels_2 = 2,
    encoder_channels_3 = 2,
    #encoder_channels_4 = 1,
    decoder_channel_size = 4,
    decoder_channels_0 = 2,
    decoder_channels_1 = 2,
    decoder_channels_2 = 1,
    #decoder_channels_3 = 1,
    latent_dim = 40,
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


def set_channel(channel_size, channel_list, target_dim, latent_dim, IsEncoder = True) -> list:
    '''
    Args:
        target_dim: input(encoder), output(decoder)
    '''
    ret = [1]
    accumulated_dim = 1


    for channel_no in range(channel_size):
        accumulated_dim *= channel_list[channel_no]
        if latent_dim * accumulated_dim > target_dim:
            break
        elif latent_dim * accumulated_dim == target_dim:
            ret += [accumulated_dim]
            break
        else:
            ret += [accumulated_dim]

    # ret += [latent_dim]
    return ret

def mlp_block(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.BatchNorm1d(output_dim),
        nn.ReLU(),
    )

class Encoder(nn.Module):
    ''' This the encoder part of VAE

    '''
    def __init__(self, config, input_dim, latent_dim):
        '''
        Args:
            input_dim: A integer indicating the size of input (in case of MNIST 28 * 28).
            hidden_dim: A integer indicating the size of hidden dimension.
            latent_dim: A integer indicating the latent size.
            n_classes: A integer indicating the number of classes. (dimension of one-hot representation of labels)
        '''
        super().__init__()
        config_channel_list = [config.encoder_channels_0,config.encoder_channels_1,config.encoder_channels_2,config.encoder_channels_3]
        self.encoder_channel_list = set_channel(config.encoder_channel_size-1,config_channel_list,input_dim,latent_dim)
        #https://michigusa-nlp.tistory.com/26
        #https://towardsdatascience.com/pytorch-how-and-when-to-use-module-sequential-modulelist-and-moduledict-7a54597b5f17
        mlp_blocks = [mlp_block(input_dim//dim_0, input_dim//dim_1)
                       for dim_0, dim_1 in zip(self.encoder_channel_list, self.encoder_channel_list[1:])]
        #unpacking
        #https://mingrammer.com/understanding-the-asterisk-of-python/#2-%EB%A6%AC%EC%8A%A4%ED%8A%B8%ED%98%95-%EC%BB%A8%ED%85%8C%EC%9D%B4%EB%84%88-%ED%83%80%EC%9E%85%EC%9D%98-%EB%8D%B0%EC%9D%B4%ED%84%B0%EB%A5%BC-%EB%B0%98%EB%B3%B5-%ED%99%95%EC%9E%A5%ED%95%98%EA%B3%A0%EC%9E%90-%ED%95%A0-%EB%95%8C
        self.encoder = nn.Sequential(*mlp_blocks)


        # self.fc1 = nn.Linear(input_dim, hidden_dim)
        # self.bn1 = nn.BatchNorm1d(hidden_dim)
        # self.fc2 = nn.Linear(hidden_dim, hidden_dim//4)
        # self.bn2 = nn.BatchNorm1d(hidden_dim//4)
        # self.fc3 = nn.Linear(hidden_dim//4, hidden_dim//(4*4))
        # self.bn3 = nn.BatchNorm1d(hidden_dim//(4*4))
        # self.fc1 = nn.Linear(input_dim, hidden_dim//4)
        # self.bn1 = nn.BatchNorm1d(hidden_dim//4)
        # self.fc2 = nn.Linear(hidden_dim//4, hidden_dim//16)
        # self.bn2 = nn.BatchNorm1d(hidden_dim//16)
        self.mu = nn.Linear(input_dim//self.encoder_channel_list[-1], latent_dim)
        self.var = nn.Linear(input_dim//self.encoder_channel_list[-1], latent_dim)

        # torch.nn.init.xavier_uniform_(self.encoder.weight)
        # torch.nn.init.xavier_uniform_(self.mu.weight)
        # torch.nn.init.xavier_uniform_(self.var.weight)
    def forward(self, x):
        # x is of shape [batch_size, input_dim + n_classes]

        hidden = self.encoder(x)
        # hidden = F.relu(self.bn1(self.fc1(x)))
        # hidden = F.relu(self.bn2(self.fc2(hidden)))
        #hidden = F.relu(self.bn3(self.fc3(hidden)))
        # hidden is of shape [batch_size, hidden_dim]

        # latent parameters
        mean = self.mu(hidden)
        # mean is of shape [batch_size, latent_dim]
        log_var = self.var(hidden)
        # log_var is of shape [batch_size, latent_dim]

        return mean, log_var


class Decoder(nn.Module):
    ''' This the decoder part of VAE

    '''
    def __init__(self, config, latent_dim, output_dim):
        '''
        Args:
            latent_dim: A integer indicating the latent size.
            hidden_dim: A integer indicating the size of hidden dimension.
            output_dim: A integer indicating the size of output (in case of MNIST 28 * 28).
            n_classes: A integer indicating the number of classes. (dimension of one-hot representation of labels)
        '''
        super().__init__()
        config_channel_list = [config.decoder_channels_0, config.decoder_channels_1, config.decoder_channels_2]
        self.decoder_channel_list = set_channel(config.decoder_channel_size-1, config_channel_list, output_dim, latent_dim,IsEncoder=False)
        #https://michigusa-nlp.tistory.com/26
        #https://towardsdatascience.com/pytorch-how-and-when-to-use-module-sequential-modulelist-and-moduledict-7a54597b5f17
        mlp_blocks = [mlp_block(latent_dim*dim_0, latent_dim*dim_1)
                       for dim_0, dim_1 in zip(self.decoder_channel_list, self.decoder_channel_list[1:])]
        self.decoder = nn.Sequential(*mlp_blocks)
        # self.fc1 = nn.Linear(latent_dim, latent_dim*2)
        # self.bn1 = nn.BatchNorm1d(latent_dim*2)
        # self.fc2 = nn.Linear(latent_dim*2, output_dim)
        # self.bn2 = nn.BatchNorm1d(output_dim)
        # self.fc3 = nn.Linear(output_dim, output_dim)
        # self.bn3 = nn.BatchNorm1d(output_dim)
        # self.latent_to_hidden = nn.Linear(latent_dim + n_classes, hidden_dim)
        # self.hidden_to_out = nn.Linear(hidden_dim, output_dim)
        self.linear = nn.Linear(latent_dim*self.decoder_channel_list[-1],output_dim)
        # self.linear2 = nn.Linear(output_dim*2, output_dim)

        # torch.nn.init.xavier_uniform_(self.decoder.weight)
        # torch.nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x):
        # x is of shape [batch_size, latent_dim + num_classes]
        hidden = self.decoder(x)
        generated_x = self.linear(hidden)
        # hidden = F.relu(self.bn1(self.fc1(x)))
        # hidden = F.relu(self.bn2(self.fc2(hidden)))
        # hidden = F.tanh(self.bn3(self.fc3(hidden)))
        # generated_x = self.linear1(hidden)
        # generated_x = self.linear2(generated_x)
        # output = F.relu(self.fc4(hidden))
        # x is of shape [batch_size, hidden_dim]
        # generated_x = self.linear(generated_x)
        # x is of shape [batch_size, output_dim]

        return generated_x


class CVAE(nn.Module):
    ''' This the VAE, which takes a encoder and decoder.

    '''
    def __init__(self, config, reference, num_beta=300):
        '''
        Args:
            input_dim: A integer indicating the size of input (in case of MNIST 28 * 28).
            hidden_dim: A integer indicating the size of hidden dimension.
            latent_dim: A integer indicating the latent size.
            n_classes: A integer indicating the number of classes. (dimension of one-hot representation of labels)
        '''
        super().__init__()

        self.reference = reference
        # self.shapeblendshape_shape = self.reference['shapeblendshape'].shape
        self.encoderBonelength = Encoder(config, config.input_dim_mesh + config.dim_bonelength, config.dim_bonelength)
        self.encoderShapestyle = Encoder(config, config.input_dim_mesh + config.dim_bonelength, config.latent_dim)
        self.decoder = Decoder(config, config.dim_bonelength + config.latent_dim, num_beta)


    # reparameterize
    # https://whereisend.tistory.com/54
    def reparamterization(self, z_mu, z_var, delta=None):
        std = torch.exp(z_var / 2)
        if delta is not None:
            eps = torch.ones(std.size()) * delta
        else:
            eps = torch.randn_like(std)
        x_sample = eps.mul(std).add_(z_mu)

        return x_sample

    def forward(self, beta, bonelength, delta=None):

        # x = torch.cat((x, y), dim=1)
        # x = torch.zeros(([beta.shape[0]] + list(self.shapeblendshape_shape)), device=beta.device, dtype=torch.float32)
        # for i in range(0, beta.shape[0]):
        #     x[i] = beta[i] * self.reference['shapeblendshape']

        # x = torch.zeros((beta.shape[0], self.shapeblendshape_shape[0], self.shapeblendshape_shape[2]), device=beta.device, dtype=torch.float32)
        # for i in range(0, beta.shape[0]):
        #     x[i] = beta[i] * self.reference['shapeblendshape']

        x = self.calculate_mesh(beta)
        # encode
        # bone length
        x_flat = torch.flatten(x, start_dim=1)
        x_flat = torch.cat((x_flat, bonelength), dim=1)
        z_bonelength_mu, z_bonelength_var = self.encoderBonelength(x_flat)
        z_bonelength = self.reparamterization(z_bonelength_mu, z_bonelength_var, delta=delta)

        # shape style
        z_shapestyle_mu, z_shapestyle_var = self.encoderShapestyle(x_flat)
        z_shapestyle = self.reparamterization(z_shapestyle_mu, z_shapestyle_var, delta=delta)

        # decode
        z = torch.cat((z_bonelength, z_shapestyle), dim=1)
        z_mu = torch.cat((z_bonelength_mu, z_shapestyle_mu), dim=1)
        z_var = torch.cat((z_bonelength_var, z_shapestyle_var), dim=1)
        generated_beta = self.decoder(z)
        generated_x = self.calculate_mesh(generated_beta)

        return x, generated_x, z_mu, z_var, z_bonelength, generated_beta

    def calculate_mesh(self, beta):
        x = torch.zeros([beta.shape[0]]+list(self.reference['shapeblendshape'].shape),device=device,dtype=torch.float32)
        for batch_idx in range(beta.shape[0]):
            x[batch_idx] = beta[batch_idx,:] * self.reference['shapeblendshape']
            #print(x.shape,self.reference['shapeblendshape'].shape)
        x = torch.sum(x, -1) + self.reference['mesh_shape_pos']

        return x

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

def calculate_bonelength_loss(bonelength, reconstructed_bonelength):
    # reconstruction loss
    RCL = F.mse_loss(reconstructed_bonelength, bonelength, size_average=False)
    # RCL = F.binary_cross_entropy(reconstructed_bonelength, bonelength, size_average=False)

    return RCL



def train(model,train_iterator,optimizer):
    global generated_beta_list
    # set the train mode
    model.train()

    # loss of the epoch
    train_loss = 0

    for i, (beta, bonelength) in enumerate(train_iterator):
        # reshape the data into [batch_size, 784]
        # print(x.shape)
        # x = x.view(-1, INPUT_DIM)
        # beta = beta.to(device)

        # y = y.view(-1, N_CLASSES)
        # bonelength = bonelength.to(device)

        # update the gradients to zero
        optimizer.zero_grad()

        # forward pass
        x, reconstructed_x, z_mu, z_var, z_bonelength, generated_beta = model(beta, bonelength)
        if e == N_EPOCHS-1:
            generated_beta_list += [[beta, generated_beta]]
        # loss
        loss = calculate_base_loss(x, reconstructed_x, z_mu, z_var)
        loss += calculate_bonelength_loss(bonelength, z_bonelength)
        #print(f'ori:\n{x}\nrecon:\n{reconstructed_x}\n')

        # backward pass
        loss.backward()
        train_loss += loss.item()

        # update the weights
        optimizer.step()

    return train_loss


def test(model,test_iterator):
    # set the evaluation mode
    model.eval()

    # test loss for the data
    test_loss = 0

    # we don't need to track the gradients, since we are not updating the parameters during evaluation / testing
    with torch.no_grad():
        for i, (beta, bonelength) in enumerate(test_iterator):
            # reshape the data
            # x = x.view(-1, INPUT_DIM)
            beta = beta.to(device)

            # y = y.view(-1, N_CLASSES)
            bonelength = bonelength.to(device)

            # forward pass
            x, reconstructed_x, z_mu, z_var, z_bonelength, generated_beta = model(beta, bonelength)

            # loss
            loss = calculate_base_loss(x, reconstructed_x, z_mu, z_var)
            loss += calculate_bonelength_loss(bonelength, z_bonelength)
            test_loss += loss.item()

    return test_loss

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


def load_trained_model(model, optimizer):
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']

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


def save_trained_model(model, train_dataset, test_dataset, train_iterator, test_iterator, optimizer, scheduler):
    global e
    e = 0
    best_test_loss = float('inf')
    example_images = []

    for e in range(N_EPOCHS):
        train_loss = train(model, train_iterator, optimizer)
        test_loss = test(model, test_iterator)

        train_loss /= len(train_dataset)
        test_loss /= len(test_dataset)

        metrics = {'train_loss': train_loss, 'test_loss': test_loss}
        wandb.log(metrics)
        # wandb.log({
        #     "Examples": example_images,
        #     "Train Loss":train_loss,
        #     "Test Loss": test_loss})


        print(f'Epoch {e}, Train Loss: {train_loss*1000:.2f}, Test Loss: {test_loss*1000:.2f}')

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


def main():
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
def load_reference(path='../data/reference.npz',device=None) -> dict:
    """Example function with PEP 484 type annotations.

    Args:
        ret['shapeblendshape]: shape is (6890,3,300)

    Returns:
        The return value. True for success, False otherwise.

    """
    sample = np.load(path, allow_pickle=True)
    ret = {key: torch.tensor(sample[key],dtype=torch.float32,device=device) for key in sample}
    return ret

def setup_trained_model(trained_time=None):
    global PATH, generated_beta_list, config

    wandb.init(config=hyperparameter_defaults, project="wandb-example")
    config = wandb.config

    if trained_time is None:
        PATH = 'C:/Users/TheOtherMotion/Documents/GitHub/STAR-Private/resources/cvae_' + stm + '.pt'
    else:
        PATH = 'C:/Users/TheOtherMotion/Documents/GitHub/STAR-Private/resources/cvae_' + trained_time+ '.pt'

    #wandb.config.update()
    transform = transformation()

    train_dataset = StarBetaBoneLengthDataset(
        path='../data/train.npz',
        transform=transform,
        device=device,
        #debug=len(generated_beta_list)
    )

    test_dataset = StarBetaBoneLengthDataset(
        path='../data/test.npz',
        transform=transform,
        device=device,
        #debug=len(generated_beta_list)
    )

    validation_dataset = StarBetaBoneLengthDataset(
        path='../data/train.npz',
        transform=transform,
        device=device,
        #debug=len(generated_beta_list)
    )

    train_iterator = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_iterator = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    validation_dataset = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


    reference = load_reference(device=device)
    model = CVAE(config=config, reference=reference).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    #http://www.gisdeveloper.co.kr/?p=8443
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    if not os.path.isfile(PATH):
        #sort, dnn style, load data, input/loss, version
        # wandb.init(config=config,project="vae-mlp-beta-(6890,3)-2")
        wandb.watch(model)
        save_trained_model(model, train_dataset, test_dataset, train_iterator, test_iterator, optimizer, scheduler)
    else:
        load_trained_model(model, optimizer)

    return model, train_iterator, test_iterator, validation_dataset


if __name__ == "__main__":
    global generated_beta_list
    generated_beta_list = []
    #for i in range(2):
    main()
    print(hyperparameter_defaults)
    #https://stackoverflow.com/questions/54268029/how-to-convert-a-pytorch-tensor-into-a-numpy-array
    for beta_pair in generated_beta_list:
        print(beta_pair[0], beta_pair[1])
        original_val = beta_pair[0].detach().cpu().numpy()[0,:]
        new_val = beta_pair[1].detach().cpu().numpy()[0,:]
        extract_obj(save_path="C:/Users/TheOtherMotion/Documents/GitHub/STAR-Private/",name="stm"+"_original",betas=original_val)
        extract_obj(save_path="C:/Users/TheOtherMotion/Documents/GitHub/STAR-Private/",name="stm"+"_new",betas=new_val)
