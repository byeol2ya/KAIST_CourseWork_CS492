''' This code contains the implementation of conditional VAE
https://github.com/graviraja/pytorch-sample-codes/blob/master/conditional_vae.py
'''
import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim.lr_scheduler

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
from utils.datasets_v1 import StarBetaBoneLengthDataset, ToTensor, Normalize
#https://greeksharifa.github.io/references/2020/06/10/wandb-usage/
import wandb

tm = time.localtime()
stm = time.strftime('%Y_%m_%d_%H_%M_%S', tm)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# print(f'device: {device}')
# BATCH_SIZE = 64         # number of data points in each batch
BATCH_SIZE = 128         # number of data points in each batch
N_EPOCHS = 50           # times to run the model on complete data
# N_EPOCHS = 1000           # times to run the model on complete data
INPUT_DIM = 300     # size of each input
HIDDEN_DIM = 128        # hidden dimension
LATENT_DIM = 28         # latent vector dimension
N_CLASSES = 14          # number of classes in the data
DATA_SIZE = 30000
lr = 1e-2               # learning rate

PATH = None


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


class Encoder(nn.Module):
    ''' This the encoder part of VAE

    '''
    def __init__(self, input_dim, hidden_dim, latent_dim, n_classes):
        '''
        Args:
            input_dim: A integer indicating the size of input (in case of MNIST 28 * 28).
            hidden_dim: A integer indicating the size of hidden dimension.
            latent_dim: A integer indicating the latent size.
            n_classes: A integer indicating the number of classes. (dimension of one-hot representation of labels)
        '''
        super().__init__()

        self.fc1 = nn.Linear(input_dim + n_classes, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim//2)
        self.bn2 = nn.BatchNorm1d(hidden_dim//2)
        self.fc3 = nn.Linear(hidden_dim//2, hidden_dim//4)
        self.bn3 = nn.BatchNorm1d(hidden_dim//4)
        self.mu = nn.Linear(hidden_dim//4, latent_dim)
        self.var = nn.Linear(hidden_dim//4, latent_dim)

    def forward(self, x):
        # x is of shape [batch_size, input_dim + n_classes]

        hidden = F.relu(self.bn1(self.fc1(x)))
        #hidden = F.relu(self.bn2(self.fc2(hidden)))
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
    def __init__(self, latent_dim, hidden_dim, output_dim, n_classes):
        '''
        Args:
            latent_dim: A integer indicating the latent size.
            hidden_dim: A integer indicating the size of hidden dimension.
            output_dim: A integer indicating the size of output (in case of MNIST 28 * 28).
            n_classes: A integer indicating the number of classes. (dimension of one-hot representation of labels)
        '''
        super().__init__()
        self.fc1 = nn.Linear(latent_dim + n_classes, hidden_dim//2)
        self.bn1 = nn.BatchNorm1d(hidden_dim//2)
        self.fc2 = nn.Linear(hidden_dim//2, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.bn3 = nn.BatchNorm1d(output_dim)
        # self.latent_to_hidden = nn.Linear(latent_dim + n_classes, hidden_dim)
        # self.hidden_to_out = nn.Linear(hidden_dim, output_dim)
        self.linear = nn.Linear(output_dim, output_dim)

    def forward(self, x):
        # x is of shape [batch_size, latent_dim + num_classes]
        hidden = F.relu(self.bn1(self.fc1(x)))
        hidden = F.relu(self.bn2(self.fc2(hidden)))
        generated_x = F.sigmoid(self.fc3(hidden))
        # output = F.relu(self.fc4(hidden))
        # x is of shape [batch_size, hidden_dim]
        # generated_x = self.linear(generated_x)
        # x is of shape [batch_size, output_dim]

        return generated_x


class CVAE(nn.Module):
    ''' This the VAE, which takes a encoder and decoder.

    '''
    def __init__(self, input_dim, hidden_dim, latent_dim, n_classes):
        '''
        Args:
            input_dim: A integer indicating the size of input (in case of MNIST 28 * 28).
            hidden_dim: A integer indicating the size of hidden dimension.
            latent_dim: A integer indicating the latent size.
            n_classes: A integer indicating the number of classes. (dimension of one-hot representation of labels)
        '''
        super().__init__()

        self.encoder = Encoder(input_dim, hidden_dim, latent_dim, n_classes)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim, n_classes)

    def forward(self, x, y, delta=None):

        x = torch.cat((x, y), dim=1)

        # encode
        z_mu, z_var = self.encoder(x)

        # sample from the distribution having latent parameters z_mu, z_var
        # reparameterize
        std = torch.exp(z_var / 2)

        eps = torch.randn_like(std)
        if delta is not None:
            #https://whereisend.tistory.com/54
            eps *= (torch.randn_like(std) * 0 + 1) * std * delta
        x_sample = eps.mul(std).add_(z_mu)

        z = torch.cat((x_sample, y), dim=1)

        # decode
        generated_x = self.decoder(z)

        return generated_x, z_mu, z_var


def calculate_loss(x, reconstructed_x, mean, log_var):
    # reconstruction loss
    # RCL = F.mse_loss(reconstructed_x, x, size_average=False)
    RCL = weighted_mse_loss(reconstructed_x,x,None)
    # RCL = F.binary_cross_entropy(reconstructed_x, x, size_average=False)
    # kl divergence loss
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    if RCL < 0:
        print(f'RCL: {RCL}, KLD: {KLD}, SUM: {RCL + KLD}')
    return RCL + KLD


def train(model,train_iterator,optimizer):
    # set the train mode
    model.train()

    # loss of the epoch
    train_loss = 0

    for i, (x, y) in enumerate(train_iterator):
        # reshape the data into [batch_size, 784]
        # print(x.shape)
        # x = x.view(-1, INPUT_DIM)
        x = x.to(device)

        # y = y.view(-1, N_CLASSES)
        y = y.to(device)

        # update the gradients to zero
        optimizer.zero_grad()

        # forward pass
        reconstructed_x, z_mu, z_var = model(x, y)

        # loss
        loss = calculate_loss(x, reconstructed_x, z_mu, z_var)
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
        for i, (x, y) in enumerate(test_iterator):
            # reshape the data
            # x = x.view(-1, INPUT_DIM)
            x = x.to(device)

            # y = y.view(-1, N_CLASSES)
            y = y.to(device)

            # forward pass
            reconstructed_x, z_mu, z_var = model(x, y)

            # loss
            loss = calculate_loss(x, reconstructed_x, z_mu, z_var)
            test_loss += loss.item()

    return test_loss


#pram: delta should be same size with LATENT_DIM or 1
def get_star(model,beta=None,delta=None):
    if delta is not None:
        assert delta.size != 1 or delta.size != LATENT_DIM, 'delta should be same size with LATENT_DIM or 1'


    transforms=transformation()
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
    best_test_loss = float('inf')
    example_images = []

    for e in range(N_EPOCHS):
        train_loss = train(model, train_iterator, optimizer)
        test_loss = test(model, test_iterator)

        train_loss /= len(train_dataset)
        test_loss /= len(test_dataset)

        wandb.log({
            "Examples": example_images,
            "Train Loss":train_loss,
            "Test Loss": test_loss})


        print(f'Epoch {e}, Train Loss: {train_loss*1000:.2f}, Test Loss: {test_loss*1000:.2f}')

        if best_test_loss > test_loss:
            best_test_loss = test_loss
            patience_counter = 1
        else:
            patience_counter += 1

        if patience_counter > 10:
            break
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

def setup_trained_model(trained_time=None):
    global PATH
    if trained_time is None:
        PATH = 'C:/Users/TheOtherMotion/Documents/GitHub/STAR-Private/resources/cvae_' + stm + '.pt'
    else:
        PATH = 'C:/Users/TheOtherMotion/Documents/GitHub/STAR-Private/resources/cvae_' + trained_time+ '.pt'

    #wandb.config.update()
    transform = transformation()

    train_dataset = StarBetaBoneLengthDataset(
        npy_file='../demo/saved_bonelength_train.npy',
        transform=transform,
        length=DATA_SIZE
    )

    test_dataset = StarBetaBoneLengthDataset(
        npy_file='../demo/saved_bonelength_test.npy',
        transform=transform,
        length=DATA_SIZE
    )

    validation_dataset = StarBetaBoneLengthDataset(
        npy_file='../demo/saved_bonelength_validation.npy',
        transform=transform)

    train_iterator = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_iterator = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    validation_dataset = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = CVAE(INPUT_DIM, HIDDEN_DIM, LATENT_DIM, N_CLASSES).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    #http://www.gisdeveloper.co.kr/?p=8443
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    if not os.path.isfile(PATH):
        wandb.init(project="wandb-tutorial")
        wandb.watch(model)
        save_trained_model(model, train_dataset, test_dataset, train_iterator, test_iterator, optimizer, scheduler)
    else:
        load_trained_model(model, optimizer)

    return model, train_iterator, test_iterator, validation_dataset


if __name__ == "__main__":
    main()
