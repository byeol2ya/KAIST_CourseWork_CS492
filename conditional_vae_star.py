''' This code contains the implementation of conditional VAE

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

from demo.datasets import StarBetaBoneLengthDataset, ToTensor, Normalize

tm = time.localtime()
stm = time.strftime('%Y_%m_%d_%H_%M_%S', tm)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# print(f'device: {device}')
# BATCH_SIZE = 64         # number of data points in each batch
BATCH_SIZE = 128         # number of data points in each batch
N_EPOCHS = 300           # times to run the model on complete data
# N_EPOCHS = 1000           # times to run the model on complete data
INPUT_DIM = 300     # size of each input
HIDDEN_DIM = 256        # hidden dimension
LATENT_DIM = 28         # latent vector dimension
N_CLASSES = 14          # number of classes in the data
DATA_SIZE = 30000
lr = 1e-2               # learning rate

PATH = None


#https://discuss.pytorch.org/t/pixelwise-weights-for-mseloss/1254/2
def weighted_mse_loss(input,target,weights):
    out = (input-target)**2
    weights = torch.zeros(out.shape, dtype=torch.float32, device=device)
    for i in range(1, 11):
        weights[:, i-1] = 1.0 + 1000.0/float(i)
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

        self.linear = nn.Linear(input_dim + n_classes, hidden_dim)
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        # x is of shape [batch_size, input_dim + n_classes]

        hidden = F.relu(self.linear(x))
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

        self.latent_to_hidden = nn.Linear(latent_dim + n_classes, hidden_dim)
        self.hidden_to_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x is of shape [batch_size, latent_dim + num_classes]
        x = F.relu(self.latent_to_hidden(x))
        # x is of shape [batch_size, hidden_dim]
        generated_x = F.sigmoid(self.hidden_to_out(x))
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

    def forward(self, x, y):

        x = torch.cat((x, y), dim=1)

        # encode
        z_mu, z_var = self.encoder(x)

        # sample from the distribution having latent parameters z_mu, z_var
        # reparameterize
        std = torch.exp(z_var / 2)
        eps = torch.randn_like(std)
        x_sample = eps.mul(std).add_(z_mu)

        z = torch.cat((x_sample, y), dim=1)

        # decode
        generated_x = self.decoder(z)

        return generated_x, z_mu, z_var


def calculate_loss(x, reconstructed_x, mean, log_var):
    # reconstruction loss
    #RCL = F.mse_loss(reconstructed_x, x, size_average=False)
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

    for e in range(N_EPOCHS):
        train_loss = train(model, train_iterator, optimizer)
        test_loss = test(model, test_iterator)

        train_loss /= len(train_dataset)
        test_loss /= len(test_dataset)

        print(f'Epoch {e}, Train Loss: {train_loss:.2f}, Test Loss: {test_loss:.2f}')

        if best_test_loss > test_loss:
            best_test_loss = test_loss
            patience_counter = 1
        else:
            patience_counter += 1

        if patience_counter > 10:
            # break
            pass
        scheduler.step()

    # https://tutorials.pytorch.kr/beginner/saving_loading_models.html
    torch.save({
        'epoch': N_EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, PATH)


def setup_trained_model(trained_time=None):
    global PATH
    if trained_time is None:
        PATH = './resources/cvae_' + stm + '.pt'
    else:
        PATH = './resources/cvae_' + trained_time+ '.pt'

    mean = 0.0
    std = 5.0

    transform = transforms.Compose([
        Normalize(mean=mean, std=std),
        #ToTensor, #왜 안써야하지?
    ])

    train_dataset = StarBetaBoneLengthDataset(
        npy_file='C:/Users/TheOtherMotion/Documents/GitHub/STAR-Private/demo/saved_bonelength_train.npy',
        transform=transform,
        length=DATA_SIZE
    )

    test_dataset = StarBetaBoneLengthDataset(
        npy_file='C:/Users/TheOtherMotion/Documents/GitHub/STAR-Private/demo/saved_bonelength_test.npy',
        transform=transform,
        length=DATA_SIZE
    )

    validation_dataset = StarBetaBoneLengthDataset(
        npy_file='C:/Users/TheOtherMotion/Documents/GitHub/STAR-Private/demo/saved_bonelength_validation.npy',
        transform=transform)

    train_iterator = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_iterator = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    validation_dataset = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = CVAE(INPUT_DIM, HIDDEN_DIM, LATENT_DIM, N_CLASSES).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    #http://www.gisdeveloper.co.kr/?p=8443
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    if not os.path.isfile(PATH):
        save_trained_model(model, train_dataset, test_dataset, train_iterator, test_iterator, optimizer, scheduler)
    load_trained_model(model, optimizer)

    return model


if __name__ == "__main__":
    setup_trained_model()
