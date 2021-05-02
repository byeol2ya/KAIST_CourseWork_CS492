import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import torch
import torch.nn as nn
from torch.nn import Parameter
import sys
sys.path.append('../')
import probtorch
from probtorch.util import expand_inputs
from tasp.MarginalObjectives import elbo
from torch.autograd.gradcheck import zero_gradients


print('probtorch:', probtorch.__version__,
      'torch:', torch.__version__,
      'cuda:', torch.cuda.is_available())

# model parameters
class NUMBER:
    def __init__(self, name, input, latent, output, hidden_encoder, hidden_decoder):
        self.input = input
        self.latent = latent
        self.output = output
        self.hidden_encoder = hidden_encoder
        self.hidden_decoder = hidden_decoder
        self.name = name

    def get_blocks(self):
        if self.name == 'encoder':
            dim0_list = [self.input] + self.hidden_encoder
            dim1_list = self.hidden_encoder + [self.latent]
            mlp_blocks = [mlp_block(dim0, dim1) for dim0, dim1 in zip(dim0_list[:-1], dim1_list[:-1])]
            last_block = mlp_block(dim0_list[-1], dim1_list[-1], activation_func=None)
        elif self.name == 'decoder':
            dim0_list = [self.latent] + self.hidden_encoder
            dim1_list = self.hidden_encoder + [self.output]
            mlp_blocks = [mlp_block(dim0, dim1) for dim0, dim1 in zip(dim0_list[:-1], dim1_list[:-1])]
            last_block = mlp_block(dim0_list[-1], dim1_list[-1], activation_func='Tanh')

        return mlp_blocks, last_block


NUM_STYLE = NUMBER(name='encoder',input=6890*3,hidden_encoder=[4096, 1024, 256],latent=10,hidden_decoder=None,output=None)
NUM_BONELENGTH = NUMBER(name='encoder',input=23,hidden_encoder=[64, 128, 32],latent=10,hidden_decoder=None,output=None)
NUM_DECODER = NUMBER(name='decoder',latent=10+10,hidden_encoder=[64, 128, 256],output=300,hidden_decoder=None,input=None)

BETA_4 = 50.0
CUDA = torch.cuda.is_available()

GAMMA = 10.0
JACOBIAN_LOSS_WEIGHT = 10.0
LAMBDA_INTRA_GROUP_ON_DIAG = 0.00  # Note these two are present in the prior matcher
LAMBDA_INTRA_GROUP_OFF_DIAG = 0.00
BETA = (  ## Collection of KL divergence loss weights ##
    1.000,  # Weight on (i), the sum of intra-group TC
    1.000,  # Weight on (ii), the sum of dimension-wise KL divergences
    1.000,  # Weight on (2), the mutual information between x & z
    BETA_4,  # Weight on (A), the inter-group TC
    0.000  # Weight on supervised error term
)

def mlp_block(input_dim, output_dim, IsBatchNorm=True, activation_func='LeakyReLU'):
    if IsBatchNorm:
        if activation_func == 'LeakyReLU':
            return nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.BatchNorm1d(output_dim),
                nn.LeakyReLU(inplace=False),
            )
        elif activation_func == 'Tanh':
            return nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.BatchNorm1d(output_dim),
                nn.Tanh(),
            )
        elif activation_func is None:
            return nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.BatchNorm1d(output_dim),
            )
        else:
            assert False
    else:
        if activation_func == 'LeakyReLU':
            return nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.LeakyReLU(inplace=False),
            )
        elif activation_func == 'Tanh':
            return nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.Tanh(),
            )
        elif activation_func is None:
            return nn.Sequential(
                nn.Linear(input_dim, output_dim),
            )
        else:
            assert False

#TODO: z_mean이 jacobian때 정상적으로 미분되고 원래 함수에는 영향을 안끼치는 지 확인
class Encoder(nn.Module):
    def __init__(self, NUM_LAYER, name_q=None, IsUseOnlyMu=False):
        super(self.__class__, self).__init__()
        self.NUM_LAYER = NUM_LAYER
        mlp_blocks, last_block = NUM_LAYER.get_blocks()
        self.enc_hidden = nn.Sequential(*mlp_blocks)

        self.z_mean = last_block
        self.z_log_std = last_block
        self.name_q = name_q
        self.IsUseOnlyMu = IsUseOnlyMu

    def forward(self, x, IsUseOnlyMu=None):
        if IsUseOnlyMu is None:
            IsUseOnlyMu = self.IsUseOnlyMu
        hiddens = self.enc_hidden(x)
        z_mean = self.z_mean(hiddens)
        z_mean.requires_grad_(True)
        z_mean.retain_grad()
        if IsUseOnlyMu:
            return z_mean
        else:
            q = probtorch.Trace()
            q.normal(z_mean.unsqueeze(0),
                     self.z_log_std(hiddens).exp(),
                     name=self.name_q)

            return q, z_mean

#TODO: change 10 to other value
class Decoder(nn.Module):
    def __init__(self, NUM_LAYER, name_p, device, name_q=None):
        super(self.__class__, self).__init__()
        self.NUM_LAYER = NUM_LAYER
        mlp_blocks, last_block = NUM_LAYER.get_blocks()
        blocks = mlp_blocks + [last_block]
        self.z_mean = torch.zeros(10).to(device)
        self.z_std = torch.ones(10).to(device)
        self.dec_image = nn.Sequential(*blocks)
        self.name_p = name_p
        if name_q is None:
            self.name_q = self.name_p

    def forward(self, q, value, z=None):
        assert q is not None or z is not None
        p = probtorch.Trace()
        if q is not None:
            z = p.normal(self.z_mean,
                         self.z_std,
                         value=q[self.name_q],
                         name=self.name_p)
            z = z.squeeze(0)
            z_cat = torch.cat((z, value), dim=1)
            beta_reconstructed = self.dec_image(z_cat)
            return p, z, beta_reconstructed
        else:
            z_cat = torch.cat((z, value), dim=1)
            beta_reconstructed = self.dec_image(z_cat)
            return z, beta_reconstructed

class CVAE(nn.Module):
    def __init__(self, reference, BATCH_SIZE, device,
                 NUM_ENCODER_STLYE=NUM_STYLE,
                 NUM_ENCODER_BONELENGTH=NUM_BONELENGTH,
                 NUM_DECODER_ALL=NUM_DECODER):
        super().__init__()
        self.NUM_ENCODER_STLYE = NUM_ENCODER_STLYE
        self.NUM_ENCODER_BONELENGTH = NUM_ENCODER_BONELENGTH
        self.NUM_DECODER_ALL = NUM_DECODER_ALL
        self.BATCH_SIZE = BATCH_SIZE
        self.device = device
        self.encoderStyle = Encoder(self.NUM_ENCODER_STLYE, 'z_Style', IsUseOnlyMu=False)
        self.encoderBonelength = Encoder(self.NUM_ENCODER_BONELENGTH, 'z_Bonelength', IsUseOnlyMu=True)
        self.decoder = Decoder(self.NUM_DECODER_ALL, name_p='z_Style', device=device)
        self.reference = reference

    def encode(self, beta, bonelength, IsUseOnlyMu=None):
        mesh = self.calculate_mesh(beta)
        mesh_flatten = torch.flatten(mesh, start_dim=1)

        ##encode
        if IsUseOnlyMu:
            mu_output = self.encoderStyle(mesh_flatten, IsUseOnlyMu=IsUseOnlyMu)  # output is q_style or mu_style
            bonelength_reduced = self.encoderBonelength(bonelength)
            bonelength_reduced.requires_grad_(True)
            bonelength_reduced.retain_grad()
            mu_output.requires_grad_(True)
            mu_output.retain_grad()
            return None, bonelength_reduced, mu_output
        else:
            output, mu_output = self.encoderStyle(mesh_flatten, IsUseOnlyMu=IsUseOnlyMu) #output is q_style or mu_style
            bonelength_reduced = self.encoderBonelength(bonelength)
            bonelength_reduced.requires_grad_(True)
            bonelength_reduced.retain_grad()
            mu_output.requires_grad_(True)
            mu_output.retain_grad()
            return output, bonelength_reduced, mu_output

    def forward(self, beta, bonelength):
        ##encode
        q_style, bonelength_reduced, mu_Style = self.encode(beta, bonelength)

        ##decode
        p, z, generated_beta = self.decoder(q_style, bonelength_reduced)
        generated_mesh = self.calculate_mesh(generated_beta)
        generated_bonelength = self.calculate_bonelength_both_from_mesh(generated_mesh)
        Loss_Jacobian = self.jacobianer(mu_S=mu_Style,mu_B=bonelength_reduced)
        return q_style, p, generated_beta, generated_bonelength, mu_Style, bonelength_reduced, Loss_Jacobian

    def bonelength_encoder_test(self, beta, delta=None):
        x = self.calculate_mesh(beta)
        # encode
        x_flat = torch.flatten(x, start_dim=1)

        z_mu, z_var = self.encoderSimple(x_flat)
        z = self.reparamterization(z_mu, z_var, delta=delta)

        # decode
        #generated_beta = self.decoderSimple(z)
        #generated_x = self.calculate_mesh(generated_beta)

        return None, None, z_mu, z_var, z, None

    def calculate_mesh(self, beta):
        _shape = [beta.shape[0]]+list(self.reference['shapeblendshape'].shape)
        x = torch.zeros(_shape,device=self.device,dtype=torch.float32)
        for batch_idx in range(beta.shape[0]):
            x[batch_idx] = beta[batch_idx,:] * self.reference['shapeblendshape']
            #print(x.shape,self.reference['shapeblendshape'].shape)
        x = torch.sum(x, -1) + self.reference['mesh_shape_pos']

        return x

    #TODO: check batchsize
    def calculate_bonelength_both_from_mesh(self, v_mesh):
        joints = torch.matmul(self.reference['jointregressor_matrix'], v_mesh)
        bone_length = self.calculate_bonelength_both(joints)  # 23
        bone_length = bone_length - 0.373924 * 2.67173

        return bone_length

    def calculate_bonelength_both(self, joints, offset=3):
        first_idx_list = [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
        second_idx_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

        return self.calculate_bonelength(vertex_list=joints, first_idx_list=first_idx_list, second_idx_list=second_idx_list, offset=offset)

    #TODO: check norm2 dtype
    def calculate_bonelength(self, vertex_list,
                        first_idx_list=[0, 1, 4, 7, 0, 3, 6, 9, 13, 16, 18, 20, 9, 12],
                        second_idx_list=[1, 4, 7, 10, 3, 6, 9, 13, 16, 18, 20, 22, 12, 15],
                        offset=3):
        local_size = len(first_idx_list)
        bonelength_list = torch.zeros([vertex_list.shape[0], local_size],device=self.device)
        vertex_list_first = torch.index_select(vertex_list, 1, torch.tensor(first_idx_list, device=self.device))
        vertex_list_second = torch.index_select(vertex_list, 1, torch.tensor(second_idx_list, device=self.device))

        temp = vertex_list_first - vertex_list_second
        bonelength_list = torch.norm(temp,dim=2)
        # bonelength_list[i] = torch.cdist(vertex_list_first, vertex_list_second, p=2)

        return bonelength_list

    def jacobianer(self, mu_B, mu_S):
        # # This is the funtion passed as an argument to the differentiation operator
        _mu_B = torch.tensor(mu_B.cpu().data.numpy(), dtype=torch.float32, requires_grad=True, device=self.device)
        _mu_S = torch.tensor(mu_S.cpu().data.numpy(), dtype=torch.float32, requires_grad=True, device=self.device)

        _, generated_beta = self.decoder(None, _mu_B, _mu_S)
        generated_mesh = self.calculate_mesh(generated_beta)
        generated_bonelength = self.calculate_bonelength_both_from_mesh(generated_mesh)
        _, mu_hat_S, mu_hat_B = self.encode(generated_beta, generated_bonelength, IsUseOnlyMu=True)  # This is mu_hat
        # print(type(mu_hat_S),type(mu_hat_B),type(mu_S),type(mu_B))

        # We need to keep the graph around since we will use it in the loss function
        # -- Jacobian dhatzE/dzI -- #
        jacobian_S_by_B = self.jacobian(inputs=_mu_B, output=mu_hat_S)
        # -- Jacobian dhatzI/dzE -- #
        jacobian_B_by_S = self.jacobian(inputs=_mu_S, output=mu_hat_B)
        # Compute the squared Frobenius norms of the Jacobians
        jacobian_dSdB = jacobian_S_by_B.pow(2).mean(dim=0).sum()
        jacobian_dBdS = jacobian_B_by_S.pow(2).mean(dim=0).sum()
        # Final penalty
        JacobianPenalty = torch.max(jacobian_dSdB, jacobian_dBdS)
        return JacobianPenalty * JACOBIAN_LOSS_WEIGHT

    def jacobian(self, inputs, output):
        assert inputs.requires_grad
        B, n = output.shape
        J = torch.zeros(n, *inputs.size()).to(self.device)  # n x B x m
        J2 = torch.zeros(n, *inputs.size()).to(self.device)  # n x B x m
        grad_output = torch.zeros(*output.size()).to(self.device)  # B x n

        for i in range(n):
            #original
            zero_gradients(inputs)
            grad_output.zero_()
            grad_output[:, i] = 1  # Across the batch, for one output
            output.backward(grad_output, create_graph=True)
            J[i] = inputs.grad

        # Cleanup
        zero_gradients(inputs)
        grad_output.zero_()
        return torch.transpose(J, dim0=0, dim1=1)  # B x n x m

        #test

        # zero_gradients(inputs)
        # grad_output.zero_()
        # output.backward(create_graph=True, retain_graph=True)
        # J2 = inputs.grad
        # J_m = J- J2

        # Cleanup
        zero_gradients(inputs)
        grad_output.zero_()
        return torch.transpose(J2, dim0=0, dim1=1)  # B x n x m

def likelihood_loss(X_hat, X):
    euc_err = (X_hat - X).pow(2).mean(dim=-1)
    return euc_err.unsqueeze(0)


def loss_function(
        model,
        ### Input AE latent vector and predicted output ###
        X,  # Deterministically encoded latent point cloud
        X_hat,  # Predicted deterministically encoded latent point cloud
        ### Encoder and decoder distributions ###
        q,  # Encoder distribution
        p,  # Decoder distribution
        ### Spectral loss arguments ###
        bonelength, # True bone length
        generated_bonelength, # Predicted bone length
        mu_S,
        mu_B,
        training_bias, # (N - 1) / (batch_size - 1)
        BATCH_SIZE,
        device
):
    # Assign the loss to the decoder distribution
    # The MSE loss corresponds to a Gaussian continuous output under log-likelihood
    p.loss(likelihood_loss, X_hat, X, name='x')
    # Hierarchically factorized ELBO function
    # hfelbo = lambda q, p: elbo(q, p, sample_dim=0, batch_dim=1, alpha=0.0, beta=BETA,
    #                            bias=training_bias, size_average=True, reduce=True)
    # L_hfvae, indiv_terms = hfelbo(q, p) # Minimize the negative elbo as a loss
    # L_hfvae = -L_hfvae
    L_hfvae = 0


    #### Covariance penalty for disentanglement ####
    # ----------------------------------------------#
    # Extract the means from the z group distirbutions
    # DIP-VAE-I covariance penalty
    mu = torch.cat((mu_S, mu_B), dim=-1).squeeze(0)  # Remove mc n_samples dim
    # Estimate the covariance matrix of mu
    M_tilde = mu - mu.mean(dim=0)  # B x L
    C_hat = (1 / (BATCH_SIZE - 1)) * M_tilde.t().mm(M_tilde)
    # For the intra-group penalties, get the intra-group covariance matrices
    C_hat_S = C_hat[0:NUM_STYLE.latent, 0:NUM_STYLE.latent]
    C_hat_B = C_hat[NUM_STYLE.latent:, :NUM_STYLE.latent]
    # For the inter-group penalties, get the inter-group covariance matrices
    C_hat_SB = C_hat[0:NUM_STYLE.latent, NUM_STYLE.latent:]  # cov(z_S, z_B)

    # Local methods for computing penalties via the L_1,1 norm
    def onDiagTerm(M):  return (M.diag() - 1).abs().sum()

    def offDiagTerm(M): return M.triu(diagonal=1).abs().sum()

    def interTerm(M):   return M.abs().sum()

    # Compute the on-diagonal intra-group penalty
    onDiagPen = onDiagTerm(C_hat_S) + onDiagTerm(C_hat_B)
    # Compute the off-diagonal intra-group penalty
    offDiagPen = offDiagTerm(C_hat_S) + offDiagTerm(C_hat_B)
    # Compute the inter-group penalty
    interPen = interTerm(C_hat_SB)
    # Final weighted penalty
    L_covarpen = GAMMA * interPen + LAMBDA_INTRA_GROUP_OFF_DIAG * offDiagPen + LAMBDA_INTRA_GROUP_ON_DIAG * onDiagPen

    # #### Jacobian loss ####
    # # P -> X -> mu -> z -> X_hat -> mu_hat
    # # Compute Jacobians: dmu_hat_S/d_mu_B and dmu_hat_B/d_mu_S
    #
    # # --- Define a function to compute the Jacobian ---#
    # def jacobian(inputs, output):
    #     assert inputs.requires_grad
    #     B, n = output.shape
    #     J = torch.zeros(n, *inputs.size()).to(device)  # n x B x m
    #     grad_output = torch.zeros(*output.size()).to(device)  # B x n
    #     for i in range(n):
    #         zero_gradients(inputs)
    #         grad_output.zero_()
    #         grad_output[:, i] = 1  # Across the batch, for one output
    #         output.backward(grad_output, create_graph=True)
    #         J[i] = inputs.grad
    #     # Cleanup
    #     zero_gradients(inputs)
    #     grad_output.zero_()
    #     return torch.transpose(J, dim0=0, dim1=1)  # B x n x m
    #
    # # --- End Jacobian calculation function ---#
    #
    # # Grab the variables wrt which we will compute the gradient
    # # Set retains_grad to true so that it actually saves the gradient
    # # Note that it is different than requires_grad, which was on
    # # As opposed to doing nothing, returning None, and giving me an error
    # # with no helpful messages whatsoever -_-'
    # # Note that retain_grad() had to be called earlier (in the encode function)
    # # mu_S = mu_S.squeeze(0) # q['z_E'].value.squeeze(0)  #mu_S.squeeze(0)
    # # mu_S.retain_grad()
    # # mu_B = mu_B.squeeze(0) # q['z_I'].value.squeeze(0)
    # # mu_B.retain_grad()
    #
    # # This is the funtion passed as an argument to the differentiation operator
    # _mu_B = torch.tensor(mu_B.cpu().data.numpy(),dtype=torch.float32, requires_grad=True, device=device)
    # _mu_S = torch.tensor(mu_S.cpu().data.numpy(),dtype=torch.float32, requires_grad=True, device=device)
    #
    # _, generated_beta = model.decoder(None, _mu_B, _mu_S)
    # generated_mesh = model.calculate_mesh(generated_beta)
    # generated_bonelength = model.calculate_bonelength_both_from_mesh(generated_mesh)
    # _, mu_hat_S,mu_hat_B = model.encode(generated_beta,generated_bonelength, IsUseOnlyMu=True)  # This is mu_hat
    # # print(type(mu_hat_S),type(mu_hat_B),type(mu_S),type(mu_B))
    #
    # # We need to keep the graph around since we will use it in the loss function
    # # -- Jacobian dhatzE/dzI -- #
    # jacobian_S_by_B = jacobian(inputs=_mu_B, output=mu_hat_S)
    # # -- Jacobian dhatzI/dzE -- #
    # jacobian_B_by_S = jacobian(inputs=_mu_S, output=mu_hat_B)
    # # Compute the squared Frobenius norms of the Jacobians
    # jacobian_dSdB = jacobian_S_by_B.pow(2).mean(dim=0).sum()
    # jacobian_dBdS = jacobian_B_by_S.pow(2).mean(dim=0).sum()
    # # Final penalty
    # JacobianPenalty = torch.max(jacobian_dSdB, jacobian_dBdS)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #--- Finally compute the total loss sum ---#
    total_loss = (   L_hfvae
               + L_covarpen)
               # + JacobianPenalty * JACOBIAN_LOSS_WEIGHT  )

    #### Indiv Loss terms ####
    #------------------------#
    # term_set = [
    #         # HFVAE terms from the KL divergence
    #         # ELBO_TERMS =
    #         #     log_likelihood:  (1) + (3),
    #         #     intra-group TCs: (i),
    #         #     dimwise KL-div:  (ii),
    #         #     MI(x, z):        (2),
    #         #     TC(z):           (A),
    #         #     T_5: semi-sup term 1
    #         #     T_6: semi-sup term 2
    #         indiv_terms[0], indiv_terms[1], indiv_terms[2], indiv_terms[3], indiv_terms[4],
    #         onDiagPen, offDiagPen, interPen, # Covariance penalty terms
    #         jacobian_dSdB, jacobian_dBdS
    # ]
    term_set=None
    term_names = [
        'log-likelihood', 'Intra-TC', 'DimWise-KL', 'MI(x,z)', 'Inter-TC',
        'ON-DIAG', 'OFF-DIAG', 'INTER-COVAR',
        'EwrtI_jacobian', 'IwrtE_jacobian'
    ]

    return total_loss, term_set, term_names

# from torchvision import datasets, transforms
# import os
#
# if not os.path.isdir(DATA_PATH):
#     os.makedirs(DATA_PATH)
#
# # train_data = torch.utils.data.DataLoader(
# #                 datasets.MNIST(DATA_PATH, train=True, download=True,
# #                                transform=transforms.ToTensor()),
# #                 batch_size=NUM_BATCH, shuffle=True)
# # test_data = torch.utils.data.DataLoader(
# #                 datasets.MNIST(DATA_PATH, train=False, download=True,
# #                                transform=transforms.ToTensor()),
# #                 batch_size=NUM_BATCH, shuffle=True)
# train_data = None
# test_data = None
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# def cuda_tensors(obj):
#     for attr in dir(obj):
#         value = getattr(obj, attr)
#         if isinstance(value, torch.Tensor):
#             setattr(obj, attr, value.to(device))
#
# enc = Encoder().to(device)
# dec = Decoder().to(device)
# enc.to(device)
# dec.to(device)
#
# # if CUDA:
# #     enc.to(device)
# #     dec.to(device)
# #     cuda_tensors(enc)
# #     cuda_tensors(dec)
#
# optimizer =  torch.optim.Adam(list(enc.parameters())+list(dec.parameters()),
#                               lr=LEARNING_RATE,
#                               betas=(BETA1, 0.999))
#
# def train(data, enc, dec, optimizer):
#     epoch_elbo = 0.0
#     enc.train()
#     dec.train()
#     N = 0
#     for b, (images, labels) in enumerate(data):
#         if images.size()[0] == NUM_BATCH:
#             N += NUM_BATCH
#             images = images.view(-1, NUM_PIXELS)
#             images = images.to(device)
#             optimizer.zero_grad()
#             q = enc(images, num_samples=NUM_SAMPLES)
#             p = dec(images, q, num_samples=NUM_SAMPLES)
#             loss = -elbo(q, p)
#             loss.backward()
#             optimizer.step()
#             loss = loss.to(device)
#             epoch_elbo -= float(loss.item())
#     return epoch_elbo / N
#
# def test(data, enc, dec):
#     enc.eval()
#     dec.eval()
#     epoch_elbo = 0.0
#     N = 0
#     for b, (images, labels) in enumerate(data):
#         if images.size()[0] == NUM_BATCH:
#             N += NUM_BATCH
#             images = images.view(-1, NUM_PIXELS)
#             images = images.to(device)
#             q = enc(images, num_samples=NUM_SAMPLES)
#             p = dec(images, q, num_samples=NUM_SAMPLES)
#             batch_elbo = elbo(q, p)
#             batch_elbo = batch_elbo.to(device)
#             epoch_elbo += float(batch_elbo.item())
#     return epoch_elbo / N
#
# import time
# from random import random
# if not RESTORE:
#     mask = {}
#     for e in range(NUM_EPOCHS):
#         train_start = time.time()
#         train_elbo = train(train_data, enc, dec, optimizer)
#         train_end = time.time()
#         test_start = time.time()
#         test_elbo = test(test_data, enc, dec)
#         test_end = time.time()
#         print('[Epoch %d] Train: ELBO %.4e (%ds) Test: ELBO %.4e (%ds)' % (
#                 e, train_elbo, train_end - train_start,
#                 test_elbo, test_end - test_start))
#
#     if not os.path.isdir(WEIGHTS_PATH):
#         os.mkdir(WEIGHTS_PATH)
#     torch.save(enc.state_dict(),
#                '%s/%s-%s-%s-enc.rar' % (WEIGHTS_PATH, MODEL_NAME, probtorch.__version__, torch.__version__))
#     torch.save(dec.state_dict(),
#                '%s/%s-%s-%s-dec.rar' % (WEIGHTS_PATH, MODEL_NAME, probtorch.__version__, torch.__version__))