import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.autograd.gradcheck import zero_gradients
import sys

sys.path.append('../')
sys.path.append('../../')

print('torch:', torch.__version__,
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


NUM_STYLE = NUMBER(name='encoder', input=6890 * 3, hidden_encoder=[4096, 1024, 256], latent=10, hidden_decoder=None,
                   output=None)
NUM_BONELENGTH = NUMBER(name='encoder', input=23, hidden_encoder=[64, 128, 32], latent=10, hidden_decoder=None,
                        output=None)
NUM_DECODER = NUMBER(name='decoder', latent=10 + 10, hidden_encoder=[64, 128, 256], output=300, hidden_decoder=None,
                     input=None)

BETA_4 = 50.0
CUDA = torch.cuda.is_available()

GAMMA = 10.0
JACOBIAN_LOSS_WEIGHT = 5.0
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


# TODO: z_mean이 jacobian때 정상적으로 미분되고 원래 함수에는 영향을 안끼치는 지 확인
class Encoder(nn.Module):
    def __init__(self, NUM_LAYER, IsUseOnlyMu=False):
        super(self.__class__, self).__init__()
        self.NUM_LAYER = NUM_LAYER
        mlp_blocks, last_block = NUM_LAYER.get_blocks()
        self.enc_hidden = nn.Sequential(*mlp_blocks)

        self.z_mean = last_block
        self.z_std = last_block
        self.IsUseOnlyMu = IsUseOnlyMu

    def forward(self, x, IsUseOnlyMu=None):
        if IsUseOnlyMu is None:
            IsUseOnlyMu = self.IsUseOnlyMu
        hiddens = self.enc_hidden(x)
        z_mean = self.z_mean(hiddens)
        z_mean.requires_grad_(True)
        z_mean.retain_grad()
        z_std = self.z_std(hiddens)
        if IsUseOnlyMu:
            return z_mean
        else:
            return z_mean, z_std


# TODO: change 10 to other value
class Decoder(nn.Module):
    def __init__(self, NUM_LAYER, device):
        super(self.__class__, self).__init__()
        self.NUM_LAYER = NUM_LAYER
        mlp_blocks, last_block = NUM_LAYER.get_blocks()
        blocks = mlp_blocks + [last_block]
        self.dec_image = nn.Sequential(*blocks)

    def forward(self, value, z):
        z_cat = torch.cat((z, value), dim=1)
        if z_cat.shape[0] == 1:
            z_cat = torch.cat((z_cat, z_cat), dim=0)
        beta_reconstructed = self.dec_image(z_cat)
        return beta_reconstructed


class CVAE(nn.Module):
    def __init__(self, reference, BATCH_SIZE, device, IsSupporter=False, IsA=True,
                 NUM_ENCODER_STLYE=NUM_STYLE,
                 NUM_ENCODER_BONELENGTH=NUM_BONELENGTH,
                 NUM_DECODER_ALL=NUM_DECODER):
        super().__init__()
        self.NUM_ENCODER_STLYE = NUM_ENCODER_STLYE
        self.NUM_ENCODER_BONELENGTH = NUM_ENCODER_BONELENGTH
        self.NUM_DECODER_ALL = NUM_DECODER_ALL
        self.BATCH_SIZE = BATCH_SIZE
        self.device = device
        self.encoderStyle = Encoder(self.NUM_ENCODER_STLYE, IsUseOnlyMu=False)
        self.encoderBonelength = Encoder(self.NUM_ENCODER_BONELENGTH, IsUseOnlyMu=True)
        self.decoder = Decoder(self.NUM_DECODER_ALL, device=device)
        self.reference = reference
        self.IsSupporter = IsSupporter
        self.IsA = IsA
        self.element_idx = None

    def encode(self, beta, bonelength, IsUseOnlyMu=None):
        mesh = self.calculate_mesh(beta)
        mesh_flatten = torch.flatten(mesh, start_dim=1)

        ##encode
        if IsUseOnlyMu:
            mu_output = self.encoderStyle(mesh_flatten, IsUseOnlyMu=IsUseOnlyMu)  # output is q_style or mu_style
            bonelength_reduced = self.encoderBonelength(bonelength)
            return None, bonelength_reduced, mu_output, None
        else:
            mu_output, std_output = self.encoderStyle(mesh_flatten,
                                                      IsUseOnlyMu=IsUseOnlyMu)  # output is q_style or mu_style
            bonelength_reduced = self.encoderBonelength(bonelength)
            return std_output, bonelength_reduced, mu_output, mesh

    def set_save_value(self, value, IsMu_B):
        self.temp_value = torch.unsqueeze(torch.tensor(value, dtype=torch.float32, device=self.device), 0)
        self.IsMu_B = IsMu_B

    # def forward_for_jacobian(self, value):
    #     assert (self.temp_value is not None), "You should use 'set_save_value' first"

    #     if self.IsMu_B:
    #         value = torch.unsqueeze(value, 0)
    #         value.requires_grad_(True)
    #         value.retain_grad()
    #         generated_beta = self.decoder(value, self.temp_value)
    #         generated_mesh = self.calculate_mesh(generated_beta)
    #         generated_bonelength = self.calculate_bonelength_both_from_mesh(generated_mesh)
    #         _, mu_hat_S, mu_hat_B, _ = self.encode(generated_beta, generated_bonelength,
    #                                                IsUseOnlyMu=True)  # This is mu_hat
    #         mu_hat_S_ = mu_hat_S[0, :]
    #         mu_hat_S_.requires_grad_(True)
    #         mu_hat_S_.retain_grad()
    #         return mu_hat_S_

    #     else:
    #         value = torch.unsqueeze(value, 0)
    #         value.requires_grad_(True)
    #         value.retain_grad()
    #         generated_beta = self.decoder(self.temp_value, value)
    #         generated_mesh = self.calculate_mesh(generated_beta)
    #         generated_bonelength = self.calculate_bonelength_both_from_mesh(generated_mesh)
    #         _, mu_hat_S, mu_hat_B, _ = self.encode(generated_beta, generated_bonelength,
    #                                                IsUseOnlyMu=True)  # This is mu_hat
    #         mu_hat_B_ = mu_hat_S[0, :]
    #         mu_hat_B_.requires_grad_(True)
    #         mu_hat_B_.retain_grad()
    #         return mu_hat_B_

    #     self.temp_value = None

    def forward_main(self, beta, bonelength):
        ##encode
        std_Style, bonelength_reduced, mu_Style, mesh = self.encode(beta, bonelength)
        z = self.reparameterization(mu_Style, std_Style)
        # self.device = z.get_device()
        ##decode
        generated_beta = self.decoder(z, bonelength_reduced)
        generated_mesh = self.calculate_mesh(generated_beta)
        generated_bonelength = self.calculate_bonelength_both_from_mesh(generated_mesh)
        # Loss_Jacobian = self.jacobianer(mu_S=mu_Style,mu_B=bonelength_reduced)
        return generated_beta, generated_bonelength, mu_Style, std_Style, bonelength_reduced, mesh, generated_mesh

    def forward_supporter(self, mu_S, mu_B):
        return self.jacobianer(mu_S, mu_B)

    def forward(self, beta, bonelength, mu_S=None, mu_B=None, element_idx=None):
        if self.IsSupporter is False:
            return self.forward_main(beta=beta, bonelength=bonelength)
        else:
            self.element_idx = element_idx
            return self.forward_supporter(mu_S=mu_S, mu_B=mu_B)

    # reparameterize
    # https://whereisend.tistory.com/54
    def reparameterization(self, z_mu, z_var, delta=None):
        std = torch.exp(z_var / 2)
        if delta is not None:
            eps = torch.ones(std.size()) * delta
        else:
            eps = torch.randn_like(std)
        x_sample = eps.mul(std).add_(z_mu)

        return x_sample

    def add_noise(self, value):
        # if torch.randint(0, 1, (1,)).item() == 0:
        temp = value[:,self.element_idx]  * 0.0001
        temp2 = value[:,self.element_idx] + temp
        eps = temp2 - value[:,self.element_idx]
        x_noised = value
        x_noised[:,self.element_idx] += eps
        # print(value)
        # print(x_noised)
        eps = eps.unsqueeze(1)
        return x_noised, eps
        # else:
        #     x_noised = value * 1.01
        #     return x_noised, 0.01

    def bonelength_encoder_test(self, beta, delta=None):
        x = self.calculate_mesh(beta)
        # encode
        x_flat = torch.flatten(x, start_dim=1)

        z_mu, z_var = self.encoderSimple(x_flat)
        z = self.reparamterization(z_mu, z_var, delta=delta)

        # decode
        # generated_beta = self.decoderSimple(z)
        # generated_x = self.calculate_mesh(generated_beta)

        return None, None, z_mu, z_var, z, None

    def calculate_mesh(self, beta):
        _shape = [beta.shape[0]] + list(self.reference['shapeblendshape'].shape)
        x = torch.zeros(_shape, device=beta.get_device(), dtype=torch.float32)
        for batch_idx in range(beta.shape[0]):
            x[batch_idx] = beta[batch_idx, :] * self.reference['shapeblendshape'].to(beta.get_device())
            # print(x.shape,self.reference['shapeblendshape'].shape)
        x = torch.sum(x, -1)
        x = x + self.reference['mesh_shape_pos'].to(x.get_device())

        return x

    # TODO: check batchsize
    def calculate_bonelength_both_from_mesh(self, v_mesh):
        joints = torch.matmul(self.reference['jointregressor_matrix'].to(v_mesh.get_device()), v_mesh)
        bone_length = self.calculate_bonelength_both(joints)  # 23
        bone_length = bone_length - 0.373924 * 2.67173

        return bone_length

    def calculate_bonelength_both(self, joints, offset=3):
        first_idx_list = [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
        second_idx_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

        return self.calculate_bonelength(vertex_list=joints, first_idx_list=first_idx_list,
                                         second_idx_list=second_idx_list, offset=offset)

    # TODO: check norm2 dtype
    def calculate_bonelength(self, vertex_list,
                             first_idx_list=[0, 1, 4, 7, 0, 3, 6, 9, 13, 16, 18, 20, 9, 12],
                             second_idx_list=[1, 4, 7, 10, 3, 6, 9, 13, 16, 18, 20, 22, 12, 15],
                             offset=3):
        local_size = len(first_idx_list)
        bonelength_list = torch.zeros([vertex_list.shape[0], local_size], device=self.device)
        vertex_list_first = torch.index_select(vertex_list, 1, torch.tensor(first_idx_list, device=self.device))
        vertex_list_second = torch.index_select(vertex_list, 1, torch.tensor(second_idx_list, device=self.device))

        temp = vertex_list_first - vertex_list_second
        bonelength_list = torch.norm(temp, dim=2)
        # bonelength_list[i] = torch.cdist(vertex_list_first, vertex_list_second, p=2)

        return bonelength_list

    def jacobianer(self, mu_S, mu_B):
        _mu_B = torch.tensor(mu_B.cpu().data.numpy(), dtype=torch.float32, requires_grad=True, device=self.device)
        _mu_S = torch.tensor(mu_S.cpu().data.numpy(), dtype=torch.float32, requires_grad=True, device=self.device)

        z_S, eps_S = self.add_noise(mu_S)
        _z_S = torch.tensor(z_S.cpu().data.numpy(), dtype=torch.float32, requires_grad=True, device=self.device)
        _eps_S = torch.tensor(eps_S.cpu().data.numpy(), dtype=torch.float32, requires_grad=True, device=self.device)
        generated_beta = self.decoder(_mu_B, _z_S)
        generated_mesh = self.calculate_mesh(generated_beta)
        generated_bonelength = self.calculate_bonelength_both_from_mesh(generated_mesh)
        _, _, mu_hat_B, _ = self.encode(generated_beta, generated_bonelength, IsUseOnlyMu=True)  # This is mu_hat

        z_B, eps_B = self.add_noise(mu_B)
        _z_B = torch.tensor(z_B.cpu().data.numpy(), dtype=torch.float32, requires_grad=True, device=self.device)
        _eps_B = torch.tensor(eps_B.cpu().data.numpy(), dtype=torch.float32, requires_grad=True, device=self.device)
        generated_beta = self.decoder(_z_B, _mu_S)
        generated_mesh = self.calculate_mesh(generated_beta)
        generated_bonelength = self.calculate_bonelength_both_from_mesh(generated_mesh)
        _, mu_hat_S, _, _ = self.encode(generated_beta, generated_bonelength, IsUseOnlyMu=True)  # This is mu_hat

        loss = torch.sum(((mu_hat_B - _mu_B) * _eps_S).pow(2) + ((mu_hat_S - _mu_S) * _eps_B).pow(2))

        return loss * JACOBIAN_LOSS_WEIGHT

        # for k in range(self.BATCH_SIZE):
        #     # # This is the funtion passed as an argument to the differentiation operator
        #     _mu_B = torch.tensor(mu_B.cpu().data.numpy()[k,:], dtype=torch.float32, requires_grad=True, device=self.device)
        #     _mu_S = torch.tensor(mu_S.cpu().data.numpy()[k,:], dtype=torch.float32, requires_grad=True, device=self.device)

        #     self.set_save_value(_mu_B,IsMu_B=True)
        #     jacobian_S_by_B = torch.autograd.functional.jacobian(self.forward_for_jacobian, _mu_S, create_graph=True)

        #     self.set_save_value(_mu_S,IsMu_B=False)
        #     jacobian_B_by_S  = torch.autograd.functional.jacobian(self.forward_for_jacobian, _mu_B, create_graph=True)

        #     # return JacobianPenalty = torch.sum(jacobian_B_by_S)
        #     return jacobian_B_by_S.pow(2).mean(dim=0).sum()

    #     _mu_B = torch.tensor(mu_B.cpu().data.numpy(), dtype=torch.float32, requires_grad=True, device=self.device)
    #     _mu_S = torch.tensor(mu_S.cpu().data.numpy(), dtype=torch.float32, requires_grad=True, device=self.device)
    #     generated_beta = self.decoder(_mu_B, _mu_S)
    #     generated_mesh = self.calculate_mesh(generated_beta)
    #     generated_bonelength = self.calculate_bonelength_both_from_mesh(generated_mesh)
    #     _, mu_hat_S, mu_hat_B, _ = self.encode(generated_beta, generated_bonelength, IsUseOnlyMu=True)  # This is mu_hat
    #     # print(type(mu_hat_S),type(mu_hat_B),type(mu_S),type(mu_B))

    #     # We need to keep the graph around since we will use it in the loss function
    #     # -- Jacobian dhatzE/dzI -- #
    #     jacobian_S_by_B = self.jacobian(inputs=_mu_B, output=mu_hat_S)
    #     # -- Jacobian dhatzI/dzE -- #
    #     jacobian_B_by_S = self.jacobian(inputs=_mu_S, output=mu_hat_B)
    #     # Compute the squared Frobenius norms of the Jacobians
    #     jacobian_dSdB = jacobian_S_by_B.pow(2).mean(dim=0).sum()
    #     jacobian_dBdS = jacobian_B_by_S.pow(2).mean(dim=0).sum()
    #     # Final penalty
    #     JacobianPenalty = torch.max(jacobian_dSdB, jacobian_dBdS)
    #     return JacobianPenalty * JACOBIAN_LOSS_WEIGHT

    # def jacobian(self, inputs, output):
    #     assert inputs.requires_grad
    #     B, n = output.shape
    #     J = torch.zeros(n, *inputs.size()).to(self.device)  # n x B x m
    #     grad_output = torch.zeros(*output.size()).to(self.device)  # B x n

    #     for i in range(n):
    #         #original
    #         zero_gradients(inputs)
    #         grad_output.zero_()
    #         grad_output[:, i] = 1  # Across the batch, for one output
    #         output.backward(grad_output, create_graph=True)
    #         J[i] = inputs.grad

    #     # Cleanup
    #     zero_gradients(inputs)
    #     grad_output.zero_()
    #     return torch.transpose(J, dim0=0, dim1=1)  # B x n x m


def likelihood_loss(X_hat, X):
    euc_err = (X_hat - X).pow(2).mean(dim=-1)
    return euc_err.unsqueeze(0)


def loss_wrapper(x, reconstructed_x, z_mu, z_var, bonelength, reconstructed_bonelength, beta, reconstructed_beta):
    # loss = calculate_base_loss(x, reconstructed_x, z_mu, z_var)
    b_beta = 1
    mean, log_var = z_mu, z_var
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    RCL_x = F.mse_loss(x, reconstructed_x) * b_beta
    # RCL_x = 0
    RCL_bone = F.mse_loss(reconstructed_bonelength, bonelength) * b_beta
    # RCL_beta = F.mse_loss(reconstructed_beta, beta)

    # loss = KLD + RCL_beta
    # loss = KLD + RCL_bone + RCL_beta
    loss = KLD + RCL_bone + RCL_x
    # loss = KLD + RCL_x + RCL_bone + RCL_beta
    # if task == 'train':
    #     print(KLD.item(), RCL_x.item(), RCL_bone.item())
    return loss, {'KLD': KLD, 'RCL_bone': RCL_bone, 'RCL_x': RCL_x}


def loss_function(
        model,
        ### Input AE latent vector and predicted output ###
        X,  # Deterministically encoded latent point cloud
        X_hat,  # Predicted deterministically encoded latent point cloud
        ### Encoder and decoder distributions ###
        ### Spectral loss arguments ###
        bonelength,  # True bone length
        generated_bonelength,  # Predicted bone length
        mu_S,
        std_S,
        mu_B,
        beta,
        generated_beta,
        training_bias,  # (N - 1) / (batch_size - 1)
        BATCH_SIZE,
        device
):
    L_base, term_set = loss_wrapper(x=X, reconstructed_x=X_hat, z_mu=mu_S, z_var=std_S, bonelength=bonelength,
                                    reconstructed_bonelength=generated_bonelength,
                                    beta=beta, reconstructed_beta=generated_beta)

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

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # --- Finally compute the total loss sum ---#
    total_loss = (L_base
                  + L_covarpen)

    #### Indiv Loss terms ####
    # ------------------------#
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
    term_names = [
        'log-likelihood', 'Intra-TC', 'DimWise-KL', 'MI(x,z)', 'Inter-TC',
        'ON-DIAG', 'OFF-DIAG', 'INTER-COVAR',
        'EwrtI_jacobian', 'IwrtE_jacobian'
    ]
    # print(L_covarpen.item())
    return total_loss, term_set, L_covarpen