import torch
import torch.nn as nn

def set_channel(channel_size, channel_list, target_dim, latent_dim, IsEncoder = True) -> list:
    '''
    Args:
        target_dim: input(encoder), output(decoder)
    '''
    ret = [1]
    accumulated_dim = 1

    if len(channel_list) < channel_size:
        loop_range = len(channel_list)
    else:
        loop_range = channel_size

    for channel_no in range(loop_range):
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

def mlp_block(input_dim, output_dim, IsBatchNorm=True, IsLeakyReLU=True):
    if IsBatchNorm:
        if IsLeakyReLU:
            return nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.BatchNorm1d(output_dim),
                nn.LeakyReLU(),
            )
        else:
            return nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.BatchNorm1d(output_dim),
            )
    else:
        if IsLeakyReLU:
            return nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.LeakyReLU(),
            )
        else:
            return nn.Sequential(
                nn.Linear(input_dim, output_dim),
            )


class Encoder(nn.Module):
    ''' This the encoder part of VAE

    '''
    def __init__(self, config, input_dim, latent_dim, encoder_channel_list=None):
        '''
        Args:
            input_dim: A integer indicating the size of input (in case of MNIST 28 * 28).
            hidden_dim: A integer indicating the size of hidden dimension.
            latent_dim: A integer indicating the latent size.
            n_classes: A integer indicating the number of classes. (dimension of one-hot representation of labels)
        '''
        super().__init__()
        if encoder_channel_list is None:
            # config_channel_list = [config.encoder_channels_0,config.encoder_channels_1,config.encoder_channels_2,config.encoder_channels_3]
            config_channel_list = [config.encoder_channels_0,config.encoder_channels_1,config.encoder_channels_2,config.encoder_channels_3,
                                   config.encoder_channels_4,config.encoder_channels_5,config.encoder_channels_6]
            self.encoder_channel_list = set_channel(config.encoder_channel_size-1,config_channel_list,input_dim,latent_dim)
        else:
            self.encoder_channel_list = encoder_channel_list
        #https://michigusa-nlp.tistory.com/26
        #https://towardsdatascience.com/pytorch-how-and-when-to-use-module-sequential-modulelist-and-moduledict-7a54597b5f17
        mlp_blocks = [mlp_block(input_dim//dim_0, input_dim//dim_1)
                       for dim_0, dim_1 in zip(self.encoder_channel_list, self.encoder_channel_list[1:])]
        #unpacking
        #https://mingrammer.com/understanding-the-asterisk-of-python/#2-%EB%A6%AC%EC%8A%A4%ED%8A%B8%ED%98%95-%EC%BB%A8%ED%85%8C%EC%9D%B4%EB%84%88-%ED%83%80%EC%9E%85%EC%9D%98-%EB%8D%B0%EC%9D%B4%ED%84%B0%EB%A5%BC-%EB%B0%98%EB%B3%B5-%ED%99%95%EC%9E%A5%ED%95%98%EA%B3%A0%EC%9E%90-%ED%95%A0-%EB%95%8C
        self.encoder = nn.Sequential(*mlp_blocks)


        # self.fc1 = nn.Linear(input_dim, hidden_dim)
        # self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.mu = nn.Linear(input_dim//self.encoder_channel_list[-1], latent_dim)
        self.var = nn.Linear(input_dim//self.encoder_channel_list[-1], latent_dim)

        # torch.nn.init.xavier_uniform_(self.encoder.weight)
        # torch.nn.init.xavier_uniform_(self.mu.weight)
        # torch.nn.init.xavier_uniform_(self.var.weight)
    def forward(self, x):
        # x is of shape [batch_size, input_dim + n_classes]

        hidden = self.encoder(x)
        # hidden = F.relu(self.bn1(self.fc1(x)))
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
        self.linear1 = nn.Linear(latent_dim*self.decoder_channel_list[-1],output_dim)
        self.linear2 = nn.Linear(output_dim,output_dim)
        # self.linear2 = nn.Linear(output_dim*2, output_dim)

        # torch.nn.init.xavier_uniform_(self.decoder.weight)
        # torch.nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x):
        # x is of shape [batch_size, latent_dim + num_classes]
        hidden = self.decoder(x)
        # hidden = torch.tanh(self.linear1(hidden))
        generated_x = self.linear2(hidden)
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
    def __init__(self, config, reference, device, num_beta=300):
        '''
        Args:
            reference (dict): it has 'mesh_shape'
                        'mesh_shape_pos'
                        'shapeblendshape'
                        'jointregressor_matrix'
        '''
        super().__init__()

        self.reference = reference
        self.device = device
        # self.init_original(config=config,num_beta=num_beta)
        self.init_simple(config=config,num_beta=num_beta)
        # self.init_reduction(config=config,num_beta=num_beta)
        # self.init_bonelength_encoder_test(config=config,num_beta=num_beta)

    def forward(self, beta, bonelength, delta=None):
        # return self.reduction(beta,None,delta=delta)
        # return self.original(beta,bonelength,delta=delta)
        return self.simple(beta,bonelength,delta=delta)
        # return self.bonelength_encoder_test(beta)

    def init_original(self, config, num_beta):
        self.encoderBonelength = Encoder(config, config.input_dim_mesh + config.dim_bonelength, config.dim_bonelength)
        self.encoderShapestyle = Encoder(config, config.input_dim_mesh + config.dim_bonelength, config.latent_dim)
        self.decoder = Decoder(config, config.dim_bonelength + config.latent_dim, num_beta)

    def init_simple(self, config, num_beta):
        self.encoderSimple = Encoder(config, config.input_dim_mesh, config.latent_dim)
        self.decoderSimple = Decoder(config, config.latent_dim, num_beta)

    def init_reduction(self, config, num_beta):
        config_channel_list = [config.encoder_channels_0,config.encoder_channels_1,config.encoder_channels_2,
                               config.encoder_channels_3,config.encoder_channels_4,config.encoder_channels_5]
        self.encoder_channel_list = set_channel(config.encoder_channel_size-1, config_channel_list, config.input_dim_mesh, config.latent_dim)
        mlp_blocks = [mlp_block(config.input_dim_mesh // dim_0, config.input_dim_mesh // dim_1)
                      for dim_0, dim_1 in zip(self.encoder_channel_list[0:2], self.encoder_channel_list[1:3])]
        mlp_blocks += [mlp_block(config.input_dim_mesh // self.encoder_channel_list[2],
                                 config.input_dim_mesh // self.encoder_channel_list[3], False, False)]
        self.encoderAE = nn.Sequential(*mlp_blocks)

        config_channel_list = [config.encoder_channels_4,config.encoder_channels_5]
        self.encoder_channel_list = set_channel(len(config_channel_list), config_channel_list,
                                                322+23, config.latent_dim)

        self.encoderDeepBonelength = Encoder(config, 322+23, config.dim_bonelength, encoder_channel_list = self.encoder_channel_list)
        self.encoderDeepShapestyle = Encoder(config, 322+23, config.latent_dim, encoder_channel_list = self.encoder_channel_list)
        self.decoder = Decoder(config, config.dim_bonelength + config.latent_dim, num_beta)

    def init_bonelength_encoder_test(self, config, num_beta):
        self.init_simple(config, num_beta)

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



    def original(self, beta, bonelength, delta=None):
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

    def simple(self, x, bonelength, delta=None):
        mesh = self.calculate_mesh(x)
        # encode
        mesh_flat = torch.flatten(mesh, start_dim=1)

        z_mu, z_var = self.encoderSimple(mesh_flat)
        z = self.reparamterization(z_mu, z_var, delta=delta)

        # decode
        generated_x = self.decoderSimple(z)
        generated_mesh = self.calculate_mesh(generated_x)
        generated_bonelength = self.

        return mesh, generated_mesh, z_mu, z_var, bonelength, generated_bonelength

    def reduction(self, beta, bonelength, delta=None):
        x = self.calculate_mesh(beta)
        # encode
        # bone length
        x_reduction = torch.flatten(x, start_dim=1)
        x_reduction = self.encoderAE(x_reduction)

        x_reduction = torch.cat((x_reduction, bonelength), dim=1)

        z_bonelength_mu, z_bonelength_var = self.encoderDeepBonelength(x_reduction)
        z_bonelength = self.reparamterization(z_bonelength_mu, z_bonelength_var, delta=delta)

        # shape style
        z_shapestyle_mu, z_shapestyle_var = self.encoderDeepShapestyle(x_reduction)
        z_shapestyle = self.reparamterization(z_shapestyle_mu, z_shapestyle_var, delta=delta)

        # decode
        z = torch.cat((z_bonelength, z_shapestyle), dim=1)
        z_mu = torch.cat((z_bonelength_mu, z_shapestyle_mu), dim=1)
        z_var = torch.cat((z_bonelength_var, z_shapestyle_var), dim=1)
        generated_beta = self.decoder(z)
        generated_x = self.calculate_mesh(generated_beta)

        return x, generated_x, z_shapestyle_mu, z_shapestyle_var, z_bonelength, generated_beta, z_bonelength_mu, z_bonelength_var

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
        x = torch.zeros([beta.shape[0]]+list(self.reference['shapeblendshape'].shape),device=self.device,dtype=torch.float32)
        for batch_idx in range(beta.shape[0]):
            x[batch_idx] = beta[batch_idx,:] * self.reference['shapeblendshape']
            #print(x.shape,self.reference['shapeblendshape'].shape)
        x = torch.sum(x, -1) + self.reference['mesh_shape_pos']

        return x

    # def calculate_bonelength_both(self):
    #     first_idx_list = [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
    #     second_idx_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    #
    #     return self.cal_bonelength(pos_xyz=pos_xyz, first_idx_list=first_idx_list, second_idx_list=second_idx_list,
    #                                offset=offset)
    #
    # def calculate_bonelength(self, pos_xyz,
    #                        first_idx_list=[0, 1, 4, 7, 0, 3, 6, 9, 13, 16, 18, 20, 9, 12],
    #                        second_idx_list=[1, 4, 7, 10, 3, 6, 9, 13, 16, 18, 20, 22, 12, 15],
    #                        offset=3):
    #         local_size = len(first_idx_list)
    #         bonelength_list = np.zeros(local_size)
    #
    #         for i in range(local_size):
    #             bonelength_list[i] = euclidean_distance(cache=pos_xyz,
    #                                                     first_idx=first_idx_list[i],
    #                                                     second_idx=second_idx_list[i],
    #                                                     cache_offset=offset)
    #
    #             if self.max_bonelength < bonelength_list[i]:
    #                 self.max_bonelength = bonelength_list[i]
    #
    #         return bonelength_list
    #
    # def caculate_joint(self):
    #     self.reference['mesh_shape_pos']
    #
    #     def cal_bonelength_both(self, pos_xyz, offset=3):
    #
    #
    #     def cal_bonelength(self, pos_xyz,
    #                        first_idx_list=[0, 1, 4, 7, 0, 3, 6, 9, 13, 16, 18, 20, 9, 12],
    #                        second_idx_list=[1, 4, 7, 10, 3, 6, 9, 13, 16, 18, 20, 22, 12, 15],
    #                        offset=3):
    #         local_size = len(first_idx_list)
    #         bonelength_list = np.zeros(local_size)
    #
    #         for i in range(local_size):
    #             bonelength_list[i] = euclidean_distance(cache=pos_xyz,
    #                                                     first_idx=first_idx_list[i],
    #                                                     second_idx=second_idx_list[i],
    #                                                     cache_offset=offset)
    #
    #             if self.max_bonelength < bonelength_list[i]:
    #                 self.max_bonelength = bonelength_list[i]
    #
    #         return bonelength_list