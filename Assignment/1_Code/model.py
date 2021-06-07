import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNetAE(nn.Module):
    def __init__(self, in_dim=3, n_points=2048):
        super(PointNetAE, self).__init__()
        self.in_dim = in_dim
        self.n_points = n_points
        
        self.mlp_dims = [1024, 512, 256, 128]
        self.n_mlp_layers = len(self.mlp_dims) - 1 # layer 개수 : 3
        assert(self.n_mlp_layers >= 1)
        
        self.dropout_prop = 0.5
        
        self.conv1 = nn.Conv1d(self.in_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        
        self.fc1 = nn.Linear(128, int(n_points/2), 1)
        self.fc2 = nn.Linear(int(n_points/2), n_points, 1)
        self.fc3 = nn.Linear(n_points, n_points*3)
        
        self.n1 = nn.BatchNorm1d(64)
        self.n2 = nn.BatchNorm1d(64)
        self.n3 = nn.BatchNorm1d(64)
        self.n4 = nn.BatchNorm1d(128)
        
        self.bn1 = nn.BatchNorm1d(int(n_points/2))
        self.bn2 = nn.BatchNorm1d(n_points)
        
        self.bn = nn.ModuleList()
        self.do = nn.ModuleList()
        
        for i in range(self.n_mlp_layers):
            if (i+1) < self.n_mlp_layers:
                self.bn.append(nn.BatchNorm1d(self.mlp_dims[i+1]))
                self.do.append(nn.Dropout(p=self.dropout_prop))
        
    def forward(self, x):
        # input : (batch_size, n_points, in_dims)
        batch_size = x.shape[0]
        n_points = x.shape[1]
        in_dims = x.shape[2]
        
        x = x.transpose(2, 1)
        
        # encoder
        x = F.relu(self.n1(self.conv1(x)))
        x = F.relu(self.n2(self.conv2(x)))
        x = F.relu(self.n3(self.conv3(x)))
        x = F.relu(self.n4(self.conv4(x)))
        
        # max pooling
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 128)
        
        global_feature = x
        
        # decoder
        x = self.do[0](F.relu(self.bn1(self.fc1(x)))) # Dropout은 classification을 위한 용도일 수 있다 ! 혹 문제가 생기면 dropout을 지워보자
        x = self.do[1](F.relu(self.bn2(self.fc2(x))))
        reconstructed_points = self.fc3(x)
        
        # do reshaping
        reconstructed_points = reconstructed_points.reshape(batch_size, n_points, in_dims)
        
        return reconstructed_points, global_feature
        