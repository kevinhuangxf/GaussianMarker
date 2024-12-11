import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import copy
import numpy as np
import torch.nn.functional as F


class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetEncoder(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, channel=3):
        super(PointNetEncoder, self).__init__()
        self.stn = STN3d(channel)
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        B, D, N = x.size()
        trans = self.stn(x)
        x = x.transpose(2, 1)
        if D > 3:
            feature = x[:, :, 3:]
            x = x[:, :, :3]
        x = torch.bmm(x, trans)
        if D > 3:
            x = torch.cat([x, feature], dim=2)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


def feature_transform_reguliarzer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss

# augmentation

def compute_snr(tensor1, tensor2):
    signal = torch.norm(tensor1)
    noise = torch.norm(tensor1 - tensor2)
    snr = 20 * torch.log10(signal / noise)
    return snr

def none(gaussians):
    return gaussians

def noise(gaussians):
    gaussians_aug = copy.deepcopy(gaussians)
    xyz = gaussians.get_xyz
    # Generate Gaussian noise with the same shape as the tensor using NumPy
    mean = 0
    std_dev = 1.0
    noise_np = np.random.normal(mean, std_dev, xyz.shape)
    # Convert the NumPy array to a PyTorch tensor
    noise_torch = torch.tensor(noise_np, dtype=torch.float32).cuda()
    # Add the noise tensor to the original tensor
    xyz_noise = xyz + noise_torch
    # xyz_noise = gaussians.get_xyz
    gaussians_aug._xyz = xyz_noise
    return gaussians_aug

def noise_to_gaussians(gaussians):
    gaussians_aug = copy.deepcopy(gaussians)
    xyz = gaussians.get_xyz
    # Generate Gaussian noise with the same shape as the tensor using NumPy
    mean = 0
    std_dev = 1.0
    noise_np = np.random.normal(mean, std_dev, xyz.shape)
    # Convert the NumPy array to a PyTorch tensor
    noise_torch = torch.tensor(noise_np, dtype=torch.float32).cuda()
    # Add the noise tensor to the original tensor
    xyz_noise = xyz + noise_torch
    # xyz_noise = gaussians.get_xyz
    gaussians_aug._xyz = xyz_noise
    return gaussians_aug

def noise_to_gaussians(gaussians):
    gaussians_aug = copy.deepcopy(gaussians)
    xyz = gaussians.get_xyz
    # Generate Gaussian noise with the same shape as the tensor using NumPy
    mean = 0
    std_dev = 1.0
    noise_np = np.random.normal(mean, std_dev, xyz.shape)
    # Convert the NumPy array to a PyTorch tensor
    noise_torch = torch.tensor(noise_np, dtype=torch.float32).cuda()
    # Add the noise tensor to the original tensor
    xyz_noise = xyz + noise_torch
    # xyz_noise = gaussians.get_xyz
    gaussians_aug._xyz = xyz_noise
    return gaussians_aug

def noise_to_gaussians_color(gaussians):
    gaussians_aug = copy.deepcopy(gaussians)
    xyz = gaussians._features_dc
    # Generate Gaussian noise with the same shape as the tensor using NumPy
    mean = 0
    std_dev = 8.0
    noise_np = np.random.normal(mean, std_dev, xyz.shape)
    # Convert the NumPy array to a PyTorch tensor
    noise_torch = torch.tensor(noise_np, dtype=torch.float32).cuda()
    # Add the noise tensor to the original tensor
    xyz_noise = xyz + noise_torch
    # xyz_noise = gaussians.get_xyz
    gaussians_aug._features_dc = xyz_noise
    return gaussians_aug

def rotate(gaussians):
    gaussians_aug = copy.deepcopy(gaussians)
    # Randomly rotate the point cloud data
    rotation_matrix = torch.randn(3, 3).cuda()
    gaussians_aug._xyz = torch.matmul(gaussians.get_xyz, rotation_matrix)
    return gaussians_aug

def translate(gaussians):
    gaussians_aug = copy.deepcopy(gaussians)
    # Randomly translate the point cloud data
    gaussians_aug._xyz = gaussians.get_xyz + torch.randn_like(gaussians.get_xyz) * 1.0
    return gaussians_aug

def crop_out(gaussians, crop_ratio=0.1):
    xyz = gaussians.get_xyz
    # Randomly crop out a region of xyz
    crop_size = int(xyz.shape[0] * crop_ratio)
    start_idx = torch.randint(0, xyz.shape[0] - crop_size, (1,))
    end_idx = start_idx + crop_size

    gaussians_aug = copy.deepcopy(gaussians)
    gaussians_aug._xyz = torch.cat((xyz[:start_idx], xyz[end_idx:]), dim=0)
    gaussians_aug._opacity = torch.cat((gaussians.get_opacity[:start_idx], gaussians.get_opacity[end_idx:]), dim=0)
    gaussians_aug._scaling = torch.cat((gaussians.get_scaling[:start_idx], gaussians.get_scaling[end_idx:]), dim=0)
    gaussians_aug._rotation = torch.cat((gaussians.get_rotation[:start_idx], gaussians.get_rotation[end_idx:]), dim=0)
    gaussians_aug._features_dc = torch.cat((gaussians._features_dc[:start_idx], gaussians._features_dc[end_idx:]), dim=0)
    return gaussians_aug
