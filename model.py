import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import h5py
import netvlad
from NMF import *

class Backbone(nn.Module):
    def __init__(self, nmf=True, mlp=False):
        super(Backbone, self).__init__()
        
        self.nmf = nmf
        self.K = 16
        self.max_iter = 50
        print("nmf: ", self.nmf)
        print("K: ", self.K)
        
        self.mlp = mlp
        print("mlp: ", self.mlp)
        
        encoder = models.resnet34(pretrained=True) # resnet34 capture only features and remove last relu and maxpool
        layers = list(encoder.children())[:-3]
        self.encoder = nn.Sequential(*layers)
        self.relu = nn.ReLU(inplace=True)
        
        self.mlp = nn.Sequential(nn.Conv2d(256, 128, kernel_size=(1, 1)),
                                 nn.ReLU(),
                                 nn.Conv2d(128, self.K, kernel_size=(1, 1)),
                                 nn.ReLU())
        
        self.net_vlad_cnn = netvlad.NetVLAD(num_clusters=64, dim=256, vladv2=True)
        self.net_vlad_nmf = netvlad.NetVLAD(num_clusters=64, dim=self.K,  vladv2=True)
        self.net_vlad_mlp = netvlad.NetVLAD(num_clusters=64, dim=self.K,  vladv2=True)
        
        initcache = "/root/I2P/I2P-v2/centroid/centroids/resnet34_ugv_data_64_desc_cen.hdf5"
        with h5py.File(initcache, mode='r') as h5:
            clsts = h5.get("centroids")[...]
            traindescs = h5.get("descriptors")[...]
            self.net_vlad_cnn.init_params(clsts, traindescs)
            # self.net_vlad_nmf.init_params(clsts, traindescs)
            del clsts, traindescs

    def forward(self, x):
        # cnn
        x = self.encoder(x)
        x = F.normalize(x, p=2, dim=1)
        
        x_mid = x
        
        if self.nmf:
            with torch.no_grad(): # nmf is not derivable
                features = x.contiguous()
                b, h, w = features.size(0), features.size(2), features.size(3)
                features = self.relu(features)
                flat_features = features.permute(0, 2, 3, 1).contiguous().view(-1, features.size(1))
                W, _ = NMF(flat_features, self.K, random_seed=1, cuda=True, max_iter=self.max_iter, verbose=False)
                isnan = torch.sum(torch.isnan(W).float())
                while isnan > 0:
                    print('nan detected. trying to resolve the nmf.')
                    W, _ = NMF(flat_features, self.K, random_seed=random.randint(0, 255), cuda=True, max_iter=self.max_iter, verbose=False)
                    isnan = torch.sum(torch.isnan(W).float())
                heatmaps = W.view(b, h, w, self.K).permute(0,3,1,2)
                heatmaps = F.normalize(heatmaps, p=2, dim=1)
                heatmaps.requires_grad = False
                x_nmf = self.net_vlad_nmf(heatmaps)
                x_nmf = F.normalize(x_nmf, p=2, dim=1)
        
        if self.mlp:
            feature_mlp = x
            feature_mlp = self.mlp(feature_mlp)
            feature_mlp = F.normalize(feature_mlp, p=2, dim=1)
            feature_mlp = self.net_vlad_mlp(feature_mlp)
            x_mlp = F.normalize(feature_mlp, p=2, dim=1)




        x = self.net_vlad_cnn(x)
        x = F.normalize(x, p=2, dim=1)
        
        if self.nmf:
            x = torch.cat((x, x_nmf), 1)
        if self.mlp:
            x = torch.cat((x, x_mlp), 1)
            
        return x_mid, x
    
class TripletLossSimple(nn.Module):
    def __init__(self, margin=0.3):
        super(TripletLossSimple, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        
        pos_dist = torch.sqrt((anchor - positive).pow(2).sum(1))
        neg_dist = torch.sqrt((anchor - negative).pow(2).sum(1))
        loss = F.relu(pos_dist-neg_dist + self.margin)
        return loss.mean()