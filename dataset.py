import os
from os.path import join, exists
from scipy.io import loadmat
import numpy as np
from random import randint, random
from collections import namedtuple
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from sklearn.neighbors import NearestNeighbors
import h5py
import random
from skimage import feature
import cv2
from matplotlib import pyplot as plt
# from utils.utils import *

def input_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
    ])

class KITTI(data.Dataset):
    def __init__(self, root="/mnt/share_disk/KITTI/dataset", seq="02", n_neg=10, resize_shape=(800, 128)):
        super().__init__()
        self.root = root
        self.n_neg = n_neg
        self.distThr = 2
        self.resize_shape = resize_shape
        
        pose_path = os.path.join(root, "poses", seq+".txt")
        self.poses = np.loadtxt(pose_path)
        self.poses = np.array([self.poses[:,3], self.poses[:,7]]).transpose()
        
        self.pose_q = []
        self.pose_db = []
        self.img_q = []
        self.img_db = []
        for i in range(3000):
            self.img_q.append('sequences/'+seq+'/image_2/'+str(1000000+i)[1:]+'.png')
            # self.img_q.append('sequences/'+seq+'/rgb/'+str(1000000+i)[1:]+'.npy')
            self.pose_q.append(self.poses[i])
        lidarn = "lidar3"
        for i in range(3000):
            self.img_db.append('sequences/'+seq+'/'+lidarn+'/'+str(1000000+i)[1:]+'.tiff')
            self.pose_db.append(self.poses[i])
            
        print(lidarn)
            
        
        self.pose_q  = np.array(self.pose_q)
        self.pose_db = np.array(self.pose_db)
        
    def __getitem__(self, index):
        resize_shape = self.resize_shape
        
        # query image
        q = np.array(Image.open(os.path.join(self.root, self.img_q[index]))).astype(np.float32)[165:]
        # q = np.load(os.path.join(self.root, self.img_q[index])).astype(np.float32).transpose([1,2,0])
        q = cv2.resize(q, resize_shape)
        # plt.imsave("crop.png", q, cmap='jet')
        q = input_transform()(q)
        
        
        # positive lidar
        pos = np.array(Image.open(os.path.join(self.root, self.img_db[index]))).astype(np.float32)
        # pos = np.load(os.path.join(self.root, self.img_db[index])).astype(np.float32)
        pos = np.array([pos]).transpose([1,2,0]).repeat(3,2).astype(np.float32)
        # pos[pos<0] = 0
        pos = pos*255
        pos = cv2.resize(pos, resize_shape)
        pos = input_transform()(pos) #[3, 55, 400]


        # negtive lidar
        hard_mine = 1
        neg = []
        diff = self.pose_q[index] - self.pose_db
        diff = np.linalg.norm(diff, 2, axis=1)
        rang = np.arange(len(diff))[diff > self.distThr]
        choice = np.random.choice(rang, 10)
        for i in range(self.n_neg-hard_mine):
            neg_one = np.array(Image.open(os.path.join(self.root, self.img_db[choice[i]]))).astype(np.float32)
            # neg_one = np.load(os.path.join(self.root, self.img_db[choice[i]]))
            neg_one = np.array([neg_one]).transpose([1,2,0]).repeat(3,2).astype(np.float32)
            # neg_one[neg_one<0] = 0
            neg_one = neg_one*255
            neg_one = cv2.resize(neg_one, resize_shape)
            neg_one = input_transform()(neg_one) #[3, 55, 400]
            neg.append(neg_one)
        rang = rang[np.argsort(diff[diff > self.distThr])]
        for i in range(hard_mine):  # hard mining
            neg_one = np.array(Image.open(os.path.join(self.root, self.img_db[rang[i]]))).astype(np.float32)
            # neg_one = np.load(os.path.join(self.root, self.img_db[rang[i]]))
            neg_one = np.array([neg_one]).transpose([1,2,0]).repeat(3,2).astype(np.float32)
            # neg_one[neg_one<0] = 0
            neg_one = neg_one*255
            neg_one = cv2.resize(neg_one, resize_shape)
            neg_one = input_transform()(neg_one) #[3, 55, 400]
            neg.append(neg_one)
        neg = np.stack(neg) # [10, 3, 55, 400] 
        
        return q, pos, neg
    
    def __len__(self):
        return len(self.img_q)

class HaomoData(data.Dataset):
    def __init__(self, root="/mnt/share_disk/HaomoData", n_neg=10, resize_shape=(800, 128)):
        super().__init__()
        self.root = root
        self.n_neg = n_neg
        self.distThr = 2
        self.resize_shape = resize_shape
        
        pose_path = os.path.join(root, "pose_kitti_fmt.json")
        self.poses = np.loadtxt(pose_path)
        self.poses = np.array([self.poses[:,4], self.poses[:,4]]).transpose()
        self.poses = self.poses - np.min(self.poses, axis=0)
        
        pair_path = os.path.join(root, "pair.txt")
        with open(pair_path, 'r') as f:
            pair = f.readlines()
            for i in range(len(pair)):
                pair[i] = pair[i].strip().split()
        self.pair = pair
        
        self.pose_q = []
        self.pose_db = []
        self.img_q = []
        self.img_db = []
        for i in range(len(self.pair)):
            self.img_q.append(self.root+'/cam/'+self.pair[i][0])
            self.pose_q.append(self.poses[int(self.pair[i][1][:5])])
        for i in range(len(self.pair)):
            self.img_db.append(self.root+'/lidar3/'+self.pair[i][1])
        self.pose_db = self.pose_q
        
    def __getitem__(self, index):
        resize_shape = self.resize_shape
        
        # query image
        q = np.array(Image.open(self.img_q[index])).astype(np.float32)[700:-650, :]
        # q = np.load(os.path.join(self.root, self.img_q[index])).astype(np.float32).transpose([1,2,0])
        q = cv2.resize(q, resize_shape)
        # plt.imsave("crop.png", q, cmap='jet')
        q = input_transform()(q)
        
        # positive lidar
        pos = np.array(Image.open(self.img_db[index][:-3]+"tiff")).astype(np.float32)
        # pos = np.load(os.path.join(self.root, self.img_db[index])).astype(np.float32)
        pos = np.array([pos]).transpose([1,2,0]).repeat(3,2).astype(np.float32)
        # pos[pos<0] = 0
        pos = pos*255
        pos = cv2.resize(pos, resize_shape)
        pos = input_transform()(pos) #[3, 55, 400]


        # negtive lidar
        neg = []
        diff = self.pose_q[index] - self.pose_db
        diff = np.linalg.norm(diff, 2, axis=1)
        rang = np.arange(len(diff))[diff > self.distThr]
        choice = np.random.choice(rang, 10)
        for i in range(self.n_neg):
            neg_one = np.array(Image.open(self.img_db[choice[i]][:-3]+"tiff")).astype(np.float32)
            # neg_one = np.load(os.path.join(self.root, self.img_db[choice[i]]))
            neg_one = np.array([neg_one]).transpose([1,2,0]).repeat(3,2).astype(np.float32)
            # neg_one[neg_one<0] = 0
            neg_one = neg_one*255
            neg_one = cv2.resize(neg_one, resize_shape)
            neg_one = input_transform()(neg_one) # [3, 55, 400]
            neg.append(neg_one)

        neg = np.stack(neg) # [10, 3, 55, 400] 
        
        return q, pos, neg
    
    def __len__(self):
        return len(self.img_q)

class KITTIPRE(data.Dataset):
    def __init__(self, root="/mnt/share_disk/KITTI/dataset", seq="00", n_neg=10, resize_shape=(800, 128)):
        super().__init__()
        self.root = root
        self.n_neg = n_neg
        self.distThr = 2
        self.resize_shape = resize_shape
        
        pose_path = os.path.join(root, "poses", seq+".txt")
        self.poses = np.loadtxt(pose_path)
        self.poses = np.array([self.poses[:,3], self.poses[:,7]]).transpose()
        
        self.pose_q = []
        self.pose_db = []
        self.img_q = []
        self.img_db = []
        for i in range(3000):
            self.img_q.append('sequences/'+seq+'/rgb/'+str(1000000+i)[1:]+'.npy')
            self.pose_q.append(self.poses[i])
        for i in range(3000):
            self.img_db.append('sequences/'+seq+'/lidar/'+str(1000000+i)[1:]+'.npy')
            self.pose_db.append(self.poses[i])
        
        self.pose_q = np.array(self.pose_q)
        self.pose_db = np.array(self.pose_db)
        
    def __getitem__(self, index):
        resize_shape = self.resize_shape
        
        # query image
        q = np.load(os.path.join(self.root, self.img_q[index])).astype(np.float32)
        q = q.transpose([1,2,0])
        q = cv2.resize(q, resize_shape)
        q = input_transform()(q)
        
        
        # positive lidar
        pos = np.load(os.path.join(self.root, self.img_db[index]))
        pos = np.array([pos]).transpose([1,2,0]).repeat(3,2).astype(np.float32)
        pos[pos<0]=0
        pos = pos/60.0*255
        pos = cv2.resize(pos, resize_shape)
        pos = input_transform()(pos) #[3, 55, 400]


        # negtive lidar
        hard_mine = 1
        neg = []
        diff = self.pose_q[index] - self.pose_db
        diff = np.linalg.norm(diff, 2, axis=1)
        rang = np.arange(len(diff))[diff > self.distThr]
        choice = np.random.choice(rang, 10)
        for i in range(self.n_neg-hard_mine):
            neg_one = np.load(os.path.join(self.root, self.img_db[choice[i]]))
            neg_one = np.array([neg_one]).transpose([1,2,0]).repeat(3,2).astype(np.float32)
            neg_one[neg_one==-1]=0
            neg_one = neg_one/60.0*255
            neg_one = cv2.resize(neg_one, resize_shape)
            neg_one = input_transform()(neg_one) #[3, 55, 400]
            neg.append(neg_one)
        rang = rang[np.argsort(diff[diff > self.distThr])]
        for i in range(hard_mine):  # hard mining
            neg_one = np.load(os.path.join(self.root, self.img_db[rang[i]]))
            neg_one = np.array([neg_one]).transpose([1,2,0]).repeat(3,2).astype(np.float32)
            neg_one[neg_one==-1]=0
            neg_one = neg_one/60.0*255
            neg_one = cv2.resize(neg_one, resize_shape)
            neg_one = input_transform()(neg_one) #[3, 55, 400]
            neg.append(neg_one)
        neg = np.stack(neg) # [10, 3, 55, 400] 
        

        return q, pos, neg
    
    def __len__(self):
        return len(self.img_q)

if __name__ == "__main__":
    dataset = KITTI()
    q, pos, neg = dataset.__getitem__(1)
    print(len(dataset))
    # print(q)