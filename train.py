from __future__ import print_function
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2,1'
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from datetime import datetime
import faiss
from dataset import *

from tensorboardX import SummaryWriter
import numpy as np
import os
from model import *
import datetime
from PIL import Image
import matplotlib.pyplot as plt
import torch.autograd as autograd
now = datetime.datetime.now()
print(now.strftime("%Y.%m.%d %H:%M"))
print("PID: ", os.getpid())
writer = SummaryWriter("logs/"+now.strftime("%Y%m%d%H%M")+"flow")

# seed = 123
# torch.manual_seed(seed)
# np.random.seed(seed)
# os.environ['PYTHONHASHSEED'] = str(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)

device = torch.device("cuda")
nmf = True
mlp = True

model = Backbone(nmf=nmf, mlp=mlp)
model = model.to(device)
lr = 0.0001
margin = 0.3
optimizer = optim.Adam(model.parameters(), lr=lr)

resume = True
K = 16
print("resume =", "\033[1;32m %s \033[0m" % resume)
pth = "/root/I2PV2/i2pv118mid3k16mlp.pth.tar"
spth = "i2pv118mid3k16mlp.pth.tar"
print(pth)
print(spth)
if resume:
    checkpoint = torch.load(pth)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

criterion = TripletLossSimple(margin).to(device)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
mid_margin = 0.03
resize_shape = (400*2, 64*2)
dataset = KITTI(resize_shape=resize_shape)
print("resize shape =", resize_shape, "margin =", margin, "lr =", lr, "mid margin = ", mid_margin)
BS = 20
dataloader = DataLoader(dataset, batch_size=BS, shuffle=False, num_workers=BS, collate_fn=None, pin_memory=False)


test_set = [0, 2, 5, 6, 8]
best_recall = [0.94, 0.68, 0.90, 0.90, 0.86]
max_recall = [0 for i in test_set]
for epoch in range(200):
    # print("epoch:", epoch)
    # print(scheduler.get_last_lr())
    loss_batch = 0
    loss_mid_batch = 0
    # pbar = tqdm(total=len(dataloader))
    print(epoch%10, end="")
    print(" ", end="")
    for index, (query, pos, neg) in enumerate(dataloader):
        # pbar.update(1)
        B, n_neg, C, H, W = neg.shape
        neg = torch.flatten(neg, start_dim=0, end_dim=1)
        model.train()
        input = torch.cat([query, pos, neg])
        input = input.to(device)
        mid, output = model(input)
        vladQ, vladP, vladN = torch.split(output, [B, B, B*n_neg])
        midQ, midP, midN = torch.split(mid, [B, B, B*n_neg])
        optimizer.zero_grad()
        
        loss = 0
        loss_mid = 0
        for i in range(BS):
            max_loss = 0
            for n in range(n_neg):
                negIx = i*n_neg + n
                loss_tmp = criterion(vladQ[i:i+1], vladP[i:i+1], vladN[negIx:negIx+1])
                if loss_tmp >= max_loss:
                    max_loss=loss_tmp
                loss_mid_tmp = F.relu(torch.mean(torch.abs(midQ[i:i+1] - midP[i:i+1])) - torch.mean(torch.abs(midQ[i:i+1] - midN[i:i+1])) + mid_margin)
                # loss_mid_tmp = F.relu(torch.mean(torch.abs(midQ[i:i+1] - midP[i:i+1])))
            loss += max_loss
            loss_mid += loss_mid_tmp
        loss /= BS
        loss_mid /= BS
        loss += loss_mid
        # print(loss)
        loss.backward()
        loss_batch += loss.item()
        loss_mid_batch += loss_mid.item()
        optimizer.step()
    optimizer.zero_grad()
    scheduler.step()
    # pbar.close()
    # print(loss_batch)
    # print(loss_batch/len(dataloader))
    
    # # draw
    # model.eval()
    # q = np.load("/mnt/share_disk/KITTI/dataset/sequences/00/rgb/000000.npy").astype(np.float32)
    # q = q.transpose([1,2,0])
    # q = cv2.resize(q, resize_shape)
    # q = input_transform()(q)
    # q = q.unsqueeze(0).cuda()
    # q = model(q, epoch)
    # pos = np.load("/mnt/share_disk/KITTI/dataset/sequences/00/lidar/000000.npy")
    # pos = np.array([pos]).transpose([1,2,0]).repeat(3,2).astype(np.float32)
    # pos[pos<0]=0
    # pos = pos/60.0*255
    # pos = cv2.resize(pos, resize_shape)
    # pos = input_transform()(pos)
    # pos = pos.unsqueeze(0).cuda()
    # pos = model(pos, epoch)
    # q = np.concatenate((q,pos))
    # plt.imsave("features/features"+str(epoch)+".png", q, cmap='jet')
    
    
    if epoch >= 1 and epoch % 10 != 0:
        continue
    print("")
    # print("*"*100)
    # print("loss_batch:", loss_batch/len(dataloader))
    # test *****************************************************
    # 加载test pair
    
    deslen = 16384
    if nmf:
        deslen += K*64
    if mlp:
        deslen += K*64
    for nnni in range(len(test_set)):
        nnn = test_set[nnni]
        query_path = "/root/I2P/I2PRIBEV/rgbway/test_query2_"+str(nnn)+".txt"
        database_path = "/root/I2P/I2PRIBEV/rgbway/test_database2_"+str(nnn)+".txt"
        # query_path = "/root/I2P/I2PRIBEV/rgbway/test_query.txt"
        # database_path = "/root/I2P/I2PRIBEV/rgbway/test_database.txt"
    
        with open(query_path, 'r') as f:
            query = f.readlines()
            for i in range(len(query)):
                query[i] = query[i].strip()
        with open(database_path, 'r') as f:
            database = f.readlines()
            for i in range(len(database)):
                database[i] = database[i].strip()
                # database[i] = database[i][:48]+database[i][49:57]+"npy"
                
        des_list = np.zeros((len(database), deslen))
        for i in range(len(database)):
            pos = np.array(Image.open(database[i])).astype(np.float32)
            # pos = np.load(database[i])
            pos = np.array([pos]).transpose([1,2,0]).repeat(3,2).astype(np.float32)
            # pos[pos<0] = 0
            pos = pos*255
            pos = cv2.resize(pos, resize_shape)
            lidar = input_transform()(pos).cuda()
            lidar = torch.unsqueeze(lidar, 0)
            model.eval()
            _, lidar = model(lidar)
            des_list[(i), :] = lidar[0, :].cpu().detach().numpy()
        des_list = des_list.astype('float32')
        quantizer = faiss.IndexFlatL2(deslen)
        faiss_index = faiss.IndexIVFFlat(quantizer, deslen, 1, faiss.METRIC_L2)
        assert not faiss_index.is_trained
        faiss_index.train(des_list)
        assert faiss_index.is_trained
        faiss_index.add(des_list)
        recog_list = []
        # print("database number:", len(database), "query number:", len(query))
            
        for i in range(len(query)):
            q = np.array(Image.open(query[i])).astype(np.float32)[165:]
            # q = np.load(query[i]).transpose(1,2,0).astype(np.float32)
            q = cv2.resize(q, resize_shape)
            q = input_transform()(q).cuda()  # [3, 55, 400]
            rgb = torch.unsqueeze(q, 0)
            model.eval()
            _, rgb = model(rgb)
            des_list_current = rgb[0, :].cpu().detach().numpy()
            D, I = faiss_index.search(des_list_current.reshape(1, -1), 1)  # top 1
            for j in range(D.shape[1]):
                one_recog = np.zeros((1,3))
                one_recog[:, 0] = i
                one_recog[:, 1] = I[:,j]
                one_recog[:, 2] = D[:,j]
                recog_list.append(one_recog)
            # print("query:"+query[i] + "---->" + "database:" + database[I[:, j][0]] + "  " + str(D[:, j]))
        t_error = []
        f = open("/mnt/share_disk/KITTI/dataset/poses/"+query[0][-21:-19]+".txt", 'r')
        poses = f.readlines()
        for j in range(len(poses)):
            poses[j] = poses[j].strip().split()
        for i in range(len(recog_list)):
            pose_query = poses[int(query[int(recog_list[i][0][0])][-10:-4])]
            pose_database = poses[int(database[int(recog_list[i][0][1])][-11:-5])]
            t_error_temp = 0
            for j in [3, 7]:
                t_error_temp += np.square(float(pose_query[j])-float(pose_database[j]))
            t_error_temp = np.sqrt(t_error_temp)
            t_error.append(t_error_temp)
        ratio = []
        for dist in [0.5, 10.0]:
            ratio.append(np.sum(np.array(t_error)<dist)/len(t_error))
        
        max_recall[nnni] = max(max_recall[nnni], ratio[-1])
        
        bestlabel = " "
        if ratio[-1] == max_recall[nnni]:
            bestlabel = "+"
        if ratio[-1] > best_recall[nnni]:
            print(epoch, "\t| %.4f" % (loss_batch/len(dataloader)), " | %.4f" % (loss_mid_batch/len(dataloader)), " |", query[0][-21:-19], " |0.5m %.4f" % ratio[0], " |10m %.4f" % ratio[-1], " |best %.4f" % max_recall[nnni], " *"+bestlabel, sep='')
        else:
            print(epoch, "\t| %.4f" % (loss_batch/len(dataloader)), " | %.4f" % (loss_mid_batch/len(dataloader)), " |", query[0][-21:-19], " |0.5m %.4f" % ratio[0], " |10m %.4f" % ratio[-1], " |best %.4f" % max_recall[nnni], "  "+bestlabel, sep='')
        writer.add_scalar('10m recall', ratio[-1], global_step=epoch)
        
    torch.save({'epoch': i, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                spth)

