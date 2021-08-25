import gc
import os
import random
import time
import warnings
warnings.simplefilter("ignore")
#import pdb
#import zipfile
#import pydicom
from albumentations import *
#from albumentations.pytorch import ToTensor
import cv2
from matplotlib import pyplot as plt
import numpy as np
#import pandas as pd
from PIL import Image, ImageFilter
import segmentation_models_pytorch as smp
from sklearn.model_selection import KFold
import tifffile as tiff
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.nn import functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, sampler
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR,CosineAnnealingWarmRestarts,StepLR,ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
from warmup_scheduler import GradualWarmupScheduler
#from lovaszloss import *
#from valid_score import *
from torchvision import transforms
import model.my_unet as my_decoder
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
#import loss_func
def set_seed(seed=2**3):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
set_seed(22)

fold = 0
nfolds = 5
reduce = 4
sz = 512
warmup_epo = 1
cosine_epo = 14
BATCH_SIZE =6
DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 30
NUM_WORKERS = 0
SEED = 22




MASKSori = '../mask/'
TRAINori = '../img/'

def binary_cross_entropy(inputs, target, weight=None, reduction='mean', smooth_eps=None, from_logits=False):
    """cross entropy loss, with support for label smoothing https://arxiv.org/abs/1512.00567"""
    smooth_eps = smooth_eps or 0
    if smooth_eps > 0:
        target = target.float()
        target.add_(smooth_eps).div_(2.)
    if from_logits:
        return F.binary_cross_entropy_with_logits(inputs, target, weight=weight, reduction=reduction)
    else:
        return F.binary_cross_entropy(inputs, target, weight=weight, reduction=reduction)


def binary_cross_entropy_with_logits(inputs, target, weight=None, reduction='mean', smooth_eps=None, from_logits=True):
    return binary_cross_entropy(inputs, target, weight, reduction, smooth_eps, from_logits)


class BCELoss(nn.BCELoss):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean', smooth_eps=None, from_logits=False):
        super(BCELoss, self).__init__(weight, size_average, reduce, reduction)
        self.smooth_eps = smooth_eps
        self.from_logits = from_logits

    def forward(self, input, target):
        return binary_cross_entropy(input, target,
                                    weight=self.weight, reduction=self.reduction,
                                    smooth_eps=self.smooth_eps, from_logits=self.from_logits)


class BCEWithLogitsLoss(BCELoss):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean', smooth_eps=None, from_logits=True):
        super(BCEWithLogitsLoss, self).__init__(weight, size_average,
                                                reduce, reduction, smooth_eps=smooth_eps, from_logits=from_logits)
class GradualWarmupSchedulerV2(GradualWarmupScheduler):
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        super(GradualWarmupSchedulerV2, self).__init__(optimizer, multiplier, total_epoch, after_scheduler)
    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

class Dice_th_pred():
    def __init__(self, ths=np.arange(0.1, 0.9, 0.01), axis=1):
        self.axis = axis
        self.ths = ths
        self.reset()

    def reset(self):
        self.inter = torch.zeros(len(self.ths))
        self.union = torch.zeros(len(self.ths))

    def accumulate(self, p, t):
        p = F.sigmoid(p).detach()
        #_,p = torch.max(p, dim=1)
        pred = p.view(-1)
        targ = t.view(-1)
        #pred, targ = flatten_check(p, t)
        for i, th in enumerate(self.ths):
            p = (pred > th).float()
            self.inter[i] += (p * targ).float().sum().item()
            self.union[i] += (p + targ).float().sum().item()
    #该装饰器的作用是像属性一样访问（只读），不用加()
    @property
    def value(self):
        dices = torch.where(self.union > 0.0, 2.0 * self.inter / self.union,
                            torch.zeros_like(self.union))
        return dices
def dice_confi(p,t):
    p = F.sigmoid(p).detach()
    pred = p.view(-1)
    targ = t.view(-1)
    p = (pred > 0.387).float()
    inter = (p * targ).float().sum().item()
    union = (p + targ).float().sum().item()
    dice_value = 2.0 * inter / (union+0.000001)
    return dice_value

def rle_encode_less_memory(img):
    # watch out for the bug
    pixels = img.T.flatten()

    # This simplified method requires first and last pixel to be zero
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] -= runs[::2]

    return ' '.join(str(x) for x in runs)


mean = np.array([0.63701, 0.47097, 0.68174])
std = np.array([0.11475, 0.16170, 0.08817])

#图像numpy转tensor，同时维度如果为2扩展到3，(1,长，宽)
def img2tensor(img, dtype: np.dtype = np.float32):
    if img.ndim == 2: img = np.expand_dims(img, 2)
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img.astype(dtype, copy=False))


class HuBMAPDataset(Dataset):
    def __init__(self, mydata = None, train=True, tfms=None):
        '''
        # 根据患者id划分数据集到5折--------------
        ids = np.array([i for i in range(131)])
        kf = KFold(n_splits=nfolds, random_state=22, shuffle=True)
        #划分训练和验证集，kf.split之后两个array，0为训练集，1为验证集
        ids = set(ids[list(kf.split(ids))[fold][0 if train else 1]])
        '''
        #获得全体png中的‘训练集’png，命名如'volume-9_71.png'
        self.fnames = mydata#[fname for fname in os.listdir(TRAINori) if int(fname.split('-')[1].split('_')[0]) in ids]   # 用于训练的ct图像文件名
        # -------------------------------------
        self.train = train
        self.tfms = tfms


    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        if fname in self.fnames:
            # 根据ct图像文件名 获取患者id与当前序号
            #patient_id = fname.split('-')[1].split('_')[0]   # 患者id
            #patient_idx = fname.split('_')[1].split('.')[0]  # 当前切片序号

            #读取BGR并转化为RGB
            #img_path = os.path.join(TRAINori, fname)
            #mask_path = os.path.join(MASKSori, fname.split('_m')[0] + '.png')
            #maskpath = os.path.join(MASKSori, fname.split('.')[0] + '_m.png')
            img = cv2.cvtColor(cv2.imread(os.path.join(TRAINori, fname),-1), cv2.COLOR_BGR2RGB)
            mask = cv2.imread(os.path.join(MASKSori, fname.split('.')[0] + '_m.png'),-1)#cv2.cvtColor(cv2.imread(os.path.join(MASKSori, fname.split('.')[0] + '_m.png')), cv2.IMREAD_GRAYSCALE)

            #img = cv2.cvtColor(cv2.imread(os.path.join(TRAINori, fname),-1), cv2.COLOR_BGR2RGB)
            #mask = cv2.imread(os.path.join(MASKSori, 'segmentation-'+patient_id+'_'+patient_idx+'.png'), cv2.IMREAD_GRAYSCALE)
            #resize
            img = cv2.resize(img, (sz, sz))
            mask = cv2.resize(mask, (sz, sz))
            # 可视化img mask
            '''
            plt.subplot(1, 2, 1)
            plt.imshow(img)
            plt.subplot(1, 2, 2)
            plt.imshow(mask*128)
            plt.show()
            '''
            if self.tfms is not None:
                augmented = self.tfms(image=img, mask=mask)
                img, mask = augmented['image'], augmented['mask']
            # return img2tensor((img / 255.0 - mean) / std), img2tensor(mask)
            # 区分liver_mask 和 tumor_mask
            '''
            plt.subplot(1, 3, 1)
            plt.imshow(img)
            plt.subplot(1, 3, 2)
            plt.imshow(mask > 0)
            plt.subplot(1, 3, 3)
            plt.imshow(mask>1)
            plt.show()
            '''
            liver_mask = (mask > 0).astype(np.int64) # liver
            tumor_mask = (mask > 1).astype(np.int64) # tumor

            return img2tensor((img / 255.0)), img2tensor(liver_mask),img2tensor(tumor_mask)

'''
            随机放射变换，对图片进行平移（translate）、缩放（scale）和旋转（roatate）
            shift_limit：图片宽高的平移因子
            scale_limit：图片缩放因子
            rotate_limit：图片旋转范围
            border_mode：OpenCV 标志，用于指定使用的外插算法（extrapolation）
            p：使用此转换的概率，默认值为 0.5
            '''
def get_transforms(*, data):
    if data == 'train_weak':
        return Compose([
            #水平镜像对称
            HorizontalFlip(),
            #垂直镜像对称
            VerticalFlip(),
            RandomRotate90(),

            ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=0.9,
                               border_mode=cv2.BORDER_REFLECT),

        ])
    elif data =='train_heavy':
        return Compose([
            Transpose(p=0.5),
            VerticalFlip(p=0.5),
            HorizontalFlip(p=0.5),
            ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
            OneOf([
                RandomGamma(),
                GaussNoise()
            ], p=0.5),
            OneOf([
                OpticalDistortion(p=0.4),
                GridDistortion(p=0.2),
                IAAPiecewiseAffine(p=0.4),
            ], p=0.5),
            OneOf([
                HueSaturationValue(10, 15, 10),
                CLAHE(clip_limit=4),
                RandomBrightnessContrast(),
            ], p=0.5),

            ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
            Cutout(max_h_size=int(256 * 0.05), max_w_size=int(256 * 0.05), num_holes=1,
                                  p=0.75),

        ])

class SegModel(nn.Module):
    def __init__(self):
        super(SegModel, self).__init__()
        # self.seg_liver = smp.Unet(
        #          encoder_name='efficientnet-b0',
        #          encoder_weights='imagenet',
        #          decoder_attention_type = 'scse',
        #          in_channels=3,
        #          classes=1)
        self.seg_tumor = smp.Unet(
            encoder_name='efficientnet-b0',
            encoder_weights='imagenet',
            decoder_attention_type='scse',
            in_channels=3,
            classes=1)

        # self.seg_liver.decoder = my_decoder.UnetDecoder(encoder_channels=self.seg_liver.encoder.out_channels,
        #                                                 attention_type='scse')
        self.seg_tumor.decoder = my_decoder.UnetDecoder(encoder_channels=self.seg_tumor.encoder.out_channels,
                                                        attention_type='scse')


        # self.seg_liver.segmentation_head = my_decoder.SegmentationHead(in_channels=4,out_channels=1)
        self.seg_tumor.segmentation_head = my_decoder.SegmentationHead(in_channels=4,out_channels=1)
        #self.seg_liver.segmentation_head
    def forward(self,x):
        # 分割肝脏
        # global_features_liver = self.seg_liver.encoder(x)
        # seg_features_liver = self.seg_liver.decoder(*global_features_liver)
        # seg_features_liver = self.seg_liver.segmentation_head(seg_features_liver)
        # 分割肿瘤
        #global_features_tumor = self.seg_tumor.encoder(torch.cat((x,seg_features_liver),1))   # img 拼接 liver_feature
        # global_features_tumor = self.seg_tumor.encoder(x*seg_features_liver)
        global_features_tumor = self.seg_tumor.encoder(x )
        seg_features_tumor = self.seg_tumor.decoder(*global_features_tumor)
        seg_features_tumor = self.seg_tumor.segmentation_head(seg_features_tumor)


        return seg_features_tumor

        # return seg_features_liver, seg_features_tumor


class BCEFocalLoss(torch.nn.Module):
  """
  二分类的Focalloss alpha 固定
  """
  def __init__(self, gamma=2, alpha=0.25, reduction='elementwise_mean'):
    super().__init__()
    self.gamma = gamma
    self.alpha = alpha
    self.reduction = reduction

  def forward(self, _input, target):
    pt = torch.sigmoid(_input)
    alpha = self.alpha
    loss = - alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - \
        (1 - alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)
    if self.reduction == 'elementwise_mean':
      loss = torch.mean(loss)
    elif self.reduction == 'sum':
      loss = torch.sum(loss)
    return loss
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        ''' 
        comment out if your model contains a sigmoid or equivalent activation layer
        smooth=1的作用
        （1）避免当|X|和|Y|都为0时，分子被0除的问题
        （2）减少过拟合
        '''
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


#img =cv2.imread('../mask/1/20181227110358935008_m.png',-1)
#--------------------
group = []
fold_names = os.listdir('../img/')
kf = KFold(n_splits=nfolds, random_state=22, shuffle=True)
for train_idx ,val_idx in kf.split(fold_names):
    trainpart = [fold_names[i] for i in train_idx]
    valpart = [fold_names[i] for i in val_idx]
    group.append([trainpart,valpart])
#--------------------
for fold in range(nfolds):
    train_paths = []
    val_paths = []

    train_folds = group[fold][0]
    val_folds = group[fold][1]

    for foldname in train_folds:
        for name in os.listdir(f'../img/{foldname}/'):
            train_paths.append(f'{foldname}/{name}')
    for foldname in val_folds:
        for name in os.listdir(f'../img/{foldname}/'):
            val_paths.append(f'{foldname}/{name}')
    break

    ds_t = HuBMAPDataset(mydata=train_paths, train=True, tfms=get_transforms(data='train_weak'))
    ds_v = HuBMAPDataset(mydata=val_paths, train=False)
    dataloader_t = torch.utils.data.DataLoader(ds_t, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,pin_memory=True)
    dataloader_v = torch.utils.data.DataLoader(ds_v, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,pin_memory=True)
    model = SegModel()
    if torch.cuda.device_count()>1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model,device_ids=[0,1])
        
    model.to(DEVICE)

    #model.seg.encoder._dropout= nn.Dropout(p=0.5, inplace=False)
    """
    optimizer = torch.optim.Adam([
        {'params': model.decoder.parameters(), 'lr': 1e-3},
        {'params': model.encoder.parameters(), 'lr': 1e-3},
    ])
    """
    optimizer = Adam(model.parameters(), lr=(1e-4) / 8, weight_decay=1e-6, amsgrad=False)

    #scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=40, T_mult=1, eta_min=1e-6, last_epoch=-1)
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cosine_epo)
    scheduler_warmup = GradualWarmupSchedulerV2(optimizer, multiplier=10, total_epoch=warmup_epo,
                                                after_scheduler=scheduler_cosine)
    #bce_smoothloss = loss_func.SoftBCEWithLogitsLoss(smooth_factor = 0.1)
    bceloss = nn.BCEWithLogitsLoss()
    #celoss = nn.CrossEntropyLoss()
    #celoss = loss_func.SoftCrossEntropyLoss()
    scaler = GradScaler()
    print(f"########FOLD: {fold}##############")
    best_score = 0.
    best_loss = 10.
    total_val_loss = []  # 记录loss参数
    # total_val_liver_score = []  # 记录liver score
    total_val_tumor_score = []  # 记录tumor score
    for epoch in range(EPOCHS):
        scheduler_warmup.step(epoch)
        # dice_l = Dice_th_pred(np.arange(0., 0.9, 0.01))
        dice_t = Dice_th_pred(np.arange(0., 0.9, 0.01))
        ###Train
        model.train()
        train_loss = 0
        # loss_weight = [0.5,0.5]  # liver tumor 各占一半
        for step,data in tqdm(enumerate(dataloader_t),total=len(dataloader_t)):
            optimizer.zero_grad()
            img, liver_mask, tumor_mask = data
            img = img.to(DEVICE)
            liver_mask = liver_mask.to(DEVICE)
            tumor_mask = tumor_mask.to(DEVICE)
            with autocast():
                # liver,tumor = model(img)
                tumor = model(img*liver_mask)
                # loss_liver = bceloss(liver,liver_mask)
                loss_tumor = bceloss(tumor,tumor_mask)
                # loss = loss_weight[0]*loss_liver+loss_weight[1]*loss_tumor
                loss = loss_tumor
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()


            train_loss += loss.item()

        train_loss /= len(dataloader_t)

        print(f"FOLD: {fold}, EPOCH: {epoch + 1}, train_loss: {train_loss}")

        ###Validation
        model.eval()
        valid_loss = 0
        for step,data in tqdm(enumerate(dataloader_v),total=len(dataloader_v)):
            img, liver_mask, tumor_mask = data
            img = img.to(DEVICE)
            liver_mask = liver_mask.to(DEVICE)
            tumor_mask = tumor_mask.to(DEVICE)
            with torch.no_grad():
                '''
                p_liver:肝脏预测mask
                p_tumor:肿瘤预测mask
                dice_l: 肝脏dice计算
                dice_t: 肿瘤dice计算
                '''
                # p_liver,p_tumor = model(img)
                p_tumor = model(img*liver_mask)
                # loss_liver = bceloss(p_liver, liver_mask)
                loss_tumor = bceloss(p_tumor, tumor_mask)
                # loss = loss_weight[0] * loss_liver + loss_weight[1] * loss_tumor
                loss = loss_tumor
                # dice_l.accumulate(p_liver, liver_mask)
                dice_t.accumulate(p_tumor, tumor_mask)
            valid_loss += loss.item()


        valid_loss /= len(dataloader_v)

        # dices_l = dice_l.value
        dices_t = dice_t.value

        # best_dice_l = dices_l.max()
        best_dice_t = dices_t.max()

        # noise_ths_l = dice_l.ths
        noise_ths_t = dice_t.ths

        # best_thr_l = noise_ths_l[dices_l.argmax()]
        best_thr_t = noise_ths_t[dices_t.argmax()]
        # 数据存入list保存
        total_val_loss.append(valid_loss)
        # total_val_liver_score.append(best_dice_l)
        total_val_tumor_score.append(best_dice_t)

        # print(f"FOLD: {fold}, EPOCH: {epoch + 1}, valid_loss: {valid_loss:.4f} ,valid liver_score: {best_dice_l:.4f},valid tumor_score: {best_dice_t:.4f},"
        #       f"valid best_liver_th:{best_thr_l:.4f},valid best_tumor_th:{best_thr_t:.4f},lr:{optimizer.param_groups[0]['lr']:.5f}"
        print(f"FOLD: {fold}, EPOCH: {epoch + 1}, valid_loss: {valid_loss:.4f} ,valid tumor_score: {best_dice_t:.4f},"
              f"valid best_tumor_th:{best_thr_t:.4f},lr:{optimizer.param_groups[0]['lr']:.5f}")
        if best_score < best_dice_t:
            best_score = best_dice_t
            # torch.save({'model':model.state_dict(),
            #             'th':[best_thr_l,best_thr_t]},
            #            f"./weight/FOLD{fold}_best_score.pth")
            torch.save({'model':model.state_dict(),
                        'th':[best_thr_t]},
                       f"./weight/FOLD{fold}_best_score.pth")
            print("save best score model")
        if valid_loss < best_loss:
            best_loss = valid_loss
            # torch.save({'model': model.state_dict(),
            #             'th': [best_thr_l, best_thr_t]},
            #            f"./weight/FOLD{fold}_best_loss.pth")
            torch.save({'model': model.state_dict(),
                        'th': [best_thr_t]},
                       f"./weight/FOLD{fold}_best_loss.pth")
            print("save best  loss model")
    np.save(f'./weight/fold_{fold}val_loss.npy',np.array(total_val_loss))
    # np.save(f'./weight/fold_{fold}val_liver_score.npy', np.array(total_val_liver_score))
    np.save(f'./weight/fold_{fold}val_tumor_score.npy', np.array(total_val_tumor_score))
