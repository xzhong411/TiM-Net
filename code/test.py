# -*- coding: utf-8 -*-
import glob

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import xlrd
import xlwt
import xlsxwriter
import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score
from xlrd import open_workbook
from xlutils.copy import copy
import os
import argparse
import time

from core.models import UNet
from core.utils import calculate_Accuracy, get_data,get_model
from pylab import *

plt.switch_backend('agg')

# --------------------------------------------------------------------------------

model_name = 'TiMNet'

# --------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='PyTorch ASOCT_Demo')
# ---------------------------
# params do not need to change
# ---------------------------
parser.add_argument('--epochs', type=int, default=250,
                    help='the epochs of this run')
parser.add_argument('--n_class', type=int, default=2,
                    help='the channel of out img, decide the num of class, ASOCT_eyes is 2/4 class')
parser.add_argument('--lr', type=float, default=0.0015,
                    help='initial learning rate')
parser.add_argument('--GroupNorm', type=bool, default=True,
                    help='decide to use the GroupNorm')
parser.add_argument('--BatchNorm', type=bool, default=False,
                    help='decide to use the BatchNorm')
# ---------------------------
# model
# ---------------------------
parser.add_argument('--data_path', type=str, default='../data/STARE',
                    help='dir of the all img')
parser.add_argument('--model_save', type=str, default='../models/stare_best_model.pth',
                    help='dir of the model.pth')
parser.add_argument('--my_description', type=str, default='dualatt',
                    help='some description define your training')
parser.add_argument('--best_model', type=str,  default='final.pth',
                    help='the pretrain model')
parser.add_argument('--batch_size', type=int, default=1,
                    help='the num of img in a batch')
parser.add_argument('--img_size', type=int, default=512,
                    help='the train img size')

# ---------------------------
# GPU
# ---------------------------
parser.add_argument('--use_gpu', type=bool, default=True,
                    help='dir of the all ori img')
parser.add_argument('--gpu_avaiable', type=str, default='3',
                    help='the gpu used')

args = parser.parse_args()


def fast_test(model, args, model_name):
    softmax_2d = nn.Softmax2d()
    EPS = 1e-12

    Background_IOU = []
    Vessel_IOU = []
    ACC = []
    SE = []
    SP = []
    AUC = []
    data_path = args.data_path
    test_img_list = glob.glob(os.path.join(data_path, 'test/image/*.png'))
    for i, img_path in enumerate(test_img_list):
        start = time.time()
        save_res_path = (img_path.replace('test/image', 'testsave'))
        # img_path = test_img_list[start:end]
        img, gt, tmp_gt, img_shape,label_ori = get_data(args.data_path, [img_path], img_size=args.img_size, gpu=args.use_gpu)
        model.eval()
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)
        out, side_5, side_6, side_7, side_8 = model(img)

        pred = np.array(out.data.cpu()[0])
        save = np.zeros(shape=(512, 512))
        save[pred[0] < 0.5] = 255
        save[pred[1] < 0.5] = 0
        cv2.imwrite(save_res_path, save)

        out = torch.log(softmax_2d(out) + EPS)

        out = F.upsample(out, size=(img_shape[0][0],img_shape[0][1]), mode='bilinear')
        out = out.cpu().data.numpy()


        y_pred =out[:,1,:,:]
        y_pred = y_pred.reshape([-1])
        ppi = np.argmax(out, 1)

        tmp_out = ppi.reshape([-1])
        tmp_gt=label_ori.reshape([-1])

        my_confusion = metrics.confusion_matrix(tmp_out, tmp_gt).astype(np.float32)
        meanIU, Acc,Se,Sp,IU = calculate_Accuracy(my_confusion)
        Auc = roc_auc_score(tmp_gt, y_pred)
        AUC.append(Auc)

        Background_IOU.append(IU[0])
        Vessel_IOU.append(IU[1])
        ACC.append(Acc)
        SE.append(Se)
        SP.append(Sp)
        end = time.time()
        print(str(i+1)+r'/'+str(len(test_img_list))+': '+'| Acc: {:.3f} | Se: {:.3f} | Sp: {:.3f} | Auc: {:.3f} |  Background_IOU: {:f}, vessel_IOU: {:f}'.format(Acc,Se,Sp,Auc,IU[0], IU[1])+'  |  time:%s'%(end-start))
    print('Acc: %s  |  Se: %s |  Sp: %s |  Auc: %s |  Background_IOU: %s |  vessel_IOU: %s '%(str(np.mean(np.stack(ACC))),str(np.mean(np.stack(SE))), str(np.mean(np.stack(SP))),str(np.mean(np.stack(AUC))),str(np.mean(np.stack(Background_IOU))),str(np.mean(np.stack(Vessel_IOU)))))

    # todo 创建excel文件
    xl = xlsxwriter.Workbook(r'../result.xls')

    # todo 添加sheet
    sheet = xl.add_worksheet('chasedb1')

    # todo 往单元格cell添加数据,索引写入
    sheet.write(12, 1, str(np.mean(np.stack(ACC))))
    sheet.write(12, 2, str(np.mean(np.stack(SE))))
    sheet.write(12, 3, str(np.mean(np.stack(SP))))
    sheet.write(12, 4, str(np.mean(np.stack(AUC))))
    sheet.write(12, 5, str(np.mean(np.stack(Background_IOU))))
    sheet.write(12, 6, str(np.mean(np.stack(Vessel_IOU))))
    # todo 关闭文件
    xl.close()



    # store test information
    with open(r'../logs/%s_%s.txt' % (model_name, args.my_description), 'a+') as f:
        f.write('Acc: %s  |  Se: %s |  Sp: %s |  Auc: %s |  Background_IOU: %s |  vessel_IOU: %s '%(str(np.mean(np.stack(ACC))),str(np.mean(np.stack(SE))), str(np.mean(np.stack(SP))),str(np.mean(np.stack(AUC))),str(np.mean(np.stack(Background_IOU))),str(np.mean(np.stack(Vessel_IOU)))))
        f.write('\n\n')

    return np.mean(np.stack(Vessel_IOU))


if __name__ == '__main__':
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] ='0,1,2'

    # model = get_model(model_name)
    # model = model(n_classes=args.n_class, bn=args.GroupNorm, BatchNorm=args.BatchNorm)
    model = UNet(n_channels=3,n_classes=args.n_class)
    model = nn.DataParallel(model)
    if args.use_gpu:
        model.cuda()
    if True:

        model.load_state_dict(torch.load(args.model_save),False)
        print('success load models: %s_%s' % (model_name, args.my_description))

    print('This model is %s_%s_%s' % (model_name, args.n_class, args.img_size))
    fast_test(model, args, model_name)
