import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
from torch.autograd import Variable
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim

import glob
import os
import shutil

from data_loader import RescaleT
from data_loader import RandomCrop
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET
from model import U2NETP

from PyQt5.QtCore import QThread, QMutex

def muti_bce_loss_fusion(bce_loss, d0, d1, d2, d3, d4, d5, d6, labels_v):
    """

    :param bce_loss:
    :param d0:
    :param d1:
    :param d2:
    :param d3:
    :param d4:
    :param d5:
    :param d6:
    :param labels_v:
    :return:
    """
    loss0 = bce_loss(d0,labels_v)
    loss1 = bce_loss(d1,labels_v)
    loss2 = bce_loss(d2,labels_v)
    loss3 = bce_loss(d3,labels_v)
    loss4 = bce_loss(d4,labels_v)
    loss5 = bce_loss(d5,labels_v)
    loss6 = bce_loss(d6,labels_v)

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
    print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n"%(loss0.data.item(),loss1.data.item(),loss2.data.item(),loss3.data.item(),loss4.data.item(),loss5.data.item(),loss6.data.item()))

    return loss0, loss


def prepare_train(data_dir, tra_image_dir, tra_label_dir, batch_size_train, image_ext=".jpg", label_ext=".png"):
    """

    :param data_dir:
    :param tra_image_dir:
    :param tra_label_dir:
    :param image_ext:
    :param label_ext:
    :return:
    """
    tra_img_name_list = glob.glob(data_dir + tra_image_dir + '*' + image_ext)
    tra_lbl_name_list = []
    for img_path in tra_img_name_list:
        img_name = img_path.split(os.sep)[-1]

        aaa = img_name.split(".")
        bbb = aaa[0:-1]
        imidx = bbb[0]
        for i in range(1, len(bbb)):
            imidx = imidx + "." + bbb[i]

        tra_lbl_name_list.append(data_dir + tra_label_dir + imidx + label_ext)

    print("---")
    print("train images: ", len(tra_img_name_list))
    print("train labels: ", len(tra_lbl_name_list))
    print("---")

    train_num = len(tra_img_name_list)

    salobj_dataset = SalObjDataset(
        img_name_list=tra_img_name_list,
        lbl_name_list=tra_lbl_name_list,
        transform=transforms.Compose([
            RescaleT(320),
            RandomCrop(288),
            ToTensorLab(flag=0)]))
    salobj_dataloader = DataLoader(salobj_dataset, batch_size=batch_size_train, shuffle=True, num_workers=0)

    return salobj_dataloader, train_num


def start_train(model_name, model_dir, epoch_num, batch_size_train, salobj_dataloader, train_num,
                initModel=None, save_frq=10, train_device='cpu'):
    """

    :param model_name:
    :param model_dir:
    :param epoch_num:
    :param batch_size_train:
    :param salobj_dataloader:
    :param train_num:
    :param initModel:
    :param save_frq:
    :return:
    """
    #------ load model -----
    if (model_name == 'u2net'):
        net = U2NET(3, 1)
    elif (model_name == 'u2netp'):
        net = U2NETP(3, 1)
    else:
        print("Not defined model")
        return

    device = "cuda" if torch.cuda.is_available() and train_device == 'cuda' else "cpu"
    #-------- load pretrained model
    if initModel is not None:
        print ("Init model is existing... ")
        if device == 'cuda':
            ckpt = torch.load(initModel)
        else:
            ckpt = torch.load(initModel, map_location='cpu')
        net.load_state_dict(ckpt)

    if device == 'cuda':
        net.cuda()

    print("---define optimizer...")
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    bce_loss = nn.BCELoss(size_average=True)

    print("---start training...")
    ite_num = 0
    running_loss = 0.0
    running_tar_loss = 0.0
    ite_num4val = 0

    for epoch in range(0, epoch_num):
        net.train()

        for i, data in enumerate(salobj_dataloader):
            ite_num = ite_num + 1
            ite_num4val = ite_num4val + 1

            inputs, labels = data['image'], data['label']

            inputs = inputs.type(torch.FloatTensor)
            labels = labels.type(torch.FloatTensor)
            print (inputs.shape)
            # wrap them in Variable
            if device == 'cuda':
                inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(),
                                                                                            requires_grad=False)
            else:
                inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

            # y zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
            loss2, loss = muti_bce_loss_fusion(bce_loss, d0, d1, d2, d3, d4, d5, d6, labels_v)

            loss.backward()
            optimizer.step()

            # # print statistics
            running_loss += loss.data.item()
            running_tar_loss += loss2.data.item()

            # del temporary outputs and loss
            del d0, d1, d2, d3, d4, d5, d6, loss2, loss

            print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f " % (
                epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val,
                running_tar_loss / ite_num4val))

            if ite_num % save_frq == 0:
                torch.save(net.state_dict(), model_dir + model_name + "_bce_itr_%d_train_%3f_tar_%3f.pth" % (
                ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))
                running_loss = 0.0
                running_tar_loss = 0.0
                net.train()  # resume train
                ite_num4val = 0

        # delete old epoches
        filelist = [fn for fn in os.listdir(model_dir) if fn.endswith('.pth')]
        f_listmp = [[(os.path.getmtime(os.path.join(model_dir, fn))), fn] for fn in filelist]
        f_listmp = sorted(f_listmp)
        if len(f_listmp) < 3:
            continue
        for idx in range(len(f_listmp)-3):
            os.remove(os.path.join(model_dir, f_listmp[idx][1]))


def U2Train(model_name = 'u2net', epoch_num = 100000, batch_size_train = 2, initModel = None, train_device = 'cpu'):

    data_dir = os.path.join(os.getcwd(), 'train_data' + os.sep)
    tra_image_dir = os.path.join('DUTS', 'DUTS-TR', 'DUTS-TR', 'im_aug' + os.sep)
    tra_label_dir = os.path.join('DUTS', 'DUTS-TR', 'DUTS-TR', 'gt_aug' + os.sep)
    model_dir = os.path.join(os.getcwd(), 'saved_models', model_name + os.sep)

    salobj_dataloader, train_num = prepare_train(data_dir=data_dir, tra_image_dir=tra_image_dir,
                tra_label_dir=tra_label_dir, batch_size_train=batch_size_train)
    start_train(model_name=model_name, model_dir=model_dir, epoch_num=epoch_num, batch_size_train=batch_size_train,
                salobj_dataloader=salobj_dataloader, train_num=train_num, initModel=initModel,
                save_frq=10, train_device=train_device)
    # save final model
    filelist = [fn for fn in os.listdir(model_dir) if fn.endswith('.pth')]
    f_listmp = [[(os.path.getmtime(os.path.join(model_dir, fn))), fn] for fn in filelist]
    f_listmp = sorted(f_listmp)
    if len(f_listmp) > 0:
        shutil.copyfile(model_dir + '/' + f_listmp[len(f_listmp)-1][1], 'final.pth')


class thrdProc3Data(QThread):
    thread_alive = False
    stop_flag = False

    def __init__(self):
        super().__init__()
        self.mainwndobj = None
        self.mutex = QMutex()
        self.initModel = None
        self.batchSize = 2
        self.epochs = 100000

    def setMainWndObj(self, mainwndobj, initModel, batSize, epochs):
        self.mainwndobj = mainwndobj
        self.initModel = initModel
        self.batchSize = batSize
        self.epochs = epochs

    def stt(self):
        U2Train('u2net', self.epochs, self.batchSize, self.initModel)

    #################################
    # main thread function
    def run(self):

        self.mutex.lock()
        self.mainwndobj.listStatus.addItem("Train Started")
        self.mutex.unlock()
        self.stt()
        self.mutex.lock()
        self.mainwndobj.listStatus.addItem("Train Finished")
        self.mainwndobj.listStatus.addItem("~~~~~~~~~")
        self.mainwndobj.listStatus.addItem("Export U2net model as 'final.pth' file.")
        self.mainwndobj.listStatus.addItem("~~~~~~~~~")
        self.mainwndobj.listStatus.addItem("--> Next step : [Test Model] or [Export For C++]")
        self.mainwndobj.train_run = False
        self.mutex.unlock()


if __name__ == "__main__":

    # -------  --------

    model_name = 'u2net' #'u2netp'

    data_dir = os.path.join(os.getcwd(), 'train_data' + os.sep)
    tra_image_dir = os.path.join('DUTS', 'DUTS-TR', 'DUTS-TR', 'im_aug' + os.sep)
    tra_label_dir = os.path.join('DUTS', 'DUTS-TR', 'DUTS-TR', 'gt_aug' + os.sep)
    model_dir = os.path.join(os.getcwd(), 'saved_models', model_name + os.sep)

    epoch_num = 100000
    batch_size_train = 1
    batch_size_val = 1
    train_num = 0
    val_num = 0

    salobj_dataloader, train_num = prepare_train(data_dir=data_dir, tra_image_dir=tra_image_dir, tra_label_dir=tra_label_dir)
    start_train(model_name=model_name, model_dir=model_dir, epoch_num=epoch_num, batch_size_train=batch_size_train,
                salobj_dataloader=salobj_dataloader, train_num=train_num, initModel='u2net.pth', save_frq=10)



