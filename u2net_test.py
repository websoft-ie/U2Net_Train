import os
import cv2
from skimage import io
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

import numpy as np
from PIL import Image
import glob

from data_loader import RescaleT
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET # full size version 173.6 MB
from model import U2NETP # small version u2net 4.7 MB

from pathlib import Path

dir_result = Path('./test_result')

# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def getPngImage(imgframe, imgAlpha):
    if imgframe is None or imgAlpha is None:
        return None
    if len(imgframe.shape) < 4:
        imgframe = cv2.cvtColor(imgframe, cv2.COLOR_BGR2BGRA)
    orih, oriw = imgframe.shape[:2]
    mskh, mskw = imgAlpha.shape[:2]
    if orih!=mskh or oriw!=mskw:
        imgAlpha = cv2.resize(imgAlpha, (oriw, orih))
    if len(imgAlpha.shape) > 2:
        imgAlpha = imgAlpha[:, :, -1]
    imgframe[:, :, 3] = imgAlpha
    return imgframe

def save_output(image_name, pred):

    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np*255).convert('RGB')
    img_name = image_name.split(os.sep)[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)

    pb_np = np.array(imo)

    fname = img_name.split(".")
    bbb = fname[0:-1]
    imidx = bbb[0]
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]

    imo.save('tmp.png')

    # imshow
    im_ori = cv2.imread(image_name)
    im_msk = cv2.imread('tmp.png')
    im_mrg = im_ori.copy()
    for ic in range(3):
        ori_f = im_ori[:, :, ic].astype(dtype=np.float32)
        msk_f = im_msk[:, :, ic].astype(dtype=np.float32)
        new_f = (ori_f * msk_f / 255.0).astype(dtype=np.uint8)
        im_mrg[:, :, ic] = new_f
    im_all = np.concatenate((im_ori, im_msk, im_mrg), axis=1)
    Path(dir_result).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dir_result / fname[0]) + '.jpg', im_all)
    return getPngImage(im_ori, im_msk), im_all


def U2Test(model_name = 'u2net',img_name_list = [], image_dir = 'train_data/DUTS/DUTS-TR/DUTS-TR/im_aug', modelpath = 'final.pth', test_device = 'cpu'):
    if len(img_name_list) == 0:
        img_name_list = glob.glob(image_dir + os.sep + '*')
    print(img_name_list)
    # --------- 2. dataloader ---------
    #1. dataloader
    test_salobj_dataset = SalObjDataset(img_name_list = img_name_list, lbl_name_list = [],
            transform=transforms.Compose([RescaleT(320), ToTensorLab(flag=0)]))
    test_salobj_dataloader = DataLoader(test_salobj_dataset, batch_size=1, shuffle=False, num_workers=0)
    # --------- 3. model define ---------
    if(model_name=='u2net'):
        print("...load U2NET---173.6 MB")
        net = U2NET(3,1)
    elif(model_name=='u2netp'):
        print("...load U2NEP---4.7 MB")
        net = U2NETP(3,1)

    device = "cuda" if torch.cuda.is_available() and test_device == 'cuda' else "cpu"
    if device == 'cuda':
        net.load_state_dict(torch.load(modelpath))
        net.cuda()
    else:
        net.load_state_dict(torch.load(modelpath, map_location='cpu'))

    net.eval()
    # --------- 4. inference for each image ---------
    img_result = []
    for i_test, data_test in enumerate(test_salobj_dataloader):
        print("inferencing:",img_name_list[i_test].split(os.sep)[-1])
        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)
        inputs_test = Variable(inputs_test)

        d1,d2,d3,d4,d5,d6,d7= net(inputs_test)

        # normalization
        pred = d1[:,0,:,:]
        pred = normPRED(pred)

        # result
        img_merged, im_all = save_output(img_name_list[i_test], pred)
        if im_all is not None:
            img_result.append(img_merged)
            # cv2.imshow("Result", im_all)
            # ky = cv2.waitKey()
            # if ky == 0 or ky == 27:
            #     break

        del d1,d2,d3,d4,d5,d6,d7

    return img_result

def U2Predict(img_in, img_out, model_name = 'u2net', modelpath = 'final.pth', test_device = 'cpu'):
    img_result = U2Test('u2net', [img_in])
    if len(img_result) > 0:
        cv2.imwrite(img_out, img_result[0])



if __name__ == "__main__":
    img_dir = 'D:/Profile/Myself/New folder'
    U2Test(model_name = 'u2net', image_dir = img_dir)
    # U2Test(model_name = 'u2net')
