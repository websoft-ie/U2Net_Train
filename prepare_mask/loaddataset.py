
import os
import cv2
import numpy as np
from prepare_mask.labelme2voc import LMe2Voc
import shutil

dir_path = os.path.abspath(os.path.dirname(__file__))

def LoadDataset(dataDir, imgList, jsonList, clsList, imgSize):

    jpgdir_tmp = 'prepare_mask/maskout/JPEGImages'
    mskdir_tmp = 'prepare_mask/maskout/SegmentationObjectPNG'
    u2jpg_tmp = 'train_data/DUTS/DUTS-TR/DUTS-TR/im_aug'
    u2msk_tmp = 'train_data/DUTS/DUTS-TR/DUTS-TR/gt_aug'

    # convert datasets to masks
    try:
        shutil.rmtree('prepare_mask/maskout')
    except Exception as e:
        pass
    with open(dir_path + '/labels.txt', 'w') as fwr:
        fwr.write('__ignore__\n_background_')
        for clsnm in clsList:
            if len(clsnm) > 0:
                fwr.write('\n' + clsnm)
        fwr.close()
    LMe2Voc(dataDir, 'prepare_mask/maskout', dir_path + '/labels.txt')

    # copy images to u2net
    jpglst_tmp = [fn for fn in os.listdir(jpgdir_tmp) if fn.lower().endswith('.jpg') or fn.lower().endswith('.png')]
    if len(jpglst_tmp) == 0:
        return
    try:
        shutil.rmtree(u2jpg_tmp)
        shutil.rmtree(u2msk_tmp)
    except Exception as e:
        pass
    os.mkdir(u2jpg_tmp)
    os.mkdir(u2msk_tmp)
    for fn in jpglst_tmp:
        imrd = cv2.imread(jpgdir_tmp + '/' + fn)
        if imrd is None:
            continue
        imrd = cv2.resize(imrd, imgSize)
        cv2.imwrite(u2jpg_tmp + '/' + fn, imrd)

    # convert masks to bw images
    msklst_tmp = [fn for fn in os.listdir(mskdir_tmp) if fn.lower().endswith('.png')]
    for fn in msklst_tmp:
        imrd = cv2.imread(mskdir_tmp + '/' + fn, cv2.IMREAD_GRAYSCALE)
        imrd = cv2.resize(np.minimum(255, (imrd.astype(np.float) * 255)).astype(np.uint8), imgSize)
        cv2.imwrite(u2msk_tmp + '/' + fn, imrd)

    try:
        shutil.rmtree('prepare_mask/maskout')
    except Exception as e:
        pass

