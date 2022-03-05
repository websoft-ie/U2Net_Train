#
# Resize all images in current folder, and export it in certain format.
# And save resized images in certain folder.
#
import cv2
import os
import pathlib
import progressbar

newsize = (320, 320)
extension = '.png'
sourcedir = './_images'
resultdir = 'D:/Work/Work_2021/2021_09/Segmentation(us)/RemoveBG/U2Net_Train/train_data/DUTS/DUTS-TR/DUTS-TR/_processed_1'
idx = 55

pathlib.Path(resultdir).mkdir(parents=True, exist_ok=True)

filelists = os.listdir(sourcedir)
fcounts = len(filelists)
if fcounts == 0:
    raise ("No Files")
prgstep = fcounts / 100.0
bar = progressbar.ProgressBar(maxval=100, \
    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
bar.start()
curstep = 0
for fls in filelists:
    curstep = curstep + 1
    bar.update(int(curstep/prgstep))
    if fls.endswith('.png') or fls.endswith('.PNG') or fls.endswith('.jpg') or fls.endswith('.jpeg'):
        imgframe = cv2.imread(os.path.join(sourcedir, fls))
        # cv2.imshow("input", imgframe)
        # cv2.waitKey(0)
        imgresized = cv2.resize(imgframe, newsize, cv2.INTER_LINEAR)
        namestr = '%.4d'%(idx) + extension
        cv2.imwrite(os.path.join(resultdir, namestr), imgresized)
        idx = idx + 1
bar.update(100)
bar.finish()
