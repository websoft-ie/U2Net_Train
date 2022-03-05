

import os
import shutil

from PyQt5 import uic, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QListWidget, QMessageBox

from prepare_mask.loaddataset import LoadDataset

from u2net_train import thrdProc3Data
from u2net_test import U2Test

class Ui(QtWidgets.QMainWindow):

    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi('mainwnd.ui', self)
        self.initUI()
        self.show()

    def initUI(self):
        self.datadir = None
        self.imglist = []
        self.jsonlist = []
        self.classestxt = None
        self.classeslist = []
        self.initModel = ""
        self.finalEpochs = 100000
        self.batchsize = 2
        self.inptSze = 320

        self.checked = False
        self.train_run = False
        self.thrdProc = thrdProc3Data()

        self.edtInitialWeights.setText(self.initModel)
        self.edtMaxepochs.setText(str(self.finalEpochs))
        self.edtBatchsize.setText(str(self.batchsize))
        self.edtInputsize.setText(str(self.inptSze))

        self.btnLoadDataset.clicked.connect(self.onBtnLoadDataset)
        self.btnLoadClasses.clicked.connect(self.onBtnLoadClassestxt)
        self.btnCheckDB.clicked.connect(self.onBtnCheckDB)
        self.btnStartTrain.clicked.connect(self.onBtnStartTrain)
        self.btnOpenmodel.clicked.connect(self.onOpenModel)
        self.btnTestModel.clicked.connect(self.onTestModel)
        self.btnExportC.clicked.connect(self.onExportC)

        self.listStatus.addItem("--> First Step : [Load Dataset] to load images and jsons")


    def onBtnLoadDataset(self):
        dirnametmp = QFileDialog.getExistingDirectory(self, 'Load Image Directory')
        if len(dirnametmp) == 0:
            return
        self.datadir = dirnametmp
        self.imglist = []
        self.jsonlist = []
        self.classeslist = []
        self.listStatus.addItem("Loading Images and Jsons from '" + self.datadir + "'")
        filelist = [fn for fn in os.listdir(self.datadir) if os.path.isfile(os.path.join(self.datadir, fn))]
        self.imglist = [fn for fn in filelist if fn[-4:].lower() == ".jpg" or fn[-4:].lower() == ".png" or fn[-4:].lower() == ".tif"]
        self.jsonlist = [fn for fn in filelist if fn[-5:].lower() == ".json"]
        self.listStatus.addItem("Totally " + str(len(self.imglist)) + " imags, " +\
                                str(len(self.jsonlist)) + " jsons loaded!")
        self.listStatus.addItem("--> Next Step : [Load Classes] to load classnames")


    def onBtnLoadClassestxt(self):
        filenametmp = QFileDialog.getOpenFileName(self, 'Load Classes txt File')
        if len(filenametmp) == 0:
            return
        self.checked = False
        self.classestxt = filenametmp[0]
        self.classeslist = []
        self.listStatus.addItem("Loading Classes from '" + self.classestxt + "'")
        with open(self.classestxt) as frd:
            data_list = [fl for fl in frd.read().split('\n') if len(fl) > 0]
            frd.close()
        self.classeslist = data_list
        self.listStatus.addItem("classes " + str(self.classeslist))

        self.listStatus.addItem("--> Next Step : Please set initial model(optional)")
        self.listStatus.addItem("--> Next Step : Please set final epochs")
        self.listStatus.addItem("--> Next Step : Please set batch size")
        self.listStatus.addItem("--> Next Step : Please set input image size")
        self.listStatus.addItem("--> Please confirm above steps again.")
        self.listStatus.addItem("--> Next Step : [Check Dataset] to prepare for Model Training")


    def onBtnCheckDB(self):
        if self.checked:
            return
        if len(self.classeslist) == 0 or len(self.jsonlist) == 0 or len(self.imglist) == 0:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Dataset doesn't loaded correctly.")
            msg.setInformativeText("Dataset Error")
            msg.setWindowTitle("Error")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
            return
        if self.train_run:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Can't check dataset while Training. Please stop training or restart program.")
            msg.setInformativeText("Checking Error")
            msg.setWindowTitle("Error")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
            return
        self.listStatus.addItem("Start Checking Dataset")
        self.inptSze = int(self.edtInputsize.text())
        LoadDataset(self.datadir, self.imglist, self.jsonlist, self.classeslist, (self.inptSze, self.inptSze))
        self.listStatus.addItem("Finished Checking Dataset")
        self.checked = True
        self.train_run = False

        self.listStatus.addItem("--> Next Step : [Start Train] to start training. Wait till finished.")


    def onOpenModel(self):
        filenametmp = QFileDialog.getOpenFileName(self, 'Load Classes txt File')[0]
        print ("Init model : " + filenametmp)
        # if len(filenametmp) < 4:
            # self.initModel = ""
        # elif filenametmp[-3:].lower() != ".pt":
            # self.initModel = ""
        # else:
        self.initModel = filenametmp
        print ("Init model : " + self.initModel)
        self.edtInitialWeights.setText(self.initModel)
        self.listStatus.addItem("Model for starting point is loaded ...")
        if len(filenametmp) < 4:
            self.listStatus.addItem("...Start from Random weights.")
        else:
            self.listStatus.addItem("..." + self.initModel)


    def onBtnStartTrain(self):
        self.train_run = True
        try:
            shutil.rmtree("runs")
        except Exception as e:
            pass
        if not self.checked:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Please check Dataset first. Do you want to use the cached data?")
            msg.setInformativeText("Dataset not checked")
            msg.setWindowTitle("Error")
            msg.setStandardButtons(QMessageBox.Ok|QMessageBox.Cancel)
            rtnsel = msg.exec_()
            if rtnsel == QMessageBox.Ok:
                self.listStatus.addItem("Model Train Started with Cached Dataset")
            else:
                return
        self.initModel = self.edtInitialWeights.text()
        if len(self.initModel) == 0:
            self.initModel = None
        self.finalEpochs = int(self.edtMaxepochs.text())
        self.batchsize = int(self.edtBatchsize.text())
        self.inptSze = int(self.edtInputsize.text())
        self.listStatus.addItem("Model Train Started. Check console output to check status")
        self.listStatus.addItem("waiting ... ... ...")
        self.thrdProc.setMainWndObj(self, self.initModel, self.batchsize, self.finalEpochs)
        self.thrdProc.start()


    def onTestModel(self):
        if not os.path.exists("final.pth"):
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Could not find trained model 'final.pth'.")
            msg.setInformativeText("Model Error")
            msg.setWindowTitle("Error")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
            return
        self.initModel = self.edtInitialWeights.text()
        self.finalEpochs = int(self.edtMaxepochs.text())
        self.batchsize = int(self.edtBatchsize.text())
        self.inptSze = int(self.edtInputsize.text())
        U2Test(test_device='cpu')
        self.listStatus.addItem("Checking model finished")


    def onExportC(self):
        self.initModel = self.edtInitialWeights.text()
        self.finalEpochs = int(self.edtMaxepochs.text())
        self.batchsize = int(self.edtBatchsize.text())
        self.inptSze = int(self.edtInputsize.text())
        if not os.path.exists("final.pth"):
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Could not find trained model 'final.pth'.")
            msg.setInformativeText("Model Error")
            msg.setWindowTitle("Error")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
            return
        cvtToScript(self.inptSze)
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText("Successfully export model. Please use 'final.torchscript.pt' file as model.")
        msg.setInformativeText("Torchscript model exported successfully.")
        msg.setWindowTitle("Ok")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()
        self.listStatus.addItem("Model Exported as <final.torchscript.pt>")


    #########################################
    def closeEvent(self, event):
        pass


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    windows = Ui()
    app.exec_()









