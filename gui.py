# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'gui.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(640, 480)
        font = QtGui.QFont()
        font.setPointSize(9)
        MainWindow.setFont(font)
        MainWindow.setAutoFillBackground(True)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.result_emo = QtWidgets.QTextBrowser(self.centralwidget)
        self.result_emo.setGeometry(QtCore.QRect(450, 45, 171, 61))
        self.result_emo.setObjectName("result_emo")
        self.recognize = QtWidgets.QPushButton(self.centralwidget)
        self.recognize.setGeometry(QtCore.QRect(170, 70, 100, 100))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.recognize.sizePolicy().hasHeightForWidth())
        self.recognize.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(17)
        self.recognize.setFont(font)
        self.recognize.setObjectName("recognize")
        self.image = QtWidgets.QLabel(self.centralwidget)
        self.image.setGeometry(QtCore.QRect(50, 210, 200, 200))
        self.image.setAcceptDrops(False)
        self.image.setAutoFillBackground(True)
        self.image.setStyleSheet("")
        self.image.setFrameShape(QtWidgets.QFrame.Box)
        self.image.setText("")
        self.image.setObjectName("image")
        self.result = QtWidgets.QLabel(self.centralwidget)
        self.result.setGeometry(QtCore.QRect(330, 40, 121, 61))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.result.setFont(font)
        self.result.setObjectName("result")
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(289, -10, 31, 481))
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.openfile = QtWidgets.QPushButton(self.centralwidget)
        self.openfile.setGeometry(QtCore.QRect(20, 70, 100, 100))
        font = QtGui.QFont()
        font.setPointSize(17)
        self.openfile.setFont(font)
        self.openfile.setObjectName("openfile")
        self.chart = QtWidgets.QLabel(self.centralwidget)
        self.chart.setGeometry(QtCore.QRect(320, 140, 300, 300))
        self.chart.setAutoFillBackground(True)
        self.chart.setStyleSheet("")
        self.chart.setFrameShape(QtWidgets.QFrame.Box)
        self.chart.setFrameShadow(QtWidgets.QFrame.Plain)
        self.chart.setText("")
        self.chart.setObjectName("chart")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "表情识别"))
        self.recognize.setText(_translate("MainWindow", "识别"))
        self.result.setText(_translate("MainWindow", "识别结果"))
        self.openfile.setText(_translate("MainWindow", "选择图片"))
