# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'window0.ui'
##
## Created by: Qt User Interface Compiler version 6.8.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QGroupBox, QHBoxLayout, QLabel,
    QMainWindow, QPushButton, QSizePolicy, QVBoxLayout,
    QWidget)
import resources_rc

class Ui_Window0(object):
    def setupUi(self, Window0):
        if not Window0.objectName():
            Window0.setObjectName(u"Window0")
        Window0.resize(1000, 600)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Window0.sizePolicy().hasHeightForWidth())
        Window0.setSizePolicy(sizePolicy)
        Window0.setStyleSheet(u"background-color: white;")
        self.centralwidget = QWidget(Window0)
        self.centralwidget.setObjectName(u"centralwidget")
        sizePolicy.setHeightForWidth(self.centralwidget.sizePolicy().hasHeightForWidth())
        self.centralwidget.setSizePolicy(sizePolicy)
        self.verticalLayout_2 = QVBoxLayout(self.centralwidget)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.toplayout = QVBoxLayout()
        self.toplayout.setObjectName(u"toplayout")
        self.topmostlayout = QHBoxLayout()
        self.topmostlayout.setSpacing(20)
        self.topmostlayout.setObjectName(u"topmostlayout")
        self.usermanualbutton = QPushButton(self.centralwidget)
        self.usermanualbutton.setObjectName(u"usermanualbutton")
        icon = QIcon()
        icon.addFile(u":/icons/question.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.usermanualbutton.setIcon(icon)
        self.usermanualbutton.setIconSize(QSize(24, 24))

        self.topmostlayout.addWidget(self.usermanualbutton)

        self.parametergroupbox = QGroupBox(self.centralwidget)
        self.parametergroupbox.setObjectName(u"parametergroupbox")

        self.topmostlayout.addWidget(self.parametergroupbox)

        self.topmostlayout.setStretch(0, 1)
        self.topmostlayout.setStretch(1, 6)

        self.toplayout.addLayout(self.topmostlayout)

        self.secondlayout = QHBoxLayout()
        self.secondlayout.setSpacing(20)
        self.secondlayout.setObjectName(u"secondlayout")
        self.recordinglistgroupbox = QGroupBox(self.centralwidget)
        self.recordinglistgroupbox.setObjectName(u"recordinglistgroupbox")

        self.secondlayout.addWidget(self.recordinglistgroupbox)

        self.recordinggroupbox = QGroupBox(self.centralwidget)
        self.recordinggroupbox.setObjectName(u"recordinggroupbox")

        self.secondlayout.addWidget(self.recordinggroupbox)

        self.secondlayout.setStretch(0, 1)
        self.secondlayout.setStretch(1, 6)

        self.toplayout.addLayout(self.secondlayout)

        self.toplayout.setStretch(0, 1)
        self.toplayout.setStretch(1, 4)

        self.verticalLayout_2.addLayout(self.toplayout)

        self.bottomlayout = QHBoxLayout()
        self.bottomlayout.setSpacing(20)
        self.bottomlayout.setObjectName(u"bottomlayout")
        self.logolabel = QLabel(self.centralwidget)
        self.logolabel.setObjectName(u"logolabel")

        self.bottomlayout.addWidget(self.logolabel)

        self.bottomrightlayout = QVBoxLayout()
        self.bottomrightlayout.setObjectName(u"bottomrightlayout")
        self.classificationgroupbox = QGroupBox(self.centralwidget)
        self.classificationgroupbox.setObjectName(u"classificationgroupbox")

        self.bottomrightlayout.addWidget(self.classificationgroupbox)

        self.consolegroupbox = QGroupBox(self.centralwidget)
        self.consolegroupbox.setObjectName(u"consolegroupbox")

        self.bottomrightlayout.addWidget(self.consolegroupbox)


        self.bottomlayout.addLayout(self.bottomrightlayout)

        self.bottomlayout.setStretch(0, 1)
        self.bottomlayout.setStretch(1, 6)

        self.verticalLayout_2.addLayout(self.bottomlayout)

        self.verticalLayout_2.setStretch(0, 1)
        self.verticalLayout_2.setStretch(1, 1)
        Window0.setCentralWidget(self.centralwidget)

        self.retranslateUi(Window0)

        QMetaObject.connectSlotsByName(Window0)
    # setupUi

    def retranslateUi(self, Window0):
        Window0.setWindowTitle(QCoreApplication.translate("Window0", u"MainWindow", None))
        self.usermanualbutton.setText("")
        self.parametergroupbox.setTitle(QCoreApplication.translate("Window0", u"Parameters for all recordings from one subject", None))
        self.recordinglistgroupbox.setTitle(QCoreApplication.translate("Window0", u"Recording list", None))
        self.recordinggroupbox.setTitle(QCoreApplication.translate("Window0", u"Data / actions for the selected recording from this subject", None))
        self.logolabel.setText(QCoreApplication.translate("Window0", u"AccuSleePy logo", None))
        self.classificationgroupbox.setTitle(QCoreApplication.translate("Window0", u"Data / actions for all recordings from this subject", None))
        self.consolegroupbox.setTitle(QCoreApplication.translate("Window0", u"Messages", None))
    # retranslateUi

