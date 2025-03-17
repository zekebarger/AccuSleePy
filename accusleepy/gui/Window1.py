# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'window1.ui'
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
from PySide6.QtWidgets import (QApplication, QCheckBox, QGridLayout, QMainWindow,
    QPushButton, QSizePolicy, QWidget)

from mplwidget import MplWidget

class Ui_Window1(object):
    def setupUi(self, Window1):
        if not Window1.objectName():
            Window1.setObjectName(u"Window1")
        Window1.resize(1000, 600)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Window1.sizePolicy().hasHeightForWidth())
        Window1.setSizePolicy(sizePolicy)
        self.centralwidget = QWidget(Window1)
        self.centralwidget.setObjectName(u"centralwidget")
        self.lowerplots = MplWidget(self.centralwidget)
        self.lowerplots.setObjectName(u"lowerplots")
        self.lowerplots.setGeometry(QRect(10, 300, 850, 280))
        sizePolicy.setHeightForWidth(self.lowerplots.sizePolicy().hasHeightForWidth())
        self.lowerplots.setSizePolicy(sizePolicy)
        self.lowerplots.setAutoFillBackground(True)
        self.upperplots = MplWidget(self.centralwidget)
        self.upperplots.setObjectName(u"upperplots")
        self.upperplots.setGeometry(QRect(10, 10, 850, 280))
        sizePolicy.setHeightForWidth(self.upperplots.sizePolicy().hasHeightForWidth())
        self.upperplots.setSizePolicy(sizePolicy)
        self.upperplots.setAutoFillBackground(True)
        self.autoscroll = QCheckBox(self.centralwidget)
        self.autoscroll.setObjectName(u"autoscroll")
        self.autoscroll.setGeometry(QRect(880, 530, 85, 20))
        self.gridLayoutWidget = QWidget(self.centralwidget)
        self.gridLayoutWidget.setObjectName(u"gridLayoutWidget")
        self.gridLayoutWidget.setGeometry(QRect(880, 300, 94, 81))
        self.eegbuttons = QGridLayout(self.gridLayoutWidget)
        self.eegbuttons.setObjectName(u"eegbuttons")
        self.eegbuttons.setContentsMargins(0, 0, 0, 0)
        self.eegzoomin = QPushButton(self.gridLayoutWidget)
        self.eegzoomin.setObjectName(u"eegzoomin")
        self.eegzoomin.setAutoRepeat(True)

        self.eegbuttons.addWidget(self.eegzoomin, 0, 0, 1, 1)

        self.eegshiftup = QPushButton(self.gridLayoutWidget)
        self.eegshiftup.setObjectName(u"eegshiftup")
        self.eegshiftup.setAutoRepeat(True)

        self.eegbuttons.addWidget(self.eegshiftup, 0, 1, 1, 1)

        self.eegzoomout = QPushButton(self.gridLayoutWidget)
        self.eegzoomout.setObjectName(u"eegzoomout")
        self.eegzoomout.setAutoRepeat(True)

        self.eegbuttons.addWidget(self.eegzoomout, 1, 0, 1, 1)

        self.eegshiftdown = QPushButton(self.gridLayoutWidget)
        self.eegshiftdown.setObjectName(u"eegshiftdown")
        self.eegshiftdown.setAutoRepeat(True)

        self.eegbuttons.addWidget(self.eegshiftdown, 1, 1, 1, 1)

        self.gridLayoutWidget_2 = QWidget(self.centralwidget)
        self.gridLayoutWidget_2.setObjectName(u"gridLayoutWidget_2")
        self.gridLayoutWidget_2.setGeometry(QRect(880, 390, 94, 81))
        self.emgbuttons = QGridLayout(self.gridLayoutWidget_2)
        self.emgbuttons.setObjectName(u"emgbuttons")
        self.emgbuttons.setContentsMargins(0, 0, 0, 0)
        self.emgzoomin = QPushButton(self.gridLayoutWidget_2)
        self.emgzoomin.setObjectName(u"emgzoomin")
        self.emgzoomin.setAutoRepeat(True)

        self.emgbuttons.addWidget(self.emgzoomin, 0, 0, 1, 1)

        self.emgshiftup = QPushButton(self.gridLayoutWidget_2)
        self.emgshiftup.setObjectName(u"emgshiftup")
        self.emgshiftup.setAutoRepeat(True)

        self.emgbuttons.addWidget(self.emgshiftup, 0, 1, 1, 1)

        self.emgzoomout = QPushButton(self.gridLayoutWidget_2)
        self.emgzoomout.setObjectName(u"emgzoomout")
        self.emgzoomout.setAutoRepeat(True)

        self.emgbuttons.addWidget(self.emgzoomout, 1, 0, 1, 1)

        self.emgshiftdown = QPushButton(self.gridLayoutWidget_2)
        self.emgshiftdown.setObjectName(u"emgshiftdown")
        self.emgshiftdown.setAutoRepeat(True)

        self.emgbuttons.addWidget(self.emgshiftdown, 1, 1, 1, 1)

        Window1.setCentralWidget(self.centralwidget)

        self.retranslateUi(Window1)

        QMetaObject.connectSlotsByName(Window1)
    # setupUi

    def retranslateUi(self, Window1):
        Window1.setWindowTitle(QCoreApplication.translate("Window1", u"MainWindow", None))
        self.autoscroll.setText(QCoreApplication.translate("Window1", u"Auto scroll", None))
        self.eegzoomin.setText(QCoreApplication.translate("Window1", u"+", None))
        self.eegshiftup.setText(QCoreApplication.translate("Window1", u"^", None))
        self.eegzoomout.setText(QCoreApplication.translate("Window1", u"-", None))
        self.eegshiftdown.setText(QCoreApplication.translate("Window1", u"v", None))
        self.emgzoomin.setText(QCoreApplication.translate("Window1", u"+", None))
        self.emgshiftup.setText(QCoreApplication.translate("Window1", u"^", None))
        self.emgzoomout.setText(QCoreApplication.translate("Window1", u"-", None))
        self.emgshiftdown.setText(QCoreApplication.translate("Window1", u"v", None))
    # retranslateUi

