# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'window1.ui'
##
## Created by: Qt User Interface Compiler version 6.8.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from mplwidget import MplWidget
from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
                            QMetaObject, QObject, QPoint, QRect, QSize, Qt,
                            QTime, QUrl)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor, QFont,
                           QFontDatabase, QGradient, QIcon, QImage,
                           QKeySequence, QLinearGradient, QPainter, QPalette,
                           QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import QApplication, QMainWindow, QSizePolicy, QWidget


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
        Window1.setCentralWidget(self.centralwidget)

        self.retranslateUi(Window1)

        QMetaObject.connectSlotsByName(Window1)
    # setupUi

    def retranslateUi(self, Window1):
        Window1.setWindowTitle(QCoreApplication.translate("Window1", u"MainWindow", None))
    # retranslateUi

