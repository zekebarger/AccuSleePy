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
from PySide6.QtWidgets import (QApplication, QCheckBox, QGridLayout, QHBoxLayout,
    QLabel, QMainWindow, QPushButton, QSizePolicy,
    QWidget)

from mplwidget import MplWidget
import resources_rc

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
        self.autoscroll.setGeometry(QRect(880, 490, 85, 20))
        self.gridLayoutWidget = QWidget(self.centralwidget)
        self.gridLayoutWidget.setObjectName(u"gridLayoutWidget")
        self.gridLayoutWidget.setGeometry(QRect(870, 300, 111, 81))
        self.eegbuttons = QGridLayout(self.gridLayoutWidget)
        self.eegbuttons.setObjectName(u"eegbuttons")
        self.eegbuttons.setContentsMargins(0, 0, 0, 0)
        self.eegzoomin = QPushButton(self.gridLayoutWidget)
        self.eegzoomin.setObjectName(u"eegzoomin")
        icon = QIcon()
        icon.addFile(u":/icons/zoom_in.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.eegzoomin.setIcon(icon)
        self.eegzoomin.setAutoRepeat(True)

        self.eegbuttons.addWidget(self.eegzoomin, 0, 0, 1, 1)

        self.eegshiftup = QPushButton(self.gridLayoutWidget)
        self.eegshiftup.setObjectName(u"eegshiftup")
        icon1 = QIcon()
        icon1.addFile(u":/icons/double_up_arrow.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.eegshiftup.setIcon(icon1)
        self.eegshiftup.setAutoRepeat(True)

        self.eegbuttons.addWidget(self.eegshiftup, 0, 1, 1, 1)

        self.eegzoomout = QPushButton(self.gridLayoutWidget)
        self.eegzoomout.setObjectName(u"eegzoomout")
        icon2 = QIcon()
        icon2.addFile(u":/icons/zoom_out.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.eegzoomout.setIcon(icon2)
        self.eegzoomout.setAutoRepeat(True)

        self.eegbuttons.addWidget(self.eegzoomout, 1, 0, 1, 1)

        self.eegshiftdown = QPushButton(self.gridLayoutWidget)
        self.eegshiftdown.setObjectName(u"eegshiftdown")
        icon3 = QIcon()
        icon3.addFile(u":/icons/double_down_arrow.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.eegshiftdown.setIcon(icon3)
        self.eegshiftdown.setAutoRepeat(True)

        self.eegbuttons.addWidget(self.eegshiftdown, 1, 1, 1, 1)

        self.gridLayoutWidget_2 = QWidget(self.centralwidget)
        self.gridLayoutWidget_2.setObjectName(u"gridLayoutWidget_2")
        self.gridLayoutWidget_2.setGeometry(QRect(870, 390, 111, 81))
        self.emgbuttons = QGridLayout(self.gridLayoutWidget_2)
        self.emgbuttons.setObjectName(u"emgbuttons")
        self.emgbuttons.setContentsMargins(0, 0, 0, 0)
        self.emgzoomin = QPushButton(self.gridLayoutWidget_2)
        self.emgzoomin.setObjectName(u"emgzoomin")
        self.emgzoomin.setIcon(icon)
        self.emgzoomin.setAutoRepeat(True)

        self.emgbuttons.addWidget(self.emgzoomin, 0, 0, 1, 1)

        self.emgshiftup = QPushButton(self.gridLayoutWidget_2)
        self.emgshiftup.setObjectName(u"emgshiftup")
        self.emgshiftup.setIcon(icon1)
        self.emgshiftup.setAutoRepeat(True)

        self.emgbuttons.addWidget(self.emgshiftup, 0, 1, 1, 1)

        self.emgzoomout = QPushButton(self.gridLayoutWidget_2)
        self.emgzoomout.setObjectName(u"emgzoomout")
        self.emgzoomout.setIcon(icon2)
        self.emgzoomout.setAutoRepeat(True)

        self.emgbuttons.addWidget(self.emgzoomout, 1, 0, 1, 1)

        self.emgshiftdown = QPushButton(self.gridLayoutWidget_2)
        self.emgshiftdown.setObjectName(u"emgshiftdown")
        self.emgshiftdown.setIcon(icon3)
        self.emgshiftdown.setAutoRepeat(True)

        self.emgbuttons.addWidget(self.emgshiftdown, 1, 1, 1, 1)

        self.epochsword = QLabel(self.centralwidget)
        self.epochsword.setObjectName(u"epochsword")
        self.epochsword.setGeometry(QRect(870, 520, 58, 16))
        self.horizontalLayoutWidget = QWidget(self.centralwidget)
        self.horizontalLayoutWidget.setObjectName(u"horizontalLayoutWidget")
        self.horizontalLayoutWidget.setGeometry(QRect(870, 540, 111, 41))
        self.horizontalLayout = QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.shownepochsminus = QPushButton(self.horizontalLayoutWidget)
        self.shownepochsminus.setObjectName(u"shownepochsminus")
        icon4 = QIcon()
        icon4.addFile(u":/icons/down_arrow.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.shownepochsminus.setIcon(icon4)
        self.shownepochsminus.setAutoRepeat(False)

        self.horizontalLayout.addWidget(self.shownepochsminus)

        self.shownepochslabel = QLabel(self.horizontalLayoutWidget)
        self.shownepochslabel.setObjectName(u"shownepochslabel")
        self.shownepochslabel.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.horizontalLayout.addWidget(self.shownepochslabel)

        self.shownepochsplus = QPushButton(self.horizontalLayoutWidget)
        self.shownepochsplus.setObjectName(u"shownepochsplus")
        icon5 = QIcon()
        icon5.addFile(u":/icons/up_arrow.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.shownepochsplus.setIcon(icon5)
        self.shownepochsplus.setAutoRepeat(False)

        self.horizontalLayout.addWidget(self.shownepochsplus)

        self.horizontalLayoutWidget_2 = QWidget(self.centralwidget)
        self.horizontalLayoutWidget_2.setObjectName(u"horizontalLayoutWidget_2")
        self.horizontalLayoutWidget_2.setGeometry(QRect(870, 110, 111, 71))
        self.horizontalLayout_2 = QHBoxLayout(self.horizontalLayoutWidget_2)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.specbrighter = QPushButton(self.horizontalLayoutWidget_2)
        self.specbrighter.setObjectName(u"specbrighter")
        icon6 = QIcon()
        icon6.addFile(u":/icons/brightness_up.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.specbrighter.setIcon(icon6)
        self.specbrighter.setAutoRepeat(True)

        self.horizontalLayout_2.addWidget(self.specbrighter)

        self.specdimmer = QPushButton(self.horizontalLayoutWidget_2)
        self.specdimmer.setObjectName(u"specdimmer")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.specdimmer.sizePolicy().hasHeightForWidth())
        self.specdimmer.setSizePolicy(sizePolicy1)
        icon7 = QIcon()
        icon7.addFile(u":/icons/brightness_down.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.specdimmer.setIcon(icon7)
        self.specdimmer.setIconSize(QSize(16, 16))
        self.specdimmer.setAutoRepeat(True)

        self.horizontalLayout_2.addWidget(self.specdimmer)

        self.horizontalLayoutWidget_3 = QWidget(self.centralwidget)
        self.horizontalLayoutWidget_3.setObjectName(u"horizontalLayoutWidget_3")
        self.horizontalLayoutWidget_3.setGeometry(QRect(870, 200, 121, 51))
        self.horizontalLayout_3 = QHBoxLayout(self.horizontalLayoutWidget_3)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.xzoomin = QPushButton(self.horizontalLayoutWidget_3)
        self.xzoomin.setObjectName(u"xzoomin")
        self.xzoomin.setIcon(icon)
        self.xzoomin.setAutoRepeat(True)

        self.horizontalLayout_3.addWidget(self.xzoomin)

        self.xzoomout = QPushButton(self.horizontalLayoutWidget_3)
        self.xzoomout.setObjectName(u"xzoomout")
        self.xzoomout.setIcon(icon2)
        self.xzoomout.setAutoRepeat(True)

        self.horizontalLayout_3.addWidget(self.xzoomout)

        self.xzoomreset = QPushButton(self.horizontalLayoutWidget_3)
        self.xzoomreset.setObjectName(u"xzoomreset")
        icon8 = QIcon()
        icon8.addFile(u":/icons/home.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.xzoomreset.setIcon(icon8)
        self.xzoomreset.setAutoRepeat(True)

        self.horizontalLayout_3.addWidget(self.xzoomreset)

        self.savebutton = QPushButton(self.centralwidget)
        self.savebutton.setObjectName(u"savebutton")
        self.savebutton.setGeometry(QRect(900, 20, 41, 51))
        icon9 = QIcon()
        icon9.addFile(u":/icons/save.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.savebutton.setIcon(icon9)
        self.savebutton.setIconSize(QSize(20, 20))
        self.savebutton.setAutoRepeat(True)
        Window1.setCentralWidget(self.centralwidget)

        self.retranslateUi(Window1)

        QMetaObject.connectSlotsByName(Window1)
    # setupUi

    def retranslateUi(self, Window1):
        Window1.setWindowTitle(QCoreApplication.translate("Window1", u"MainWindow", None))
#if QT_CONFIG(tooltip)
        self.autoscroll.setToolTip(QCoreApplication.translate("Window1", u"Step forward when setting brain state", None))
#endif // QT_CONFIG(tooltip)
        self.autoscroll.setText(QCoreApplication.translate("Window1", u"Auto scroll", None))
#if QT_CONFIG(tooltip)
        self.eegzoomin.setToolTip(QCoreApplication.translate("Window1", u"Scale signal up", None))
#endif // QT_CONFIG(tooltip)
        self.eegzoomin.setText("")
#if QT_CONFIG(tooltip)
        self.eegshiftup.setToolTip(QCoreApplication.translate("Window1", u"Shift signal up", None))
#endif // QT_CONFIG(tooltip)
        self.eegshiftup.setText("")
#if QT_CONFIG(tooltip)
        self.eegzoomout.setToolTip(QCoreApplication.translate("Window1", u"Scale signal down", None))
#endif // QT_CONFIG(tooltip)
        self.eegzoomout.setText("")
#if QT_CONFIG(tooltip)
        self.eegshiftdown.setToolTip(QCoreApplication.translate("Window1", u"Shift signal down", None))
#endif // QT_CONFIG(tooltip)
        self.eegshiftdown.setText("")
        self.emgzoomin.setText("")
        self.emgshiftup.setText("")
        self.emgzoomout.setText("")
        self.emgshiftdown.setText("")
        self.epochsword.setText(QCoreApplication.translate("Window1", u"Epochs:", None))
        self.shownepochsminus.setText("")
        self.shownepochslabel.setText(QCoreApplication.translate("Window1", u"5", None))
        self.shownepochsplus.setText("")
#if QT_CONFIG(tooltip)
        self.specbrighter.setToolTip(QCoreApplication.translate("Window1", u"Increase brightness", None))
#endif // QT_CONFIG(tooltip)
        self.specbrighter.setText("")
#if QT_CONFIG(tooltip)
        self.specdimmer.setToolTip(QCoreApplication.translate("Window1", u"Decrease brightness", None))
#endif // QT_CONFIG(tooltip)
        self.specdimmer.setText("")
#if QT_CONFIG(tooltip)
        self.xzoomin.setToolTip(QCoreApplication.translate("Window1", u"Zoom in (+)", None))
#endif // QT_CONFIG(tooltip)
        self.xzoomin.setText("")
#if QT_CONFIG(tooltip)
        self.xzoomout.setToolTip(QCoreApplication.translate("Window1", u"Zoom out (-)", None))
#endif // QT_CONFIG(tooltip)
        self.xzoomout.setText("")
#if QT_CONFIG(tooltip)
        self.xzoomreset.setToolTip(QCoreApplication.translate("Window1", u"Reset zoom", None))
#endif // QT_CONFIG(tooltip)
        self.xzoomreset.setText("")
#if QT_CONFIG(tooltip)
        self.savebutton.setToolTip(QCoreApplication.translate("Window1", u"Save labels (Ctrl+S)", None))
#endif // QT_CONFIG(tooltip)
        self.savebutton.setText("")
    # retranslateUi

