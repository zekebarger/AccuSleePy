# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'viewer_window.ui'
##
## Created by: Qt User Interface Compiler version 6.8.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

import resources_rc
from mplwidget import MplWidget
from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
                            QMetaObject, QObject, QPoint, QRect, QSize, Qt,
                            QTime, QUrl)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor, QFont,
                           QFontDatabase, QGradient, QIcon, QImage,
                           QKeySequence, QLinearGradient, QPainter, QPalette,
                           QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QCheckBox, QDialog, QFrame,
                               QGridLayout, QHBoxLayout, QLabel, QLayout,
                               QPushButton, QSizePolicy, QSpacerItem,
                               QVBoxLayout, QWidget)


class Ui_ViewerWindow(object):
    def setupUi(self, ViewerWindow):
        if not ViewerWindow.objectName():
            ViewerWindow.setObjectName(u"ViewerWindow")
        ViewerWindow.resize(1200, 700)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(ViewerWindow.sizePolicy().hasHeightForWidth())
        ViewerWindow.setSizePolicy(sizePolicy)
        ViewerWindow.setStyleSheet(u"background-color: white;")
        self.horizontalLayout = QHBoxLayout(ViewerWindow)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.all_plots = QVBoxLayout()
        self.all_plots.setSpacing(1)
        self.all_plots.setObjectName(u"all_plots")
        self.all_plots.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        self.upperfigure = MplWidget(ViewerWindow)
        self.upperfigure.setObjectName(u"upperfigure")
        sizePolicy.setHeightForWidth(self.upperfigure.sizePolicy().hasHeightForWidth())
        self.upperfigure.setSizePolicy(sizePolicy)
        self.upperfigure.setAutoFillBackground(False)

        self.all_plots.addWidget(self.upperfigure)

        self.line = QFrame(ViewerWindow)
        self.line.setObjectName(u"line")
        self.line.setLineWidth(2)
        self.line.setFrameShape(QFrame.Shape.HLine)
        self.line.setFrameShadow(QFrame.Shadow.Sunken)

        self.all_plots.addWidget(self.line)

        self.lowerfigure = MplWidget(ViewerWindow)
        self.lowerfigure.setObjectName(u"lowerfigure")
        sizePolicy.setHeightForWidth(self.lowerfigure.sizePolicy().hasHeightForWidth())
        self.lowerfigure.setSizePolicy(sizePolicy)
        self.lowerfigure.setAutoFillBackground(False)

        self.all_plots.addWidget(self.lowerfigure)

        self.all_plots.setStretch(0, 50)
        self.all_plots.setStretch(1, 1)
        self.all_plots.setStretch(2, 50)

        self.horizontalLayout.addLayout(self.all_plots)

        self.all_controls = QVBoxLayout()
        self.all_controls.setSpacing(20)
        self.all_controls.setObjectName(u"all_controls")
        self.all_controls.setSizeConstraint(QLayout.SizeConstraint.SetDefaultConstraint)
        self.top_plot_buttons = QVBoxLayout()
        self.top_plot_buttons.setObjectName(u"top_plot_buttons")
        self.topcontroltopspacer = QSpacerItem(5, 10, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)

        self.top_plot_buttons.addItem(self.topcontroltopspacer)

        self.save_help_buttons = QHBoxLayout()
        self.save_help_buttons.setSpacing(20)
        self.save_help_buttons.setObjectName(u"save_help_buttons")
        self.save_help_buttons.setSizeConstraint(QLayout.SizeConstraint.SetDefaultConstraint)
        self.savebutton = QPushButton(ViewerWindow)
        self.savebutton.setObjectName(u"savebutton")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.savebutton.sizePolicy().hasHeightForWidth())
        self.savebutton.setSizePolicy(sizePolicy1)
        self.savebutton.setStyleSheet(u"background-color: rgb(237, 241, 241);")
        icon = QIcon()
        icon.addFile(u":/icons/save.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.savebutton.setIcon(icon)
        self.savebutton.setIconSize(QSize(28, 28))
        self.savebutton.setAutoRepeat(True)

        self.save_help_buttons.addWidget(self.savebutton)

        self.helpbutton = QPushButton(ViewerWindow)
        self.helpbutton.setObjectName(u"helpbutton")
        sizePolicy1.setHeightForWidth(self.helpbutton.sizePolicy().hasHeightForWidth())
        self.helpbutton.setSizePolicy(sizePolicy1)
        self.helpbutton.setStyleSheet(u"background-color: rgb(237, 241, 241);")
        icon1 = QIcon()
        icon1.addFile(u":/icons/question.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.helpbutton.setIcon(icon1)
        self.helpbutton.setIconSize(QSize(28, 28))
        self.helpbutton.setAutoRepeat(True)

        self.save_help_buttons.addWidget(self.helpbutton)


        self.top_plot_buttons.addLayout(self.save_help_buttons)

        self.topcontrolmidspacer = QSpacerItem(5, 5, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Preferred)

        self.top_plot_buttons.addItem(self.topcontrolmidspacer)

        self.zoom_and_brightness = QVBoxLayout()
        self.zoom_and_brightness.setObjectName(u"zoom_and_brightness")
        self.brightness_buttons = QHBoxLayout()
        self.brightness_buttons.setObjectName(u"brightness_buttons")
        self.specbrighter = QPushButton(ViewerWindow)
        self.specbrighter.setObjectName(u"specbrighter")
        sizePolicy1.setHeightForWidth(self.specbrighter.sizePolicy().hasHeightForWidth())
        self.specbrighter.setSizePolicy(sizePolicy1)
        self.specbrighter.setStyleSheet(u"background-color: rgb(237, 241, 241);")
        icon2 = QIcon()
        icon2.addFile(u":/icons/brightness_up.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.specbrighter.setIcon(icon2)
        self.specbrighter.setIconSize(QSize(24, 24))
        self.specbrighter.setAutoRepeat(True)

        self.brightness_buttons.addWidget(self.specbrighter)

        self.specdimmer = QPushButton(ViewerWindow)
        self.specdimmer.setObjectName(u"specdimmer")
        sizePolicy1.setHeightForWidth(self.specdimmer.sizePolicy().hasHeightForWidth())
        self.specdimmer.setSizePolicy(sizePolicy1)
        self.specdimmer.setStyleSheet(u"background-color: rgb(237, 241, 241);")
        icon3 = QIcon()
        icon3.addFile(u":/icons/brightness_down.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.specdimmer.setIcon(icon3)
        self.specdimmer.setIconSize(QSize(24, 24))
        self.specdimmer.setAutoRepeat(True)

        self.brightness_buttons.addWidget(self.specdimmer)


        self.zoom_and_brightness.addLayout(self.brightness_buttons)

        self.zoom_buttons = QHBoxLayout()
        self.zoom_buttons.setObjectName(u"zoom_buttons")
        self.zoom_buttons.setSizeConstraint(QLayout.SizeConstraint.SetDefaultConstraint)
        self.zoom_buttons.setContentsMargins(2, -1, 2, -1)
        self.xzoomin = QPushButton(ViewerWindow)
        self.xzoomin.setObjectName(u"xzoomin")
        sizePolicy1.setHeightForWidth(self.xzoomin.sizePolicy().hasHeightForWidth())
        self.xzoomin.setSizePolicy(sizePolicy1)
        self.xzoomin.setStyleSheet(u"background-color: rgb(237, 241, 241);")
        icon4 = QIcon()
        icon4.addFile(u":/icons/zoom_in.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.xzoomin.setIcon(icon4)
        self.xzoomin.setIconSize(QSize(20, 20))
        self.xzoomin.setAutoRepeat(True)

        self.zoom_buttons.addWidget(self.xzoomin)

        self.xzoomout = QPushButton(ViewerWindow)
        self.xzoomout.setObjectName(u"xzoomout")
        sizePolicy1.setHeightForWidth(self.xzoomout.sizePolicy().hasHeightForWidth())
        self.xzoomout.setSizePolicy(sizePolicy1)
        self.xzoomout.setStyleSheet(u"background-color: rgb(237, 241, 241);")
        icon5 = QIcon()
        icon5.addFile(u":/icons/zoom_out.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.xzoomout.setIcon(icon5)
        self.xzoomout.setIconSize(QSize(20, 20))
        self.xzoomout.setAutoRepeat(True)

        self.zoom_buttons.addWidget(self.xzoomout)

        self.xzoomreset = QPushButton(ViewerWindow)
        self.xzoomreset.setObjectName(u"xzoomreset")
        sizePolicy1.setHeightForWidth(self.xzoomreset.sizePolicy().hasHeightForWidth())
        self.xzoomreset.setSizePolicy(sizePolicy1)
        self.xzoomreset.setStyleSheet(u"background-color: rgb(237, 241, 241);")
        icon6 = QIcon()
        icon6.addFile(u":/icons/home.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.xzoomreset.setIcon(icon6)
        self.xzoomreset.setIconSize(QSize(20, 20))
        self.xzoomreset.setAutoRepeat(True)

        self.zoom_buttons.addWidget(self.xzoomreset)


        self.zoom_and_brightness.addLayout(self.zoom_buttons)


        self.top_plot_buttons.addLayout(self.zoom_and_brightness)

        self.topcontrolbottomspacer = QSpacerItem(5, 5, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Preferred)

        self.top_plot_buttons.addItem(self.topcontrolbottomspacer)

        self.top_plot_buttons.setStretch(0, 1)
        self.top_plot_buttons.setStretch(1, 10)
        self.top_plot_buttons.setStretch(2, 3)
        self.top_plot_buttons.setStretch(3, 10)
        self.top_plot_buttons.setStretch(4, 10)

        self.all_controls.addLayout(self.top_plot_buttons)

        self.allcontrolmidspacer = QSpacerItem(5, 5, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Preferred)

        self.all_controls.addItem(self.allcontrolmidspacer)

        self.bottom_plot_buttons = QVBoxLayout()
        self.bottom_plot_buttons.setObjectName(u"bottom_plot_buttons")
        self.verticalSpacer_3 = QSpacerItem(5, 5, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Preferred)

        self.bottom_plot_buttons.addItem(self.verticalSpacer_3)

        self.eeg_emg_buttons = QVBoxLayout()
        self.eeg_emg_buttons.setSpacing(20)
        self.eeg_emg_buttons.setObjectName(u"eeg_emg_buttons")
        self.eegbuttons = QGridLayout()
        self.eegbuttons.setObjectName(u"eegbuttons")
        self.eegbuttons.setHorizontalSpacing(10)
        self.eegbuttons.setVerticalSpacing(20)
        self.eegshiftup = QPushButton(ViewerWindow)
        self.eegshiftup.setObjectName(u"eegshiftup")
        sizePolicy1.setHeightForWidth(self.eegshiftup.sizePolicy().hasHeightForWidth())
        self.eegshiftup.setSizePolicy(sizePolicy1)
        self.eegshiftup.setStyleSheet(u"background-color: rgb(237, 241, 241);")
        icon7 = QIcon()
        icon7.addFile(u":/icons/double_up_arrow.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.eegshiftup.setIcon(icon7)
        self.eegshiftup.setIconSize(QSize(20, 20))
        self.eegshiftup.setAutoRepeat(True)

        self.eegbuttons.addWidget(self.eegshiftup, 0, 1, 1, 1)

        self.eegzoomout = QPushButton(ViewerWindow)
        self.eegzoomout.setObjectName(u"eegzoomout")
        sizePolicy1.setHeightForWidth(self.eegzoomout.sizePolicy().hasHeightForWidth())
        self.eegzoomout.setSizePolicy(sizePolicy1)
        self.eegzoomout.setStyleSheet(u"background-color: rgb(237, 241, 241);")
        self.eegzoomout.setIcon(icon5)
        self.eegzoomout.setIconSize(QSize(20, 20))
        self.eegzoomout.setAutoRepeat(True)

        self.eegbuttons.addWidget(self.eegzoomout, 1, 0, 1, 1)

        self.eegshiftdown = QPushButton(ViewerWindow)
        self.eegshiftdown.setObjectName(u"eegshiftdown")
        sizePolicy1.setHeightForWidth(self.eegshiftdown.sizePolicy().hasHeightForWidth())
        self.eegshiftdown.setSizePolicy(sizePolicy1)
        self.eegshiftdown.setStyleSheet(u"background-color: rgb(237, 241, 241);")
        icon8 = QIcon()
        icon8.addFile(u":/icons/double_down_arrow.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.eegshiftdown.setIcon(icon8)
        self.eegshiftdown.setIconSize(QSize(20, 20))
        self.eegshiftdown.setAutoRepeat(True)

        self.eegbuttons.addWidget(self.eegshiftdown, 1, 1, 1, 1)

        self.eegzoomin = QPushButton(ViewerWindow)
        self.eegzoomin.setObjectName(u"eegzoomin")
        sizePolicy1.setHeightForWidth(self.eegzoomin.sizePolicy().hasHeightForWidth())
        self.eegzoomin.setSizePolicy(sizePolicy1)
        self.eegzoomin.setStyleSheet(u"background-color: rgb(237, 241, 241);")
        self.eegzoomin.setIcon(icon4)
        self.eegzoomin.setIconSize(QSize(20, 20))
        self.eegzoomin.setAutoRepeat(True)

        self.eegbuttons.addWidget(self.eegzoomin, 0, 0, 1, 1)


        self.eeg_emg_buttons.addLayout(self.eegbuttons)

        self.verticalSpacer_5 = QSpacerItem(5, 5, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Preferred)

        self.eeg_emg_buttons.addItem(self.verticalSpacer_5)

        self.emgbuttons = QGridLayout()
        self.emgbuttons.setObjectName(u"emgbuttons")
        self.emgbuttons.setHorizontalSpacing(10)
        self.emgbuttons.setVerticalSpacing(20)
        self.emgzoomin = QPushButton(ViewerWindow)
        self.emgzoomin.setObjectName(u"emgzoomin")
        sizePolicy1.setHeightForWidth(self.emgzoomin.sizePolicy().hasHeightForWidth())
        self.emgzoomin.setSizePolicy(sizePolicy1)
        self.emgzoomin.setStyleSheet(u"background-color: rgb(237, 241, 241);")
        self.emgzoomin.setIcon(icon4)
        self.emgzoomin.setIconSize(QSize(20, 20))
        self.emgzoomin.setAutoRepeat(True)

        self.emgbuttons.addWidget(self.emgzoomin, 0, 0, 1, 1)

        self.emgshiftdown = QPushButton(ViewerWindow)
        self.emgshiftdown.setObjectName(u"emgshiftdown")
        sizePolicy1.setHeightForWidth(self.emgshiftdown.sizePolicy().hasHeightForWidth())
        self.emgshiftdown.setSizePolicy(sizePolicy1)
        self.emgshiftdown.setStyleSheet(u"background-color: rgb(237, 241, 241);")
        self.emgshiftdown.setIcon(icon8)
        self.emgshiftdown.setIconSize(QSize(20, 20))
        self.emgshiftdown.setAutoRepeat(True)

        self.emgbuttons.addWidget(self.emgshiftdown, 1, 1, 1, 1)

        self.emgshiftup = QPushButton(ViewerWindow)
        self.emgshiftup.setObjectName(u"emgshiftup")
        sizePolicy1.setHeightForWidth(self.emgshiftup.sizePolicy().hasHeightForWidth())
        self.emgshiftup.setSizePolicy(sizePolicy1)
        self.emgshiftup.setStyleSheet(u"background-color: rgb(237, 241, 241);")
        self.emgshiftup.setIcon(icon7)
        self.emgshiftup.setIconSize(QSize(20, 20))
        self.emgshiftup.setAutoRepeat(True)

        self.emgbuttons.addWidget(self.emgshiftup, 0, 1, 1, 1)

        self.emgzoomout = QPushButton(ViewerWindow)
        self.emgzoomout.setObjectName(u"emgzoomout")
        sizePolicy1.setHeightForWidth(self.emgzoomout.sizePolicy().hasHeightForWidth())
        self.emgzoomout.setSizePolicy(sizePolicy1)
        self.emgzoomout.setStyleSheet(u"background-color: rgb(237, 241, 241);")
        self.emgzoomout.setIcon(icon5)
        self.emgzoomout.setIconSize(QSize(20, 20))
        self.emgzoomout.setAutoRepeat(True)

        self.emgbuttons.addWidget(self.emgzoomout, 1, 0, 1, 1)


        self.eeg_emg_buttons.addLayout(self.emgbuttons)


        self.bottom_plot_buttons.addLayout(self.eeg_emg_buttons)

        self.verticalSpacer = QSpacerItem(5, 5, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Preferred)

        self.bottom_plot_buttons.addItem(self.verticalSpacer)

        self.autoscroll_layout = QVBoxLayout()
        self.autoscroll_layout.setObjectName(u"autoscroll_layout")
        self.autoscroll_layout.setContentsMargins(30, -1, 20, -1)
        self.autoscroll = QCheckBox(ViewerWindow)
        self.autoscroll.setObjectName(u"autoscroll")
        sizePolicy1.setHeightForWidth(self.autoscroll.sizePolicy().hasHeightForWidth())
        self.autoscroll.setSizePolicy(sizePolicy1)

        self.autoscroll_layout.addWidget(self.autoscroll)


        self.bottom_plot_buttons.addLayout(self.autoscroll_layout)

        self.verticalSpacer_2 = QSpacerItem(5, 5, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Preferred)

        self.bottom_plot_buttons.addItem(self.verticalSpacer_2)

        self.epoch_controls = QGridLayout()
        self.epoch_controls.setObjectName(u"epoch_controls")
        self.epoch_controls.setHorizontalSpacing(2)
        self.epoch_controls.setVerticalSpacing(10)
        self.shownepochslabel = QLabel(ViewerWindow)
        self.shownepochslabel.setObjectName(u"shownepochslabel")
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.shownepochslabel.sizePolicy().hasHeightForWidth())
        self.shownepochslabel.setSizePolicy(sizePolicy2)
        self.shownepochslabel.setAlignment(Qt.AlignmentFlag.AlignLeading|Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)

        self.epoch_controls.addWidget(self.shownepochslabel, 0, 1, 1, 1)

        self.epochsword = QLabel(ViewerWindow)
        self.epochsword.setObjectName(u"epochsword")
        sizePolicy1.setHeightForWidth(self.epochsword.sizePolicy().hasHeightForWidth())
        self.epochsword.setSizePolicy(sizePolicy1)

        self.epoch_controls.addWidget(self.epochsword, 0, 0, 1, 1)

        self.shownepochsplus = QPushButton(ViewerWindow)
        self.shownepochsplus.setObjectName(u"shownepochsplus")
        sizePolicy1.setHeightForWidth(self.shownepochsplus.sizePolicy().hasHeightForWidth())
        self.shownepochsplus.setSizePolicy(sizePolicy1)
        self.shownepochsplus.setStyleSheet(u"background-color: rgb(237, 241, 241);")
        icon9 = QIcon()
        icon9.addFile(u":/icons/up_arrow.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.shownepochsplus.setIcon(icon9)
        self.shownepochsplus.setIconSize(QSize(20, 20))
        self.shownepochsplus.setAutoRepeat(False)

        self.epoch_controls.addWidget(self.shownepochsplus, 1, 1, 1, 1)

        self.shownepochsminus = QPushButton(ViewerWindow)
        self.shownepochsminus.setObjectName(u"shownepochsminus")
        sizePolicy1.setHeightForWidth(self.shownepochsminus.sizePolicy().hasHeightForWidth())
        self.shownepochsminus.setSizePolicy(sizePolicy1)
        self.shownepochsminus.setStyleSheet(u"background-color: rgb(237, 241, 241);")
        icon10 = QIcon()
        icon10.addFile(u":/icons/down_arrow.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.shownepochsminus.setIcon(icon10)
        self.shownepochsminus.setIconSize(QSize(20, 20))
        self.shownepochsminus.setAutoRepeat(False)

        self.epoch_controls.addWidget(self.shownepochsminus, 1, 0, 1, 1)

        self.epoch_controls.setColumnStretch(0, 1)
        self.epoch_controls.setColumnStretch(1, 1)

        self.bottom_plot_buttons.addLayout(self.epoch_controls)

        self.verticalSpacer_4 = QSpacerItem(5, 5, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)

        self.bottom_plot_buttons.addItem(self.verticalSpacer_4)

        self.bottom_plot_buttons.setStretch(0, 1)
        self.bottom_plot_buttons.setStretch(1, 6)
        self.bottom_plot_buttons.setStretch(2, 1)
        self.bottom_plot_buttons.setStretch(3, 3)
        self.bottom_plot_buttons.setStretch(4, 1)
        self.bottom_plot_buttons.setStretch(5, 4)
        self.bottom_plot_buttons.setStretch(6, 1)

        self.all_controls.addLayout(self.bottom_plot_buttons)

        self.all_controls.setStretch(0, 50)
        self.all_controls.setStretch(1, 1)
        self.all_controls.setStretch(2, 50)

        self.horizontalLayout.addLayout(self.all_controls)

        self.horizontalLayout.setStretch(0, 20)
        self.horizontalLayout.setStretch(1, 1)

        self.retranslateUi(ViewerWindow)

        QMetaObject.connectSlotsByName(ViewerWindow)
    # setupUi

    def retranslateUi(self, ViewerWindow):
#if QT_CONFIG(tooltip)
        self.savebutton.setToolTip(QCoreApplication.translate("ViewerWindow", u"Save labels (Ctrl+S)", None))
#endif // QT_CONFIG(tooltip)
        self.savebutton.setText("")
#if QT_CONFIG(tooltip)
        self.helpbutton.setToolTip(QCoreApplication.translate("ViewerWindow", u"User manual", None))
#endif // QT_CONFIG(tooltip)
        self.helpbutton.setText("")
#if QT_CONFIG(tooltip)
        self.specbrighter.setToolTip(QCoreApplication.translate("ViewerWindow", u"Increase brightness", None))
#endif // QT_CONFIG(tooltip)
        self.specbrighter.setText("")
#if QT_CONFIG(tooltip)
        self.specdimmer.setToolTip(QCoreApplication.translate("ViewerWindow", u"Decrease brightness", None))
#endif // QT_CONFIG(tooltip)
        self.specdimmer.setText("")
#if QT_CONFIG(tooltip)
        self.xzoomin.setToolTip(QCoreApplication.translate("ViewerWindow", u"Zoom in (+)", None))
#endif // QT_CONFIG(tooltip)
        self.xzoomin.setText("")
#if QT_CONFIG(tooltip)
        self.xzoomout.setToolTip(QCoreApplication.translate("ViewerWindow", u"Zoom out (-)", None))
#endif // QT_CONFIG(tooltip)
        self.xzoomout.setText("")
#if QT_CONFIG(tooltip)
        self.xzoomreset.setToolTip(QCoreApplication.translate("ViewerWindow", u"Reset zoom", None))
#endif // QT_CONFIG(tooltip)
        self.xzoomreset.setText("")
#if QT_CONFIG(tooltip)
        self.eegshiftup.setToolTip(QCoreApplication.translate("ViewerWindow", u"Shift signal up", None))
#endif // QT_CONFIG(tooltip)
        self.eegshiftup.setText("")
#if QT_CONFIG(tooltip)
        self.eegzoomout.setToolTip(QCoreApplication.translate("ViewerWindow", u"Scale signal down", None))
#endif // QT_CONFIG(tooltip)
        self.eegzoomout.setText("")
#if QT_CONFIG(tooltip)
        self.eegshiftdown.setToolTip(QCoreApplication.translate("ViewerWindow", u"Shift signal down", None))
#endif // QT_CONFIG(tooltip)
        self.eegshiftdown.setText("")
#if QT_CONFIG(tooltip)
        self.eegzoomin.setToolTip(QCoreApplication.translate("ViewerWindow", u"Scale signal up", None))
#endif // QT_CONFIG(tooltip)
        self.eegzoomin.setText("")
        self.emgzoomin.setText("")
        self.emgshiftdown.setText("")
        self.emgshiftup.setText("")
        self.emgzoomout.setText("")
#if QT_CONFIG(tooltip)
        self.autoscroll.setToolTip(QCoreApplication.translate("ViewerWindow", u"Step forward when setting brain state", None))
#endif // QT_CONFIG(tooltip)
        self.autoscroll.setText(QCoreApplication.translate("ViewerWindow", u"Auto scroll", None))
        self.shownepochslabel.setText(QCoreApplication.translate("ViewerWindow", u"5", None))
        self.epochsword.setText(QCoreApplication.translate("ViewerWindow", u"Epochs:", None))
        self.shownepochsplus.setText("")
        self.shownepochsminus.setText("")
        pass
    # retranslateUi

