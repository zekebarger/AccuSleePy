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
from PySide6.QtWidgets import (QApplication, QCheckBox, QDoubleSpinBox, QGridLayout,
    QGroupBox, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QMainWindow, QPushButton, QSizePolicy,
    QSpacerItem, QTextBrowser, QVBoxLayout, QWidget)
import resources_rc

class Ui_Window0(object):
    def setupUi(self, Window0):
        if not Window0.objectName():
            Window0.setObjectName(u"Window0")
        Window0.resize(1079, 686)
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
        self.gridLayout = QGridLayout(self.centralwidget)
        self.gridLayout.setSpacing(20)
        self.gridLayout.setObjectName(u"gridLayout")
        self.gridLayout.setContentsMargins(5, 5, 5, 5)
        self.groupBox_3 = QGroupBox(self.centralwidget)
        self.groupBox_3.setObjectName(u"groupBox_3")
        sizePolicy.setHeightForWidth(self.groupBox_3.sizePolicy().hasHeightForWidth())
        self.groupBox_3.setSizePolicy(sizePolicy)
        self.verticalLayout_3 = QVBoxLayout(self.groupBox_3)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.horizontalLayout_7 = QHBoxLayout()
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.loadcalibrationbutton = QPushButton(self.groupBox_3)
        self.loadcalibrationbutton.setObjectName(u"loadcalibrationbutton")

        self.horizontalLayout_7.addWidget(self.loadcalibrationbutton)

        self.calibrationfiletext = QLabel(self.groupBox_3)
        self.calibrationfiletext.setObjectName(u"calibrationfiletext")

        self.horizontalLayout_7.addWidget(self.calibrationfiletext)

        self.horizontalLayout_7.setStretch(0, 1)
        self.horizontalLayout_7.setStretch(1, 5)

        self.verticalLayout_3.addLayout(self.horizontalLayout_7)

        self.horizontalLayout_9 = QHBoxLayout()
        self.horizontalLayout_9.setObjectName(u"horizontalLayout_9")
        self.pushButton_2 = QPushButton(self.groupBox_3)
        self.pushButton_2.setObjectName(u"pushButton_2")

        self.horizontalLayout_9.addWidget(self.pushButton_2)

        self.classificationmodeltext = QLabel(self.groupBox_3)
        self.classificationmodeltext.setObjectName(u"classificationmodeltext")

        self.horizontalLayout_9.addWidget(self.classificationmodeltext)

        self.horizontalLayout_9.setStretch(0, 1)
        self.horizontalLayout_9.setStretch(1, 5)

        self.verticalLayout_3.addLayout(self.horizontalLayout_9)

        self.horizontalLayout_8 = QHBoxLayout()
        self.horizontalLayout_8.setObjectName(u"horizontalLayout_8")
        self.scoreallbutton = QPushButton(self.groupBox_3)
        self.scoreallbutton.setObjectName(u"scoreallbutton")

        self.horizontalLayout_8.addWidget(self.scoreallbutton)

        self.scoreallstatus = QLabel(self.groupBox_3)
        self.scoreallstatus.setObjectName(u"scoreallstatus")

        self.horizontalLayout_8.addWidget(self.scoreallstatus)

        self.horizontalSpacer_5 = QSpacerItem(10, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_8.addItem(self.horizontalSpacer_5)

        self.verticalLayout_4 = QVBoxLayout()
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.overwritecheckbox = QCheckBox(self.groupBox_3)
        self.overwritecheckbox.setObjectName(u"overwritecheckbox")

        self.verticalLayout_4.addWidget(self.overwritecheckbox)

        self.boutlengthlayout = QHBoxLayout()
        self.boutlengthlayout.setObjectName(u"boutlengthlayout")
        self.boutlengthlabel = QLabel(self.groupBox_3)
        self.boutlengthlabel.setObjectName(u"boutlengthlabel")

        self.boutlengthlayout.addWidget(self.boutlengthlabel)

        self.boutlengthinput = QDoubleSpinBox(self.groupBox_3)
        self.boutlengthinput.setObjectName(u"boutlengthinput")
        self.boutlengthinput.setDecimals(3)
        self.boutlengthinput.setMaximum(1000.000000000000000)

        self.boutlengthlayout.addWidget(self.boutlengthinput)


        self.verticalLayout_4.addLayout(self.boutlengthlayout)


        self.horizontalLayout_8.addLayout(self.verticalLayout_4)


        self.verticalLayout_3.addLayout(self.horizontalLayout_8)


        self.gridLayout.addWidget(self.groupBox_3, 2, 1, 1, 1)

        self.parametergroupbox = QGroupBox(self.centralwidget)
        self.parametergroupbox.setObjectName(u"parametergroupbox")
        self.horizontalLayout_2 = QHBoxLayout(self.parametergroupbox)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalSpacer_3 = QSpacerItem(20, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacer_3)

        self.samplingratelayout = QHBoxLayout()
        self.samplingratelayout.setSpacing(5)
        self.samplingratelayout.setObjectName(u"samplingratelayout")
        self.samplingratelabel = QLabel(self.parametergroupbox)
        self.samplingratelabel.setObjectName(u"samplingratelabel")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.samplingratelabel.sizePolicy().hasHeightForWidth())
        self.samplingratelabel.setSizePolicy(sizePolicy1)

        self.samplingratelayout.addWidget(self.samplingratelabel)

        self.samplingrateinput = QDoubleSpinBox(self.parametergroupbox)
        self.samplingrateinput.setObjectName(u"samplingrateinput")
        sizePolicy1.setHeightForWidth(self.samplingrateinput.sizePolicy().hasHeightForWidth())
        self.samplingrateinput.setSizePolicy(sizePolicy1)
        self.samplingrateinput.setMinimum(0.000000000000000)
        self.samplingrateinput.setMaximum(100000.000000000000000)

        self.samplingratelayout.addWidget(self.samplingrateinput)

        self.samplingratelayout.setStretch(0, 4)
        self.samplingratelayout.setStretch(1, 2)

        self.horizontalLayout_2.addLayout(self.samplingratelayout)

        self.horizontalSpacer_2 = QSpacerItem(20, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacer_2)

        self.epochlengthlayout = QHBoxLayout()
        self.epochlengthlayout.setSpacing(5)
        self.epochlengthlayout.setObjectName(u"epochlengthlayout")
        self.epochlengthlabel = QLabel(self.parametergroupbox)
        self.epochlengthlabel.setObjectName(u"epochlengthlabel")
        sizePolicy1.setHeightForWidth(self.epochlengthlabel.sizePolicy().hasHeightForWidth())
        self.epochlengthlabel.setSizePolicy(sizePolicy1)

        self.epochlengthlayout.addWidget(self.epochlengthlabel)

        self.epochlengthinput = QDoubleSpinBox(self.parametergroupbox)
        self.epochlengthinput.setObjectName(u"epochlengthinput")
        sizePolicy1.setHeightForWidth(self.epochlengthinput.sizePolicy().hasHeightForWidth())
        self.epochlengthinput.setSizePolicy(sizePolicy1)
        self.epochlengthinput.setMaximum(100000.000000000000000)
        self.epochlengthinput.setSingleStep(0.500000000000000)

        self.epochlengthlayout.addWidget(self.epochlengthinput)

        self.epochlengthlayout.setStretch(0, 4)
        self.epochlengthlayout.setStretch(1, 2)

        self.horizontalLayout_2.addLayout(self.epochlengthlayout)

        self.horizontalSpacer = QSpacerItem(20, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacer)


        self.gridLayout.addWidget(self.parametergroupbox, 0, 1, 1, 1)

        self.groupBox = QGroupBox(self.centralwidget)
        self.groupBox.setObjectName(u"groupBox")
        sizePolicy.setHeightForWidth(self.groupBox.sizePolicy().hasHeightForWidth())
        self.groupBox.setSizePolicy(sizePolicy)
        self.verticalLayout = QVBoxLayout(self.groupBox)
        self.verticalLayout.setSpacing(5)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setSpacing(20)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.addbutton = QPushButton(self.groupBox)
        self.addbutton.setObjectName(u"addbutton")
        sizePolicy1.setHeightForWidth(self.addbutton.sizePolicy().hasHeightForWidth())
        self.addbutton.setSizePolicy(sizePolicy1)

        self.horizontalLayout_3.addWidget(self.addbutton)

        self.removebutton = QPushButton(self.groupBox)
        self.removebutton.setObjectName(u"removebutton")
        sizePolicy1.setHeightForWidth(self.removebutton.sizePolicy().hasHeightForWidth())
        self.removebutton.setSizePolicy(sizePolicy1)

        self.horizontalLayout_3.addWidget(self.removebutton)

        self.horizontalLayout_3.setStretch(0, 1)
        self.horizontalLayout_3.setStretch(1, 1)

        self.verticalLayout.addLayout(self.horizontalLayout_3)

        self.listWidget = QListWidget(self.groupBox)
        self.listWidget.setObjectName(u"listWidget")
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.listWidget.sizePolicy().hasHeightForWidth())
        self.listWidget.setSizePolicy(sizePolicy2)

        self.verticalLayout.addWidget(self.listWidget)

        self.verticalLayout.setStretch(0, 1)
        self.verticalLayout.setStretch(1, 5)

        self.gridLayout.addWidget(self.groupBox, 1, 0, 1, 1)

        self.groupBox_2 = QGroupBox(self.centralwidget)
        self.groupBox_2.setObjectName(u"groupBox_2")
        sizePolicy.setHeightForWidth(self.groupBox_2.sizePolicy().hasHeightForWidth())
        self.groupBox_2.setSizePolicy(sizePolicy)
        self.verticalLayout_2 = QVBoxLayout(self.groupBox_2)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.recordingfilebutton = QPushButton(self.groupBox_2)
        self.recordingfilebutton.setObjectName(u"recordingfilebutton")

        self.horizontalLayout_4.addWidget(self.recordingfilebutton)

        self.recordingfiletext = QLabel(self.groupBox_2)
        self.recordingfiletext.setObjectName(u"recordingfiletext")

        self.horizontalLayout_4.addWidget(self.recordingfiletext)

        self.horizontalLayout_4.setStretch(0, 1)
        self.horizontalLayout_4.setStretch(1, 5)

        self.verticalLayout_2.addLayout(self.horizontalLayout_4)

        self.horizontalLayout_6 = QHBoxLayout()
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.labelfilebutton = QPushButton(self.groupBox_2)
        self.labelfilebutton.setObjectName(u"labelfilebutton")

        self.horizontalLayout_6.addWidget(self.labelfilebutton)

        self.labelfiletext = QLabel(self.groupBox_2)
        self.labelfiletext.setObjectName(u"labelfiletext")

        self.horizontalLayout_6.addWidget(self.labelfiletext)

        self.horizontalLayout_6.setStretch(0, 1)
        self.horizontalLayout_6.setStretch(1, 5)

        self.verticalLayout_2.addLayout(self.horizontalLayout_6)

        self.horizontalLayout_5 = QHBoxLayout()
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.pushButton_3 = QPushButton(self.groupBox_2)
        self.pushButton_3.setObjectName(u"pushButton_3")

        self.horizontalLayout_5.addWidget(self.pushButton_3)

        self.manualscorestatus = QLabel(self.groupBox_2)
        self.manualscorestatus.setObjectName(u"manualscorestatus")

        self.horizontalLayout_5.addWidget(self.manualscorestatus)

        self.horizontalSpacer_4 = QSpacerItem(10, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_5.addItem(self.horizontalSpacer_4)

        self.pushButton = QPushButton(self.groupBox_2)
        self.pushButton.setObjectName(u"pushButton")

        self.horizontalLayout_5.addWidget(self.pushButton)

        self.calibrationstatus = QLabel(self.groupBox_2)
        self.calibrationstatus.setObjectName(u"calibrationstatus")

        self.horizontalLayout_5.addWidget(self.calibrationstatus)

        self.horizontalLayout_5.setStretch(0, 5)
        self.horizontalLayout_5.setStretch(1, 5)
        self.horizontalLayout_5.setStretch(2, 4)
        self.horizontalLayout_5.setStretch(3, 5)
        self.horizontalLayout_5.setStretch(4, 5)

        self.verticalLayout_2.addLayout(self.horizontalLayout_5)


        self.gridLayout.addWidget(self.groupBox_2, 1, 1, 1, 1)

        self.usermanualbutton = QPushButton(self.centralwidget)
        self.usermanualbutton.setObjectName(u"usermanualbutton")
        sizePolicy1.setHeightForWidth(self.usermanualbutton.sizePolicy().hasHeightForWidth())
        self.usermanualbutton.setSizePolicy(sizePolicy1)

        self.gridLayout.addWidget(self.usermanualbutton, 0, 0, 1, 1)

        self.groupBox_4 = QGroupBox(self.centralwidget)
        self.groupBox_4.setObjectName(u"groupBox_4")
        self.gridLayout_2 = QGridLayout(self.groupBox_4)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.messagebox = QTextBrowser(self.groupBox_4)
        self.messagebox.setObjectName(u"messagebox")
        sizePolicy.setHeightForWidth(self.messagebox.sizePolicy().hasHeightForWidth())
        self.messagebox.setSizePolicy(sizePolicy)

        self.gridLayout_2.addWidget(self.messagebox, 0, 0, 1, 1)


        self.gridLayout.addWidget(self.groupBox_4, 3, 1, 1, 1)

        self.gridLayout.setRowStretch(0, 1)
        self.gridLayout.setRowStretch(1, 5)
        self.gridLayout.setRowStretch(2, 4)
        self.gridLayout.setRowStretch(3, 3)
        self.gridLayout.setColumnStretch(0, 1)
        self.gridLayout.setColumnStretch(1, 6)
        Window0.setCentralWidget(self.centralwidget)

        self.retranslateUi(Window0)

        QMetaObject.connectSlotsByName(Window0)
    # setupUi

    def retranslateUi(self, Window0):
        Window0.setWindowTitle(QCoreApplication.translate("Window0", u"MainWindow", None))
        self.groupBox_3.setTitle(QCoreApplication.translate("Window0", u"Data / actions for all recordings from this subject", None))
        self.loadcalibrationbutton.setText(QCoreApplication.translate("Window0", u"Load calibration file", None))
        self.calibrationfiletext.setText("")
        self.pushButton_2.setText(QCoreApplication.translate("Window0", u"Load classification model", None))
        self.classificationmodeltext.setText("")
        self.scoreallbutton.setText(QCoreApplication.translate("Window0", u"Score all automatically", None))
        self.scoreallstatus.setText("")
        self.overwritecheckbox.setText(QCoreApplication.translate("Window0", u"Only overwrite undefined epochs", None))
        self.boutlengthlabel.setText(QCoreApplication.translate("Window0", u"Minimum bout length (sec):", None))
        self.parametergroupbox.setTitle(QCoreApplication.translate("Window0", u"Parameters for all recordings from one subject", None))
        self.samplingratelabel.setText(QCoreApplication.translate("Window0", u"Sampling rate (Hz):", None))
        self.epochlengthlabel.setText(QCoreApplication.translate("Window0", u"Epoch length (sec):", None))
        self.groupBox.setTitle(QCoreApplication.translate("Window0", u"Recording list", None))
        self.addbutton.setText(QCoreApplication.translate("Window0", u"add", None))
        self.removebutton.setText(QCoreApplication.translate("Window0", u"remove", None))
        self.groupBox_2.setTitle(QCoreApplication.translate("Window0", u"Data / actions for the selected recording from this subject", None))
        self.recordingfilebutton.setText(QCoreApplication.translate("Window0", u"Select recording file", None))
        self.recordingfiletext.setText("")
        self.labelfilebutton.setText(QCoreApplication.translate("Window0", u"Select label file", None))
        self.labelfiletext.setText("")
        self.pushButton_3.setText(QCoreApplication.translate("Window0", u"Score manually", None))
        self.manualscorestatus.setText("")
        self.pushButton.setText(QCoreApplication.translate("Window0", u"Create calibration file", None))
        self.calibrationstatus.setText("")
        self.usermanualbutton.setText(QCoreApplication.translate("Window0", u"User manual", None))
        self.groupBox_4.setTitle(QCoreApplication.translate("Window0", u"Messages", None))
    # retranslateUi

