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
        self.gridLayout.setContentsMargins(10, 10, 10, 10)
        self.usermanualbutton = QPushButton(self.centralwidget)
        self.usermanualbutton.setObjectName(u"usermanualbutton")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.usermanualbutton.sizePolicy().hasHeightForWidth())
        self.usermanualbutton.setSizePolicy(sizePolicy1)

        self.gridLayout.addWidget(self.usermanualbutton, 0, 0, 1, 1)

        self.parametergroupbox = QGroupBox(self.centralwidget)
        self.parametergroupbox.setObjectName(u"parametergroupbox")
        self.horizontalLayout_2 = QHBoxLayout(self.parametergroupbox)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.epochlengthlayout = QHBoxLayout()
        self.epochlengthlayout.setSpacing(10)
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

        self.horizontalSpacer_3 = QSpacerItem(20, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        self.epochlengthlayout.addItem(self.horizontalSpacer_3)

        self.epochlengthlayout.setStretch(0, 1)
        self.epochlengthlayout.setStretch(1, 1)
        self.epochlengthlayout.setStretch(2, 7)

        self.horizontalLayout_2.addLayout(self.epochlengthlayout)


        self.gridLayout.addWidget(self.parametergroupbox, 0, 1, 1, 1)

        self.recordinglistgroupbox = QGroupBox(self.centralwidget)
        self.recordinglistgroupbox.setObjectName(u"recordinglistgroupbox")
        sizePolicy.setHeightForWidth(self.recordinglistgroupbox.sizePolicy().hasHeightForWidth())
        self.recordinglistgroupbox.setSizePolicy(sizePolicy)
        self.recordinglistgroupbox.setStyleSheet(u"")
        self.verticalLayout = QVBoxLayout(self.recordinglistgroupbox)
        self.verticalLayout.setSpacing(5)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setSpacing(20)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.add_button = QPushButton(self.recordinglistgroupbox)
        self.add_button.setObjectName(u"add_button")
        sizePolicy1.setHeightForWidth(self.add_button.sizePolicy().hasHeightForWidth())
        self.add_button.setSizePolicy(sizePolicy1)

        self.horizontalLayout_3.addWidget(self.add_button)

        self.remove_button = QPushButton(self.recordinglistgroupbox)
        self.remove_button.setObjectName(u"remove_button")
        sizePolicy1.setHeightForWidth(self.remove_button.sizePolicy().hasHeightForWidth())
        self.remove_button.setSizePolicy(sizePolicy1)

        self.horizontalLayout_3.addWidget(self.remove_button)

        self.horizontalLayout_3.setStretch(0, 1)
        self.horizontalLayout_3.setStretch(1, 1)

        self.verticalLayout.addLayout(self.horizontalLayout_3)

        self.recording_list_widget = QListWidget(self.recordinglistgroupbox)
        self.recording_list_widget.setObjectName(u"recording_list_widget")
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.recording_list_widget.sizePolicy().hasHeightForWidth())
        self.recording_list_widget.setSizePolicy(sizePolicy2)

        self.verticalLayout.addWidget(self.recording_list_widget)

        self.verticalLayout.setStretch(0, 1)
        self.verticalLayout.setStretch(1, 5)

        self.gridLayout.addWidget(self.recordinglistgroupbox, 1, 0, 1, 1)

        self.recordingactionsgroupbox = QVBoxLayout()
        self.recordingactionsgroupbox.setSpacing(20)
        self.recordingactionsgroupbox.setObjectName(u"recordingactionsgroupbox")
        self.thisrecordinggroupbox = QGroupBox(self.centralwidget)
        self.thisrecordinggroupbox.setObjectName(u"thisrecordinggroupbox")
        sizePolicy.setHeightForWidth(self.thisrecordinggroupbox.sizePolicy().hasHeightForWidth())
        self.thisrecordinggroupbox.setSizePolicy(sizePolicy)
        self.verticalLayout_2 = QVBoxLayout(self.thisrecordinggroupbox)
        self.verticalLayout_2.setSpacing(10)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.samplingratelayout = QHBoxLayout()
        self.samplingratelayout.setSpacing(10)
        self.samplingratelayout.setObjectName(u"samplingratelayout")
        self.samplingratelabel = QLabel(self.thisrecordinggroupbox)
        self.samplingratelabel.setObjectName(u"samplingratelabel")
        sizePolicy1.setHeightForWidth(self.samplingratelabel.sizePolicy().hasHeightForWidth())
        self.samplingratelabel.setSizePolicy(sizePolicy1)

        self.samplingratelayout.addWidget(self.samplingratelabel)

        self.samplingrateinput = QDoubleSpinBox(self.thisrecordinggroupbox)
        self.samplingrateinput.setObjectName(u"samplingrateinput")
        sizePolicy1.setHeightForWidth(self.samplingrateinput.sizePolicy().hasHeightForWidth())
        self.samplingrateinput.setSizePolicy(sizePolicy1)
        self.samplingrateinput.setMinimum(0.000000000000000)
        self.samplingrateinput.setMaximum(100000.000000000000000)

        self.samplingratelayout.addWidget(self.samplingrateinput)

        self.horizontalSpacer_2 = QSpacerItem(20, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        self.samplingratelayout.addItem(self.horizontalSpacer_2)

        self.samplingratelayout.setStretch(0, 1)
        self.samplingratelayout.setStretch(1, 1)
        self.samplingratelayout.setStretch(2, 7)

        self.verticalLayout_2.addLayout(self.samplingratelayout)

        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.recordingfilebutton = QPushButton(self.thisrecordinggroupbox)
        self.recordingfilebutton.setObjectName(u"recordingfilebutton")
        sizePolicy3 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.recordingfilebutton.sizePolicy().hasHeightForWidth())
        self.recordingfilebutton.setSizePolicy(sizePolicy3)

        self.horizontalLayout_4.addWidget(self.recordingfilebutton)

        self.recordingfiletext = QLabel(self.thisrecordinggroupbox)
        self.recordingfiletext.setObjectName(u"recordingfiletext")

        self.horizontalLayout_4.addWidget(self.recordingfiletext)

        self.horizontalLayout_4.setStretch(0, 2)
        self.horizontalLayout_4.setStretch(1, 7)

        self.verticalLayout_2.addLayout(self.horizontalLayout_4)

        self.horizontalLayout_6 = QHBoxLayout()
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.labelfilebutton = QPushButton(self.thisrecordinggroupbox)
        self.labelfilebutton.setObjectName(u"labelfilebutton")
        sizePolicy3.setHeightForWidth(self.labelfilebutton.sizePolicy().hasHeightForWidth())
        self.labelfilebutton.setSizePolicy(sizePolicy3)
        self.labelfilebutton.setBaseSize(QSize(0, 0))

        self.horizontalLayout_6.addWidget(self.labelfilebutton)

        self.labelfiletext = QLabel(self.thisrecordinggroupbox)
        self.labelfiletext.setObjectName(u"labelfiletext")

        self.horizontalLayout_6.addWidget(self.labelfiletext)

        self.horizontalLayout_6.setStretch(0, 2)
        self.horizontalLayout_6.setStretch(1, 7)

        self.verticalLayout_2.addLayout(self.horizontalLayout_6)

        self.horizontalLayout_5 = QHBoxLayout()
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.pushButton_3 = QPushButton(self.thisrecordinggroupbox)
        self.pushButton_3.setObjectName(u"pushButton_3")
        sizePolicy1.setHeightForWidth(self.pushButton_3.sizePolicy().hasHeightForWidth())
        self.pushButton_3.setSizePolicy(sizePolicy1)

        self.horizontalLayout_5.addWidget(self.pushButton_3)

        self.manualscorestatus = QLabel(self.thisrecordinggroupbox)
        self.manualscorestatus.setObjectName(u"manualscorestatus")

        self.horizontalLayout_5.addWidget(self.manualscorestatus)

        self.horizontalSpacer_4 = QSpacerItem(10, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_5.addItem(self.horizontalSpacer_4)

        self.pushButton = QPushButton(self.thisrecordinggroupbox)
        self.pushButton.setObjectName(u"pushButton")
        sizePolicy1.setHeightForWidth(self.pushButton.sizePolicy().hasHeightForWidth())
        self.pushButton.setSizePolicy(sizePolicy1)

        self.horizontalLayout_5.addWidget(self.pushButton)

        self.calibrationstatus = QLabel(self.thisrecordinggroupbox)
        self.calibrationstatus.setObjectName(u"calibrationstatus")

        self.horizontalLayout_5.addWidget(self.calibrationstatus)

        self.horizontalLayout_5.setStretch(0, 2)
        self.horizontalLayout_5.setStretch(1, 2)
        self.horizontalLayout_5.setStretch(2, 1)
        self.horizontalLayout_5.setStretch(3, 2)
        self.horizontalLayout_5.setStretch(4, 2)

        self.verticalLayout_2.addLayout(self.horizontalLayout_5)

        self.verticalLayout_2.setStretch(1, 1)
        self.verticalLayout_2.setStretch(2, 1)
        self.verticalLayout_2.setStretch(3, 1)

        self.recordingactionsgroupbox.addWidget(self.thisrecordinggroupbox)

        self.allrecordingsgroupbox = QGroupBox(self.centralwidget)
        self.allrecordingsgroupbox.setObjectName(u"allrecordingsgroupbox")
        sizePolicy.setHeightForWidth(self.allrecordingsgroupbox.sizePolicy().hasHeightForWidth())
        self.allrecordingsgroupbox.setSizePolicy(sizePolicy)
        self.verticalLayout_3 = QVBoxLayout(self.allrecordingsgroupbox)
        self.verticalLayout_3.setSpacing(10)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_7 = QHBoxLayout()
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.loadcalibrationbutton = QPushButton(self.allrecordingsgroupbox)
        self.loadcalibrationbutton.setObjectName(u"loadcalibrationbutton")
        sizePolicy3.setHeightForWidth(self.loadcalibrationbutton.sizePolicy().hasHeightForWidth())
        self.loadcalibrationbutton.setSizePolicy(sizePolicy3)

        self.horizontalLayout_7.addWidget(self.loadcalibrationbutton)

        self.calibrationfiletext = QLabel(self.allrecordingsgroupbox)
        self.calibrationfiletext.setObjectName(u"calibrationfiletext")

        self.horizontalLayout_7.addWidget(self.calibrationfiletext)

        self.horizontalLayout_7.setStretch(0, 2)
        self.horizontalLayout_7.setStretch(1, 7)

        self.verticalLayout_3.addLayout(self.horizontalLayout_7)

        self.horizontalLayout_9 = QHBoxLayout()
        self.horizontalLayout_9.setObjectName(u"horizontalLayout_9")
        self.pushButton_2 = QPushButton(self.allrecordingsgroupbox)
        self.pushButton_2.setObjectName(u"pushButton_2")
        sizePolicy3.setHeightForWidth(self.pushButton_2.sizePolicy().hasHeightForWidth())
        self.pushButton_2.setSizePolicy(sizePolicy3)

        self.horizontalLayout_9.addWidget(self.pushButton_2)

        self.classificationmodeltext = QLabel(self.allrecordingsgroupbox)
        self.classificationmodeltext.setObjectName(u"classificationmodeltext")

        self.horizontalLayout_9.addWidget(self.classificationmodeltext)

        self.horizontalLayout_9.setStretch(0, 2)
        self.horizontalLayout_9.setStretch(1, 7)

        self.verticalLayout_3.addLayout(self.horizontalLayout_9)

        self.horizontalLayout_8 = QHBoxLayout()
        self.horizontalLayout_8.setObjectName(u"horizontalLayout_8")
        self.scoreallbutton = QPushButton(self.allrecordingsgroupbox)
        self.scoreallbutton.setObjectName(u"scoreallbutton")
        sizePolicy1.setHeightForWidth(self.scoreallbutton.sizePolicy().hasHeightForWidth())
        self.scoreallbutton.setSizePolicy(sizePolicy1)

        self.horizontalLayout_8.addWidget(self.scoreallbutton)

        self.scoreallstatus = QLabel(self.allrecordingsgroupbox)
        self.scoreallstatus.setObjectName(u"scoreallstatus")

        self.horizontalLayout_8.addWidget(self.scoreallstatus)

        self.horizontalSpacer_5 = QSpacerItem(10, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_8.addItem(self.horizontalSpacer_5)

        self.verticalLayout_4 = QVBoxLayout()
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.overwritecheckbox = QCheckBox(self.allrecordingsgroupbox)
        self.overwritecheckbox.setObjectName(u"overwritecheckbox")
        sizePolicy1.setHeightForWidth(self.overwritecheckbox.sizePolicy().hasHeightForWidth())
        self.overwritecheckbox.setSizePolicy(sizePolicy1)

        self.verticalLayout_4.addWidget(self.overwritecheckbox)

        self.boutlengthlayout = QHBoxLayout()
        self.boutlengthlayout.setSpacing(5)
        self.boutlengthlayout.setObjectName(u"boutlengthlayout")
        self.boutlengthlabel = QLabel(self.allrecordingsgroupbox)
        self.boutlengthlabel.setObjectName(u"boutlengthlabel")
        sizePolicy1.setHeightForWidth(self.boutlengthlabel.sizePolicy().hasHeightForWidth())
        self.boutlengthlabel.setSizePolicy(sizePolicy1)

        self.boutlengthlayout.addWidget(self.boutlengthlabel)

        self.boutlengthinput = QDoubleSpinBox(self.allrecordingsgroupbox)
        self.boutlengthinput.setObjectName(u"boutlengthinput")
        sizePolicy1.setHeightForWidth(self.boutlengthinput.sizePolicy().hasHeightForWidth())
        self.boutlengthinput.setSizePolicy(sizePolicy1)
        self.boutlengthinput.setDecimals(3)
        self.boutlengthinput.setMaximum(1000.000000000000000)

        self.boutlengthlayout.addWidget(self.boutlengthinput)


        self.verticalLayout_4.addLayout(self.boutlengthlayout)


        self.horizontalLayout_8.addLayout(self.verticalLayout_4)


        self.verticalLayout_3.addLayout(self.horizontalLayout_8)

        self.verticalLayout_3.setStretch(0, 3)
        self.verticalLayout_3.setStretch(1, 3)
        self.verticalLayout_3.setStretch(2, 4)

        self.recordingactionsgroupbox.addWidget(self.allrecordingsgroupbox)

        self.recordingactionsgroupbox.setStretch(0, 1)
        self.recordingactionsgroupbox.setStretch(1, 1)

        self.gridLayout.addLayout(self.recordingactionsgroupbox, 1, 1, 1, 1)

        self.messagesgroupbox = QGroupBox(self.centralwidget)
        self.messagesgroupbox.setObjectName(u"messagesgroupbox")
        self.gridLayout_2 = QGridLayout(self.messagesgroupbox)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.gridLayout_2.setContentsMargins(10, 10, 10, 10)
        self.message_area = QTextBrowser(self.messagesgroupbox)
        self.message_area.setObjectName(u"message_area")

        self.gridLayout_2.addWidget(self.message_area, 0, 0, 1, 1)


        self.gridLayout.addWidget(self.messagesgroupbox, 2, 1, 1, 1)

        self.gridLayout.setRowStretch(0, 1)
        self.gridLayout.setRowStretch(1, 5)
        self.gridLayout.setRowStretch(2, 2)
        self.gridLayout.setColumnStretch(0, 1)
        self.gridLayout.setColumnStretch(1, 5)
        Window0.setCentralWidget(self.centralwidget)

        self.retranslateUi(Window0)

        QMetaObject.connectSlotsByName(Window0)
    # setupUi

    def retranslateUi(self, Window0):
        Window0.setWindowTitle(QCoreApplication.translate("Window0", u"MainWindow", None))
        self.usermanualbutton.setText(QCoreApplication.translate("Window0", u"User manual", None))
        self.parametergroupbox.setTitle(QCoreApplication.translate("Window0", u"Parameters for all recordings from one subject", None))
        self.epochlengthlabel.setText(QCoreApplication.translate("Window0", u"Epoch length (sec):", None))
        self.recordinglistgroupbox.setTitle(QCoreApplication.translate("Window0", u"Recording list", None))
        self.add_button.setText(QCoreApplication.translate("Window0", u"add", None))
        self.remove_button.setText(QCoreApplication.translate("Window0", u"remove", None))
        self.thisrecordinggroupbox.setTitle(QCoreApplication.translate("Window0", u"Data / actions for the selected recording from this subject", None))
        self.samplingratelabel.setText(QCoreApplication.translate("Window0", u"Sampling rate (Hz):", None))
        self.recordingfilebutton.setText(QCoreApplication.translate("Window0", u"Select recording file", None))
        self.recordingfiletext.setText("")
        self.labelfilebutton.setText(QCoreApplication.translate("Window0", u"Select label file", None))
        self.labelfiletext.setText("")
        self.pushButton_3.setText(QCoreApplication.translate("Window0", u"Score manually", None))
        self.manualscorestatus.setText("")
        self.pushButton.setText(QCoreApplication.translate("Window0", u"Create calibration file", None))
        self.calibrationstatus.setText("")
        self.allrecordingsgroupbox.setTitle(QCoreApplication.translate("Window0", u"Data / actions for all recordings from this subject", None))
        self.loadcalibrationbutton.setText(QCoreApplication.translate("Window0", u"Load calibration file", None))
        self.calibrationfiletext.setText("")
        self.pushButton_2.setText(QCoreApplication.translate("Window0", u"Load classification model", None))
        self.classificationmodeltext.setText("")
        self.scoreallbutton.setText(QCoreApplication.translate("Window0", u"Score all automatically", None))
        self.scoreallstatus.setText("")
        self.overwritecheckbox.setText(QCoreApplication.translate("Window0", u"Only overwrite undefined epochs", None))
        self.boutlengthlabel.setText(QCoreApplication.translate("Window0", u"Minimum bout length (sec):", None))
        self.messagesgroupbox.setTitle(QCoreApplication.translate("Window0", u"Messages", None))
    # retranslateUi

