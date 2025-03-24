# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'primary_window.ui'
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

class Ui_PrimaryWindow(object):
    def setupUi(self, PrimaryWindow):
        if not PrimaryWindow.objectName():
            PrimaryWindow.setObjectName(u"PrimaryWindow")
        PrimaryWindow.resize(1034, 686)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(PrimaryWindow.sizePolicy().hasHeightForWidth())
        PrimaryWindow.setSizePolicy(sizePolicy)
        PrimaryWindow.setStyleSheet(u"background-color: white;")
        self.centralwidget = QWidget(PrimaryWindow)
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

        self.epoch_length_input = QDoubleSpinBox(self.parametergroupbox)
        self.epoch_length_input.setObjectName(u"epoch_length_input")
        sizePolicy1.setHeightForWidth(self.epoch_length_input.sizePolicy().hasHeightForWidth())
        self.epoch_length_input.setSizePolicy(sizePolicy1)
        self.epoch_length_input.setMaximum(100000.000000000000000)
        self.epoch_length_input.setSingleStep(0.500000000000000)

        self.epochlengthlayout.addWidget(self.epoch_length_input)

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
        self.selected_recording_groupbox = QGroupBox(self.centralwidget)
        self.selected_recording_groupbox.setObjectName(u"selected_recording_groupbox")
        sizePolicy.setHeightForWidth(self.selected_recording_groupbox.sizePolicy().hasHeightForWidth())
        self.selected_recording_groupbox.setSizePolicy(sizePolicy)
        self.verticalLayout_2 = QVBoxLayout(self.selected_recording_groupbox)
        self.verticalLayout_2.setSpacing(10)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.samplingratelayout = QHBoxLayout()
        self.samplingratelayout.setSpacing(10)
        self.samplingratelayout.setObjectName(u"samplingratelayout")
        self.samplingratelabel = QLabel(self.selected_recording_groupbox)
        self.samplingratelabel.setObjectName(u"samplingratelabel")
        sizePolicy1.setHeightForWidth(self.samplingratelabel.sizePolicy().hasHeightForWidth())
        self.samplingratelabel.setSizePolicy(sizePolicy1)

        self.samplingratelayout.addWidget(self.samplingratelabel)

        self.sampling_rate_input = QDoubleSpinBox(self.selected_recording_groupbox)
        self.sampling_rate_input.setObjectName(u"sampling_rate_input")
        sizePolicy1.setHeightForWidth(self.sampling_rate_input.sizePolicy().hasHeightForWidth())
        self.sampling_rate_input.setSizePolicy(sizePolicy1)
        self.sampling_rate_input.setMinimum(0.000000000000000)
        self.sampling_rate_input.setMaximum(100000.000000000000000)

        self.samplingratelayout.addWidget(self.sampling_rate_input)

        self.horizontalSpacer_2 = QSpacerItem(20, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        self.samplingratelayout.addItem(self.horizontalSpacer_2)

        self.samplingratelayout.setStretch(0, 1)
        self.samplingratelayout.setStretch(1, 1)
        self.samplingratelayout.setStretch(2, 7)

        self.verticalLayout_2.addLayout(self.samplingratelayout)

        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.recording_file_button = QPushButton(self.selected_recording_groupbox)
        self.recording_file_button.setObjectName(u"recording_file_button")
        sizePolicy3 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.recording_file_button.sizePolicy().hasHeightForWidth())
        self.recording_file_button.setSizePolicy(sizePolicy3)

        self.horizontalLayout_4.addWidget(self.recording_file_button)

        self.recording_file_label = QLabel(self.selected_recording_groupbox)
        self.recording_file_label.setObjectName(u"recording_file_label")

        self.horizontalLayout_4.addWidget(self.recording_file_label)

        self.horizontalLayout_4.setStretch(0, 5)
        self.horizontalLayout_4.setStretch(1, 12)

        self.verticalLayout_2.addLayout(self.horizontalLayout_4)

        self.horizontalLayout_6 = QHBoxLayout()
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.select_or_create_layout = QHBoxLayout()
        self.select_or_create_layout.setSpacing(5)
        self.select_or_create_layout.setObjectName(u"select_or_create_layout")
        self.select_label_button = QPushButton(self.selected_recording_groupbox)
        self.select_label_button.setObjectName(u"select_label_button")
        sizePolicy3.setHeightForWidth(self.select_label_button.sizePolicy().hasHeightForWidth())
        self.select_label_button.setSizePolicy(sizePolicy3)
        self.select_label_button.setBaseSize(QSize(0, 0))

        self.select_or_create_layout.addWidget(self.select_label_button)

        self.or_label = QLabel(self.selected_recording_groupbox)
        self.or_label.setObjectName(u"or_label")
        sizePolicy3.setHeightForWidth(self.or_label.sizePolicy().hasHeightForWidth())
        self.or_label.setSizePolicy(sizePolicy3)
        self.or_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.select_or_create_layout.addWidget(self.or_label)

        self.create_label_button = QPushButton(self.selected_recording_groupbox)
        self.create_label_button.setObjectName(u"create_label_button")
        sizePolicy3.setHeightForWidth(self.create_label_button.sizePolicy().hasHeightForWidth())
        self.create_label_button.setSizePolicy(sizePolicy3)

        self.select_or_create_layout.addWidget(self.create_label_button)

        self.label_text = QLabel(self.selected_recording_groupbox)
        self.label_text.setObjectName(u"label_text")
        sizePolicy3.setHeightForWidth(self.label_text.sizePolicy().hasHeightForWidth())
        self.label_text.setSizePolicy(sizePolicy3)

        self.select_or_create_layout.addWidget(self.label_text)

        self.select_or_create_layout.setStretch(0, 3)
        self.select_or_create_layout.setStretch(1, 1)
        self.select_or_create_layout.setStretch(2, 3)

        self.horizontalLayout_6.addLayout(self.select_or_create_layout)

        self.label_file_label = QLabel(self.selected_recording_groupbox)
        self.label_file_label.setObjectName(u"label_file_label")

        self.horizontalLayout_6.addWidget(self.label_file_label)

        self.horizontalLayout_6.setStretch(0, 5)
        self.horizontalLayout_6.setStretch(1, 12)

        self.verticalLayout_2.addLayout(self.horizontalLayout_6)

        self.horizontalLayout_5 = QHBoxLayout()
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.manual_scoring_button = QPushButton(self.selected_recording_groupbox)
        self.manual_scoring_button.setObjectName(u"manual_scoring_button")
        sizePolicy1.setHeightForWidth(self.manual_scoring_button.sizePolicy().hasHeightForWidth())
        self.manual_scoring_button.setSizePolicy(sizePolicy1)

        self.horizontalLayout_5.addWidget(self.manual_scoring_button)

        self.manual_scoring_status = QLabel(self.selected_recording_groupbox)
        self.manual_scoring_status.setObjectName(u"manual_scoring_status")

        self.horizontalLayout_5.addWidget(self.manual_scoring_status)

        self.horizontalSpacer_4 = QSpacerItem(10, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_5.addItem(self.horizontalSpacer_4)

        self.create_calibration_button = QPushButton(self.selected_recording_groupbox)
        self.create_calibration_button.setObjectName(u"create_calibration_button")
        sizePolicy1.setHeightForWidth(self.create_calibration_button.sizePolicy().hasHeightForWidth())
        self.create_calibration_button.setSizePolicy(sizePolicy1)

        self.horizontalLayout_5.addWidget(self.create_calibration_button)

        self.calibration_status = QLabel(self.selected_recording_groupbox)
        self.calibration_status.setObjectName(u"calibration_status")

        self.horizontalLayout_5.addWidget(self.calibration_status)

        self.horizontalLayout_5.setStretch(0, 2)
        self.horizontalLayout_5.setStretch(1, 2)
        self.horizontalLayout_5.setStretch(2, 1)
        self.horizontalLayout_5.setStretch(3, 2)
        self.horizontalLayout_5.setStretch(4, 2)

        self.verticalLayout_2.addLayout(self.horizontalLayout_5)

        self.verticalLayout_2.setStretch(1, 1)
        self.verticalLayout_2.setStretch(2, 1)
        self.verticalLayout_2.setStretch(3, 1)

        self.recordingactionsgroupbox.addWidget(self.selected_recording_groupbox)

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
        self.load_calibration_button = QPushButton(self.allrecordingsgroupbox)
        self.load_calibration_button.setObjectName(u"load_calibration_button")
        sizePolicy3.setHeightForWidth(self.load_calibration_button.sizePolicy().hasHeightForWidth())
        self.load_calibration_button.setSizePolicy(sizePolicy3)

        self.horizontalLayout_7.addWidget(self.load_calibration_button)

        self.calibration_file_label = QLabel(self.allrecordingsgroupbox)
        self.calibration_file_label.setObjectName(u"calibration_file_label")

        self.horizontalLayout_7.addWidget(self.calibration_file_label)

        self.horizontalLayout_7.setStretch(0, 5)
        self.horizontalLayout_7.setStretch(1, 12)

        self.verticalLayout_3.addLayout(self.horizontalLayout_7)

        self.horizontalLayout_9 = QHBoxLayout()
        self.horizontalLayout_9.setObjectName(u"horizontalLayout_9")
        self.load_model_button = QPushButton(self.allrecordingsgroupbox)
        self.load_model_button.setObjectName(u"load_model_button")
        sizePolicy3.setHeightForWidth(self.load_model_button.sizePolicy().hasHeightForWidth())
        self.load_model_button.setSizePolicy(sizePolicy3)

        self.horizontalLayout_9.addWidget(self.load_model_button)

        self.model_label = QLabel(self.allrecordingsgroupbox)
        self.model_label.setObjectName(u"model_label")

        self.horizontalLayout_9.addWidget(self.model_label)

        self.horizontalLayout_9.setStretch(0, 5)
        self.horizontalLayout_9.setStretch(1, 12)

        self.verticalLayout_3.addLayout(self.horizontalLayout_9)

        self.horizontalLayout_8 = QHBoxLayout()
        self.horizontalLayout_8.setObjectName(u"horizontalLayout_8")
        self.score_all_button = QPushButton(self.allrecordingsgroupbox)
        self.score_all_button.setObjectName(u"score_all_button")
        sizePolicy1.setHeightForWidth(self.score_all_button.sizePolicy().hasHeightForWidth())
        self.score_all_button.setSizePolicy(sizePolicy1)

        self.horizontalLayout_8.addWidget(self.score_all_button)

        self.score_all_status = QLabel(self.allrecordingsgroupbox)
        self.score_all_status.setObjectName(u"score_all_status")

        self.horizontalLayout_8.addWidget(self.score_all_status)

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

        self.bout_length_input = QDoubleSpinBox(self.allrecordingsgroupbox)
        self.bout_length_input.setObjectName(u"bout_length_input")
        sizePolicy1.setHeightForWidth(self.bout_length_input.sizePolicy().hasHeightForWidth())
        self.bout_length_input.setSizePolicy(sizePolicy1)
        self.bout_length_input.setDecimals(2)
        self.bout_length_input.setMaximum(1000.000000000000000)
        self.bout_length_input.setValue(5.000000000000000)

        self.boutlengthlayout.addWidget(self.bout_length_input)


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
        sizePolicy.setHeightForWidth(self.message_area.sizePolicy().hasHeightForWidth())
        self.message_area.setSizePolicy(sizePolicy)
        self.message_area.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)

        self.gridLayout_2.addWidget(self.message_area, 0, 0, 1, 1)


        self.gridLayout.addWidget(self.messagesgroupbox, 2, 1, 1, 1)

        self.gridLayout.setRowStretch(0, 1)
        self.gridLayout.setRowStretch(1, 5)
        self.gridLayout.setRowStretch(2, 2)
        self.gridLayout.setColumnStretch(0, 1)
        self.gridLayout.setColumnStretch(1, 5)
        PrimaryWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(PrimaryWindow)

        QMetaObject.connectSlotsByName(PrimaryWindow)
    # setupUi

    def retranslateUi(self, PrimaryWindow):
        PrimaryWindow.setWindowTitle(QCoreApplication.translate("PrimaryWindow", u"MainWindow", None))
        self.usermanualbutton.setText(QCoreApplication.translate("PrimaryWindow", u"User manual", None))
        self.parametergroupbox.setTitle(QCoreApplication.translate("PrimaryWindow", u"Parameters for all recordings from one subject", None))
        self.epochlengthlabel.setText(QCoreApplication.translate("PrimaryWindow", u"Epoch length (sec):", None))
        self.recordinglistgroupbox.setTitle(QCoreApplication.translate("PrimaryWindow", u"Recording list", None))
        self.add_button.setText(QCoreApplication.translate("PrimaryWindow", u"add", None))
        self.remove_button.setText(QCoreApplication.translate("PrimaryWindow", u"remove", None))
        self.selected_recording_groupbox.setTitle(QCoreApplication.translate("PrimaryWindow", u"Data / actions for the selected recording (Recording 1) from this subject", None))
        self.samplingratelabel.setText(QCoreApplication.translate("PrimaryWindow", u"Sampling rate (Hz):", None))
        self.recording_file_button.setText(QCoreApplication.translate("PrimaryWindow", u"Select recording file", None))
        self.recording_file_label.setText("")
        self.select_label_button.setText(QCoreApplication.translate("PrimaryWindow", u"Select", None))
        self.or_label.setText(QCoreApplication.translate("PrimaryWindow", u"or", None))
        self.create_label_button.setText(QCoreApplication.translate("PrimaryWindow", u"create", None))
        self.label_text.setText(QCoreApplication.translate("PrimaryWindow", u"label file", None))
        self.label_file_label.setText("")
        self.manual_scoring_button.setText(QCoreApplication.translate("PrimaryWindow", u"Score manually", None))
        self.manual_scoring_status.setText("")
        self.create_calibration_button.setText(QCoreApplication.translate("PrimaryWindow", u"Create calibration file", None))
        self.calibration_status.setText("")
        self.allrecordingsgroupbox.setTitle(QCoreApplication.translate("PrimaryWindow", u"Data / actions for all recordings from this subject", None))
        self.load_calibration_button.setText(QCoreApplication.translate("PrimaryWindow", u"Load calibration file", None))
        self.calibration_file_label.setText("")
        self.load_model_button.setText(QCoreApplication.translate("PrimaryWindow", u"Load classification model", None))
        self.model_label.setText("")
        self.score_all_button.setText(QCoreApplication.translate("PrimaryWindow", u"Score all automatically", None))
        self.score_all_status.setText("")
        self.overwritecheckbox.setText(QCoreApplication.translate("PrimaryWindow", u"Only overwrite undefined epochs", None))
        self.boutlengthlabel.setText(QCoreApplication.translate("PrimaryWindow", u"Minimum bout length (sec):", None))
        self.messagesgroupbox.setTitle(QCoreApplication.translate("PrimaryWindow", u"Messages", None))
    # retranslateUi

