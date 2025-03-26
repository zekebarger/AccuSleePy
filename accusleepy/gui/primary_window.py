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
    QListWidgetItem, QMainWindow, QPushButton, QRadioButton,
    QSizePolicy, QSpacerItem, QSpinBox, QTabWidget,
    QTextBrowser, QVBoxLayout, QWidget)
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
        PrimaryWindow.setStyleSheet(u"")
        self.centralwidget = QWidget(PrimaryWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        sizePolicy.setHeightForWidth(self.centralwidget.sizePolicy().hasHeightForWidth())
        self.centralwidget.setSizePolicy(sizePolicy)
        self.verticalLayout_5 = QVBoxLayout(self.centralwidget)
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.verticalLayout_5.setContentsMargins(0, 10, 0, 0)
        self.upper_tab_widget = QTabWidget(self.centralwidget)
        self.upper_tab_widget.setObjectName(u"upper_tab_widget")
        self.scoring_tab = QWidget()
        self.scoring_tab.setObjectName(u"scoring_tab")
        self.gridLayout_3 = QGridLayout(self.scoring_tab)
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.gridLayout_3.setContentsMargins(0, 0, 0, 0)
        self.scoring_tab_layout = QGridLayout()
        self.scoring_tab_layout.setSpacing(20)
        self.scoring_tab_layout.setObjectName(u"scoring_tab_layout")
        self.scoring_tab_layout.setContentsMargins(10, 10, 10, 10)
        self.messagesgroupbox = QGroupBox(self.scoring_tab)
        self.messagesgroupbox.setObjectName(u"messagesgroupbox")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.messagesgroupbox.sizePolicy().hasHeightForWidth())
        self.messagesgroupbox.setSizePolicy(sizePolicy1)
        font = QFont()
        font.setPointSize(13)
        self.messagesgroupbox.setFont(font)
        self.messagesgroupbox.setStyleSheet(u"")
        self.gridLayout_2 = QGridLayout(self.messagesgroupbox)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.gridLayout_2.setContentsMargins(5, 5, 5, 5)
        self.message_area = QTextBrowser(self.messagesgroupbox)
        self.message_area.setObjectName(u"message_area")
        sizePolicy1.setHeightForWidth(self.message_area.sizePolicy().hasHeightForWidth())
        self.message_area.setSizePolicy(sizePolicy1)
        self.message_area.setStyleSheet(u"background-color: white;")
        self.message_area.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)

        self.gridLayout_2.addWidget(self.message_area, 0, 0, 1, 1)


        self.scoring_tab_layout.addWidget(self.messagesgroupbox, 1, 1, 1, 1)

        self.recordingactionsgroupbox = QVBoxLayout()
        self.recordingactionsgroupbox.setSpacing(35)
        self.recordingactionsgroupbox.setObjectName(u"recordingactionsgroupbox")
        self.selected_recording_groupbox = QGroupBox(self.scoring_tab)
        self.selected_recording_groupbox.setObjectName(u"selected_recording_groupbox")
        sizePolicy.setHeightForWidth(self.selected_recording_groupbox.sizePolicy().hasHeightForWidth())
        self.selected_recording_groupbox.setSizePolicy(sizePolicy)
        self.selected_recording_groupbox.setFont(font)
        self.verticalLayout_2 = QVBoxLayout(self.selected_recording_groupbox)
        self.verticalLayout_2.setSpacing(10)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(5, 5, 5, 5)
        self.samplingratelayout = QHBoxLayout()
        self.samplingratelayout.setSpacing(10)
        self.samplingratelayout.setObjectName(u"samplingratelayout")
        self.samplingratelabel = QLabel(self.selected_recording_groupbox)
        self.samplingratelabel.setObjectName(u"samplingratelabel")
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.samplingratelabel.sizePolicy().hasHeightForWidth())
        self.samplingratelabel.setSizePolicy(sizePolicy2)
        self.samplingratelabel.setStyleSheet(u"background-color: transparent;")

        self.samplingratelayout.addWidget(self.samplingratelabel)

        self.sampling_rate_input = QDoubleSpinBox(self.selected_recording_groupbox)
        self.sampling_rate_input.setObjectName(u"sampling_rate_input")
        sizePolicy2.setHeightForWidth(self.sampling_rate_input.sizePolicy().hasHeightForWidth())
        self.sampling_rate_input.setSizePolicy(sizePolicy2)
        self.sampling_rate_input.setStyleSheet(u"background-color: white;")
        self.sampling_rate_input.setMinimum(0.000000000000000)
        self.sampling_rate_input.setMaximum(100000.000000000000000)

        self.samplingratelayout.addWidget(self.sampling_rate_input)

        self.horizontalSpacer_2 = QSpacerItem(20, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        self.samplingratelayout.addItem(self.horizontalSpacer_2)

        self.samplingratelayout.setStretch(0, 1)
        self.samplingratelayout.setStretch(1, 1)
        self.samplingratelayout.setStretch(2, 7)

        self.verticalLayout_2.addLayout(self.samplingratelayout)

        self.select_recording_layout = QHBoxLayout()
        self.select_recording_layout.setObjectName(u"select_recording_layout")
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setSpacing(5)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.recording_file_button = QPushButton(self.selected_recording_groupbox)
        self.recording_file_button.setObjectName(u"recording_file_button")
        sizePolicy3 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.recording_file_button.sizePolicy().hasHeightForWidth())
        self.recording_file_button.setSizePolicy(sizePolicy3)

        self.horizontalLayout.addWidget(self.recording_file_button)


        self.select_recording_layout.addLayout(self.horizontalLayout)

        self.recording_file_label = QLabel(self.selected_recording_groupbox)
        self.recording_file_label.setObjectName(u"recording_file_label")
        sizePolicy4 = QSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        sizePolicy4.setHorizontalStretch(0)
        sizePolicy4.setVerticalStretch(0)
        sizePolicy4.setHeightForWidth(self.recording_file_label.sizePolicy().hasHeightForWidth())
        self.recording_file_label.setSizePolicy(sizePolicy4)
        self.recording_file_label.setAcceptDrops(True)
        self.recording_file_label.setStyleSheet(u"background-color: white;")
        self.recording_file_label.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.select_recording_layout.addWidget(self.recording_file_label)

        self.select_recording_layout.setStretch(0, 5)
        self.select_recording_layout.setStretch(1, 12)

        self.verticalLayout_2.addLayout(self.select_recording_layout)

        self.label_file_layout = QHBoxLayout()
        self.label_file_layout.setObjectName(u"label_file_layout")
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
        self.or_label.setStyleSheet(u"background-color: transparent;")
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
        self.label_text.setStyleSheet(u"background-color: transparent;")

        self.select_or_create_layout.addWidget(self.label_text)

        self.select_or_create_layout.setStretch(0, 3)
        self.select_or_create_layout.setStretch(1, 1)
        self.select_or_create_layout.setStretch(2, 3)
        self.select_or_create_layout.setStretch(3, 3)

        self.label_file_layout.addLayout(self.select_or_create_layout)

        self.label_file_label = QLabel(self.selected_recording_groupbox)
        self.label_file_label.setObjectName(u"label_file_label")
        sizePolicy4.setHeightForWidth(self.label_file_label.sizePolicy().hasHeightForWidth())
        self.label_file_label.setSizePolicy(sizePolicy4)
        self.label_file_label.setAcceptDrops(True)
        self.label_file_label.setStyleSheet(u"background-color: white;")
        self.label_file_label.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.label_file_layout.addWidget(self.label_file_label)

        self.label_file_layout.setStretch(0, 5)
        self.label_file_layout.setStretch(1, 12)

        self.verticalLayout_2.addLayout(self.label_file_layout)

        self.manual_scoring_layout = QHBoxLayout()
        self.manual_scoring_layout.setObjectName(u"manual_scoring_layout")
        self.manual_scoring_button = QPushButton(self.selected_recording_groupbox)
        self.manual_scoring_button.setObjectName(u"manual_scoring_button")
        sizePolicy2.setHeightForWidth(self.manual_scoring_button.sizePolicy().hasHeightForWidth())
        self.manual_scoring_button.setSizePolicy(sizePolicy2)

        self.manual_scoring_layout.addWidget(self.manual_scoring_button)

        self.manual_scoring_status = QLabel(self.selected_recording_groupbox)
        self.manual_scoring_status.setObjectName(u"manual_scoring_status")
        self.manual_scoring_status.setStyleSheet(u"background-color: transparent;")

        self.manual_scoring_layout.addWidget(self.manual_scoring_status)

        self.create_calibration_button = QPushButton(self.selected_recording_groupbox)
        self.create_calibration_button.setObjectName(u"create_calibration_button")
        sizePolicy2.setHeightForWidth(self.create_calibration_button.sizePolicy().hasHeightForWidth())
        self.create_calibration_button.setSizePolicy(sizePolicy2)

        self.manual_scoring_layout.addWidget(self.create_calibration_button)

        self.calibration_status = QLabel(self.selected_recording_groupbox)
        self.calibration_status.setObjectName(u"calibration_status")
        self.calibration_status.setStyleSheet(u"background-color: transparent;")

        self.manual_scoring_layout.addWidget(self.calibration_status)

        self.horizontalSpacer_4 = QSpacerItem(10, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        self.manual_scoring_layout.addItem(self.horizontalSpacer_4)

        self.manual_scoring_layout.setStretch(0, 2)
        self.manual_scoring_layout.setStretch(1, 3)
        self.manual_scoring_layout.setStretch(2, 2)
        self.manual_scoring_layout.setStretch(3, 3)
        self.manual_scoring_layout.setStretch(4, 1)

        self.verticalLayout_2.addLayout(self.manual_scoring_layout)

        self.load_calibration_layout = QHBoxLayout()
        self.load_calibration_layout.setObjectName(u"load_calibration_layout")
        self.horizontalLayout_10 = QHBoxLayout()
        self.horizontalLayout_10.setSpacing(5)
        self.horizontalLayout_10.setObjectName(u"horizontalLayout_10")
        self.select_calibration_button = QPushButton(self.selected_recording_groupbox)
        self.select_calibration_button.setObjectName(u"select_calibration_button")
        sizePolicy3.setHeightForWidth(self.select_calibration_button.sizePolicy().hasHeightForWidth())
        self.select_calibration_button.setSizePolicy(sizePolicy3)

        self.horizontalLayout_10.addWidget(self.select_calibration_button)


        self.load_calibration_layout.addLayout(self.horizontalLayout_10)

        self.calibration_file_label = QLabel(self.selected_recording_groupbox)
        self.calibration_file_label.setObjectName(u"calibration_file_label")
        sizePolicy4.setHeightForWidth(self.calibration_file_label.sizePolicy().hasHeightForWidth())
        self.calibration_file_label.setSizePolicy(sizePolicy4)
        self.calibration_file_label.setAcceptDrops(True)
        self.calibration_file_label.setStyleSheet(u"background-color: white;")
        self.calibration_file_label.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.load_calibration_layout.addWidget(self.calibration_file_label)

        self.load_calibration_layout.setStretch(0, 5)
        self.load_calibration_layout.setStretch(1, 12)

        self.verticalLayout_2.addLayout(self.load_calibration_layout)

        self.verticalLayout_2.setStretch(0, 1)
        self.verticalLayout_2.setStretch(1, 1)
        self.verticalLayout_2.setStretch(2, 1)
        self.verticalLayout_2.setStretch(3, 1)
        self.verticalLayout_2.setStretch(4, 1)

        self.recordingactionsgroupbox.addWidget(self.selected_recording_groupbox)

        self.lower_tab_widget = QTabWidget(self.scoring_tab)
        self.lower_tab_widget.setObjectName(u"lower_tab_widget")
        self.classification_tab = QWidget()
        self.classification_tab.setObjectName(u"classification_tab")
        self.classification_tab.setStyleSheet(u"")
        self.gridLayout = QGridLayout(self.classification_tab)
        self.gridLayout.setObjectName(u"gridLayout")
        self.gridLayout.setVerticalSpacing(10)
        self.gridLayout.setContentsMargins(5, 5, 5, 5)
        self.gridLayout_4 = QGridLayout()
        self.gridLayout_4.setObjectName(u"gridLayout_4")
        self.score_all_layout = QHBoxLayout()
        self.score_all_layout.setObjectName(u"score_all_layout")
        self.score_all_button = QPushButton(self.classification_tab)
        self.score_all_button.setObjectName(u"score_all_button")
        sizePolicy2.setHeightForWidth(self.score_all_button.sizePolicy().hasHeightForWidth())
        self.score_all_button.setSizePolicy(sizePolicy2)

        self.score_all_layout.addWidget(self.score_all_button)

        self.score_all_status = QLabel(self.classification_tab)
        self.score_all_status.setObjectName(u"score_all_status")
        self.score_all_status.setStyleSheet(u"background-color: transparent;")

        self.score_all_layout.addWidget(self.score_all_status)

        self.verticalLayout_4 = QVBoxLayout()
        self.verticalLayout_4.setSpacing(10)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.overwritecheckbox = QCheckBox(self.classification_tab)
        self.overwritecheckbox.setObjectName(u"overwritecheckbox")
        sizePolicy2.setHeightForWidth(self.overwritecheckbox.sizePolicy().hasHeightForWidth())
        self.overwritecheckbox.setSizePolicy(sizePolicy2)
        self.overwritecheckbox.setStyleSheet(u"background-color: transparent;")

        self.verticalLayout_4.addWidget(self.overwritecheckbox)

        self.boutlengthlayout = QHBoxLayout()
        self.boutlengthlayout.setSpacing(5)
        self.boutlengthlayout.setObjectName(u"boutlengthlayout")
        self.boutlengthlabel = QLabel(self.classification_tab)
        self.boutlengthlabel.setObjectName(u"boutlengthlabel")
        sizePolicy2.setHeightForWidth(self.boutlengthlabel.sizePolicy().hasHeightForWidth())
        self.boutlengthlabel.setSizePolicy(sizePolicy2)
        self.boutlengthlabel.setStyleSheet(u"background-color: transparent;")

        self.boutlengthlayout.addWidget(self.boutlengthlabel)

        self.bout_length_input = QDoubleSpinBox(self.classification_tab)
        self.bout_length_input.setObjectName(u"bout_length_input")
        sizePolicy2.setHeightForWidth(self.bout_length_input.sizePolicy().hasHeightForWidth())
        self.bout_length_input.setSizePolicy(sizePolicy2)
        self.bout_length_input.setStyleSheet(u"background-color: white;")
        self.bout_length_input.setDecimals(2)
        self.bout_length_input.setMaximum(1000.000000000000000)
        self.bout_length_input.setValue(5.000000000000000)

        self.boutlengthlayout.addWidget(self.bout_length_input)


        self.verticalLayout_4.addLayout(self.boutlengthlayout)


        self.score_all_layout.addLayout(self.verticalLayout_4)

        self.horizontalSpacer_5 = QSpacerItem(10, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        self.score_all_layout.addItem(self.horizontalSpacer_5)

        self.score_all_layout.setStretch(0, 3)
        self.score_all_layout.setStretch(1, 3)
        self.score_all_layout.setStretch(2, 4)
        self.score_all_layout.setStretch(3, 2)

        self.gridLayout_4.addLayout(self.score_all_layout, 1, 0, 1, 1)

        self.load_model_layout = QHBoxLayout()
        self.load_model_layout.setObjectName(u"load_model_layout")
        self.horizontalLayout_11 = QHBoxLayout()
        self.horizontalLayout_11.setSpacing(5)
        self.horizontalLayout_11.setObjectName(u"horizontalLayout_11")
        self.load_model_button = QPushButton(self.classification_tab)
        self.load_model_button.setObjectName(u"load_model_button")
        sizePolicy3.setHeightForWidth(self.load_model_button.sizePolicy().hasHeightForWidth())
        self.load_model_button.setSizePolicy(sizePolicy3)

        self.horizontalLayout_11.addWidget(self.load_model_button)


        self.load_model_layout.addLayout(self.horizontalLayout_11)

        self.model_label = QLabel(self.classification_tab)
        self.model_label.setObjectName(u"model_label")
        sizePolicy4.setHeightForWidth(self.model_label.sizePolicy().hasHeightForWidth())
        self.model_label.setSizePolicy(sizePolicy4)
        self.model_label.setAcceptDrops(True)
        self.model_label.setStyleSheet(u"background-color: white;")
        self.model_label.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.load_model_layout.addWidget(self.model_label)

        self.load_model_layout.setStretch(0, 5)
        self.load_model_layout.setStretch(1, 12)

        self.gridLayout_4.addLayout(self.load_model_layout, 0, 0, 1, 1)


        self.gridLayout.addLayout(self.gridLayout_4, 0, 0, 1, 1)

        self.gridLayout.setColumnStretch(0, 2)
        self.lower_tab_widget.addTab(self.classification_tab, "")
        self.model_training_tab = QWidget()
        self.model_training_tab.setObjectName(u"model_training_tab")
        self.gridLayout_7 = QGridLayout(self.model_training_tab)
        self.gridLayout_7.setObjectName(u"gridLayout_7")
        self.gridLayout_7.setContentsMargins(0, 0, 0, 0)
        self.model_training_layout = QGridLayout()
        self.model_training_layout.setObjectName(u"model_training_layout")
        self.top_training_layout = QHBoxLayout()
        self.top_training_layout.setSpacing(10)
        self.top_training_layout.setObjectName(u"top_training_layout")
        self.horizontalLayout_5 = QHBoxLayout()
        self.horizontalLayout_5.setSpacing(5)
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.label = QLabel(self.model_training_tab)
        self.label.setObjectName(u"label")
        sizePolicy2.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy2)

        self.horizontalLayout_5.addWidget(self.label)

        self.image_number_input = QSpinBox(self.model_training_tab)
        self.image_number_input.setObjectName(u"image_number_input")
        sizePolicy2.setHeightForWidth(self.image_number_input.sizePolicy().hasHeightForWidth())
        self.image_number_input.setSizePolicy(sizePolicy2)
        self.image_number_input.setStyleSheet(u"background-color: white;")
        self.image_number_input.setMinimum(1)
        self.image_number_input.setMaximum(999)
        self.image_number_input.setValue(9)

        self.horizontalLayout_5.addWidget(self.image_number_input)


        self.top_training_layout.addLayout(self.horizontalLayout_5)

        self.horizontalSpacer = QSpacerItem(10, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        self.top_training_layout.addItem(self.horizontalSpacer)

        self.delete_image_box = QCheckBox(self.model_training_tab)
        self.delete_image_box.setObjectName(u"delete_image_box")
        sizePolicy2.setHeightForWidth(self.delete_image_box.sizePolicy().hasHeightForWidth())
        self.delete_image_box.setSizePolicy(sizePolicy2)
        self.delete_image_box.setChecked(True)

        self.top_training_layout.addWidget(self.delete_image_box)

        self.horizontalSpacer_6 = QSpacerItem(10, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        self.top_training_layout.addItem(self.horizontalSpacer_6)

        self.horizontalLayout_6 = QHBoxLayout()
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.label_2 = QLabel(self.model_training_tab)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setEnabled(False)
        sizePolicy2.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy2)

        self.horizontalLayout_6.addWidget(self.label_2)

        self.default_type_button = QRadioButton(self.model_training_tab)
        self.default_type_button.setObjectName(u"default_type_button")
        self.default_type_button.setEnabled(False)
        sizePolicy2.setHeightForWidth(self.default_type_button.sizePolicy().hasHeightForWidth())
        self.default_type_button.setSizePolicy(sizePolicy2)
        self.default_type_button.setChecked(True)

        self.horizontalLayout_6.addWidget(self.default_type_button)

        self.real_time_button = QRadioButton(self.model_training_tab)
        self.real_time_button.setObjectName(u"real_time_button")
        self.real_time_button.setEnabled(False)
        sizePolicy2.setHeightForWidth(self.real_time_button.sizePolicy().hasHeightForWidth())
        self.real_time_button.setSizePolicy(sizePolicy2)

        self.horizontalLayout_6.addWidget(self.real_time_button)

        self.horizontalLayout_6.setStretch(0, 2)
        self.horizontalLayout_6.setStretch(1, 3)
        self.horizontalLayout_6.setStretch(2, 3)

        self.top_training_layout.addLayout(self.horizontalLayout_6)

        self.horizontalSpacer_3 = QSpacerItem(10, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        self.top_training_layout.addItem(self.horizontalSpacer_3)

        self.top_training_layout.setStretch(0, 2)
        self.top_training_layout.setStretch(1, 1)
        self.top_training_layout.setStretch(2, 2)
        self.top_training_layout.setStretch(3, 1)
        self.top_training_layout.setStretch(4, 3)
        self.top_training_layout.setStretch(5, 1)

        self.model_training_layout.addLayout(self.top_training_layout, 0, 0, 1, 1)

        self.bottom_training_layout = QHBoxLayout()
        self.bottom_training_layout.setObjectName(u"bottom_training_layout")
        self.horizontalSpacer_7 = QSpacerItem(10, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        self.bottom_training_layout.addItem(self.horizontalSpacer_7)

        self.train_model_button = QPushButton(self.model_training_tab)
        self.train_model_button.setObjectName(u"train_model_button")
        sizePolicy2.setHeightForWidth(self.train_model_button.sizePolicy().hasHeightForWidth())
        self.train_model_button.setSizePolicy(sizePolicy2)

        self.bottom_training_layout.addWidget(self.train_model_button)

        self.horizontalSpacer_8 = QSpacerItem(10, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        self.bottom_training_layout.addItem(self.horizontalSpacer_8)


        self.model_training_layout.addLayout(self.bottom_training_layout, 2, 0, 1, 1)

        self.middle_training_layout = QHBoxLayout()
        self.middle_training_layout.setObjectName(u"middle_training_layout")
        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.training_folder_button = QPushButton(self.model_training_tab)
        self.training_folder_button.setObjectName(u"training_folder_button")
        sizePolicy3.setHeightForWidth(self.training_folder_button.sizePolicy().hasHeightForWidth())
        self.training_folder_button.setSizePolicy(sizePolicy3)

        self.horizontalLayout_2.addWidget(self.training_folder_button)


        self.middle_training_layout.addLayout(self.horizontalLayout_2)

        self.image_folder_label = QLabel(self.model_training_tab)
        self.image_folder_label.setObjectName(u"image_folder_label")
        sizePolicy4.setHeightForWidth(self.image_folder_label.sizePolicy().hasHeightForWidth())
        self.image_folder_label.setSizePolicy(sizePolicy4)
        self.image_folder_label.setStyleSheet(u"background-color: white;")

        self.middle_training_layout.addWidget(self.image_folder_label)

        self.middle_training_layout.setStretch(0, 5)
        self.middle_training_layout.setStretch(1, 12)

        self.model_training_layout.addLayout(self.middle_training_layout, 1, 0, 1, 1)

        self.model_training_layout.setRowStretch(0, 1)
        self.model_training_layout.setRowStretch(1, 1)
        self.model_training_layout.setRowStretch(2, 1)

        self.gridLayout_7.addLayout(self.model_training_layout, 0, 0, 1, 1)

        self.lower_tab_widget.addTab(self.model_training_tab, "")

        self.recordingactionsgroupbox.addWidget(self.lower_tab_widget)

        self.recordingactionsgroupbox.setStretch(0, 2)

        self.scoring_tab_layout.addLayout(self.recordingactionsgroupbox, 0, 1, 1, 1)

        self.left_col_layout = QVBoxLayout()
        self.left_col_layout.setSpacing(25)
        self.left_col_layout.setObjectName(u"left_col_layout")
        self.left_col_layout.setContentsMargins(5, 5, 5, 5)
        self.epoch_length_layout = QVBoxLayout()
        self.epoch_length_layout.setSpacing(5)
        self.epoch_length_layout.setObjectName(u"epoch_length_layout")
        self.epochlengthlabel = QLabel(self.scoring_tab)
        self.epochlengthlabel.setObjectName(u"epochlengthlabel")
        sizePolicy2.setHeightForWidth(self.epochlengthlabel.sizePolicy().hasHeightForWidth())
        self.epochlengthlabel.setSizePolicy(sizePolicy2)
        self.epochlengthlabel.setStyleSheet(u"background-color: transparent;")

        self.epoch_length_layout.addWidget(self.epochlengthlabel)

        self.epoch_length_input = QDoubleSpinBox(self.scoring_tab)
        self.epoch_length_input.setObjectName(u"epoch_length_input")
        sizePolicy2.setHeightForWidth(self.epoch_length_input.sizePolicy().hasHeightForWidth())
        self.epoch_length_input.setSizePolicy(sizePolicy2)
        self.epoch_length_input.setStyleSheet(u"background-color: white;")
        self.epoch_length_input.setMaximum(100000.000000000000000)
        self.epoch_length_input.setSingleStep(0.500000000000000)

        self.epoch_length_layout.addWidget(self.epoch_length_input)


        self.left_col_layout.addLayout(self.epoch_length_layout)

        self.recordinglistgroupbox = QGroupBox(self.scoring_tab)
        self.recordinglistgroupbox.setObjectName(u"recordinglistgroupbox")
        sizePolicy.setHeightForWidth(self.recordinglistgroupbox.sizePolicy().hasHeightForWidth())
        self.recordinglistgroupbox.setSizePolicy(sizePolicy)
        self.recordinglistgroupbox.setFont(font)
        self.recordinglistgroupbox.setStyleSheet(u"")
        self.verticalLayout = QVBoxLayout(self.recordinglistgroupbox)
        self.verticalLayout.setSpacing(5)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(5, 5, 5, 5)
        self.add_remove_layout = QHBoxLayout()
        self.add_remove_layout.setSpacing(20)
        self.add_remove_layout.setObjectName(u"add_remove_layout")
        self.add_button = QPushButton(self.recordinglistgroupbox)
        self.add_button.setObjectName(u"add_button")
        sizePolicy2.setHeightForWidth(self.add_button.sizePolicy().hasHeightForWidth())
        self.add_button.setSizePolicy(sizePolicy2)

        self.add_remove_layout.addWidget(self.add_button)

        self.remove_button = QPushButton(self.recordinglistgroupbox)
        self.remove_button.setObjectName(u"remove_button")
        sizePolicy2.setHeightForWidth(self.remove_button.sizePolicy().hasHeightForWidth())
        self.remove_button.setSizePolicy(sizePolicy2)

        self.add_remove_layout.addWidget(self.remove_button)

        self.add_remove_layout.setStretch(0, 1)
        self.add_remove_layout.setStretch(1, 1)

        self.verticalLayout.addLayout(self.add_remove_layout)

        self.recording_list_widget = QListWidget(self.recordinglistgroupbox)
        self.recording_list_widget.setObjectName(u"recording_list_widget")
        sizePolicy5 = QSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
        sizePolicy5.setHorizontalStretch(0)
        sizePolicy5.setVerticalStretch(0)
        sizePolicy5.setHeightForWidth(self.recording_list_widget.sizePolicy().hasHeightForWidth())
        self.recording_list_widget.setSizePolicy(sizePolicy5)
        self.recording_list_widget.setStyleSheet(u"background-color: white;")

        self.verticalLayout.addWidget(self.recording_list_widget)

        self.verticalLayout.setStretch(0, 1)
        self.verticalLayout.setStretch(1, 5)

        self.left_col_layout.addWidget(self.recordinglistgroupbox)

        self.user_manual_layout = QHBoxLayout()
        self.user_manual_layout.setObjectName(u"user_manual_layout")
        self.user_manual_button = QPushButton(self.scoring_tab)
        self.user_manual_button.setObjectName(u"user_manual_button")
        sizePolicy2.setHeightForWidth(self.user_manual_button.sizePolicy().hasHeightForWidth())
        self.user_manual_button.setSizePolicy(sizePolicy2)
        self.user_manual_button.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        icon = QIcon()
        icon.addFile(u":/icons/question.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.user_manual_button.setIcon(icon)
        self.user_manual_button.setIconSize(QSize(24, 24))

        self.user_manual_layout.addWidget(self.user_manual_button)


        self.left_col_layout.addLayout(self.user_manual_layout)

        self.left_col_layout.setStretch(0, 2)
        self.left_col_layout.setStretch(1, 10)

        self.scoring_tab_layout.addLayout(self.left_col_layout, 0, 0, 1, 1)

        self.scoring_tab_layout.setRowStretch(0, 2)
        self.scoring_tab_layout.setRowStretch(1, 1)
        self.scoring_tab_layout.setColumnStretch(0, 1)
        self.scoring_tab_layout.setColumnStretch(1, 10)

        self.gridLayout_3.addLayout(self.scoring_tab_layout, 0, 0, 1, 1)

        self.upper_tab_widget.addTab(self.scoring_tab, "")
        self.settings_tab = QWidget()
        self.settings_tab.setObjectName(u"settings_tab")
        self.gridLayout_5 = QGridLayout(self.settings_tab)
        self.gridLayout_5.setObjectName(u"gridLayout_5")
        self.gridLayout_5.setContentsMargins(0, 0, 0, 0)
        self.settings_tab_layout = QGridLayout()
        self.settings_tab_layout.setObjectName(u"settings_tab_layout")

        self.gridLayout_5.addLayout(self.settings_tab_layout, 0, 0, 1, 1)

        self.upper_tab_widget.addTab(self.settings_tab, "")

        self.verticalLayout_5.addWidget(self.upper_tab_widget)

        PrimaryWindow.setCentralWidget(self.centralwidget)
        QWidget.setTabOrder(self.epoch_length_input, self.sampling_rate_input)
        QWidget.setTabOrder(self.sampling_rate_input, self.recording_file_button)
        QWidget.setTabOrder(self.recording_file_button, self.select_label_button)
        QWidget.setTabOrder(self.select_label_button, self.create_label_button)
        QWidget.setTabOrder(self.create_label_button, self.manual_scoring_button)
        QWidget.setTabOrder(self.manual_scoring_button, self.create_calibration_button)
        QWidget.setTabOrder(self.create_calibration_button, self.add_button)
        QWidget.setTabOrder(self.add_button, self.remove_button)
        QWidget.setTabOrder(self.remove_button, self.select_calibration_button)
        QWidget.setTabOrder(self.select_calibration_button, self.load_model_button)
        QWidget.setTabOrder(self.load_model_button, self.score_all_button)
        QWidget.setTabOrder(self.score_all_button, self.overwritecheckbox)
        QWidget.setTabOrder(self.overwritecheckbox, self.bout_length_input)
        QWidget.setTabOrder(self.bout_length_input, self.user_manual_button)
        QWidget.setTabOrder(self.user_manual_button, self.recording_list_widget)
        QWidget.setTabOrder(self.recording_list_widget, self.message_area)

        self.retranslateUi(PrimaryWindow)

        self.upper_tab_widget.setCurrentIndex(0)
        self.lower_tab_widget.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(PrimaryWindow)
    # setupUi

    def retranslateUi(self, PrimaryWindow):
        PrimaryWindow.setWindowTitle(QCoreApplication.translate("PrimaryWindow", u"MainWindow", None))
        self.messagesgroupbox.setTitle(QCoreApplication.translate("PrimaryWindow", u"Messages", None))
        self.selected_recording_groupbox.setTitle(QCoreApplication.translate("PrimaryWindow", u"Data / actions for Recording 1", None))
        self.samplingratelabel.setText(QCoreApplication.translate("PrimaryWindow", u"Sampling rate (Hz):", None))
        self.recording_file_button.setText(QCoreApplication.translate("PrimaryWindow", u"Select recording file", None))
        self.recording_file_label.setText("")
#if QT_CONFIG(tooltip)
        self.select_label_button.setToolTip(QCoreApplication.translate("PrimaryWindow", u"Select existing label file", None))
#endif // QT_CONFIG(tooltip)
        self.select_label_button.setText(QCoreApplication.translate("PrimaryWindow", u"Select", None))
        self.or_label.setText(QCoreApplication.translate("PrimaryWindow", u"or", None))
#if QT_CONFIG(tooltip)
        self.create_label_button.setToolTip(QCoreApplication.translate("PrimaryWindow", u"Choose filename for new label file", None))
#endif // QT_CONFIG(tooltip)
        self.create_label_button.setText(QCoreApplication.translate("PrimaryWindow", u"create", None))
        self.label_text.setText(QCoreApplication.translate("PrimaryWindow", u"label file", None))
        self.label_file_label.setText("")
#if QT_CONFIG(tooltip)
        self.manual_scoring_button.setToolTip(QCoreApplication.translate("PrimaryWindow", u"View and edit brain state labels for this recording", None))
#endif // QT_CONFIG(tooltip)
        self.manual_scoring_button.setText(QCoreApplication.translate("PrimaryWindow", u"Score manually", None))
        self.manual_scoring_status.setText("")
        self.create_calibration_button.setText(QCoreApplication.translate("PrimaryWindow", u"Create calibration file", None))
        self.calibration_status.setText("")
        self.select_calibration_button.setText(QCoreApplication.translate("PrimaryWindow", u"Select calibration file", None))
        self.calibration_file_label.setText("")
#if QT_CONFIG(tooltip)
        self.score_all_button.setToolTip(QCoreApplication.translate("PrimaryWindow", u"Use classification model to score all recordings", None))
#endif // QT_CONFIG(tooltip)
        self.score_all_button.setText(QCoreApplication.translate("PrimaryWindow", u"Score all automatically", None))
        self.score_all_status.setText("")
        self.overwritecheckbox.setText(QCoreApplication.translate("PrimaryWindow", u"Only overwrite undefined epochs", None))
        self.boutlengthlabel.setText(QCoreApplication.translate("PrimaryWindow", u"Minimum bout length (sec):", None))
        self.load_model_button.setText(QCoreApplication.translate("PrimaryWindow", u"Load classification model", None))
        self.model_label.setText("")
        self.lower_tab_widget.setTabText(self.lower_tab_widget.indexOf(self.classification_tab), QCoreApplication.translate("PrimaryWindow", u"Classification", None))
        self.label.setText(QCoreApplication.translate("PrimaryWindow", u"Epochs per image:", None))
        self.delete_image_box.setText(QCoreApplication.translate("PrimaryWindow", u"Delete images after training", None))
        self.label_2.setText(QCoreApplication.translate("PrimaryWindow", u"Model type:", None))
        self.default_type_button.setText(QCoreApplication.translate("PrimaryWindow", u"Default", None))
        self.real_time_button.setText(QCoreApplication.translate("PrimaryWindow", u"Real-time", None))
        self.train_model_button.setText(QCoreApplication.translate("PrimaryWindow", u"Train classification model", None))
#if QT_CONFIG(tooltip)
        self.training_folder_button.setToolTip(QCoreApplication.translate("PrimaryWindow", u"A temporary folder will be created here", None))
#endif // QT_CONFIG(tooltip)
        self.training_folder_button.setText(QCoreApplication.translate("PrimaryWindow", u"Set training image directory", None))
        self.image_folder_label.setText("")
        self.lower_tab_widget.setTabText(self.lower_tab_widget.indexOf(self.model_training_tab), QCoreApplication.translate("PrimaryWindow", u"Model training", None))
        self.epochlengthlabel.setText(QCoreApplication.translate("PrimaryWindow", u"Epoch length (sec):", None))
        self.recordinglistgroupbox.setTitle(QCoreApplication.translate("PrimaryWindow", u"Recording list", None))
        self.add_button.setText(QCoreApplication.translate("PrimaryWindow", u"add", None))
        self.remove_button.setText(QCoreApplication.translate("PrimaryWindow", u"remove", None))
#if QT_CONFIG(tooltip)
        self.user_manual_button.setToolTip(QCoreApplication.translate("PrimaryWindow", u"User manual", None))
#endif // QT_CONFIG(tooltip)
        self.user_manual_button.setText("")
        self.upper_tab_widget.setTabText(self.upper_tab_widget.indexOf(self.scoring_tab), QCoreApplication.translate("PrimaryWindow", u"Sleep scoring", None))
        self.upper_tab_widget.setTabText(self.upper_tab_widget.indexOf(self.settings_tab), QCoreApplication.translate("PrimaryWindow", u"Settings", None))
    # retranslateUi

