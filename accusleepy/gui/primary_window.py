# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'primary_window.ui'
##
## Created by: Qt User Interface Compiler version 6.7.3
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
from PySide6.QtWidgets import (QApplication, QCheckBox, QDoubleSpinBox, QFrame,
    QGridLayout, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QListWidget, QListWidgetItem, QMainWindow,
    QPushButton, QRadioButton, QSizePolicy, QSpacerItem,
    QSpinBox, QTabWidget, QTextBrowser, QVBoxLayout,
    QWidget)
import accusleepy.gui.resources_rc  # noqa F401

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
        palette = QPalette()
        brush = QBrush(QColor(223, 226, 226, 255))
        brush.setStyle(Qt.SolidPattern)
        palette.setBrush(QPalette.Active, QPalette.Window, brush)
        palette.setBrush(QPalette.Inactive, QPalette.Window, brush)
        palette.setBrush(QPalette.Disabled, QPalette.Base, brush)
        palette.setBrush(QPalette.Disabled, QPalette.Window, brush)
        PrimaryWindow.setPalette(palette)
        self.centralwidget = QWidget(PrimaryWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        sizePolicy.setHeightForWidth(self.centralwidget.sizePolicy().hasHeightForWidth())
        self.centralwidget.setSizePolicy(sizePolicy)
        self.verticalLayout_5 = QVBoxLayout(self.centralwidget)
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.verticalLayout_5.setContentsMargins(0, 10, 0, 0)
        self.upper_tab_widget = QTabWidget(self.centralwidget)
        self.upper_tab_widget.setObjectName(u"upper_tab_widget")
        self.upper_tab_widget.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
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
        self.messagesgroupbox.setStyleSheet(u"")
        self.gridLayout_2 = QGridLayout(self.messagesgroupbox)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.gridLayout_2.setContentsMargins(5, 5, 5, 5)
        self.message_area = QTextBrowser(self.messagesgroupbox)
        self.message_area.setObjectName(u"message_area")
        sizePolicy1.setHeightForWidth(self.message_area.sizePolicy().hasHeightForWidth())
        self.message_area.setSizePolicy(sizePolicy1)
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

        self.samplingratelayout.addWidget(self.samplingratelabel)

        self.sampling_rate_input = QDoubleSpinBox(self.selected_recording_groupbox)
        self.sampling_rate_input.setObjectName(u"sampling_rate_input")
        sizePolicy2.setHeightForWidth(self.sampling_rate_input.sizePolicy().hasHeightForWidth())
        self.sampling_rate_input.setSizePolicy(sizePolicy2)
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
        sizePolicy3.setHeightForWidth(self.manual_scoring_button.sizePolicy().hasHeightForWidth())
        self.manual_scoring_button.setSizePolicy(sizePolicy3)

        self.manual_scoring_layout.addWidget(self.manual_scoring_button)

        self.manual_scoring_status = QLabel(self.selected_recording_groupbox)
        self.manual_scoring_status.setObjectName(u"manual_scoring_status")
        self.manual_scoring_status.setStyleSheet(u"background-color: transparent;")

        self.manual_scoring_layout.addWidget(self.manual_scoring_status)

        self.create_calibration_button = QPushButton(self.selected_recording_groupbox)
        self.create_calibration_button.setObjectName(u"create_calibration_button")
        sizePolicy3.setHeightForWidth(self.create_calibration_button.sizePolicy().hasHeightForWidth())
        self.create_calibration_button.setSizePolicy(sizePolicy3)

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
        sizePolicy3.setHeightForWidth(self.score_all_button.sizePolicy().hasHeightForWidth())
        self.score_all_button.setSizePolicy(sizePolicy3)

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
        self.image_number_input.setMinimum(9)
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
        sizePolicy2.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy2)

        self.horizontalLayout_6.addWidget(self.label_2)

        self.default_type_button = QRadioButton(self.model_training_tab)
        self.default_type_button.setObjectName(u"default_type_button")
        sizePolicy2.setHeightForWidth(self.default_type_button.sizePolicy().hasHeightForWidth())
        self.default_type_button.setSizePolicy(sizePolicy2)
        self.default_type_button.setChecked(True)

        self.horizontalLayout_6.addWidget(self.default_type_button)

        self.real_time_button = QRadioButton(self.model_training_tab)
        self.real_time_button.setObjectName(u"real_time_button")
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
        sizePolicy3.setHeightForWidth(self.train_model_button.sizePolicy().hasHeightForWidth())
        self.train_model_button.setSizePolicy(sizePolicy3)

        self.bottom_training_layout.addWidget(self.train_model_button)

        self.horizontalSpacer_8 = QSpacerItem(10, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        self.bottom_training_layout.addItem(self.horizontalSpacer_8)

        self.bottom_training_layout.setStretch(0, 2)
        self.bottom_training_layout.setStretch(1, 1)
        self.bottom_training_layout.setStretch(2, 2)

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
        self.left_col_layout.setSpacing(20)
        self.left_col_layout.setObjectName(u"left_col_layout")
        self.left_col_layout.setContentsMargins(5, 5, 5, 5)
        self.epoch_length_layout = QVBoxLayout()
        self.epoch_length_layout.setSpacing(5)
        self.epoch_length_layout.setObjectName(u"epoch_length_layout")
        self.epochlengthlabel = QLabel(self.scoring_tab)
        self.epochlengthlabel.setObjectName(u"epochlengthlabel")
        sizePolicy2.setHeightForWidth(self.epochlengthlabel.sizePolicy().hasHeightForWidth())
        self.epochlengthlabel.setSizePolicy(sizePolicy2)

        self.epoch_length_layout.addWidget(self.epochlengthlabel)

        self.epoch_length_input = QDoubleSpinBox(self.scoring_tab)
        self.epoch_length_input.setObjectName(u"epoch_length_input")
        sizePolicy2.setHeightForWidth(self.epoch_length_input.sizePolicy().hasHeightForWidth())
        self.epoch_length_input.setSizePolicy(sizePolicy2)
        self.epoch_length_input.setMaximum(100000.000000000000000)
        self.epoch_length_input.setSingleStep(0.500000000000000)

        self.epoch_length_layout.addWidget(self.epoch_length_input)


        self.left_col_layout.addLayout(self.epoch_length_layout)

        self.recordinglistgroupbox = QGroupBox(self.scoring_tab)
        self.recordinglistgroupbox.setObjectName(u"recordinglistgroupbox")
        sizePolicy.setHeightForWidth(self.recordinglistgroupbox.sizePolicy().hasHeightForWidth())
        self.recordinglistgroupbox.setSizePolicy(sizePolicy)
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

        self.horizontalLayout_59 = QHBoxLayout()
        self.horizontalLayout_59.setObjectName(u"horizontalLayout_59")
        self.export_button = QPushButton(self.recordinglistgroupbox)
        self.export_button.setObjectName(u"export_button")
        sizePolicy2.setHeightForWidth(self.export_button.sizePolicy().hasHeightForWidth())
        self.export_button.setSizePolicy(sizePolicy2)

        self.horizontalLayout_59.addWidget(self.export_button)

        self.import_button = QPushButton(self.recordinglistgroupbox)
        self.import_button.setObjectName(u"import_button")
        sizePolicy2.setHeightForWidth(self.import_button.sizePolicy().hasHeightForWidth())
        self.import_button.setSizePolicy(sizePolicy2)

        self.horizontalLayout_59.addWidget(self.import_button)


        self.verticalLayout.addLayout(self.horizontalLayout_59)

        self.verticalLayout.setStretch(0, 1)
        self.verticalLayout.setStretch(1, 5)
        self.verticalLayout.setStretch(2, 1)

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
        self.left_col_layout.setStretch(2, 1)

        self.scoring_tab_layout.addLayout(self.left_col_layout, 0, 0, 1, 1)

        self.verticalLayout_7 = QVBoxLayout()
        self.verticalLayout_7.setObjectName(u"verticalLayout_7")
        self.frame = QFrame(self.scoring_tab)
        self.frame.setObjectName(u"frame")
        sizePolicy.setHeightForWidth(self.frame.sizePolicy().hasHeightForWidth())
        self.frame.setSizePolicy(sizePolicy)
        self.frame.setStyleSheet(u"background-color: transparent;")
        self.frame.setFrameShape(QFrame.Shape.NoFrame)
        self.frame.setFrameShadow(QFrame.Shadow.Raised)
        self.accusleepy2 = QLabel(self.frame)
        self.accusleepy2.setObjectName(u"accusleepy2")
        self.accusleepy2.setGeometry(QRect(11, 75, 160, 60))
        sizePolicy2.setHeightForWidth(self.accusleepy2.sizePolicy().hasHeightForWidth())
        self.accusleepy2.setSizePolicy(sizePolicy2)
        font = QFont()
        font.setPointSize(22)
        font.setBold(True)
        font.setItalic(True)
        self.accusleepy2.setFont(font)
        self.accusleepy2.setStyleSheet(u"background-color: transparent;\n"
"color: rgb(130, 169, 68);")
        self.accusleepy2.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.accusleepy3 = QLabel(self.frame)
        self.accusleepy3.setObjectName(u"accusleepy3")
        self.accusleepy3.setGeometry(QRect(13, 77, 160, 60))
        self.accusleepy3.setFont(font)
        self.accusleepy3.setStyleSheet(u"background-color: transparent;\n"
"color: rgb(46, 63, 150);")
        self.accusleepy3.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.accusleepy1 = QLabel(self.frame)
        self.accusleepy1.setObjectName(u"accusleepy1")
        self.accusleepy1.setGeometry(QRect(9, 73, 160, 60))
        self.accusleepy1.setFont(font)
        self.accusleepy1.setStyleSheet(u"background-color: transparent;\n"
"color: rgb(244, 195, 68);")
        self.accusleepy1.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.accusleepy1.raise_()
        self.accusleepy2.raise_()
        self.accusleepy3.raise_()

        self.verticalLayout_7.addWidget(self.frame)


        self.scoring_tab_layout.addLayout(self.verticalLayout_7, 1, 0, 1, 1)

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
        self.settings_tab_layout.setHorizontalSpacing(20)
        self.settings_tab_layout.setVerticalSpacing(10)
        self.settings_tab_layout.setContentsMargins(20, 20, 20, -1)
        self.verticalLayout_3 = QVBoxLayout()
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.settings_text = QLabel(self.settings_tab)
        self.settings_text.setObjectName(u"settings_text")
        self.settings_text.setStyleSheet(u"background-color: white;")
        self.settings_text.setMargin(16)

        self.verticalLayout_3.addWidget(self.settings_text)


        self.settings_tab_layout.addLayout(self.verticalLayout_3, 0, 1, 1, 1)

        self.verticalLayout_6 = QVBoxLayout()
        self.verticalLayout_6.setObjectName(u"verticalLayout_6")
        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setSpacing(10)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.label_15 = QLabel(self.settings_tab)
        self.label_15.setObjectName(u"label_15")
        sizePolicy3.setHeightForWidth(self.label_15.sizePolicy().hasHeightForWidth())
        self.label_15.setSizePolicy(sizePolicy3)
        self.label_15.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.horizontalLayout_3.addWidget(self.label_15)

        self.label_14 = QLabel(self.settings_tab)
        self.label_14.setObjectName(u"label_14")
        sizePolicy3.setHeightForWidth(self.label_14.sizePolicy().hasHeightForWidth())
        self.label_14.setSizePolicy(sizePolicy3)
        self.label_14.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.horizontalLayout_3.addWidget(self.label_14)

        self.label_16 = QLabel(self.settings_tab)
        self.label_16.setObjectName(u"label_16")
        sizePolicy3.setHeightForWidth(self.label_16.sizePolicy().hasHeightForWidth())
        self.label_16.setSizePolicy(sizePolicy3)
        self.label_16.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.horizontalLayout_3.addWidget(self.label_16)

        self.label_13 = QLabel(self.settings_tab)
        self.label_13.setObjectName(u"label_13")
        sizePolicy3.setHeightForWidth(self.label_13.sizePolicy().hasHeightForWidth())
        self.label_13.setSizePolicy(sizePolicy3)
        self.label_13.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.horizontalLayout_3.addWidget(self.label_13)

        self.label_18 = QLabel(self.settings_tab)
        self.label_18.setObjectName(u"label_18")
        sizePolicy6 = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        sizePolicy6.setHorizontalStretch(0)
        sizePolicy6.setVerticalStretch(0)
        sizePolicy6.setHeightForWidth(self.label_18.sizePolicy().hasHeightForWidth())
        self.label_18.setSizePolicy(sizePolicy6)
        self.label_18.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.horizontalLayout_3.addWidget(self.label_18)

        self.horizontalLayout_3.setStretch(0, 3)
        self.horizontalLayout_3.setStretch(1, 3)
        self.horizontalLayout_3.setStretch(2, 4)
        self.horizontalLayout_3.setStretch(3, 3)
        self.horizontalLayout_3.setStretch(4, 3)

        self.verticalLayout_6.addLayout(self.horizontalLayout_3)

        self.horizontalLayout_17 = QHBoxLayout()
        self.horizontalLayout_17.setSpacing(10)
        self.horizontalLayout_17.setObjectName(u"horizontalLayout_17")
        self.horizontalLayout_19 = QHBoxLayout()
        self.horizontalLayout_19.setObjectName(u"horizontalLayout_19")
        self.label_12 = QLabel(self.settings_tab)
        self.label_12.setObjectName(u"label_12")
        sizePolicy.setHeightForWidth(self.label_12.sizePolicy().hasHeightForWidth())
        self.label_12.setSizePolicy(sizePolicy)
        font1 = QFont()
        font1.setPointSize(16)
        self.label_12.setFont(font1)
        self.label_12.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.horizontalLayout_19.addWidget(self.label_12)


        self.horizontalLayout_17.addLayout(self.horizontalLayout_19)

        self.horizontalLayout_20 = QHBoxLayout()
        self.horizontalLayout_20.setObjectName(u"horizontalLayout_20")
        self.horizontalSpacer_12 = QSpacerItem(5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_20.addItem(self.horizontalSpacer_12)

        self.enable_state_1 = QCheckBox(self.settings_tab)
        self.enable_state_1.setObjectName(u"enable_state_1")
        sizePolicy2.setHeightForWidth(self.enable_state_1.sizePolicy().hasHeightForWidth())
        self.enable_state_1.setSizePolicy(sizePolicy2)

        self.horizontalLayout_20.addWidget(self.enable_state_1)

        self.horizontalSpacer_11 = QSpacerItem(5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_20.addItem(self.horizontalSpacer_11)


        self.horizontalLayout_17.addLayout(self.horizontalLayout_20)

        self.state_name_1 = QLineEdit(self.settings_tab)
        self.state_name_1.setObjectName(u"state_name_1")
        sizePolicy3.setHeightForWidth(self.state_name_1.sizePolicy().hasHeightForWidth())
        self.state_name_1.setSizePolicy(sizePolicy3)
        self.state_name_1.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self.horizontalLayout_17.addWidget(self.state_name_1)

        self.horizontalLayout_21 = QHBoxLayout()
        self.horizontalLayout_21.setObjectName(u"horizontalLayout_21")
        self.horizontalSpacer_14 = QSpacerItem(5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_21.addItem(self.horizontalSpacer_14)

        self.state_scored_1 = QCheckBox(self.settings_tab)
        self.state_scored_1.setObjectName(u"state_scored_1")
        sizePolicy2.setHeightForWidth(self.state_scored_1.sizePolicy().hasHeightForWidth())
        self.state_scored_1.setSizePolicy(sizePolicy2)
        self.state_scored_1.setChecked(True)

        self.horizontalLayout_21.addWidget(self.state_scored_1)

        self.horizontalSpacer_13 = QSpacerItem(5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_21.addItem(self.horizontalSpacer_13)


        self.horizontalLayout_17.addLayout(self.horizontalLayout_21)

        self.horizontalLayout_22 = QHBoxLayout()
        self.horizontalLayout_22.setObjectName(u"horizontalLayout_22")
        self.horizontalSpacer_10 = QSpacerItem(5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_22.addItem(self.horizontalSpacer_10)

        self.state_frequency_1 = QDoubleSpinBox(self.settings_tab)
        self.state_frequency_1.setObjectName(u"state_frequency_1")
        self.state_frequency_1.setMaximum(1.000000000000000)
        self.state_frequency_1.setSingleStep(0.010000000000000)

        self.horizontalLayout_22.addWidget(self.state_frequency_1)

        self.horizontalSpacer_51 = QSpacerItem(5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_22.addItem(self.horizontalSpacer_51)


        self.horizontalLayout_17.addLayout(self.horizontalLayout_22)

        self.horizontalLayout_17.setStretch(0, 3)
        self.horizontalLayout_17.setStretch(1, 3)
        self.horizontalLayout_17.setStretch(2, 4)
        self.horizontalLayout_17.setStretch(3, 3)
        self.horizontalLayout_17.setStretch(4, 3)

        self.verticalLayout_6.addLayout(self.horizontalLayout_17)

        self.horizontalLayout_16 = QHBoxLayout()
        self.horizontalLayout_16.setSpacing(10)
        self.horizontalLayout_16.setObjectName(u"horizontalLayout_16")
        self.horizontalLayout_23 = QHBoxLayout()
        self.horizontalLayout_23.setObjectName(u"horizontalLayout_23")
        self.label_11 = QLabel(self.settings_tab)
        self.label_11.setObjectName(u"label_11")
        sizePolicy.setHeightForWidth(self.label_11.sizePolicy().hasHeightForWidth())
        self.label_11.setSizePolicy(sizePolicy)
        self.label_11.setFont(font1)
        self.label_11.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.horizontalLayout_23.addWidget(self.label_11)


        self.horizontalLayout_16.addLayout(self.horizontalLayout_23)

        self.horizontalLayout_24 = QHBoxLayout()
        self.horizontalLayout_24.setObjectName(u"horizontalLayout_24")
        self.horizontalSpacer_16 = QSpacerItem(5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_24.addItem(self.horizontalSpacer_16)

        self.enable_state_2 = QCheckBox(self.settings_tab)
        self.enable_state_2.setObjectName(u"enable_state_2")
        sizePolicy2.setHeightForWidth(self.enable_state_2.sizePolicy().hasHeightForWidth())
        self.enable_state_2.setSizePolicy(sizePolicy2)

        self.horizontalLayout_24.addWidget(self.enable_state_2)

        self.horizontalSpacer_15 = QSpacerItem(5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_24.addItem(self.horizontalSpacer_15)


        self.horizontalLayout_16.addLayout(self.horizontalLayout_24)

        self.state_name_2 = QLineEdit(self.settings_tab)
        self.state_name_2.setObjectName(u"state_name_2")
        sizePolicy3.setHeightForWidth(self.state_name_2.sizePolicy().hasHeightForWidth())
        self.state_name_2.setSizePolicy(sizePolicy3)

        self.horizontalLayout_16.addWidget(self.state_name_2)

        self.horizontalLayout_26 = QHBoxLayout()
        self.horizontalLayout_26.setObjectName(u"horizontalLayout_26")
        self.horizontalSpacer_18 = QSpacerItem(5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_26.addItem(self.horizontalSpacer_18)

        self.state_scored_2 = QCheckBox(self.settings_tab)
        self.state_scored_2.setObjectName(u"state_scored_2")
        sizePolicy2.setHeightForWidth(self.state_scored_2.sizePolicy().hasHeightForWidth())
        self.state_scored_2.setSizePolicy(sizePolicy2)
        self.state_scored_2.setChecked(True)

        self.horizontalLayout_26.addWidget(self.state_scored_2)

        self.horizontalSpacer_17 = QSpacerItem(5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_26.addItem(self.horizontalSpacer_17)


        self.horizontalLayout_16.addLayout(self.horizontalLayout_26)

        self.horizontalLayout_25 = QHBoxLayout()
        self.horizontalLayout_25.setObjectName(u"horizontalLayout_25")
        self.horizontalSpacer_52 = QSpacerItem(5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_25.addItem(self.horizontalSpacer_52)

        self.state_frequency_2 = QDoubleSpinBox(self.settings_tab)
        self.state_frequency_2.setObjectName(u"state_frequency_2")
        self.state_frequency_2.setMaximum(1.000000000000000)
        self.state_frequency_2.setSingleStep(0.010000000000000)

        self.horizontalLayout_25.addWidget(self.state_frequency_2)

        self.horizontalSpacer_53 = QSpacerItem(5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_25.addItem(self.horizontalSpacer_53)


        self.horizontalLayout_16.addLayout(self.horizontalLayout_25)

        self.horizontalLayout_16.setStretch(0, 3)
        self.horizontalLayout_16.setStretch(1, 3)
        self.horizontalLayout_16.setStretch(2, 4)
        self.horizontalLayout_16.setStretch(3, 3)
        self.horizontalLayout_16.setStretch(4, 3)

        self.verticalLayout_6.addLayout(self.horizontalLayout_16)

        self.horizontalLayout_15 = QHBoxLayout()
        self.horizontalLayout_15.setSpacing(10)
        self.horizontalLayout_15.setObjectName(u"horizontalLayout_15")
        self.horizontalLayout_28 = QHBoxLayout()
        self.horizontalLayout_28.setObjectName(u"horizontalLayout_28")
        self.label_10 = QLabel(self.settings_tab)
        self.label_10.setObjectName(u"label_10")
        sizePolicy.setHeightForWidth(self.label_10.sizePolicy().hasHeightForWidth())
        self.label_10.setSizePolicy(sizePolicy)
        self.label_10.setFont(font1)
        self.label_10.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.horizontalLayout_28.addWidget(self.label_10)


        self.horizontalLayout_15.addLayout(self.horizontalLayout_28)

        self.horizontalLayout_29 = QHBoxLayout()
        self.horizontalLayout_29.setObjectName(u"horizontalLayout_29")
        self.horizontalSpacer_20 = QSpacerItem(5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_29.addItem(self.horizontalSpacer_20)

        self.enable_state_3 = QCheckBox(self.settings_tab)
        self.enable_state_3.setObjectName(u"enable_state_3")
        sizePolicy2.setHeightForWidth(self.enable_state_3.sizePolicy().hasHeightForWidth())
        self.enable_state_3.setSizePolicy(sizePolicy2)

        self.horizontalLayout_29.addWidget(self.enable_state_3)

        self.horizontalSpacer_19 = QSpacerItem(5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_29.addItem(self.horizontalSpacer_19)


        self.horizontalLayout_15.addLayout(self.horizontalLayout_29)

        self.state_name_3 = QLineEdit(self.settings_tab)
        self.state_name_3.setObjectName(u"state_name_3")
        sizePolicy3.setHeightForWidth(self.state_name_3.sizePolicy().hasHeightForWidth())
        self.state_name_3.setSizePolicy(sizePolicy3)

        self.horizontalLayout_15.addWidget(self.state_name_3)

        self.horizontalLayout_30 = QHBoxLayout()
        self.horizontalLayout_30.setObjectName(u"horizontalLayout_30")
        self.horizontalSpacer_22 = QSpacerItem(5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_30.addItem(self.horizontalSpacer_22)

        self.state_scored_3 = QCheckBox(self.settings_tab)
        self.state_scored_3.setObjectName(u"state_scored_3")
        sizePolicy2.setHeightForWidth(self.state_scored_3.sizePolicy().hasHeightForWidth())
        self.state_scored_3.setSizePolicy(sizePolicy2)
        self.state_scored_3.setChecked(True)

        self.horizontalLayout_30.addWidget(self.state_scored_3)

        self.horizontalSpacer_21 = QSpacerItem(5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_30.addItem(self.horizontalSpacer_21)


        self.horizontalLayout_15.addLayout(self.horizontalLayout_30)

        self.horizontalLayout_27 = QHBoxLayout()
        self.horizontalLayout_27.setObjectName(u"horizontalLayout_27")
        self.horizontalSpacer_55 = QSpacerItem(5, 20, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_27.addItem(self.horizontalSpacer_55)

        self.state_frequency_3 = QDoubleSpinBox(self.settings_tab)
        self.state_frequency_3.setObjectName(u"state_frequency_3")
        self.state_frequency_3.setMaximum(1.000000000000000)
        self.state_frequency_3.setSingleStep(0.010000000000000)

        self.horizontalLayout_27.addWidget(self.state_frequency_3)

        self.horizontalSpacer_54 = QSpacerItem(5, 20, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_27.addItem(self.horizontalSpacer_54)


        self.horizontalLayout_15.addLayout(self.horizontalLayout_27)

        self.horizontalLayout_15.setStretch(0, 3)
        self.horizontalLayout_15.setStretch(1, 3)
        self.horizontalLayout_15.setStretch(2, 4)
        self.horizontalLayout_15.setStretch(3, 3)
        self.horizontalLayout_15.setStretch(4, 3)

        self.verticalLayout_6.addLayout(self.horizontalLayout_15)

        self.horizontalLayout_14 = QHBoxLayout()
        self.horizontalLayout_14.setSpacing(10)
        self.horizontalLayout_14.setObjectName(u"horizontalLayout_14")
        self.horizontalLayout_31 = QHBoxLayout()
        self.horizontalLayout_31.setObjectName(u"horizontalLayout_31")
        self.label_9 = QLabel(self.settings_tab)
        self.label_9.setObjectName(u"label_9")
        sizePolicy.setHeightForWidth(self.label_9.sizePolicy().hasHeightForWidth())
        self.label_9.setSizePolicy(sizePolicy)
        self.label_9.setFont(font1)
        self.label_9.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.horizontalLayout_31.addWidget(self.label_9)


        self.horizontalLayout_14.addLayout(self.horizontalLayout_31)

        self.horizontalLayout_45 = QHBoxLayout()
        self.horizontalLayout_45.setObjectName(u"horizontalLayout_45")
        self.horizontalSpacer_24 = QSpacerItem(5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_45.addItem(self.horizontalSpacer_24)

        self.enable_state_4 = QCheckBox(self.settings_tab)
        self.enable_state_4.setObjectName(u"enable_state_4")
        sizePolicy2.setHeightForWidth(self.enable_state_4.sizePolicy().hasHeightForWidth())
        self.enable_state_4.setSizePolicy(sizePolicy2)

        self.horizontalLayout_45.addWidget(self.enable_state_4)

        self.horizontalSpacer_23 = QSpacerItem(5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_45.addItem(self.horizontalSpacer_23)


        self.horizontalLayout_14.addLayout(self.horizontalLayout_45)

        self.state_name_4 = QLineEdit(self.settings_tab)
        self.state_name_4.setObjectName(u"state_name_4")
        sizePolicy3.setHeightForWidth(self.state_name_4.sizePolicy().hasHeightForWidth())
        self.state_name_4.setSizePolicy(sizePolicy3)

        self.horizontalLayout_14.addWidget(self.state_name_4)

        self.horizontalLayout_52 = QHBoxLayout()
        self.horizontalLayout_52.setObjectName(u"horizontalLayout_52")
        self.horizontalSpacer_26 = QSpacerItem(5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_52.addItem(self.horizontalSpacer_26)

        self.state_scored_4 = QCheckBox(self.settings_tab)
        self.state_scored_4.setObjectName(u"state_scored_4")
        sizePolicy2.setHeightForWidth(self.state_scored_4.sizePolicy().hasHeightForWidth())
        self.state_scored_4.setSizePolicy(sizePolicy2)
        self.state_scored_4.setChecked(True)

        self.horizontalLayout_52.addWidget(self.state_scored_4)

        self.horizontalSpacer_25 = QSpacerItem(5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_52.addItem(self.horizontalSpacer_25)


        self.horizontalLayout_14.addLayout(self.horizontalLayout_52)

        self.horizontalLayout_38 = QHBoxLayout()
        self.horizontalLayout_38.setObjectName(u"horizontalLayout_38")
        self.horizontalSpacer_57 = QSpacerItem(5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_38.addItem(self.horizontalSpacer_57)

        self.state_frequency_4 = QDoubleSpinBox(self.settings_tab)
        self.state_frequency_4.setObjectName(u"state_frequency_4")
        self.state_frequency_4.setMaximum(1.000000000000000)
        self.state_frequency_4.setSingleStep(0.010000000000000)

        self.horizontalLayout_38.addWidget(self.state_frequency_4)

        self.horizontalSpacer_56 = QSpacerItem(5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_38.addItem(self.horizontalSpacer_56)


        self.horizontalLayout_14.addLayout(self.horizontalLayout_38)

        self.horizontalLayout_14.setStretch(0, 3)
        self.horizontalLayout_14.setStretch(1, 3)
        self.horizontalLayout_14.setStretch(2, 4)
        self.horizontalLayout_14.setStretch(3, 3)
        self.horizontalLayout_14.setStretch(4, 3)

        self.verticalLayout_6.addLayout(self.horizontalLayout_14)

        self.horizontalLayout_13 = QHBoxLayout()
        self.horizontalLayout_13.setSpacing(10)
        self.horizontalLayout_13.setObjectName(u"horizontalLayout_13")
        self.horizontalLayout_32 = QHBoxLayout()
        self.horizontalLayout_32.setObjectName(u"horizontalLayout_32")
        self.label_8 = QLabel(self.settings_tab)
        self.label_8.setObjectName(u"label_8")
        sizePolicy.setHeightForWidth(self.label_8.sizePolicy().hasHeightForWidth())
        self.label_8.setSizePolicy(sizePolicy)
        self.label_8.setFont(font1)
        self.label_8.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.horizontalLayout_32.addWidget(self.label_8)


        self.horizontalLayout_13.addLayout(self.horizontalLayout_32)

        self.horizontalLayout_46 = QHBoxLayout()
        self.horizontalLayout_46.setObjectName(u"horizontalLayout_46")
        self.horizontalSpacer_29 = QSpacerItem(5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_46.addItem(self.horizontalSpacer_29)

        self.enable_state_5 = QCheckBox(self.settings_tab)
        self.enable_state_5.setObjectName(u"enable_state_5")
        sizePolicy2.setHeightForWidth(self.enable_state_5.sizePolicy().hasHeightForWidth())
        self.enable_state_5.setSizePolicy(sizePolicy2)

        self.horizontalLayout_46.addWidget(self.enable_state_5)

        self.horizontalSpacer_30 = QSpacerItem(5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_46.addItem(self.horizontalSpacer_30)


        self.horizontalLayout_13.addLayout(self.horizontalLayout_46)

        self.state_name_5 = QLineEdit(self.settings_tab)
        self.state_name_5.setObjectName(u"state_name_5")
        sizePolicy3.setHeightForWidth(self.state_name_5.sizePolicy().hasHeightForWidth())
        self.state_name_5.setSizePolicy(sizePolicy3)

        self.horizontalLayout_13.addWidget(self.state_name_5)

        self.horizontalLayout_53 = QHBoxLayout()
        self.horizontalLayout_53.setObjectName(u"horizontalLayout_53")
        self.horizontalSpacer_27 = QSpacerItem(5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_53.addItem(self.horizontalSpacer_27)

        self.state_scored_5 = QCheckBox(self.settings_tab)
        self.state_scored_5.setObjectName(u"state_scored_5")
        sizePolicy2.setHeightForWidth(self.state_scored_5.sizePolicy().hasHeightForWidth())
        self.state_scored_5.setSizePolicy(sizePolicy2)
        self.state_scored_5.setChecked(True)

        self.horizontalLayout_53.addWidget(self.state_scored_5)

        self.horizontalSpacer_28 = QSpacerItem(5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_53.addItem(self.horizontalSpacer_28)


        self.horizontalLayout_13.addLayout(self.horizontalLayout_53)

        self.horizontalLayout_39 = QHBoxLayout()
        self.horizontalLayout_39.setObjectName(u"horizontalLayout_39")
        self.horizontalSpacer_59 = QSpacerItem(5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_39.addItem(self.horizontalSpacer_59)

        self.state_frequency_5 = QDoubleSpinBox(self.settings_tab)
        self.state_frequency_5.setObjectName(u"state_frequency_5")
        self.state_frequency_5.setMaximum(1.000000000000000)
        self.state_frequency_5.setSingleStep(0.010000000000000)

        self.horizontalLayout_39.addWidget(self.state_frequency_5)

        self.horizontalSpacer_58 = QSpacerItem(5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_39.addItem(self.horizontalSpacer_58)


        self.horizontalLayout_13.addLayout(self.horizontalLayout_39)

        self.horizontalLayout_13.setStretch(0, 3)
        self.horizontalLayout_13.setStretch(1, 3)
        self.horizontalLayout_13.setStretch(2, 4)
        self.horizontalLayout_13.setStretch(3, 3)
        self.horizontalLayout_13.setStretch(4, 3)

        self.verticalLayout_6.addLayout(self.horizontalLayout_13)

        self.horizontalLayout_12 = QHBoxLayout()
        self.horizontalLayout_12.setSpacing(10)
        self.horizontalLayout_12.setObjectName(u"horizontalLayout_12")
        self.horizontalLayout_33 = QHBoxLayout()
        self.horizontalLayout_33.setObjectName(u"horizontalLayout_33")
        self.label_7 = QLabel(self.settings_tab)
        self.label_7.setObjectName(u"label_7")
        sizePolicy.setHeightForWidth(self.label_7.sizePolicy().hasHeightForWidth())
        self.label_7.setSizePolicy(sizePolicy)
        self.label_7.setFont(font1)
        self.label_7.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.horizontalLayout_33.addWidget(self.label_7)


        self.horizontalLayout_12.addLayout(self.horizontalLayout_33)

        self.horizontalLayout_47 = QHBoxLayout()
        self.horizontalLayout_47.setObjectName(u"horizontalLayout_47")
        self.horizontalSpacer_32 = QSpacerItem(5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_47.addItem(self.horizontalSpacer_32)

        self.enable_state_6 = QCheckBox(self.settings_tab)
        self.enable_state_6.setObjectName(u"enable_state_6")
        sizePolicy2.setHeightForWidth(self.enable_state_6.sizePolicy().hasHeightForWidth())
        self.enable_state_6.setSizePolicy(sizePolicy2)

        self.horizontalLayout_47.addWidget(self.enable_state_6)

        self.horizontalSpacer_31 = QSpacerItem(5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_47.addItem(self.horizontalSpacer_31)


        self.horizontalLayout_12.addLayout(self.horizontalLayout_47)

        self.state_name_6 = QLineEdit(self.settings_tab)
        self.state_name_6.setObjectName(u"state_name_6")
        sizePolicy3.setHeightForWidth(self.state_name_6.sizePolicy().hasHeightForWidth())
        self.state_name_6.setSizePolicy(sizePolicy3)

        self.horizontalLayout_12.addWidget(self.state_name_6)

        self.horizontalLayout_54 = QHBoxLayout()
        self.horizontalLayout_54.setObjectName(u"horizontalLayout_54")
        self.horizontalSpacer_34 = QSpacerItem(5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_54.addItem(self.horizontalSpacer_34)

        self.state_scored_6 = QCheckBox(self.settings_tab)
        self.state_scored_6.setObjectName(u"state_scored_6")
        sizePolicy2.setHeightForWidth(self.state_scored_6.sizePolicy().hasHeightForWidth())
        self.state_scored_6.setSizePolicy(sizePolicy2)
        self.state_scored_6.setChecked(True)

        self.horizontalLayout_54.addWidget(self.state_scored_6)

        self.horizontalSpacer_33 = QSpacerItem(5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_54.addItem(self.horizontalSpacer_33)


        self.horizontalLayout_12.addLayout(self.horizontalLayout_54)

        self.horizontalLayout_40 = QHBoxLayout()
        self.horizontalLayout_40.setObjectName(u"horizontalLayout_40")
        self.horizontalSpacer_61 = QSpacerItem(5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_40.addItem(self.horizontalSpacer_61)

        self.state_frequency_6 = QDoubleSpinBox(self.settings_tab)
        self.state_frequency_6.setObjectName(u"state_frequency_6")
        self.state_frequency_6.setMaximum(1.000000000000000)
        self.state_frequency_6.setSingleStep(0.010000000000000)

        self.horizontalLayout_40.addWidget(self.state_frequency_6)

        self.horizontalSpacer_60 = QSpacerItem(5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_40.addItem(self.horizontalSpacer_60)


        self.horizontalLayout_12.addLayout(self.horizontalLayout_40)

        self.horizontalLayout_12.setStretch(0, 3)
        self.horizontalLayout_12.setStretch(1, 3)
        self.horizontalLayout_12.setStretch(2, 4)
        self.horizontalLayout_12.setStretch(3, 3)
        self.horizontalLayout_12.setStretch(4, 3)

        self.verticalLayout_6.addLayout(self.horizontalLayout_12)

        self.horizontalLayout_9 = QHBoxLayout()
        self.horizontalLayout_9.setSpacing(10)
        self.horizontalLayout_9.setObjectName(u"horizontalLayout_9")
        self.horizontalLayout_34 = QHBoxLayout()
        self.horizontalLayout_34.setObjectName(u"horizontalLayout_34")
        self.label_6 = QLabel(self.settings_tab)
        self.label_6.setObjectName(u"label_6")
        sizePolicy.setHeightForWidth(self.label_6.sizePolicy().hasHeightForWidth())
        self.label_6.setSizePolicy(sizePolicy)
        self.label_6.setFont(font1)
        self.label_6.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.horizontalLayout_34.addWidget(self.label_6)


        self.horizontalLayout_9.addLayout(self.horizontalLayout_34)

        self.horizontalLayout_48 = QHBoxLayout()
        self.horizontalLayout_48.setObjectName(u"horizontalLayout_48")
        self.horizontalSpacer_36 = QSpacerItem(5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_48.addItem(self.horizontalSpacer_36)

        self.enable_state_7 = QCheckBox(self.settings_tab)
        self.enable_state_7.setObjectName(u"enable_state_7")
        sizePolicy2.setHeightForWidth(self.enable_state_7.sizePolicy().hasHeightForWidth())
        self.enable_state_7.setSizePolicy(sizePolicy2)

        self.horizontalLayout_48.addWidget(self.enable_state_7)

        self.horizontalSpacer_35 = QSpacerItem(5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_48.addItem(self.horizontalSpacer_35)


        self.horizontalLayout_9.addLayout(self.horizontalLayout_48)

        self.state_name_7 = QLineEdit(self.settings_tab)
        self.state_name_7.setObjectName(u"state_name_7")
        sizePolicy3.setHeightForWidth(self.state_name_7.sizePolicy().hasHeightForWidth())
        self.state_name_7.setSizePolicy(sizePolicy3)

        self.horizontalLayout_9.addWidget(self.state_name_7)

        self.horizontalLayout_55 = QHBoxLayout()
        self.horizontalLayout_55.setObjectName(u"horizontalLayout_55")
        self.horizontalSpacer_38 = QSpacerItem(5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_55.addItem(self.horizontalSpacer_38)

        self.state_scored_7 = QCheckBox(self.settings_tab)
        self.state_scored_7.setObjectName(u"state_scored_7")
        sizePolicy2.setHeightForWidth(self.state_scored_7.sizePolicy().hasHeightForWidth())
        self.state_scored_7.setSizePolicy(sizePolicy2)
        self.state_scored_7.setChecked(True)

        self.horizontalLayout_55.addWidget(self.state_scored_7)

        self.horizontalSpacer_37 = QSpacerItem(5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_55.addItem(self.horizontalSpacer_37)


        self.horizontalLayout_9.addLayout(self.horizontalLayout_55)

        self.horizontalLayout_41 = QHBoxLayout()
        self.horizontalLayout_41.setObjectName(u"horizontalLayout_41")
        self.horizontalSpacer_63 = QSpacerItem(5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_41.addItem(self.horizontalSpacer_63)

        self.state_frequency_7 = QDoubleSpinBox(self.settings_tab)
        self.state_frequency_7.setObjectName(u"state_frequency_7")
        self.state_frequency_7.setMaximum(1.000000000000000)
        self.state_frequency_7.setSingleStep(0.010000000000000)

        self.horizontalLayout_41.addWidget(self.state_frequency_7)

        self.horizontalSpacer_62 = QSpacerItem(5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_41.addItem(self.horizontalSpacer_62)


        self.horizontalLayout_9.addLayout(self.horizontalLayout_41)

        self.horizontalLayout_9.setStretch(0, 3)
        self.horizontalLayout_9.setStretch(1, 3)
        self.horizontalLayout_9.setStretch(2, 4)
        self.horizontalLayout_9.setStretch(3, 3)
        self.horizontalLayout_9.setStretch(4, 3)

        self.verticalLayout_6.addLayout(self.horizontalLayout_9)

        self.horizontalLayout_8 = QHBoxLayout()
        self.horizontalLayout_8.setSpacing(10)
        self.horizontalLayout_8.setObjectName(u"horizontalLayout_8")
        self.horizontalLayout_35 = QHBoxLayout()
        self.horizontalLayout_35.setObjectName(u"horizontalLayout_35")
        self.label_5 = QLabel(self.settings_tab)
        self.label_5.setObjectName(u"label_5")
        sizePolicy.setHeightForWidth(self.label_5.sizePolicy().hasHeightForWidth())
        self.label_5.setSizePolicy(sizePolicy)
        self.label_5.setFont(font1)
        self.label_5.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.horizontalLayout_35.addWidget(self.label_5)


        self.horizontalLayout_8.addLayout(self.horizontalLayout_35)

        self.horizontalLayout_49 = QHBoxLayout()
        self.horizontalLayout_49.setObjectName(u"horizontalLayout_49")
        self.horizontalSpacer_40 = QSpacerItem(5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_49.addItem(self.horizontalSpacer_40)

        self.enable_state_8 = QCheckBox(self.settings_tab)
        self.enable_state_8.setObjectName(u"enable_state_8")
        sizePolicy2.setHeightForWidth(self.enable_state_8.sizePolicy().hasHeightForWidth())
        self.enable_state_8.setSizePolicy(sizePolicy2)

        self.horizontalLayout_49.addWidget(self.enable_state_8)

        self.horizontalSpacer_39 = QSpacerItem(5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_49.addItem(self.horizontalSpacer_39)


        self.horizontalLayout_8.addLayout(self.horizontalLayout_49)

        self.state_name_8 = QLineEdit(self.settings_tab)
        self.state_name_8.setObjectName(u"state_name_8")
        sizePolicy3.setHeightForWidth(self.state_name_8.sizePolicy().hasHeightForWidth())
        self.state_name_8.setSizePolicy(sizePolicy3)

        self.horizontalLayout_8.addWidget(self.state_name_8)

        self.horizontalLayout_56 = QHBoxLayout()
        self.horizontalLayout_56.setObjectName(u"horizontalLayout_56")
        self.horizontalSpacer_42 = QSpacerItem(5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_56.addItem(self.horizontalSpacer_42)

        self.state_scored_8 = QCheckBox(self.settings_tab)
        self.state_scored_8.setObjectName(u"state_scored_8")
        sizePolicy2.setHeightForWidth(self.state_scored_8.sizePolicy().hasHeightForWidth())
        self.state_scored_8.setSizePolicy(sizePolicy2)
        self.state_scored_8.setChecked(True)

        self.horizontalLayout_56.addWidget(self.state_scored_8)

        self.horizontalSpacer_41 = QSpacerItem(5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_56.addItem(self.horizontalSpacer_41)


        self.horizontalLayout_8.addLayout(self.horizontalLayout_56)

        self.horizontalLayout_42 = QHBoxLayout()
        self.horizontalLayout_42.setObjectName(u"horizontalLayout_42")
        self.horizontalSpacer_65 = QSpacerItem(5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_42.addItem(self.horizontalSpacer_65)

        self.state_frequency_8 = QDoubleSpinBox(self.settings_tab)
        self.state_frequency_8.setObjectName(u"state_frequency_8")
        self.state_frequency_8.setMaximum(1.000000000000000)
        self.state_frequency_8.setSingleStep(0.010000000000000)

        self.horizontalLayout_42.addWidget(self.state_frequency_8)

        self.horizontalSpacer_64 = QSpacerItem(5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_42.addItem(self.horizontalSpacer_64)


        self.horizontalLayout_8.addLayout(self.horizontalLayout_42)

        self.horizontalLayout_8.setStretch(0, 3)
        self.horizontalLayout_8.setStretch(1, 3)
        self.horizontalLayout_8.setStretch(2, 4)
        self.horizontalLayout_8.setStretch(3, 3)
        self.horizontalLayout_8.setStretch(4, 3)

        self.verticalLayout_6.addLayout(self.horizontalLayout_8)

        self.horizontalLayout_7 = QHBoxLayout()
        self.horizontalLayout_7.setSpacing(10)
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.horizontalLayout_36 = QHBoxLayout()
        self.horizontalLayout_36.setObjectName(u"horizontalLayout_36")
        self.label_4 = QLabel(self.settings_tab)
        self.label_4.setObjectName(u"label_4")
        sizePolicy.setHeightForWidth(self.label_4.sizePolicy().hasHeightForWidth())
        self.label_4.setSizePolicy(sizePolicy)
        self.label_4.setFont(font1)
        self.label_4.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.horizontalLayout_36.addWidget(self.label_4)


        self.horizontalLayout_7.addLayout(self.horizontalLayout_36)

        self.horizontalLayout_50 = QHBoxLayout()
        self.horizontalLayout_50.setObjectName(u"horizontalLayout_50")
        self.horizontalSpacer_44 = QSpacerItem(5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_50.addItem(self.horizontalSpacer_44)

        self.enable_state_9 = QCheckBox(self.settings_tab)
        self.enable_state_9.setObjectName(u"enable_state_9")
        sizePolicy2.setHeightForWidth(self.enable_state_9.sizePolicy().hasHeightForWidth())
        self.enable_state_9.setSizePolicy(sizePolicy2)

        self.horizontalLayout_50.addWidget(self.enable_state_9)

        self.horizontalSpacer_43 = QSpacerItem(5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_50.addItem(self.horizontalSpacer_43)


        self.horizontalLayout_7.addLayout(self.horizontalLayout_50)

        self.state_name_9 = QLineEdit(self.settings_tab)
        self.state_name_9.setObjectName(u"state_name_9")
        sizePolicy3.setHeightForWidth(self.state_name_9.sizePolicy().hasHeightForWidth())
        self.state_name_9.setSizePolicy(sizePolicy3)

        self.horizontalLayout_7.addWidget(self.state_name_9)

        self.horizontalLayout_57 = QHBoxLayout()
        self.horizontalLayout_57.setObjectName(u"horizontalLayout_57")
        self.horizontalSpacer_46 = QSpacerItem(5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_57.addItem(self.horizontalSpacer_46)

        self.state_scored_9 = QCheckBox(self.settings_tab)
        self.state_scored_9.setObjectName(u"state_scored_9")
        sizePolicy2.setHeightForWidth(self.state_scored_9.sizePolicy().hasHeightForWidth())
        self.state_scored_9.setSizePolicy(sizePolicy2)
        self.state_scored_9.setChecked(True)

        self.horizontalLayout_57.addWidget(self.state_scored_9)

        self.horizontalSpacer_45 = QSpacerItem(5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_57.addItem(self.horizontalSpacer_45)


        self.horizontalLayout_7.addLayout(self.horizontalLayout_57)

        self.horizontalLayout_43 = QHBoxLayout()
        self.horizontalLayout_43.setObjectName(u"horizontalLayout_43")
        self.horizontalSpacer_67 = QSpacerItem(5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_43.addItem(self.horizontalSpacer_67)

        self.state_frequency_9 = QDoubleSpinBox(self.settings_tab)
        self.state_frequency_9.setObjectName(u"state_frequency_9")
        self.state_frequency_9.setMaximum(1.000000000000000)
        self.state_frequency_9.setSingleStep(0.010000000000000)

        self.horizontalLayout_43.addWidget(self.state_frequency_9)

        self.horizontalSpacer_66 = QSpacerItem(5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_43.addItem(self.horizontalSpacer_66)


        self.horizontalLayout_7.addLayout(self.horizontalLayout_43)

        self.horizontalLayout_7.setStretch(0, 3)
        self.horizontalLayout_7.setStretch(1, 3)
        self.horizontalLayout_7.setStretch(2, 4)
        self.horizontalLayout_7.setStretch(3, 3)
        self.horizontalLayout_7.setStretch(4, 3)

        self.verticalLayout_6.addLayout(self.horizontalLayout_7)

        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setSpacing(10)
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.horizontalLayout_37 = QHBoxLayout()
        self.horizontalLayout_37.setObjectName(u"horizontalLayout_37")
        self.label_3 = QLabel(self.settings_tab)
        self.label_3.setObjectName(u"label_3")
        sizePolicy.setHeightForWidth(self.label_3.sizePolicy().hasHeightForWidth())
        self.label_3.setSizePolicy(sizePolicy)
        self.label_3.setFont(font1)
        self.label_3.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.horizontalLayout_37.addWidget(self.label_3)


        self.horizontalLayout_4.addLayout(self.horizontalLayout_37)

        self.horizontalLayout_51 = QHBoxLayout()
        self.horizontalLayout_51.setObjectName(u"horizontalLayout_51")
        self.horizontalSpacer_48 = QSpacerItem(5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_51.addItem(self.horizontalSpacer_48)

        self.enable_state_0 = QCheckBox(self.settings_tab)
        self.enable_state_0.setObjectName(u"enable_state_0")
        sizePolicy2.setHeightForWidth(self.enable_state_0.sizePolicy().hasHeightForWidth())
        self.enable_state_0.setSizePolicy(sizePolicy2)

        self.horizontalLayout_51.addWidget(self.enable_state_0)

        self.horizontalSpacer_47 = QSpacerItem(5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_51.addItem(self.horizontalSpacer_47)


        self.horizontalLayout_4.addLayout(self.horizontalLayout_51)

        self.state_name_0 = QLineEdit(self.settings_tab)
        self.state_name_0.setObjectName(u"state_name_0")
        sizePolicy3.setHeightForWidth(self.state_name_0.sizePolicy().hasHeightForWidth())
        self.state_name_0.setSizePolicy(sizePolicy3)

        self.horizontalLayout_4.addWidget(self.state_name_0)

        self.horizontalLayout_58 = QHBoxLayout()
        self.horizontalLayout_58.setObjectName(u"horizontalLayout_58")
        self.horizontalSpacer_49 = QSpacerItem(5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_58.addItem(self.horizontalSpacer_49)

        self.state_scored_0 = QCheckBox(self.settings_tab)
        self.state_scored_0.setObjectName(u"state_scored_0")
        sizePolicy2.setHeightForWidth(self.state_scored_0.sizePolicy().hasHeightForWidth())
        self.state_scored_0.setSizePolicy(sizePolicy2)
        self.state_scored_0.setChecked(True)

        self.horizontalLayout_58.addWidget(self.state_scored_0)

        self.horizontalSpacer_50 = QSpacerItem(5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_58.addItem(self.horizontalSpacer_50)


        self.horizontalLayout_4.addLayout(self.horizontalLayout_58)

        self.horizontalLayout_44 = QHBoxLayout()
        self.horizontalLayout_44.setObjectName(u"horizontalLayout_44")
        self.horizontalSpacer_69 = QSpacerItem(5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_44.addItem(self.horizontalSpacer_69)

        self.state_frequency_0 = QDoubleSpinBox(self.settings_tab)
        self.state_frequency_0.setObjectName(u"state_frequency_0")
        self.state_frequency_0.setMaximum(1.000000000000000)
        self.state_frequency_0.setSingleStep(0.010000000000000)

        self.horizontalLayout_44.addWidget(self.state_frequency_0)

        self.horizontalSpacer_68 = QSpacerItem(5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_44.addItem(self.horizontalSpacer_68)


        self.horizontalLayout_4.addLayout(self.horizontalLayout_44)

        self.horizontalLayout_4.setStretch(0, 3)
        self.horizontalLayout_4.setStretch(1, 3)
        self.horizontalLayout_4.setStretch(2, 4)
        self.horizontalLayout_4.setStretch(3, 3)
        self.horizontalLayout_4.setStretch(4, 3)

        self.verticalLayout_6.addLayout(self.horizontalLayout_4)

        self.horizontalLayout_18 = QHBoxLayout()
        self.horizontalLayout_18.setSpacing(10)
        self.horizontalLayout_18.setObjectName(u"horizontalLayout_18")
        self.horizontalSpacer_9 = QSpacerItem(10, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_18.addItem(self.horizontalSpacer_9)

        self.save_config_button = QPushButton(self.settings_tab)
        self.save_config_button.setObjectName(u"save_config_button")
        sizePolicy2.setHeightForWidth(self.save_config_button.sizePolicy().hasHeightForWidth())
        self.save_config_button.setSizePolicy(sizePolicy2)

        self.horizontalLayout_18.addWidget(self.save_config_button)

        self.save_config_status = QLabel(self.settings_tab)
        self.save_config_status.setObjectName(u"save_config_status")
        sizePolicy3.setHeightForWidth(self.save_config_status.sizePolicy().hasHeightForWidth())
        self.save_config_status.setSizePolicy(sizePolicy3)
        self.save_config_status.setStyleSheet(u"background-color: transparent;")

        self.horizontalLayout_18.addWidget(self.save_config_status)

        self.horizontalLayout_18.setStretch(0, 3)
        self.horizontalLayout_18.setStretch(1, 1)
        self.horizontalLayout_18.setStretch(2, 3)

        self.verticalLayout_6.addLayout(self.horizontalLayout_18)

        self.verticalLayout_6.setStretch(0, 2)
        self.verticalLayout_6.setStretch(2, 2)
        self.verticalLayout_6.setStretch(3, 2)
        self.verticalLayout_6.setStretch(4, 2)
        self.verticalLayout_6.setStretch(5, 2)
        self.verticalLayout_6.setStretch(6, 2)
        self.verticalLayout_6.setStretch(7, 2)
        self.verticalLayout_6.setStretch(8, 2)
        self.verticalLayout_6.setStretch(9, 2)
        self.verticalLayout_6.setStretch(10, 2)
        self.verticalLayout_6.setStretch(11, 3)

        self.settings_tab_layout.addLayout(self.verticalLayout_6, 0, 0, 1, 1)

        self.verticalSpacer = QSpacerItem(5, 30, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.settings_tab_layout.addItem(self.verticalSpacer, 1, 1, 1, 1)

        self.settings_tab_layout.setColumnStretch(0, 1)
        self.settings_tab_layout.setColumnStretch(1, 1)

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
        self.export_button.setText(QCoreApplication.translate("PrimaryWindow", u"export", None))
        self.import_button.setText(QCoreApplication.translate("PrimaryWindow", u"import", None))
#if QT_CONFIG(tooltip)
        self.user_manual_button.setToolTip(QCoreApplication.translate("PrimaryWindow", u"User manual", None))
#endif // QT_CONFIG(tooltip)
        self.user_manual_button.setText("")
        self.accusleepy2.setText(QCoreApplication.translate("PrimaryWindow", u"AccuSleePy", None))
        self.accusleepy3.setText(QCoreApplication.translate("PrimaryWindow", u"AccuSleePy", None))
        self.accusleepy1.setText(QCoreApplication.translate("PrimaryWindow", u"AccuSleePy", None))
        self.upper_tab_widget.setTabText(self.upper_tab_widget.indexOf(self.scoring_tab), QCoreApplication.translate("PrimaryWindow", u"Sleep scoring", None))
        self.settings_text.setText("")
        self.label_15.setText(QCoreApplication.translate("PrimaryWindow", u"Digit", None))
        self.label_14.setText(QCoreApplication.translate("PrimaryWindow", u"Enabled", None))
        self.label_16.setText(QCoreApplication.translate("PrimaryWindow", u"Name", None))
        self.label_13.setText(QCoreApplication.translate("PrimaryWindow", u"Scored", None))
        self.label_18.setText(QCoreApplication.translate("PrimaryWindow", u"Frequency", None))
        self.label_12.setText(QCoreApplication.translate("PrimaryWindow", u"1", None))
        self.enable_state_1.setText("")
        self.state_scored_1.setText("")
        self.label_11.setText(QCoreApplication.translate("PrimaryWindow", u"2", None))
        self.enable_state_2.setText("")
        self.state_scored_2.setText("")
        self.label_10.setText(QCoreApplication.translate("PrimaryWindow", u"3", None))
        self.enable_state_3.setText("")
        self.state_scored_3.setText("")
        self.label_9.setText(QCoreApplication.translate("PrimaryWindow", u"4", None))
        self.enable_state_4.setText("")
        self.state_scored_4.setText("")
        self.label_8.setText(QCoreApplication.translate("PrimaryWindow", u"5", None))
        self.enable_state_5.setText("")
        self.state_scored_5.setText("")
        self.label_7.setText(QCoreApplication.translate("PrimaryWindow", u"6", None))
        self.enable_state_6.setText("")
        self.state_scored_6.setText("")
        self.label_6.setText(QCoreApplication.translate("PrimaryWindow", u"7", None))
        self.enable_state_7.setText("")
        self.state_scored_7.setText("")
        self.label_5.setText(QCoreApplication.translate("PrimaryWindow", u"8", None))
        self.enable_state_8.setText("")
        self.state_scored_8.setText("")
        self.label_4.setText(QCoreApplication.translate("PrimaryWindow", u"9", None))
        self.enable_state_9.setText("")
        self.state_scored_9.setText("")
        self.label_3.setText(QCoreApplication.translate("PrimaryWindow", u"0", None))
        self.enable_state_0.setText("")
        self.state_scored_0.setText("")
        self.save_config_button.setText(QCoreApplication.translate("PrimaryWindow", u"Save", None))
        self.save_config_status.setText("")
        self.upper_tab_widget.setTabText(self.upper_tab_widget.indexOf(self.settings_tab), QCoreApplication.translate("PrimaryWindow", u"Settings", None))
    # retranslateUi

