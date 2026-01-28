# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'primary_window.ui'
##
## Created by: Qt User Interface Compiler version 6.9.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import QCoreApplication, QMetaObject, QRect, QSize, Qt
from PySide6.QtGui import QBrush, QColor, QFont, QIcon, QPalette
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSpacerItem,
    QSpinBox,
    QStackedWidget,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)
import accusleepy.gui.resources_rc  # noqa F401


class Ui_PrimaryWindow(object):
    def setupUi(self, PrimaryWindow):
        if not PrimaryWindow.objectName():
            PrimaryWindow.setObjectName("PrimaryWindow")
        PrimaryWindow.resize(1034, 660)
        sizePolicy = QSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(PrimaryWindow.sizePolicy().hasHeightForWidth())
        PrimaryWindow.setSizePolicy(sizePolicy)
        palette = QPalette()
        brush = QBrush(QColor(223, 226, 226, 255))
        brush.setStyle(Qt.BrushStyle.SolidPattern)
        palette.setBrush(QPalette.ColorGroup.Active, QPalette.ColorRole.Window, brush)
        palette.setBrush(QPalette.ColorGroup.Inactive, QPalette.ColorRole.Window, brush)
        palette.setBrush(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Base, brush)
        palette.setBrush(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Window, brush)
        PrimaryWindow.setPalette(palette)
        self.centralwidget = QWidget(PrimaryWindow)
        self.centralwidget.setObjectName("centralwidget")
        sizePolicy.setHeightForWidth(
            self.centralwidget.sizePolicy().hasHeightForWidth()
        )
        self.centralwidget.setSizePolicy(sizePolicy)
        self.verticalLayout_5 = QVBoxLayout(self.centralwidget)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.verticalLayout_5.setContentsMargins(0, 10, 0, 0)
        self.upper_tab_widget = QTabWidget(self.centralwidget)
        self.upper_tab_widget.setObjectName("upper_tab_widget")
        self.upper_tab_widget.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.scoring_tab = QWidget()
        self.scoring_tab.setObjectName("scoring_tab")
        self.gridLayout_3 = QGridLayout(self.scoring_tab)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.gridLayout_3.setContentsMargins(0, 0, 0, 0)
        self.scoring_tab_layout = QGridLayout()
        self.scoring_tab_layout.setSpacing(20)
        self.scoring_tab_layout.setObjectName("scoring_tab_layout")
        self.scoring_tab_layout.setContentsMargins(10, 10, 10, 10)
        self.left_scoring_vlayout = QVBoxLayout()
        self.left_scoring_vlayout.setSpacing(20)
        self.left_scoring_vlayout.setObjectName("left_scoring_vlayout")
        self.left_scoring_vlayout.setContentsMargins(-1, -1, -1, 20)
        self.epoch_length_layout = QVBoxLayout()
        self.epoch_length_layout.setSpacing(5)
        self.epoch_length_layout.setObjectName("epoch_length_layout")
        self.epoch_length_layout.setContentsMargins(5, -1, -1, -1)
        self.epochlengthlabel = QLabel(self.scoring_tab)
        self.epochlengthlabel.setObjectName("epochlengthlabel")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(
            self.epochlengthlabel.sizePolicy().hasHeightForWidth()
        )
        self.epochlengthlabel.setSizePolicy(sizePolicy1)

        self.epoch_length_layout.addWidget(self.epochlengthlabel)

        self.epoch_length_input = QDoubleSpinBox(self.scoring_tab)
        self.epoch_length_input.setObjectName("epoch_length_input")
        sizePolicy1.setHeightForWidth(
            self.epoch_length_input.sizePolicy().hasHeightForWidth()
        )
        self.epoch_length_input.setSizePolicy(sizePolicy1)
        self.epoch_length_input.setAlignment(
            Qt.AlignmentFlag.AlignLeading
            | Qt.AlignmentFlag.AlignLeft
            | Qt.AlignmentFlag.AlignVCenter
        )
        self.epoch_length_input.setMaximum(100000.000000000000000)
        self.epoch_length_input.setSingleStep(0.500000000000000)

        self.epoch_length_layout.addWidget(self.epoch_length_input)

        self.left_scoring_vlayout.addLayout(self.epoch_length_layout)

        self.recordinglistgroupbox = QGroupBox(self.scoring_tab)
        self.recordinglistgroupbox.setObjectName("recordinglistgroupbox")
        sizePolicy2 = QSizePolicy(
            QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred
        )
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(
            self.recordinglistgroupbox.sizePolicy().hasHeightForWidth()
        )
        self.recordinglistgroupbox.setSizePolicy(sizePolicy2)
        self.recordinglistgroupbox.setStyleSheet("")
        self.verticalLayout = QVBoxLayout(self.recordinglistgroupbox)
        self.verticalLayout.setSpacing(5)
        self.verticalLayout.setObjectName("verticalLayout")
        self.verticalLayout.setContentsMargins(5, 5, 5, 5)
        self.add_remove_layout = QHBoxLayout()
        self.add_remove_layout.setSpacing(20)
        self.add_remove_layout.setObjectName("add_remove_layout")
        self.add_button = QPushButton(self.recordinglistgroupbox)
        self.add_button.setObjectName("add_button")
        sizePolicy1.setHeightForWidth(self.add_button.sizePolicy().hasHeightForWidth())
        self.add_button.setSizePolicy(sizePolicy1)

        self.add_remove_layout.addWidget(self.add_button)

        self.remove_button = QPushButton(self.recordinglistgroupbox)
        self.remove_button.setObjectName("remove_button")
        sizePolicy1.setHeightForWidth(
            self.remove_button.sizePolicy().hasHeightForWidth()
        )
        self.remove_button.setSizePolicy(sizePolicy1)

        self.add_remove_layout.addWidget(self.remove_button)

        self.add_remove_layout.setStretch(0, 1)
        self.add_remove_layout.setStretch(1, 1)

        self.verticalLayout.addLayout(self.add_remove_layout)

        self.recording_list_widget = QListWidget(self.recordinglistgroupbox)
        self.recording_list_widget.setObjectName("recording_list_widget")
        sizePolicy.setHeightForWidth(
            self.recording_list_widget.sizePolicy().hasHeightForWidth()
        )
        self.recording_list_widget.setSizePolicy(sizePolicy)
        self.recording_list_widget.setStyleSheet("background-color: white;")

        self.verticalLayout.addWidget(self.recording_list_widget)

        self.horizontalLayout_59 = QHBoxLayout()
        self.horizontalLayout_59.setObjectName("horizontalLayout_59")
        self.export_button = QPushButton(self.recordinglistgroupbox)
        self.export_button.setObjectName("export_button")
        sizePolicy1.setHeightForWidth(
            self.export_button.sizePolicy().hasHeightForWidth()
        )
        self.export_button.setSizePolicy(sizePolicy1)

        self.horizontalLayout_59.addWidget(self.export_button)

        self.import_button = QPushButton(self.recordinglistgroupbox)
        self.import_button.setObjectName("import_button")
        sizePolicy1.setHeightForWidth(
            self.import_button.sizePolicy().hasHeightForWidth()
        )
        self.import_button.setSizePolicy(sizePolicy1)

        self.horizontalLayout_59.addWidget(self.import_button)

        self.verticalLayout.addLayout(self.horizontalLayout_59)

        self.verticalLayout.setStretch(0, 1)
        self.verticalLayout.setStretch(1, 7)
        self.verticalLayout.setStretch(2, 1)

        self.left_scoring_vlayout.addWidget(self.recordinglistgroupbox)

        self.logo_and_version_layout = QVBoxLayout()
        self.logo_and_version_layout.setObjectName("logo_and_version_layout")
        self.logo_layout = QVBoxLayout()
        self.logo_layout.setObjectName("logo_layout")
        self.frame = QFrame(self.scoring_tab)
        self.frame.setObjectName("frame")
        sizePolicy1.setHeightForWidth(self.frame.sizePolicy().hasHeightForWidth())
        self.frame.setSizePolicy(sizePolicy1)
        self.frame.setMinimumSize(QSize(180, 80))
        self.frame.setStyleSheet("background-color: transparent;")
        self.frame.setFrameShape(QFrame.Shape.NoFrame)
        self.frame.setFrameShadow(QFrame.Shadow.Raised)
        self.accusleepy2 = QLabel(self.frame)
        self.accusleepy2.setObjectName("accusleepy2")
        self.accusleepy2.setGeometry(QRect(11, 15, 160, 50))
        sizePolicy1.setHeightForWidth(self.accusleepy2.sizePolicy().hasHeightForWidth())
        self.accusleepy2.setSizePolicy(sizePolicy1)
        font = QFont()
        font.setPointSize(21)
        font.setBold(True)
        font.setItalic(True)
        self.accusleepy2.setFont(font)
        self.accusleepy2.setStyleSheet(
            "background-color: transparent;\ncolor: rgb(130, 169, 68);"
        )
        self.accusleepy2.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.accusleepy3 = QLabel(self.frame)
        self.accusleepy3.setObjectName("accusleepy3")
        self.accusleepy3.setGeometry(QRect(13, 17, 160, 50))
        sizePolicy1.setHeightForWidth(self.accusleepy3.sizePolicy().hasHeightForWidth())
        self.accusleepy3.setSizePolicy(sizePolicy1)
        self.accusleepy3.setFont(font)
        self.accusleepy3.setStyleSheet(
            "background-color: transparent;\ncolor: rgb(46, 63, 150);"
        )
        self.accusleepy3.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.accusleepy1 = QLabel(self.frame)
        self.accusleepy1.setObjectName("accusleepy1")
        self.accusleepy1.setGeometry(QRect(9, 13, 160, 50))
        sizePolicy1.setHeightForWidth(self.accusleepy1.sizePolicy().hasHeightForWidth())
        self.accusleepy1.setSizePolicy(sizePolicy1)
        self.accusleepy1.setFont(font)
        self.accusleepy1.setStyleSheet(
            "background-color: transparent;\ncolor: rgb(244, 195, 68);"
        )
        self.accusleepy1.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.accusleepy1.raise_()
        self.accusleepy2.raise_()
        self.accusleepy3.raise_()

        self.logo_layout.addWidget(self.frame)

        self.logo_and_version_layout.addLayout(self.logo_layout)

        self.version_label = QLabel(self.scoring_tab)
        self.version_label.setObjectName("version_label")
        sizePolicy3 = QSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed
        )
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(
            self.version_label.sizePolicy().hasHeightForWidth()
        )
        self.version_label.setSizePolicy(sizePolicy3)
        self.version_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.logo_and_version_layout.addWidget(self.version_label)

        self.left_scoring_vlayout.addLayout(self.logo_and_version_layout)

        self.user_manual_layout = QHBoxLayout()
        self.user_manual_layout.setObjectName("user_manual_layout")
        self.user_manual_button = QPushButton(self.scoring_tab)
        self.user_manual_button.setObjectName("user_manual_button")
        sizePolicy1.setHeightForWidth(
            self.user_manual_button.sizePolicy().hasHeightForWidth()
        )
        self.user_manual_button.setSizePolicy(sizePolicy1)
        self.user_manual_button.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        icon = QIcon()
        icon.addFile(
            ":/icons/question.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off
        )
        self.user_manual_button.setIcon(icon)
        self.user_manual_button.setIconSize(QSize(24, 24))

        self.user_manual_layout.addWidget(self.user_manual_button)

        self.left_scoring_vlayout.addLayout(self.user_manual_layout)

        self.scoring_tab_layout.addLayout(self.left_scoring_vlayout, 0, 0, 1, 1)

        self.right_scoring_vlayout = QVBoxLayout()
        self.right_scoring_vlayout.setSpacing(30)
        self.right_scoring_vlayout.setObjectName("right_scoring_vlayout")
        self.upper_right_layout = QVBoxLayout()
        self.upper_right_layout.setSpacing(30)
        self.upper_right_layout.setObjectName("upper_right_layout")
        self.selected_recording_groupbox = QGroupBox(self.scoring_tab)
        self.selected_recording_groupbox.setObjectName("selected_recording_groupbox")
        sizePolicy.setHeightForWidth(
            self.selected_recording_groupbox.sizePolicy().hasHeightForWidth()
        )
        self.selected_recording_groupbox.setSizePolicy(sizePolicy)
        self.verticalLayout_2 = QVBoxLayout(self.selected_recording_groupbox)
        self.verticalLayout_2.setSpacing(10)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(5, 5, 5, 5)
        self.samplingratelayout = QHBoxLayout()
        self.samplingratelayout.setSpacing(10)
        self.samplingratelayout.setObjectName("samplingratelayout")
        self.samplingratelabel = QLabel(self.selected_recording_groupbox)
        self.samplingratelabel.setObjectName("samplingratelabel")
        sizePolicy1.setHeightForWidth(
            self.samplingratelabel.sizePolicy().hasHeightForWidth()
        )
        self.samplingratelabel.setSizePolicy(sizePolicy1)

        self.samplingratelayout.addWidget(self.samplingratelabel)

        self.sampling_rate_input = QDoubleSpinBox(self.selected_recording_groupbox)
        self.sampling_rate_input.setObjectName("sampling_rate_input")
        sizePolicy1.setHeightForWidth(
            self.sampling_rate_input.sizePolicy().hasHeightForWidth()
        )
        self.sampling_rate_input.setSizePolicy(sizePolicy1)
        self.sampling_rate_input.setAlignment(
            Qt.AlignmentFlag.AlignLeading
            | Qt.AlignmentFlag.AlignLeft
            | Qt.AlignmentFlag.AlignVCenter
        )
        self.sampling_rate_input.setMinimum(0.000000000000000)
        self.sampling_rate_input.setMaximum(100000.000000000000000)

        self.samplingratelayout.addWidget(self.sampling_rate_input)

        self.horizontalSpacer_2 = QSpacerItem(
            20, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )

        self.samplingratelayout.addItem(self.horizontalSpacer_2)

        self.samplingratelayout.setStretch(0, 1)
        self.samplingratelayout.setStretch(1, 1)
        self.samplingratelayout.setStretch(2, 7)

        self.verticalLayout_2.addLayout(self.samplingratelayout)

        self.select_recording_layout = QHBoxLayout()
        self.select_recording_layout.setSpacing(10)
        self.select_recording_layout.setObjectName("select_recording_layout")
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setSpacing(5)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.recording_file_button = QPushButton(self.selected_recording_groupbox)
        self.recording_file_button.setObjectName("recording_file_button")
        sizePolicy3.setHeightForWidth(
            self.recording_file_button.sizePolicy().hasHeightForWidth()
        )
        self.recording_file_button.setSizePolicy(sizePolicy3)

        self.horizontalLayout.addWidget(self.recording_file_button)

        self.select_recording_layout.addLayout(self.horizontalLayout)

        self.recording_file_label = QLabel(self.selected_recording_groupbox)
        self.recording_file_label.setObjectName("recording_file_label")
        sizePolicy4 = QSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        sizePolicy4.setHorizontalStretch(0)
        sizePolicy4.setVerticalStretch(0)
        sizePolicy4.setHeightForWidth(
            self.recording_file_label.sizePolicy().hasHeightForWidth()
        )
        self.recording_file_label.setSizePolicy(sizePolicy4)
        self.recording_file_label.setAcceptDrops(True)
        self.recording_file_label.setStyleSheet(
            "background-color: rgb(240, 242, 255); border: 1px solid gray;"
        )
        self.recording_file_label.setAlignment(
            Qt.AlignmentFlag.AlignRight
            | Qt.AlignmentFlag.AlignTrailing
            | Qt.AlignmentFlag.AlignVCenter
        )
        self.recording_file_label.setMargin(4)

        self.select_recording_layout.addWidget(self.recording_file_label)

        self.select_recording_layout.setStretch(0, 5)
        self.select_recording_layout.setStretch(1, 12)

        self.verticalLayout_2.addLayout(self.select_recording_layout)

        self.label_file_layout = QHBoxLayout()
        self.label_file_layout.setSpacing(10)
        self.label_file_layout.setObjectName("label_file_layout")
        self.select_or_create_layout = QHBoxLayout()
        self.select_or_create_layout.setSpacing(5)
        self.select_or_create_layout.setObjectName("select_or_create_layout")
        self.select_label_button = QPushButton(self.selected_recording_groupbox)
        self.select_label_button.setObjectName("select_label_button")
        sizePolicy3.setHeightForWidth(
            self.select_label_button.sizePolicy().hasHeightForWidth()
        )
        self.select_label_button.setSizePolicy(sizePolicy3)
        self.select_label_button.setBaseSize(QSize(0, 0))

        self.select_or_create_layout.addWidget(self.select_label_button)

        self.or_label = QLabel(self.selected_recording_groupbox)
        self.or_label.setObjectName("or_label")
        sizePolicy3.setHeightForWidth(self.or_label.sizePolicy().hasHeightForWidth())
        self.or_label.setSizePolicy(sizePolicy3)
        self.or_label.setStyleSheet("background-color: transparent;")
        self.or_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.select_or_create_layout.addWidget(self.or_label)

        self.create_label_button = QPushButton(self.selected_recording_groupbox)
        self.create_label_button.setObjectName("create_label_button")
        sizePolicy3.setHeightForWidth(
            self.create_label_button.sizePolicy().hasHeightForWidth()
        )
        self.create_label_button.setSizePolicy(sizePolicy3)

        self.select_or_create_layout.addWidget(self.create_label_button)

        self.label_text = QLabel(self.selected_recording_groupbox)
        self.label_text.setObjectName("label_text")
        sizePolicy3.setHeightForWidth(self.label_text.sizePolicy().hasHeightForWidth())
        self.label_text.setSizePolicy(sizePolicy3)
        self.label_text.setStyleSheet("background-color: transparent;")

        self.select_or_create_layout.addWidget(self.label_text)

        self.select_or_create_layout.setStretch(0, 3)
        self.select_or_create_layout.setStretch(1, 1)
        self.select_or_create_layout.setStretch(2, 3)
        self.select_or_create_layout.setStretch(3, 3)

        self.label_file_layout.addLayout(self.select_or_create_layout)

        self.label_file_label = QLabel(self.selected_recording_groupbox)
        self.label_file_label.setObjectName("label_file_label")
        sizePolicy4.setHeightForWidth(
            self.label_file_label.sizePolicy().hasHeightForWidth()
        )
        self.label_file_label.setSizePolicy(sizePolicy4)
        self.label_file_label.setAcceptDrops(True)
        self.label_file_label.setStyleSheet(
            "background-color: rgb(240, 242, 255); border: 1px solid gray;"
        )
        self.label_file_label.setAlignment(
            Qt.AlignmentFlag.AlignRight
            | Qt.AlignmentFlag.AlignTrailing
            | Qt.AlignmentFlag.AlignVCenter
        )
        self.label_file_label.setMargin(4)

        self.label_file_layout.addWidget(self.label_file_label)

        self.label_file_layout.setStretch(0, 5)
        self.label_file_layout.setStretch(1, 12)

        self.verticalLayout_2.addLayout(self.label_file_layout)

        self.manual_scoring_layout = QHBoxLayout()
        self.manual_scoring_layout.setObjectName("manual_scoring_layout")
        self.manual_scoring_button = QPushButton(self.selected_recording_groupbox)
        self.manual_scoring_button.setObjectName("manual_scoring_button")
        sizePolicy3.setHeightForWidth(
            self.manual_scoring_button.sizePolicy().hasHeightForWidth()
        )
        self.manual_scoring_button.setSizePolicy(sizePolicy3)

        self.manual_scoring_layout.addWidget(self.manual_scoring_button)

        self.manual_scoring_status = QLabel(self.selected_recording_groupbox)
        self.manual_scoring_status.setObjectName("manual_scoring_status")
        sizePolicy3.setHeightForWidth(
            self.manual_scoring_status.sizePolicy().hasHeightForWidth()
        )
        self.manual_scoring_status.setSizePolicy(sizePolicy3)
        self.manual_scoring_status.setStyleSheet("background-color: transparent;")
        self.manual_scoring_status.setMargin(4)

        self.manual_scoring_layout.addWidget(self.manual_scoring_status)

        self.create_calibration_button = QPushButton(self.selected_recording_groupbox)
        self.create_calibration_button.setObjectName("create_calibration_button")
        sizePolicy3.setHeightForWidth(
            self.create_calibration_button.sizePolicy().hasHeightForWidth()
        )
        self.create_calibration_button.setSizePolicy(sizePolicy3)

        self.manual_scoring_layout.addWidget(self.create_calibration_button)

        self.calibration_status = QLabel(self.selected_recording_groupbox)
        self.calibration_status.setObjectName("calibration_status")
        sizePolicy3.setHeightForWidth(
            self.calibration_status.sizePolicy().hasHeightForWidth()
        )
        self.calibration_status.setSizePolicy(sizePolicy3)
        self.calibration_status.setStyleSheet("background-color: transparent;")
        self.calibration_status.setMargin(4)

        self.manual_scoring_layout.addWidget(self.calibration_status)

        self.horizontalSpacer_4 = QSpacerItem(
            10, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )

        self.manual_scoring_layout.addItem(self.horizontalSpacer_4)

        self.manual_scoring_layout.setStretch(0, 2)
        self.manual_scoring_layout.setStretch(1, 3)
        self.manual_scoring_layout.setStretch(2, 2)
        self.manual_scoring_layout.setStretch(3, 3)
        self.manual_scoring_layout.setStretch(4, 1)

        self.verticalLayout_2.addLayout(self.manual_scoring_layout)

        self.load_calibration_layout = QHBoxLayout()
        self.load_calibration_layout.setSpacing(10)
        self.load_calibration_layout.setObjectName("load_calibration_layout")
        self.horizontalLayout_10 = QHBoxLayout()
        self.horizontalLayout_10.setSpacing(5)
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        self.select_calibration_button = QPushButton(self.selected_recording_groupbox)
        self.select_calibration_button.setObjectName("select_calibration_button")
        sizePolicy3.setHeightForWidth(
            self.select_calibration_button.sizePolicy().hasHeightForWidth()
        )
        self.select_calibration_button.setSizePolicy(sizePolicy3)

        self.horizontalLayout_10.addWidget(self.select_calibration_button)

        self.load_calibration_layout.addLayout(self.horizontalLayout_10)

        self.calibration_file_label = QLabel(self.selected_recording_groupbox)
        self.calibration_file_label.setObjectName("calibration_file_label")
        sizePolicy4.setHeightForWidth(
            self.calibration_file_label.sizePolicy().hasHeightForWidth()
        )
        self.calibration_file_label.setSizePolicy(sizePolicy4)
        self.calibration_file_label.setAcceptDrops(True)
        self.calibration_file_label.setStyleSheet(
            "background-color: rgb(240, 242, 255); border: 1px solid gray;"
        )
        self.calibration_file_label.setAlignment(
            Qt.AlignmentFlag.AlignRight
            | Qt.AlignmentFlag.AlignTrailing
            | Qt.AlignmentFlag.AlignVCenter
        )
        self.calibration_file_label.setMargin(4)

        self.load_calibration_layout.addWidget(self.calibration_file_label)

        self.load_calibration_layout.setStretch(0, 5)
        self.load_calibration_layout.setStretch(1, 12)

        self.verticalLayout_2.addLayout(self.load_calibration_layout)

        self.verticalLayout_2.setStretch(0, 1)
        self.verticalLayout_2.setStretch(1, 1)
        self.verticalLayout_2.setStretch(2, 1)
        self.verticalLayout_2.setStretch(3, 1)
        self.verticalLayout_2.setStretch(4, 1)

        self.upper_right_layout.addWidget(self.selected_recording_groupbox)

        self.lower_tab_widget = QTabWidget(self.scoring_tab)
        self.lower_tab_widget.setObjectName("lower_tab_widget")
        sizePolicy.setHeightForWidth(
            self.lower_tab_widget.sizePolicy().hasHeightForWidth()
        )
        self.lower_tab_widget.setSizePolicy(sizePolicy)
        self.classification_tab = QWidget()
        self.classification_tab.setObjectName("classification_tab")
        self.classification_tab.setStyleSheet("")
        self.gridLayout = QGridLayout(self.classification_tab)
        self.gridLayout.setObjectName("gridLayout")
        self.gridLayout.setVerticalSpacing(10)
        self.gridLayout.setContentsMargins(5, 5, 5, 5)
        self.gridLayout_4 = QGridLayout()
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.gridLayout_4.setVerticalSpacing(10)
        self.gridLayout_4.setContentsMargins(0, 5, 0, 3)
        self.load_model_layout = QHBoxLayout()
        self.load_model_layout.setSpacing(10)
        self.load_model_layout.setObjectName("load_model_layout")
        self.horizontalLayout_11 = QHBoxLayout()
        self.horizontalLayout_11.setSpacing(5)
        self.horizontalLayout_11.setObjectName("horizontalLayout_11")
        self.load_model_button = QPushButton(self.classification_tab)
        self.load_model_button.setObjectName("load_model_button")
        sizePolicy3.setHeightForWidth(
            self.load_model_button.sizePolicy().hasHeightForWidth()
        )
        self.load_model_button.setSizePolicy(sizePolicy3)

        self.horizontalLayout_11.addWidget(self.load_model_button)

        self.load_model_layout.addLayout(self.horizontalLayout_11)

        self.model_label = QLabel(self.classification_tab)
        self.model_label.setObjectName("model_label")
        sizePolicy4.setHeightForWidth(self.model_label.sizePolicy().hasHeightForWidth())
        self.model_label.setSizePolicy(sizePolicy4)
        self.model_label.setAcceptDrops(True)
        self.model_label.setStyleSheet(
            "background-color: rgb(240, 242, 255); border: 1px solid gray;"
        )
        self.model_label.setAlignment(
            Qt.AlignmentFlag.AlignRight
            | Qt.AlignmentFlag.AlignTrailing
            | Qt.AlignmentFlag.AlignVCenter
        )
        self.model_label.setMargin(4)

        self.load_model_layout.addWidget(self.model_label)

        self.load_model_layout.setStretch(0, 5)
        self.load_model_layout.setStretch(1, 12)

        self.gridLayout_4.addLayout(self.load_model_layout, 0, 0, 1, 1)

        self.score_all_layout = QHBoxLayout()
        self.score_all_layout.setObjectName("score_all_layout")
        self.score_all_button = QPushButton(self.classification_tab)
        self.score_all_button.setObjectName("score_all_button")
        sizePolicy3.setHeightForWidth(
            self.score_all_button.sizePolicy().hasHeightForWidth()
        )
        self.score_all_button.setSizePolicy(sizePolicy3)

        self.score_all_layout.addWidget(self.score_all_button)

        self.score_all_status = QLabel(self.classification_tab)
        self.score_all_status.setObjectName("score_all_status")
        sizePolicy3.setHeightForWidth(
            self.score_all_status.sizePolicy().hasHeightForWidth()
        )
        self.score_all_status.setSizePolicy(sizePolicy3)
        self.score_all_status.setStyleSheet("background-color: transparent;")
        self.score_all_status.setMargin(4)

        self.score_all_layout.addWidget(self.score_all_status)

        self.horizontalSpacer_5 = QSpacerItem(
            10, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )

        self.score_all_layout.addItem(self.horizontalSpacer_5)

        self.score_all_layout.setStretch(0, 3)
        self.score_all_layout.setStretch(1, 3)
        self.score_all_layout.setStretch(2, 4)

        self.gridLayout_4.addLayout(self.score_all_layout, 2, 0, 1, 1)

        self.classification_options_layout = QHBoxLayout()
        self.classification_options_layout.setSpacing(10)
        self.classification_options_layout.setObjectName(
            "classification_options_layout"
        )
        self.overwritecheckbox = QCheckBox(self.classification_tab)
        self.overwritecheckbox.setObjectName("overwritecheckbox")
        sizePolicy1.setHeightForWidth(
            self.overwritecheckbox.sizePolicy().hasHeightForWidth()
        )
        self.overwritecheckbox.setSizePolicy(sizePolicy1)

        self.classification_options_layout.addWidget(self.overwritecheckbox)

        self.horizontalSpacer_77 = QSpacerItem(
            10, 10, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )

        self.classification_options_layout.addItem(self.horizontalSpacer_77)

        self.save_confidence_checkbox = QCheckBox(self.classification_tab)
        self.save_confidence_checkbox.setObjectName("save_confidence_checkbox")
        sizePolicy1.setHeightForWidth(
            self.save_confidence_checkbox.sizePolicy().hasHeightForWidth()
        )
        self.save_confidence_checkbox.setSizePolicy(sizePolicy1)

        self.classification_options_layout.addWidget(self.save_confidence_checkbox)

        self.horizontalSpacer_76 = QSpacerItem(
            10, 10, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )

        self.classification_options_layout.addItem(self.horizontalSpacer_76)

        self.boutlengthlayout = QHBoxLayout()
        self.boutlengthlayout.setSpacing(5)
        self.boutlengthlayout.setObjectName("boutlengthlayout")
        self.horizontalSpacer_74 = QSpacerItem(
            10, 10, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
        )

        self.boutlengthlayout.addItem(self.horizontalSpacer_74)

        self.boutlengthlabel = QLabel(self.classification_tab)
        self.boutlengthlabel.setObjectName("boutlengthlabel")
        sizePolicy1.setHeightForWidth(
            self.boutlengthlabel.sizePolicy().hasHeightForWidth()
        )
        self.boutlengthlabel.setSizePolicy(sizePolicy1)
        self.boutlengthlabel.setStyleSheet("background-color: transparent;")

        self.boutlengthlayout.addWidget(self.boutlengthlabel)

        self.bout_length_input = QDoubleSpinBox(self.classification_tab)
        self.bout_length_input.setObjectName("bout_length_input")
        sizePolicy1.setHeightForWidth(
            self.bout_length_input.sizePolicy().hasHeightForWidth()
        )
        self.bout_length_input.setSizePolicy(sizePolicy1)
        self.bout_length_input.setAlignment(
            Qt.AlignmentFlag.AlignLeading
            | Qt.AlignmentFlag.AlignLeft
            | Qt.AlignmentFlag.AlignVCenter
        )
        self.bout_length_input.setDecimals(2)
        self.bout_length_input.setMaximum(1000.000000000000000)
        self.bout_length_input.setValue(5.000000000000000)

        self.boutlengthlayout.addWidget(self.bout_length_input)

        self.horizontalSpacer_75 = QSpacerItem(
            10, 10, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
        )

        self.boutlengthlayout.addItem(self.horizontalSpacer_75)

        self.classification_options_layout.addLayout(self.boutlengthlayout)

        self.classification_options_layout.setStretch(0, 5)
        self.classification_options_layout.setStretch(1, 1)
        self.classification_options_layout.setStretch(2, 5)
        self.classification_options_layout.setStretch(3, 1)
        self.classification_options_layout.setStretch(4, 7)

        self.gridLayout_4.addLayout(self.classification_options_layout, 1, 0, 1, 1)

        self.gridLayout.addLayout(self.gridLayout_4, 0, 0, 1, 1)

        self.gridLayout.setColumnStretch(0, 2)
        self.lower_tab_widget.addTab(self.classification_tab, "")
        self.model_training_tab = QWidget()
        self.model_training_tab.setObjectName("model_training_tab")
        self.gridLayout_7 = QGridLayout(self.model_training_tab)
        self.gridLayout_7.setObjectName("gridLayout_7")
        self.gridLayout_7.setContentsMargins(0, 0, 0, 0)
        self.model_training_layout = QGridLayout()
        self.model_training_layout.setObjectName("model_training_layout")
        self.model_training_layout.setVerticalSpacing(10)
        self.model_training_layout.setContentsMargins(5, 5, 5, 25)
        self.top_training_layout = QHBoxLayout()
        self.top_training_layout.setSpacing(10)
        self.top_training_layout.setObjectName("top_training_layout")
        self.horizontalLayout_5 = QHBoxLayout()
        self.horizontalLayout_5.setSpacing(5)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.label = QLabel(self.model_training_tab)
        self.label.setObjectName("label")
        sizePolicy1.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy1)

        self.horizontalLayout_5.addWidget(self.label)

        self.image_number_input = QSpinBox(self.model_training_tab)
        self.image_number_input.setObjectName("image_number_input")
        sizePolicy1.setHeightForWidth(
            self.image_number_input.sizePolicy().hasHeightForWidth()
        )
        self.image_number_input.setSizePolicy(sizePolicy1)
        self.image_number_input.setMinimum(9)
        self.image_number_input.setMaximum(999)
        self.image_number_input.setValue(9)

        self.horizontalLayout_5.addWidget(self.image_number_input)

        self.top_training_layout.addLayout(self.horizontalLayout_5)

        self.horizontalSpacer = QSpacerItem(
            10, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )

        self.top_training_layout.addItem(self.horizontalSpacer)

        self.delete_image_box = QCheckBox(self.model_training_tab)
        self.delete_image_box.setObjectName("delete_image_box")
        sizePolicy1.setHeightForWidth(
            self.delete_image_box.sizePolicy().hasHeightForWidth()
        )
        self.delete_image_box.setSizePolicy(sizePolicy1)
        self.delete_image_box.setChecked(True)

        self.top_training_layout.addWidget(self.delete_image_box)

        self.horizontalSpacer_6 = QSpacerItem(
            10, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )

        self.top_training_layout.addItem(self.horizontalSpacer_6)

        self.horizontalLayout_6 = QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.label_2 = QLabel(self.model_training_tab)
        self.label_2.setObjectName("label_2")
        sizePolicy1.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy1)

        self.horizontalLayout_6.addWidget(self.label_2)

        self.default_type_button = QRadioButton(self.model_training_tab)
        self.default_type_button.setObjectName("default_type_button")
        sizePolicy1.setHeightForWidth(
            self.default_type_button.sizePolicy().hasHeightForWidth()
        )
        self.default_type_button.setSizePolicy(sizePolicy1)
        self.default_type_button.setChecked(True)

        self.horizontalLayout_6.addWidget(self.default_type_button)

        self.real_time_button = QRadioButton(self.model_training_tab)
        self.real_time_button.setObjectName("real_time_button")
        sizePolicy1.setHeightForWidth(
            self.real_time_button.sizePolicy().hasHeightForWidth()
        )
        self.real_time_button.setSizePolicy(sizePolicy1)

        self.horizontalLayout_6.addWidget(self.real_time_button)

        self.horizontalLayout_6.setStretch(0, 2)
        self.horizontalLayout_6.setStretch(1, 3)
        self.horizontalLayout_6.setStretch(2, 3)

        self.top_training_layout.addLayout(self.horizontalLayout_6)

        self.horizontalSpacer_3 = QSpacerItem(
            10, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )

        self.top_training_layout.addItem(self.horizontalSpacer_3)

        self.top_training_layout.setStretch(0, 2)
        self.top_training_layout.setStretch(1, 1)
        self.top_training_layout.setStretch(2, 2)
        self.top_training_layout.setStretch(3, 1)
        self.top_training_layout.setStretch(4, 3)
        self.top_training_layout.setStretch(5, 1)

        self.model_training_layout.addLayout(self.top_training_layout, 0, 0, 1, 1)

        self.bottom_training_layout = QHBoxLayout()
        self.bottom_training_layout.setObjectName("bottom_training_layout")
        self.train_model_button = QPushButton(self.model_training_tab)
        self.train_model_button.setObjectName("train_model_button")
        sizePolicy3.setHeightForWidth(
            self.train_model_button.sizePolicy().hasHeightForWidth()
        )
        self.train_model_button.setSizePolicy(sizePolicy3)

        self.bottom_training_layout.addWidget(self.train_model_button)

        self.horizontalSpacer_8 = QSpacerItem(
            10, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )

        self.bottom_training_layout.addItem(self.horizontalSpacer_8)

        self.calibrate_checkbox = QCheckBox(self.model_training_tab)
        self.calibrate_checkbox.setObjectName("calibrate_checkbox")
        sizePolicy1.setHeightForWidth(
            self.calibrate_checkbox.sizePolicy().hasHeightForWidth()
        )
        self.calibrate_checkbox.setSizePolicy(sizePolicy1)
        self.calibrate_checkbox.setChecked(True)

        self.bottom_training_layout.addWidget(self.calibrate_checkbox)

        self.horizontalSpacer_7 = QSpacerItem(
            10, 10, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )

        self.bottom_training_layout.addItem(self.horizontalSpacer_7)

        self.calibrate_label = QLabel(self.model_training_tab)
        self.calibrate_label.setObjectName("calibrate_label")
        sizePolicy1.setHeightForWidth(
            self.calibrate_label.sizePolicy().hasHeightForWidth()
        )
        self.calibrate_label.setSizePolicy(sizePolicy1)

        self.bottom_training_layout.addWidget(self.calibrate_label)

        self.calibration_spinbox = QSpinBox(self.model_training_tab)
        self.calibration_spinbox.setObjectName("calibration_spinbox")
        sizePolicy1.setHeightForWidth(
            self.calibration_spinbox.sizePolicy().hasHeightForWidth()
        )
        self.calibration_spinbox.setSizePolicy(sizePolicy1)
        self.calibration_spinbox.setMinimum(10)
        self.calibration_spinbox.setMaximum(50)
        self.calibration_spinbox.setValue(15)

        self.bottom_training_layout.addWidget(self.calibration_spinbox)

        self.horizontalSpacer_78 = QSpacerItem(
            10, 10, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )

        self.bottom_training_layout.addItem(self.horizontalSpacer_78)

        self.bottom_training_layout.setStretch(0, 6)
        self.bottom_training_layout.setStretch(1, 5)
        self.bottom_training_layout.setStretch(2, 3)
        self.bottom_training_layout.setStretch(3, 1)
        self.bottom_training_layout.setStretch(4, 3)
        self.bottom_training_layout.setStretch(5, 1)
        self.bottom_training_layout.setStretch(6, 1)

        self.model_training_layout.addLayout(self.bottom_training_layout, 1, 0, 1, 1)

        self.model_training_layout.setRowStretch(0, 1)
        self.model_training_layout.setRowStretch(1, 1)

        self.gridLayout_7.addLayout(self.model_training_layout, 0, 0, 1, 1)

        self.lower_tab_widget.addTab(self.model_training_tab, "")

        self.upper_right_layout.addWidget(self.lower_tab_widget)

        self.right_scoring_vlayout.addLayout(self.upper_right_layout)

        self.messagesgroupbox = QGroupBox(self.scoring_tab)
        self.messagesgroupbox.setObjectName("messagesgroupbox")
        sizePolicy5 = QSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding
        )
        sizePolicy5.setHorizontalStretch(0)
        sizePolicy5.setVerticalStretch(0)
        sizePolicy5.setHeightForWidth(
            self.messagesgroupbox.sizePolicy().hasHeightForWidth()
        )
        self.messagesgroupbox.setSizePolicy(sizePolicy5)
        self.messagesgroupbox.setStyleSheet("")
        self.gridLayout_2 = QGridLayout(self.messagesgroupbox)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.gridLayout_2.setContentsMargins(5, 5, 5, 5)
        self.message_area = QTextBrowser(self.messagesgroupbox)
        self.message_area.setObjectName("message_area")
        sizePolicy5.setHeightForWidth(
            self.message_area.sizePolicy().hasHeightForWidth()
        )
        self.message_area.setSizePolicy(sizePolicy5)
        self.message_area.setTextInteractionFlags(
            Qt.TextInteractionFlag.NoTextInteraction
        )

        self.gridLayout_2.addWidget(self.message_area, 0, 0, 1, 1)

        self.right_scoring_vlayout.addWidget(self.messagesgroupbox)

        self.scoring_tab_layout.addLayout(self.right_scoring_vlayout, 0, 1, 1, 1)

        self.scoring_tab_layout.setRowStretch(0, 2)
        self.scoring_tab_layout.setColumnStretch(0, 1)
        self.scoring_tab_layout.setColumnStretch(1, 6)

        self.gridLayout_3.addLayout(self.scoring_tab_layout, 0, 0, 1, 1)

        self.upper_tab_widget.addTab(self.scoring_tab, "")
        self.settings_tab = QWidget()
        self.settings_tab.setObjectName("settings_tab")
        self.gridLayout_5 = QGridLayout(self.settings_tab)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.gridLayout_5.setContentsMargins(0, 0, 0, 0)
        self.settings_tab_layout = QGridLayout()
        self.settings_tab_layout.setObjectName("settings_tab_layout")
        self.settings_tab_layout.setHorizontalSpacing(20)
        self.settings_tab_layout.setVerticalSpacing(10)
        self.settings_tab_layout.setContentsMargins(20, 20, 20, -1)
        self.verticalLayout_3 = QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.saveconfig_layout = QHBoxLayout()
        self.saveconfig_layout.setObjectName("saveconfig_layout")
        self.horizontalSpacer_79 = QSpacerItem(
            40, 20, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )

        self.saveconfig_layout.addItem(self.horizontalSpacer_79)

        self.save_config_button = QPushButton(self.settings_tab)
        self.save_config_button.setObjectName("save_config_button")
        sizePolicy1.setHeightForWidth(
            self.save_config_button.sizePolicy().hasHeightForWidth()
        )
        self.save_config_button.setSizePolicy(sizePolicy1)

        self.saveconfig_layout.addWidget(self.save_config_button)

        self.save_config_status = QLabel(self.settings_tab)
        self.save_config_status.setObjectName("save_config_status")
        sizePolicy3.setHeightForWidth(
            self.save_config_status.sizePolicy().hasHeightForWidth()
        )
        self.save_config_status.setSizePolicy(sizePolicy3)
        self.save_config_status.setStyleSheet("background-color: transparent;")

        self.saveconfig_layout.addWidget(self.save_config_status)

        self.saveconfig_layout.setStretch(0, 10)
        self.saveconfig_layout.setStretch(1, 2)
        self.saveconfig_layout.setStretch(2, 10)

        self.verticalLayout_3.addLayout(self.saveconfig_layout)

        self.verticalSpacer_2 = QSpacerItem(
            20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding
        )

        self.verticalLayout_3.addItem(self.verticalSpacer_2)

        self.settings_tab_layout.addLayout(self.verticalLayout_3, 2, 0, 1, 1)

        self.settings_layout = QVBoxLayout()
        self.settings_layout.setObjectName("settings_layout")
        self.settings_combo_layout = QHBoxLayout()
        self.settings_combo_layout.setSpacing(20)
        self.settings_combo_layout.setObjectName("settings_combo_layout")
        self.settings_combobox = QComboBox(self.settings_tab)
        self.settings_combobox.addItem("")
        self.settings_combobox.addItem("")
        self.settings_combobox.addItem("")
        self.settings_combobox.addItem("")
        self.settings_combobox.setObjectName("settings_combobox")
        sizePolicy3.setHeightForWidth(
            self.settings_combobox.sizePolicy().hasHeightForWidth()
        )
        self.settings_combobox.setSizePolicy(sizePolicy3)

        self.settings_combo_layout.addWidget(self.settings_combobox)

        self.horizontalSpacer_89 = QSpacerItem(
            5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )

        self.settings_combo_layout.addItem(self.horizontalSpacer_89)

        self.settings_combo_layout.setStretch(0, 1)
        self.settings_combo_layout.setStretch(1, 3)

        self.settings_layout.addLayout(self.settings_combo_layout)

        self.settings_stack = QStackedWidget(self.settings_tab)
        self.settings_stack.setObjectName("settings_stack")
        self.brain_state_page = QWidget()
        self.brain_state_page.setObjectName("brain_state_page")
        self.horizontalLayout_80 = QHBoxLayout(self.brain_state_page)
        self.horizontalLayout_80.setSpacing(20)
        self.horizontalLayout_80.setObjectName("horizontalLayout_80")
        self.settings_controls_layout = QVBoxLayout()
        self.settings_controls_layout.setSpacing(10)
        self.settings_controls_layout.setObjectName("settings_controls_layout")
        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setSpacing(10)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_15 = QLabel(self.brain_state_page)
        self.label_15.setObjectName("label_15")
        sizePolicy3.setHeightForWidth(self.label_15.sizePolicy().hasHeightForWidth())
        self.label_15.setSizePolicy(sizePolicy3)
        self.label_15.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.horizontalLayout_3.addWidget(self.label_15)

        self.label_14 = QLabel(self.brain_state_page)
        self.label_14.setObjectName("label_14")
        sizePolicy3.setHeightForWidth(self.label_14.sizePolicy().hasHeightForWidth())
        self.label_14.setSizePolicy(sizePolicy3)
        self.label_14.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.horizontalLayout_3.addWidget(self.label_14)

        self.label_16 = QLabel(self.brain_state_page)
        self.label_16.setObjectName("label_16")
        sizePolicy3.setHeightForWidth(self.label_16.sizePolicy().hasHeightForWidth())
        self.label_16.setSizePolicy(sizePolicy3)
        self.label_16.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.horizontalLayout_3.addWidget(self.label_16)

        self.label_13 = QLabel(self.brain_state_page)
        self.label_13.setObjectName("label_13")
        sizePolicy3.setHeightForWidth(self.label_13.sizePolicy().hasHeightForWidth())
        self.label_13.setSizePolicy(sizePolicy3)
        self.label_13.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.horizontalLayout_3.addWidget(self.label_13)

        self.label_18 = QLabel(self.brain_state_page)
        self.label_18.setObjectName("label_18")
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
        self.horizontalLayout_3.setStretch(4, 4)

        self.settings_controls_layout.addLayout(self.horizontalLayout_3)

        self.horizontalLayout_17 = QHBoxLayout()
        self.horizontalLayout_17.setSpacing(10)
        self.horizontalLayout_17.setObjectName("horizontalLayout_17")
        self.horizontalLayout_19 = QHBoxLayout()
        self.horizontalLayout_19.setObjectName("horizontalLayout_19")
        self.label_12 = QLabel(self.brain_state_page)
        self.label_12.setObjectName("label_12")
        sizePolicy.setHeightForWidth(self.label_12.sizePolicy().hasHeightForWidth())
        self.label_12.setSizePolicy(sizePolicy)
        font1 = QFont()
        font1.setPointSize(16)
        self.label_12.setFont(font1)
        self.label_12.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.horizontalLayout_19.addWidget(self.label_12)

        self.horizontalLayout_17.addLayout(self.horizontalLayout_19)

        self.horizontalLayout_20 = QHBoxLayout()
        self.horizontalLayout_20.setObjectName("horizontalLayout_20")
        self.horizontalSpacer_12 = QSpacerItem(
            5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )

        self.horizontalLayout_20.addItem(self.horizontalSpacer_12)

        self.enable_state_1 = QCheckBox(self.brain_state_page)
        self.enable_state_1.setObjectName("enable_state_1")
        sizePolicy1.setHeightForWidth(
            self.enable_state_1.sizePolicy().hasHeightForWidth()
        )
        self.enable_state_1.setSizePolicy(sizePolicy1)

        self.horizontalLayout_20.addWidget(self.enable_state_1)

        self.horizontalSpacer_11 = QSpacerItem(
            5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )

        self.horizontalLayout_20.addItem(self.horizontalSpacer_11)

        self.horizontalLayout_17.addLayout(self.horizontalLayout_20)

        self.state_name_1 = QLineEdit(self.brain_state_page)
        self.state_name_1.setObjectName("state_name_1")
        sizePolicy3.setHeightForWidth(
            self.state_name_1.sizePolicy().hasHeightForWidth()
        )
        self.state_name_1.setSizePolicy(sizePolicy3)
        self.state_name_1.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self.horizontalLayout_17.addWidget(self.state_name_1)

        self.horizontalLayout_21 = QHBoxLayout()
        self.horizontalLayout_21.setObjectName("horizontalLayout_21")
        self.horizontalSpacer_14 = QSpacerItem(
            5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )

        self.horizontalLayout_21.addItem(self.horizontalSpacer_14)

        self.state_scored_1 = QCheckBox(self.brain_state_page)
        self.state_scored_1.setObjectName("state_scored_1")
        sizePolicy1.setHeightForWidth(
            self.state_scored_1.sizePolicy().hasHeightForWidth()
        )
        self.state_scored_1.setSizePolicy(sizePolicy1)
        self.state_scored_1.setChecked(True)

        self.horizontalLayout_21.addWidget(self.state_scored_1)

        self.horizontalSpacer_13 = QSpacerItem(
            5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )

        self.horizontalLayout_21.addItem(self.horizontalSpacer_13)

        self.horizontalLayout_17.addLayout(self.horizontalLayout_21)

        self.horizontalLayout_22 = QHBoxLayout()
        self.horizontalLayout_22.setObjectName("horizontalLayout_22")
        self.horizontalSpacer_10 = QSpacerItem(
            5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )

        self.horizontalLayout_22.addItem(self.horizontalSpacer_10)

        self.state_frequency_1 = QDoubleSpinBox(self.brain_state_page)
        self.state_frequency_1.setObjectName("state_frequency_1")
        self.state_frequency_1.setMaximum(1.000000000000000)
        self.state_frequency_1.setSingleStep(0.010000000000000)

        self.horizontalLayout_22.addWidget(self.state_frequency_1)

        self.horizontalSpacer_51 = QSpacerItem(
            5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )

        self.horizontalLayout_22.addItem(self.horizontalSpacer_51)

        self.horizontalLayout_17.addLayout(self.horizontalLayout_22)

        self.horizontalLayout_17.setStretch(0, 3)
        self.horizontalLayout_17.setStretch(1, 3)
        self.horizontalLayout_17.setStretch(2, 4)
        self.horizontalLayout_17.setStretch(3, 3)
        self.horizontalLayout_17.setStretch(4, 4)

        self.settings_controls_layout.addLayout(self.horizontalLayout_17)

        self.horizontalLayout_16 = QHBoxLayout()
        self.horizontalLayout_16.setSpacing(10)
        self.horizontalLayout_16.setObjectName("horizontalLayout_16")
        self.horizontalLayout_23 = QHBoxLayout()
        self.horizontalLayout_23.setObjectName("horizontalLayout_23")
        self.label_11 = QLabel(self.brain_state_page)
        self.label_11.setObjectName("label_11")
        sizePolicy.setHeightForWidth(self.label_11.sizePolicy().hasHeightForWidth())
        self.label_11.setSizePolicy(sizePolicy)
        self.label_11.setFont(font1)
        self.label_11.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.horizontalLayout_23.addWidget(self.label_11)

        self.horizontalLayout_16.addLayout(self.horizontalLayout_23)

        self.horizontalLayout_24 = QHBoxLayout()
        self.horizontalLayout_24.setObjectName("horizontalLayout_24")
        self.horizontalSpacer_16 = QSpacerItem(
            5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )

        self.horizontalLayout_24.addItem(self.horizontalSpacer_16)

        self.enable_state_2 = QCheckBox(self.brain_state_page)
        self.enable_state_2.setObjectName("enable_state_2")
        sizePolicy1.setHeightForWidth(
            self.enable_state_2.sizePolicy().hasHeightForWidth()
        )
        self.enable_state_2.setSizePolicy(sizePolicy1)

        self.horizontalLayout_24.addWidget(self.enable_state_2)

        self.horizontalSpacer_15 = QSpacerItem(
            5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )

        self.horizontalLayout_24.addItem(self.horizontalSpacer_15)

        self.horizontalLayout_16.addLayout(self.horizontalLayout_24)

        self.state_name_2 = QLineEdit(self.brain_state_page)
        self.state_name_2.setObjectName("state_name_2")
        sizePolicy3.setHeightForWidth(
            self.state_name_2.sizePolicy().hasHeightForWidth()
        )
        self.state_name_2.setSizePolicy(sizePolicy3)

        self.horizontalLayout_16.addWidget(self.state_name_2)

        self.horizontalLayout_26 = QHBoxLayout()
        self.horizontalLayout_26.setObjectName("horizontalLayout_26")
        self.horizontalSpacer_18 = QSpacerItem(
            5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )

        self.horizontalLayout_26.addItem(self.horizontalSpacer_18)

        self.state_scored_2 = QCheckBox(self.brain_state_page)
        self.state_scored_2.setObjectName("state_scored_2")
        sizePolicy1.setHeightForWidth(
            self.state_scored_2.sizePolicy().hasHeightForWidth()
        )
        self.state_scored_2.setSizePolicy(sizePolicy1)
        self.state_scored_2.setChecked(True)

        self.horizontalLayout_26.addWidget(self.state_scored_2)

        self.horizontalSpacer_17 = QSpacerItem(
            5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )

        self.horizontalLayout_26.addItem(self.horizontalSpacer_17)

        self.horizontalLayout_16.addLayout(self.horizontalLayout_26)

        self.horizontalLayout_25 = QHBoxLayout()
        self.horizontalLayout_25.setObjectName("horizontalLayout_25")
        self.horizontalSpacer_52 = QSpacerItem(
            5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )

        self.horizontalLayout_25.addItem(self.horizontalSpacer_52)

        self.state_frequency_2 = QDoubleSpinBox(self.brain_state_page)
        self.state_frequency_2.setObjectName("state_frequency_2")
        self.state_frequency_2.setMaximum(1.000000000000000)
        self.state_frequency_2.setSingleStep(0.010000000000000)

        self.horizontalLayout_25.addWidget(self.state_frequency_2)

        self.horizontalSpacer_53 = QSpacerItem(
            5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )

        self.horizontalLayout_25.addItem(self.horizontalSpacer_53)

        self.horizontalLayout_16.addLayout(self.horizontalLayout_25)

        self.horizontalLayout_16.setStretch(0, 3)
        self.horizontalLayout_16.setStretch(1, 3)
        self.horizontalLayout_16.setStretch(2, 4)
        self.horizontalLayout_16.setStretch(3, 3)
        self.horizontalLayout_16.setStretch(4, 4)

        self.settings_controls_layout.addLayout(self.horizontalLayout_16)

        self.horizontalLayout_15 = QHBoxLayout()
        self.horizontalLayout_15.setSpacing(10)
        self.horizontalLayout_15.setObjectName("horizontalLayout_15")
        self.horizontalLayout_28 = QHBoxLayout()
        self.horizontalLayout_28.setObjectName("horizontalLayout_28")
        self.label_10 = QLabel(self.brain_state_page)
        self.label_10.setObjectName("label_10")
        sizePolicy.setHeightForWidth(self.label_10.sizePolicy().hasHeightForWidth())
        self.label_10.setSizePolicy(sizePolicy)
        self.label_10.setFont(font1)
        self.label_10.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.horizontalLayout_28.addWidget(self.label_10)

        self.horizontalLayout_15.addLayout(self.horizontalLayout_28)

        self.horizontalLayout_29 = QHBoxLayout()
        self.horizontalLayout_29.setObjectName("horizontalLayout_29")
        self.horizontalSpacer_20 = QSpacerItem(
            5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )

        self.horizontalLayout_29.addItem(self.horizontalSpacer_20)

        self.enable_state_3 = QCheckBox(self.brain_state_page)
        self.enable_state_3.setObjectName("enable_state_3")
        sizePolicy1.setHeightForWidth(
            self.enable_state_3.sizePolicy().hasHeightForWidth()
        )
        self.enable_state_3.setSizePolicy(sizePolicy1)

        self.horizontalLayout_29.addWidget(self.enable_state_3)

        self.horizontalSpacer_19 = QSpacerItem(
            5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )

        self.horizontalLayout_29.addItem(self.horizontalSpacer_19)

        self.horizontalLayout_15.addLayout(self.horizontalLayout_29)

        self.state_name_3 = QLineEdit(self.brain_state_page)
        self.state_name_3.setObjectName("state_name_3")
        sizePolicy3.setHeightForWidth(
            self.state_name_3.sizePolicy().hasHeightForWidth()
        )
        self.state_name_3.setSizePolicy(sizePolicy3)

        self.horizontalLayout_15.addWidget(self.state_name_3)

        self.horizontalLayout_30 = QHBoxLayout()
        self.horizontalLayout_30.setObjectName("horizontalLayout_30")
        self.horizontalSpacer_22 = QSpacerItem(
            5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )

        self.horizontalLayout_30.addItem(self.horizontalSpacer_22)

        self.state_scored_3 = QCheckBox(self.brain_state_page)
        self.state_scored_3.setObjectName("state_scored_3")
        sizePolicy1.setHeightForWidth(
            self.state_scored_3.sizePolicy().hasHeightForWidth()
        )
        self.state_scored_3.setSizePolicy(sizePolicy1)
        self.state_scored_3.setChecked(True)

        self.horizontalLayout_30.addWidget(self.state_scored_3)

        self.horizontalSpacer_21 = QSpacerItem(
            5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )

        self.horizontalLayout_30.addItem(self.horizontalSpacer_21)

        self.horizontalLayout_15.addLayout(self.horizontalLayout_30)

        self.horizontalLayout_27 = QHBoxLayout()
        self.horizontalLayout_27.setObjectName("horizontalLayout_27")
        self.horizontalSpacer_55 = QSpacerItem(
            5, 20, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )

        self.horizontalLayout_27.addItem(self.horizontalSpacer_55)

        self.state_frequency_3 = QDoubleSpinBox(self.brain_state_page)
        self.state_frequency_3.setObjectName("state_frequency_3")
        self.state_frequency_3.setMaximum(1.000000000000000)
        self.state_frequency_3.setSingleStep(0.010000000000000)

        self.horizontalLayout_27.addWidget(self.state_frequency_3)

        self.horizontalSpacer_54 = QSpacerItem(
            5, 20, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )

        self.horizontalLayout_27.addItem(self.horizontalSpacer_54)

        self.horizontalLayout_15.addLayout(self.horizontalLayout_27)

        self.horizontalLayout_15.setStretch(0, 3)
        self.horizontalLayout_15.setStretch(1, 3)
        self.horizontalLayout_15.setStretch(2, 4)
        self.horizontalLayout_15.setStretch(3, 3)
        self.horizontalLayout_15.setStretch(4, 4)

        self.settings_controls_layout.addLayout(self.horizontalLayout_15)

        self.horizontalLayout_14 = QHBoxLayout()
        self.horizontalLayout_14.setSpacing(10)
        self.horizontalLayout_14.setObjectName("horizontalLayout_14")
        self.horizontalLayout_31 = QHBoxLayout()
        self.horizontalLayout_31.setObjectName("horizontalLayout_31")
        self.label_9 = QLabel(self.brain_state_page)
        self.label_9.setObjectName("label_9")
        sizePolicy.setHeightForWidth(self.label_9.sizePolicy().hasHeightForWidth())
        self.label_9.setSizePolicy(sizePolicy)
        self.label_9.setFont(font1)
        self.label_9.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.horizontalLayout_31.addWidget(self.label_9)

        self.horizontalLayout_14.addLayout(self.horizontalLayout_31)

        self.horizontalLayout_45 = QHBoxLayout()
        self.horizontalLayout_45.setObjectName("horizontalLayout_45")
        self.horizontalSpacer_24 = QSpacerItem(
            5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )

        self.horizontalLayout_45.addItem(self.horizontalSpacer_24)

        self.enable_state_4 = QCheckBox(self.brain_state_page)
        self.enable_state_4.setObjectName("enable_state_4")
        sizePolicy1.setHeightForWidth(
            self.enable_state_4.sizePolicy().hasHeightForWidth()
        )
        self.enable_state_4.setSizePolicy(sizePolicy1)

        self.horizontalLayout_45.addWidget(self.enable_state_4)

        self.horizontalSpacer_23 = QSpacerItem(
            5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )

        self.horizontalLayout_45.addItem(self.horizontalSpacer_23)

        self.horizontalLayout_14.addLayout(self.horizontalLayout_45)

        self.state_name_4 = QLineEdit(self.brain_state_page)
        self.state_name_4.setObjectName("state_name_4")
        sizePolicy3.setHeightForWidth(
            self.state_name_4.sizePolicy().hasHeightForWidth()
        )
        self.state_name_4.setSizePolicy(sizePolicy3)

        self.horizontalLayout_14.addWidget(self.state_name_4)

        self.horizontalLayout_52 = QHBoxLayout()
        self.horizontalLayout_52.setObjectName("horizontalLayout_52")
        self.horizontalSpacer_26 = QSpacerItem(
            5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )

        self.horizontalLayout_52.addItem(self.horizontalSpacer_26)

        self.state_scored_4 = QCheckBox(self.brain_state_page)
        self.state_scored_4.setObjectName("state_scored_4")
        sizePolicy1.setHeightForWidth(
            self.state_scored_4.sizePolicy().hasHeightForWidth()
        )
        self.state_scored_4.setSizePolicy(sizePolicy1)
        self.state_scored_4.setChecked(True)

        self.horizontalLayout_52.addWidget(self.state_scored_4)

        self.horizontalSpacer_25 = QSpacerItem(
            5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )

        self.horizontalLayout_52.addItem(self.horizontalSpacer_25)

        self.horizontalLayout_14.addLayout(self.horizontalLayout_52)

        self.horizontalLayout_38 = QHBoxLayout()
        self.horizontalLayout_38.setObjectName("horizontalLayout_38")
        self.horizontalSpacer_57 = QSpacerItem(
            5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )

        self.horizontalLayout_38.addItem(self.horizontalSpacer_57)

        self.state_frequency_4 = QDoubleSpinBox(self.brain_state_page)
        self.state_frequency_4.setObjectName("state_frequency_4")
        self.state_frequency_4.setMaximum(1.000000000000000)
        self.state_frequency_4.setSingleStep(0.010000000000000)

        self.horizontalLayout_38.addWidget(self.state_frequency_4)

        self.horizontalSpacer_56 = QSpacerItem(
            5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )

        self.horizontalLayout_38.addItem(self.horizontalSpacer_56)

        self.horizontalLayout_14.addLayout(self.horizontalLayout_38)

        self.horizontalLayout_14.setStretch(0, 3)
        self.horizontalLayout_14.setStretch(1, 3)
        self.horizontalLayout_14.setStretch(2, 4)
        self.horizontalLayout_14.setStretch(3, 3)
        self.horizontalLayout_14.setStretch(4, 4)

        self.settings_controls_layout.addLayout(self.horizontalLayout_14)

        self.horizontalLayout_13 = QHBoxLayout()
        self.horizontalLayout_13.setSpacing(10)
        self.horizontalLayout_13.setObjectName("horizontalLayout_13")
        self.horizontalLayout_32 = QHBoxLayout()
        self.horizontalLayout_32.setObjectName("horizontalLayout_32")
        self.label_8 = QLabel(self.brain_state_page)
        self.label_8.setObjectName("label_8")
        sizePolicy.setHeightForWidth(self.label_8.sizePolicy().hasHeightForWidth())
        self.label_8.setSizePolicy(sizePolicy)
        self.label_8.setFont(font1)
        self.label_8.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.horizontalLayout_32.addWidget(self.label_8)

        self.horizontalLayout_13.addLayout(self.horizontalLayout_32)

        self.horizontalLayout_46 = QHBoxLayout()
        self.horizontalLayout_46.setObjectName("horizontalLayout_46")
        self.horizontalSpacer_29 = QSpacerItem(
            5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )

        self.horizontalLayout_46.addItem(self.horizontalSpacer_29)

        self.enable_state_5 = QCheckBox(self.brain_state_page)
        self.enable_state_5.setObjectName("enable_state_5")
        sizePolicy1.setHeightForWidth(
            self.enable_state_5.sizePolicy().hasHeightForWidth()
        )
        self.enable_state_5.setSizePolicy(sizePolicy1)

        self.horizontalLayout_46.addWidget(self.enable_state_5)

        self.horizontalSpacer_30 = QSpacerItem(
            5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )

        self.horizontalLayout_46.addItem(self.horizontalSpacer_30)

        self.horizontalLayout_13.addLayout(self.horizontalLayout_46)

        self.state_name_5 = QLineEdit(self.brain_state_page)
        self.state_name_5.setObjectName("state_name_5")
        sizePolicy3.setHeightForWidth(
            self.state_name_5.sizePolicy().hasHeightForWidth()
        )
        self.state_name_5.setSizePolicy(sizePolicy3)

        self.horizontalLayout_13.addWidget(self.state_name_5)

        self.horizontalLayout_53 = QHBoxLayout()
        self.horizontalLayout_53.setObjectName("horizontalLayout_53")
        self.horizontalSpacer_27 = QSpacerItem(
            5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )

        self.horizontalLayout_53.addItem(self.horizontalSpacer_27)

        self.state_scored_5 = QCheckBox(self.brain_state_page)
        self.state_scored_5.setObjectName("state_scored_5")
        sizePolicy1.setHeightForWidth(
            self.state_scored_5.sizePolicy().hasHeightForWidth()
        )
        self.state_scored_5.setSizePolicy(sizePolicy1)
        self.state_scored_5.setChecked(True)

        self.horizontalLayout_53.addWidget(self.state_scored_5)

        self.horizontalSpacer_28 = QSpacerItem(
            5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )

        self.horizontalLayout_53.addItem(self.horizontalSpacer_28)

        self.horizontalLayout_13.addLayout(self.horizontalLayout_53)

        self.horizontalLayout_39 = QHBoxLayout()
        self.horizontalLayout_39.setObjectName("horizontalLayout_39")
        self.horizontalSpacer_59 = QSpacerItem(
            5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )

        self.horizontalLayout_39.addItem(self.horizontalSpacer_59)

        self.state_frequency_5 = QDoubleSpinBox(self.brain_state_page)
        self.state_frequency_5.setObjectName("state_frequency_5")
        self.state_frequency_5.setMaximum(1.000000000000000)
        self.state_frequency_5.setSingleStep(0.010000000000000)

        self.horizontalLayout_39.addWidget(self.state_frequency_5)

        self.horizontalSpacer_58 = QSpacerItem(
            5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )

        self.horizontalLayout_39.addItem(self.horizontalSpacer_58)

        self.horizontalLayout_13.addLayout(self.horizontalLayout_39)

        self.horizontalLayout_13.setStretch(0, 3)
        self.horizontalLayout_13.setStretch(1, 3)
        self.horizontalLayout_13.setStretch(2, 4)
        self.horizontalLayout_13.setStretch(3, 3)
        self.horizontalLayout_13.setStretch(4, 4)

        self.settings_controls_layout.addLayout(self.horizontalLayout_13)

        self.horizontalLayout_12 = QHBoxLayout()
        self.horizontalLayout_12.setSpacing(10)
        self.horizontalLayout_12.setObjectName("horizontalLayout_12")
        self.horizontalLayout_33 = QHBoxLayout()
        self.horizontalLayout_33.setObjectName("horizontalLayout_33")
        self.label_7 = QLabel(self.brain_state_page)
        self.label_7.setObjectName("label_7")
        sizePolicy.setHeightForWidth(self.label_7.sizePolicy().hasHeightForWidth())
        self.label_7.setSizePolicy(sizePolicy)
        self.label_7.setFont(font1)
        self.label_7.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.horizontalLayout_33.addWidget(self.label_7)

        self.horizontalLayout_12.addLayout(self.horizontalLayout_33)

        self.horizontalLayout_47 = QHBoxLayout()
        self.horizontalLayout_47.setObjectName("horizontalLayout_47")
        self.horizontalSpacer_32 = QSpacerItem(
            5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )

        self.horizontalLayout_47.addItem(self.horizontalSpacer_32)

        self.enable_state_6 = QCheckBox(self.brain_state_page)
        self.enable_state_6.setObjectName("enable_state_6")
        sizePolicy1.setHeightForWidth(
            self.enable_state_6.sizePolicy().hasHeightForWidth()
        )
        self.enable_state_6.setSizePolicy(sizePolicy1)

        self.horizontalLayout_47.addWidget(self.enable_state_6)

        self.horizontalSpacer_31 = QSpacerItem(
            5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )

        self.horizontalLayout_47.addItem(self.horizontalSpacer_31)

        self.horizontalLayout_12.addLayout(self.horizontalLayout_47)

        self.state_name_6 = QLineEdit(self.brain_state_page)
        self.state_name_6.setObjectName("state_name_6")
        sizePolicy3.setHeightForWidth(
            self.state_name_6.sizePolicy().hasHeightForWidth()
        )
        self.state_name_6.setSizePolicy(sizePolicy3)

        self.horizontalLayout_12.addWidget(self.state_name_6)

        self.horizontalLayout_54 = QHBoxLayout()
        self.horizontalLayout_54.setObjectName("horizontalLayout_54")
        self.horizontalSpacer_34 = QSpacerItem(
            5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )

        self.horizontalLayout_54.addItem(self.horizontalSpacer_34)

        self.state_scored_6 = QCheckBox(self.brain_state_page)
        self.state_scored_6.setObjectName("state_scored_6")
        sizePolicy1.setHeightForWidth(
            self.state_scored_6.sizePolicy().hasHeightForWidth()
        )
        self.state_scored_6.setSizePolicy(sizePolicy1)
        self.state_scored_6.setChecked(True)

        self.horizontalLayout_54.addWidget(self.state_scored_6)

        self.horizontalSpacer_33 = QSpacerItem(
            5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )

        self.horizontalLayout_54.addItem(self.horizontalSpacer_33)

        self.horizontalLayout_12.addLayout(self.horizontalLayout_54)

        self.horizontalLayout_40 = QHBoxLayout()
        self.horizontalLayout_40.setObjectName("horizontalLayout_40")
        self.horizontalSpacer_61 = QSpacerItem(
            5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )

        self.horizontalLayout_40.addItem(self.horizontalSpacer_61)

        self.state_frequency_6 = QDoubleSpinBox(self.brain_state_page)
        self.state_frequency_6.setObjectName("state_frequency_6")
        self.state_frequency_6.setMaximum(1.000000000000000)
        self.state_frequency_6.setSingleStep(0.010000000000000)

        self.horizontalLayout_40.addWidget(self.state_frequency_6)

        self.horizontalSpacer_60 = QSpacerItem(
            5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )

        self.horizontalLayout_40.addItem(self.horizontalSpacer_60)

        self.horizontalLayout_12.addLayout(self.horizontalLayout_40)

        self.horizontalLayout_12.setStretch(0, 3)
        self.horizontalLayout_12.setStretch(1, 3)
        self.horizontalLayout_12.setStretch(2, 4)
        self.horizontalLayout_12.setStretch(3, 3)
        self.horizontalLayout_12.setStretch(4, 4)

        self.settings_controls_layout.addLayout(self.horizontalLayout_12)

        self.horizontalLayout_9 = QHBoxLayout()
        self.horizontalLayout_9.setSpacing(10)
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.horizontalLayout_34 = QHBoxLayout()
        self.horizontalLayout_34.setObjectName("horizontalLayout_34")
        self.label_6 = QLabel(self.brain_state_page)
        self.label_6.setObjectName("label_6")
        sizePolicy.setHeightForWidth(self.label_6.sizePolicy().hasHeightForWidth())
        self.label_6.setSizePolicy(sizePolicy)
        self.label_6.setFont(font1)
        self.label_6.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.horizontalLayout_34.addWidget(self.label_6)

        self.horizontalLayout_9.addLayout(self.horizontalLayout_34)

        self.horizontalLayout_48 = QHBoxLayout()
        self.horizontalLayout_48.setObjectName("horizontalLayout_48")
        self.horizontalSpacer_36 = QSpacerItem(
            5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )

        self.horizontalLayout_48.addItem(self.horizontalSpacer_36)

        self.enable_state_7 = QCheckBox(self.brain_state_page)
        self.enable_state_7.setObjectName("enable_state_7")
        sizePolicy1.setHeightForWidth(
            self.enable_state_7.sizePolicy().hasHeightForWidth()
        )
        self.enable_state_7.setSizePolicy(sizePolicy1)

        self.horizontalLayout_48.addWidget(self.enable_state_7)

        self.horizontalSpacer_35 = QSpacerItem(
            5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )

        self.horizontalLayout_48.addItem(self.horizontalSpacer_35)

        self.horizontalLayout_9.addLayout(self.horizontalLayout_48)

        self.state_name_7 = QLineEdit(self.brain_state_page)
        self.state_name_7.setObjectName("state_name_7")
        sizePolicy3.setHeightForWidth(
            self.state_name_7.sizePolicy().hasHeightForWidth()
        )
        self.state_name_7.setSizePolicy(sizePolicy3)

        self.horizontalLayout_9.addWidget(self.state_name_7)

        self.horizontalLayout_55 = QHBoxLayout()
        self.horizontalLayout_55.setObjectName("horizontalLayout_55")
        self.horizontalSpacer_38 = QSpacerItem(
            5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )

        self.horizontalLayout_55.addItem(self.horizontalSpacer_38)

        self.state_scored_7 = QCheckBox(self.brain_state_page)
        self.state_scored_7.setObjectName("state_scored_7")
        sizePolicy1.setHeightForWidth(
            self.state_scored_7.sizePolicy().hasHeightForWidth()
        )
        self.state_scored_7.setSizePolicy(sizePolicy1)
        self.state_scored_7.setChecked(True)

        self.horizontalLayout_55.addWidget(self.state_scored_7)

        self.horizontalSpacer_37 = QSpacerItem(
            5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )

        self.horizontalLayout_55.addItem(self.horizontalSpacer_37)

        self.horizontalLayout_9.addLayout(self.horizontalLayout_55)

        self.horizontalLayout_41 = QHBoxLayout()
        self.horizontalLayout_41.setObjectName("horizontalLayout_41")
        self.horizontalSpacer_63 = QSpacerItem(
            5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )

        self.horizontalLayout_41.addItem(self.horizontalSpacer_63)

        self.state_frequency_7 = QDoubleSpinBox(self.brain_state_page)
        self.state_frequency_7.setObjectName("state_frequency_7")
        self.state_frequency_7.setMaximum(1.000000000000000)
        self.state_frequency_7.setSingleStep(0.010000000000000)

        self.horizontalLayout_41.addWidget(self.state_frequency_7)

        self.horizontalSpacer_62 = QSpacerItem(
            5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )

        self.horizontalLayout_41.addItem(self.horizontalSpacer_62)

        self.horizontalLayout_9.addLayout(self.horizontalLayout_41)

        self.horizontalLayout_9.setStretch(0, 3)
        self.horizontalLayout_9.setStretch(1, 3)
        self.horizontalLayout_9.setStretch(2, 4)
        self.horizontalLayout_9.setStretch(3, 3)
        self.horizontalLayout_9.setStretch(4, 4)

        self.settings_controls_layout.addLayout(self.horizontalLayout_9)

        self.horizontalLayout_8 = QHBoxLayout()
        self.horizontalLayout_8.setSpacing(10)
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.horizontalLayout_35 = QHBoxLayout()
        self.horizontalLayout_35.setObjectName("horizontalLayout_35")
        self.label_5 = QLabel(self.brain_state_page)
        self.label_5.setObjectName("label_5")
        sizePolicy.setHeightForWidth(self.label_5.sizePolicy().hasHeightForWidth())
        self.label_5.setSizePolicy(sizePolicy)
        self.label_5.setFont(font1)
        self.label_5.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.horizontalLayout_35.addWidget(self.label_5)

        self.horizontalLayout_8.addLayout(self.horizontalLayout_35)

        self.horizontalLayout_49 = QHBoxLayout()
        self.horizontalLayout_49.setObjectName("horizontalLayout_49")
        self.horizontalSpacer_40 = QSpacerItem(
            5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )

        self.horizontalLayout_49.addItem(self.horizontalSpacer_40)

        self.enable_state_8 = QCheckBox(self.brain_state_page)
        self.enable_state_8.setObjectName("enable_state_8")
        sizePolicy1.setHeightForWidth(
            self.enable_state_8.sizePolicy().hasHeightForWidth()
        )
        self.enable_state_8.setSizePolicy(sizePolicy1)

        self.horizontalLayout_49.addWidget(self.enable_state_8)

        self.horizontalSpacer_39 = QSpacerItem(
            5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )

        self.horizontalLayout_49.addItem(self.horizontalSpacer_39)

        self.horizontalLayout_8.addLayout(self.horizontalLayout_49)

        self.state_name_8 = QLineEdit(self.brain_state_page)
        self.state_name_8.setObjectName("state_name_8")
        sizePolicy3.setHeightForWidth(
            self.state_name_8.sizePolicy().hasHeightForWidth()
        )
        self.state_name_8.setSizePolicy(sizePolicy3)

        self.horizontalLayout_8.addWidget(self.state_name_8)

        self.horizontalLayout_56 = QHBoxLayout()
        self.horizontalLayout_56.setObjectName("horizontalLayout_56")
        self.horizontalSpacer_42 = QSpacerItem(
            5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )

        self.horizontalLayout_56.addItem(self.horizontalSpacer_42)

        self.state_scored_8 = QCheckBox(self.brain_state_page)
        self.state_scored_8.setObjectName("state_scored_8")
        sizePolicy1.setHeightForWidth(
            self.state_scored_8.sizePolicy().hasHeightForWidth()
        )
        self.state_scored_8.setSizePolicy(sizePolicy1)
        self.state_scored_8.setChecked(True)

        self.horizontalLayout_56.addWidget(self.state_scored_8)

        self.horizontalSpacer_41 = QSpacerItem(
            5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )

        self.horizontalLayout_56.addItem(self.horizontalSpacer_41)

        self.horizontalLayout_8.addLayout(self.horizontalLayout_56)

        self.horizontalLayout_42 = QHBoxLayout()
        self.horizontalLayout_42.setObjectName("horizontalLayout_42")
        self.horizontalSpacer_65 = QSpacerItem(
            5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )

        self.horizontalLayout_42.addItem(self.horizontalSpacer_65)

        self.state_frequency_8 = QDoubleSpinBox(self.brain_state_page)
        self.state_frequency_8.setObjectName("state_frequency_8")
        self.state_frequency_8.setMaximum(1.000000000000000)
        self.state_frequency_8.setSingleStep(0.010000000000000)

        self.horizontalLayout_42.addWidget(self.state_frequency_8)

        self.horizontalSpacer_64 = QSpacerItem(
            5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )

        self.horizontalLayout_42.addItem(self.horizontalSpacer_64)

        self.horizontalLayout_8.addLayout(self.horizontalLayout_42)

        self.horizontalLayout_8.setStretch(0, 3)
        self.horizontalLayout_8.setStretch(1, 3)
        self.horizontalLayout_8.setStretch(2, 4)
        self.horizontalLayout_8.setStretch(3, 3)
        self.horizontalLayout_8.setStretch(4, 4)

        self.settings_controls_layout.addLayout(self.horizontalLayout_8)

        self.horizontalLayout_7 = QHBoxLayout()
        self.horizontalLayout_7.setSpacing(10)
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.horizontalLayout_36 = QHBoxLayout()
        self.horizontalLayout_36.setObjectName("horizontalLayout_36")
        self.label_4 = QLabel(self.brain_state_page)
        self.label_4.setObjectName("label_4")
        sizePolicy.setHeightForWidth(self.label_4.sizePolicy().hasHeightForWidth())
        self.label_4.setSizePolicy(sizePolicy)
        self.label_4.setFont(font1)
        self.label_4.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.horizontalLayout_36.addWidget(self.label_4)

        self.horizontalLayout_7.addLayout(self.horizontalLayout_36)

        self.horizontalLayout_50 = QHBoxLayout()
        self.horizontalLayout_50.setObjectName("horizontalLayout_50")
        self.horizontalSpacer_44 = QSpacerItem(
            5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )

        self.horizontalLayout_50.addItem(self.horizontalSpacer_44)

        self.enable_state_9 = QCheckBox(self.brain_state_page)
        self.enable_state_9.setObjectName("enable_state_9")
        sizePolicy1.setHeightForWidth(
            self.enable_state_9.sizePolicy().hasHeightForWidth()
        )
        self.enable_state_9.setSizePolicy(sizePolicy1)

        self.horizontalLayout_50.addWidget(self.enable_state_9)

        self.horizontalSpacer_43 = QSpacerItem(
            5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )

        self.horizontalLayout_50.addItem(self.horizontalSpacer_43)

        self.horizontalLayout_7.addLayout(self.horizontalLayout_50)

        self.state_name_9 = QLineEdit(self.brain_state_page)
        self.state_name_9.setObjectName("state_name_9")
        sizePolicy3.setHeightForWidth(
            self.state_name_9.sizePolicy().hasHeightForWidth()
        )
        self.state_name_9.setSizePolicy(sizePolicy3)

        self.horizontalLayout_7.addWidget(self.state_name_9)

        self.horizontalLayout_57 = QHBoxLayout()
        self.horizontalLayout_57.setObjectName("horizontalLayout_57")
        self.horizontalSpacer_46 = QSpacerItem(
            5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )

        self.horizontalLayout_57.addItem(self.horizontalSpacer_46)

        self.state_scored_9 = QCheckBox(self.brain_state_page)
        self.state_scored_9.setObjectName("state_scored_9")
        sizePolicy1.setHeightForWidth(
            self.state_scored_9.sizePolicy().hasHeightForWidth()
        )
        self.state_scored_9.setSizePolicy(sizePolicy1)
        self.state_scored_9.setChecked(True)

        self.horizontalLayout_57.addWidget(self.state_scored_9)

        self.horizontalSpacer_45 = QSpacerItem(
            5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )

        self.horizontalLayout_57.addItem(self.horizontalSpacer_45)

        self.horizontalLayout_7.addLayout(self.horizontalLayout_57)

        self.horizontalLayout_43 = QHBoxLayout()
        self.horizontalLayout_43.setObjectName("horizontalLayout_43")
        self.horizontalSpacer_67 = QSpacerItem(
            5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )

        self.horizontalLayout_43.addItem(self.horizontalSpacer_67)

        self.state_frequency_9 = QDoubleSpinBox(self.brain_state_page)
        self.state_frequency_9.setObjectName("state_frequency_9")
        self.state_frequency_9.setMaximum(1.000000000000000)
        self.state_frequency_9.setSingleStep(0.010000000000000)

        self.horizontalLayout_43.addWidget(self.state_frequency_9)

        self.horizontalSpacer_66 = QSpacerItem(
            5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )

        self.horizontalLayout_43.addItem(self.horizontalSpacer_66)

        self.horizontalLayout_7.addLayout(self.horizontalLayout_43)

        self.horizontalLayout_7.setStretch(0, 3)
        self.horizontalLayout_7.setStretch(1, 3)
        self.horizontalLayout_7.setStretch(2, 4)
        self.horizontalLayout_7.setStretch(3, 3)
        self.horizontalLayout_7.setStretch(4, 4)

        self.settings_controls_layout.addLayout(self.horizontalLayout_7)

        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setSpacing(10)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.horizontalLayout_37 = QHBoxLayout()
        self.horizontalLayout_37.setObjectName("horizontalLayout_37")
        self.label_3 = QLabel(self.brain_state_page)
        self.label_3.setObjectName("label_3")
        sizePolicy.setHeightForWidth(self.label_3.sizePolicy().hasHeightForWidth())
        self.label_3.setSizePolicy(sizePolicy)
        self.label_3.setFont(font1)
        self.label_3.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.horizontalLayout_37.addWidget(self.label_3)

        self.horizontalLayout_4.addLayout(self.horizontalLayout_37)

        self.horizontalLayout_51 = QHBoxLayout()
        self.horizontalLayout_51.setObjectName("horizontalLayout_51")
        self.horizontalSpacer_48 = QSpacerItem(
            5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )

        self.horizontalLayout_51.addItem(self.horizontalSpacer_48)

        self.enable_state_0 = QCheckBox(self.brain_state_page)
        self.enable_state_0.setObjectName("enable_state_0")
        sizePolicy1.setHeightForWidth(
            self.enable_state_0.sizePolicy().hasHeightForWidth()
        )
        self.enable_state_0.setSizePolicy(sizePolicy1)

        self.horizontalLayout_51.addWidget(self.enable_state_0)

        self.horizontalSpacer_47 = QSpacerItem(
            5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )

        self.horizontalLayout_51.addItem(self.horizontalSpacer_47)

        self.horizontalLayout_4.addLayout(self.horizontalLayout_51)

        self.state_name_0 = QLineEdit(self.brain_state_page)
        self.state_name_0.setObjectName("state_name_0")
        sizePolicy3.setHeightForWidth(
            self.state_name_0.sizePolicy().hasHeightForWidth()
        )
        self.state_name_0.setSizePolicy(sizePolicy3)

        self.horizontalLayout_4.addWidget(self.state_name_0)

        self.horizontalLayout_58 = QHBoxLayout()
        self.horizontalLayout_58.setObjectName("horizontalLayout_58")
        self.horizontalSpacer_49 = QSpacerItem(
            5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )

        self.horizontalLayout_58.addItem(self.horizontalSpacer_49)

        self.state_scored_0 = QCheckBox(self.brain_state_page)
        self.state_scored_0.setObjectName("state_scored_0")
        sizePolicy1.setHeightForWidth(
            self.state_scored_0.sizePolicy().hasHeightForWidth()
        )
        self.state_scored_0.setSizePolicy(sizePolicy1)
        self.state_scored_0.setChecked(True)

        self.horizontalLayout_58.addWidget(self.state_scored_0)

        self.horizontalSpacer_50 = QSpacerItem(
            5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )

        self.horizontalLayout_58.addItem(self.horizontalSpacer_50)

        self.horizontalLayout_4.addLayout(self.horizontalLayout_58)

        self.horizontalLayout_44 = QHBoxLayout()
        self.horizontalLayout_44.setObjectName("horizontalLayout_44")
        self.horizontalSpacer_69 = QSpacerItem(
            5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )

        self.horizontalLayout_44.addItem(self.horizontalSpacer_69)

        self.state_frequency_0 = QDoubleSpinBox(self.brain_state_page)
        self.state_frequency_0.setObjectName("state_frequency_0")
        self.state_frequency_0.setMaximum(1.000000000000000)
        self.state_frequency_0.setSingleStep(0.010000000000000)

        self.horizontalLayout_44.addWidget(self.state_frequency_0)

        self.horizontalSpacer_68 = QSpacerItem(
            5, 5, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )

        self.horizontalLayout_44.addItem(self.horizontalSpacer_68)

        self.horizontalLayout_4.addLayout(self.horizontalLayout_44)

        self.horizontalLayout_4.setStretch(0, 3)
        self.horizontalLayout_4.setStretch(1, 3)
        self.horizontalLayout_4.setStretch(2, 4)
        self.horizontalLayout_4.setStretch(3, 3)
        self.horizontalLayout_4.setStretch(4, 4)

        self.settings_controls_layout.addLayout(self.horizontalLayout_4)

        self.verticalSpacer_3 = QSpacerItem(
            5, 5, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding
        )

        self.settings_controls_layout.addItem(self.verticalSpacer_3)

        self.horizontalLayout_80.addLayout(self.settings_controls_layout)

        self.settings_description_layout = QVBoxLayout()
        self.settings_description_layout.setObjectName("settings_description_layout")
        self.settings_text = QLabel(self.brain_state_page)
        self.settings_text.setObjectName("settings_text")
        self.settings_text.setStyleSheet("background-color: white;")
        self.settings_text.setTextFormat(Qt.TextFormat.MarkdownText)
        self.settings_text.setAlignment(
            Qt.AlignmentFlag.AlignLeading
            | Qt.AlignmentFlag.AlignLeft
            | Qt.AlignmentFlag.AlignTop
        )
        self.settings_text.setWordWrap(True)
        self.settings_text.setMargin(12)

        self.settings_description_layout.addWidget(self.settings_text)

        self.horizontalLayout_80.addLayout(self.settings_description_layout)

        self.horizontalLayout_80.setStretch(0, 1)
        self.horizontalLayout_80.setStretch(1, 1)
        self.settings_stack.addWidget(self.brain_state_page)
        self.ui_default_page = QWidget()
        self.ui_default_page.setObjectName("ui_default_page")
        self.horizontalLayout_81 = QHBoxLayout(self.ui_default_page)
        self.horizontalLayout_81.setSpacing(20)
        self.horizontalLayout_81.setObjectName("horizontalLayout_81")
        self.verticalLayout_6 = QVBoxLayout()
        self.verticalLayout_6.setSpacing(20)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.primary_defaults_groupbox = QGroupBox(self.ui_default_page)
        self.primary_defaults_groupbox.setObjectName("primary_defaults_groupbox")
        self.verticalLayout_4 = QVBoxLayout(self.primary_defaults_groupbox)
        self.verticalLayout_4.setSpacing(10)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.default_epoch_layout = QHBoxLayout()
        self.default_epoch_layout.setObjectName("default_epoch_layout")
        self.horizontalLayout_60 = QHBoxLayout()
        self.horizontalLayout_60.setObjectName("horizontalLayout_60")
        self.label_17 = QLabel(self.primary_defaults_groupbox)
        self.label_17.setObjectName("label_17")
        sizePolicy1.setHeightForWidth(self.label_17.sizePolicy().hasHeightForWidth())
        self.label_17.setSizePolicy(sizePolicy1)

        self.horizontalLayout_60.addWidget(self.label_17)

        self.default_epoch_input = QDoubleSpinBox(self.primary_defaults_groupbox)
        self.default_epoch_input.setObjectName("default_epoch_input")
        sizePolicy1.setHeightForWidth(
            self.default_epoch_input.sizePolicy().hasHeightForWidth()
        )
        self.default_epoch_input.setSizePolicy(sizePolicy1)
        self.default_epoch_input.setMaximum(100000.000000000000000)
        self.default_epoch_input.setSingleStep(0.500000000000000)

        self.horizontalLayout_60.addWidget(self.default_epoch_input)

        self.default_epoch_layout.addLayout(self.horizontalLayout_60)

        self.horizontalSpacer_70 = QSpacerItem(
            10, 10, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
        )

        self.default_epoch_layout.addItem(self.horizontalSpacer_70)

        self.verticalLayout_4.addLayout(self.default_epoch_layout)

        self.horizontalLayout_75 = QHBoxLayout()
        self.horizontalLayout_75.setObjectName("horizontalLayout_75")
        self.horizontalLayout_76 = QHBoxLayout()
        self.horizontalLayout_76.setObjectName("horizontalLayout_76")
        self.overwrite_default_checkbox = QCheckBox(self.primary_defaults_groupbox)
        self.overwrite_default_checkbox.setObjectName("overwrite_default_checkbox")

        self.horizontalLayout_76.addWidget(self.overwrite_default_checkbox)

        self.horizontalLayout_75.addLayout(self.horizontalLayout_76)

        self.horizontalSpacer_88 = QSpacerItem(
            40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
        )

        self.horizontalLayout_75.addItem(self.horizontalSpacer_88)

        self.verticalLayout_4.addLayout(self.horizontalLayout_75)

        self.confidence_score_setting_layout = QHBoxLayout()
        self.confidence_score_setting_layout.setObjectName(
            "confidence_score_setting_layout"
        )
        self.horizontalLayout_62 = QHBoxLayout()
        self.horizontalLayout_62.setObjectName("horizontalLayout_62")
        self.confidence_setting_checkbox = QCheckBox(self.primary_defaults_groupbox)
        self.confidence_setting_checkbox.setObjectName("confidence_setting_checkbox")
        sizePolicy1.setHeightForWidth(
            self.confidence_setting_checkbox.sizePolicy().hasHeightForWidth()
        )
        self.confidence_setting_checkbox.setSizePolicy(sizePolicy1)
        self.confidence_setting_checkbox.setChecked(True)

        self.horizontalLayout_62.addWidget(self.confidence_setting_checkbox)

        self.confidence_score_setting_layout.addLayout(self.horizontalLayout_62)

        self.horizontalSpacer_72 = QSpacerItem(
            10, 10, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
        )

        self.confidence_score_setting_layout.addItem(self.horizontalSpacer_72)

        self.verticalLayout_4.addLayout(self.confidence_score_setting_layout)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.horizontalLayout_18 = QHBoxLayout()
        self.horizontalLayout_18.setObjectName("horizontalLayout_18")
        self.label_19 = QLabel(self.primary_defaults_groupbox)
        self.label_19.setObjectName("label_19")
        sizePolicy1.setHeightForWidth(self.label_19.sizePolicy().hasHeightForWidth())
        self.label_19.setSizePolicy(sizePolicy1)

        self.horizontalLayout_18.addWidget(self.label_19)

        self.default_min_bout_length_spinbox = QDoubleSpinBox(
            self.primary_defaults_groupbox
        )
        self.default_min_bout_length_spinbox.setObjectName(
            "default_min_bout_length_spinbox"
        )
        sizePolicy1.setHeightForWidth(
            self.default_min_bout_length_spinbox.sizePolicy().hasHeightForWidth()
        )
        self.default_min_bout_length_spinbox.setSizePolicy(sizePolicy1)
        self.default_min_bout_length_spinbox.setMaximum(1000.000000000000000)
        self.default_min_bout_length_spinbox.setValue(5.000000000000000)

        self.horizontalLayout_18.addWidget(self.default_min_bout_length_spinbox)

        self.horizontalLayout_2.addLayout(self.horizontalLayout_18)

        self.horizontalSpacer_90 = QSpacerItem(
            5, 5, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
        )

        self.horizontalLayout_2.addItem(self.horizontalSpacer_90)

        self.verticalLayout_4.addLayout(self.horizontalLayout_2)

        self.verticalLayout_6.addWidget(self.primary_defaults_groupbox)

        self.manual_defaults_groupbox = QGroupBox(self.ui_default_page)
        self.manual_defaults_groupbox.setObjectName("manual_defaults_groupbox")
        self.verticalLayout_7 = QVBoxLayout(self.manual_defaults_groupbox)
        self.verticalLayout_7.setSpacing(10)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.epochs_to_show_layout = QHBoxLayout()
        self.epochs_to_show_layout.setObjectName("epochs_to_show_layout")
        self.horizontalLayout_78 = QHBoxLayout()
        self.horizontalLayout_78.setObjectName("horizontalLayout_78")
        self.label_20 = QLabel(self.manual_defaults_groupbox)
        self.label_20.setObjectName("label_20")
        sizePolicy1.setHeightForWidth(self.label_20.sizePolicy().hasHeightForWidth())
        self.label_20.setSizePolicy(sizePolicy1)

        self.horizontalLayout_78.addWidget(self.label_20)

        self.epochs_to_show_spinbox = QSpinBox(self.manual_defaults_groupbox)
        self.epochs_to_show_spinbox.setObjectName("epochs_to_show_spinbox")
        self.epochs_to_show_spinbox.setMinimum(3)
        self.epochs_to_show_spinbox.setSingleStep(2)
        self.epochs_to_show_spinbox.setValue(5)

        self.horizontalLayout_78.addWidget(self.epochs_to_show_spinbox)

        self.epochs_to_show_layout.addLayout(self.horizontalLayout_78)

        self.horizontalSpacer_92 = QSpacerItem(
            40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
        )

        self.epochs_to_show_layout.addItem(self.horizontalSpacer_92)

        self.verticalLayout_7.addLayout(self.epochs_to_show_layout)

        self.autoscroll_layout = QHBoxLayout()
        self.autoscroll_layout.setObjectName("autoscroll_layout")
        self.horizontalLayout_77 = QHBoxLayout()
        self.horizontalLayout_77.setObjectName("horizontalLayout_77")
        self.autoscroll_checkbox = QCheckBox(self.manual_defaults_groupbox)
        self.autoscroll_checkbox.setObjectName("autoscroll_checkbox")
        sizePolicy1.setHeightForWidth(
            self.autoscroll_checkbox.sizePolicy().hasHeightForWidth()
        )
        self.autoscroll_checkbox.setSizePolicy(sizePolicy1)

        self.horizontalLayout_77.addWidget(self.autoscroll_checkbox)

        self.autoscroll_layout.addLayout(self.horizontalLayout_77)

        self.horizontalSpacer_91 = QSpacerItem(
            10, 10, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
        )

        self.autoscroll_layout.addItem(self.horizontalSpacer_91)

        self.verticalLayout_7.addLayout(self.autoscroll_layout)

        self.verticalLayout_6.addWidget(self.manual_defaults_groupbox)

        self.verticalSpacer_5 = QSpacerItem(
            5, 5, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding
        )

        self.verticalLayout_6.addItem(self.verticalSpacer_5)

        self.horizontalLayout_81.addLayout(self.verticalLayout_6)

        self.ui_default_description_layout = QVBoxLayout()
        self.ui_default_description_layout.setObjectName(
            "ui_default_description_layout"
        )
        self.ui_default_description_label = QLabel(self.ui_default_page)
        self.ui_default_description_label.setObjectName("ui_default_description_label")
        self.ui_default_description_label.setStyleSheet("background-color: white;")
        self.ui_default_description_label.setTextFormat(Qt.TextFormat.MarkdownText)
        self.ui_default_description_label.setAlignment(
            Qt.AlignmentFlag.AlignLeading
            | Qt.AlignmentFlag.AlignLeft
            | Qt.AlignmentFlag.AlignTop
        )
        self.ui_default_description_label.setWordWrap(True)
        self.ui_default_description_label.setMargin(12)

        self.ui_default_description_layout.addWidget(self.ui_default_description_label)

        self.horizontalLayout_81.addLayout(self.ui_default_description_layout)

        self.horizontalLayout_81.setStretch(0, 1)
        self.horizontalLayout_81.setStretch(1, 1)
        self.settings_stack.addWidget(self.ui_default_page)
        self.emg_filter_page = QWidget()
        self.emg_filter_page.setObjectName("emg_filter_page")
        self.horizontalLayout_82 = QHBoxLayout(self.emg_filter_page)
        self.horizontalLayout_82.setSpacing(20)
        self.horizontalLayout_82.setObjectName("horizontalLayout_82")
        self.verticalLayout_13 = QVBoxLayout()
        self.verticalLayout_13.setSpacing(10)
        self.verticalLayout_13.setObjectName("verticalLayout_13")
        self.filter_order_layout = QHBoxLayout()
        self.filter_order_layout.setObjectName("filter_order_layout")
        self.horizontalLayout_63 = QHBoxLayout()
        self.horizontalLayout_63.setObjectName("horizontalLayout_63")
        self.filter_order_label = QLabel(self.emg_filter_page)
        self.filter_order_label.setObjectName("filter_order_label")
        sizePolicy1.setHeightForWidth(
            self.filter_order_label.sizePolicy().hasHeightForWidth()
        )
        self.filter_order_label.setSizePolicy(sizePolicy1)

        self.horizontalLayout_63.addWidget(self.filter_order_label)

        self.emg_order_spinbox = QSpinBox(self.emg_filter_page)
        self.emg_order_spinbox.setObjectName("emg_order_spinbox")
        sizePolicy1.setHeightForWidth(
            self.emg_order_spinbox.sizePolicy().hasHeightForWidth()
        )
        self.emg_order_spinbox.setSizePolicy(sizePolicy1)
        self.emg_order_spinbox.setMinimum(1)
        self.emg_order_spinbox.setMaximum(10)
        self.emg_order_spinbox.setValue(8)

        self.horizontalLayout_63.addWidget(self.emg_order_spinbox)

        self.filter_order_layout.addLayout(self.horizontalLayout_63)

        self.horizontalSpacer_9 = QSpacerItem(
            10, 10, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )

        self.filter_order_layout.addItem(self.horizontalSpacer_9)

        self.filter_order_layout.setStretch(0, 1)
        self.filter_order_layout.setStretch(1, 3)

        self.verticalLayout_13.addLayout(self.filter_order_layout)

        self.bp_lower_layout = QHBoxLayout()
        self.bp_lower_layout.setObjectName("bp_lower_layout")
        self.horizontalLayout_65 = QHBoxLayout()
        self.horizontalLayout_65.setObjectName("horizontalLayout_65")
        self.bp_lower_label = QLabel(self.emg_filter_page)
        self.bp_lower_label.setObjectName("bp_lower_label")
        sizePolicy1.setHeightForWidth(
            self.bp_lower_label.sizePolicy().hasHeightForWidth()
        )
        self.bp_lower_label.setSizePolicy(sizePolicy1)

        self.horizontalLayout_65.addWidget(self.bp_lower_label)

        self.bp_lower_spinbox = QDoubleSpinBox(self.emg_filter_page)
        self.bp_lower_spinbox.setObjectName("bp_lower_spinbox")
        self.bp_lower_spinbox.setMaximum(999.990000000000009)
        self.bp_lower_spinbox.setValue(20.000000000000000)

        self.horizontalLayout_65.addWidget(self.bp_lower_spinbox)

        self.bp_lower_layout.addLayout(self.horizontalLayout_65)

        self.horizontalSpacer_73 = QSpacerItem(
            10, 10, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
        )

        self.bp_lower_layout.addItem(self.horizontalSpacer_73)

        self.verticalLayout_13.addLayout(self.bp_lower_layout)

        self.bp_upper_layout = QHBoxLayout()
        self.bp_upper_layout.setObjectName("bp_upper_layout")
        self.horizontalLayout_61 = QHBoxLayout()
        self.horizontalLayout_61.setObjectName("horizontalLayout_61")
        self.bp_upper_label = QLabel(self.emg_filter_page)
        self.bp_upper_label.setObjectName("bp_upper_label")
        sizePolicy1.setHeightForWidth(
            self.bp_upper_label.sizePolicy().hasHeightForWidth()
        )
        self.bp_upper_label.setSizePolicy(sizePolicy1)

        self.horizontalLayout_61.addWidget(self.bp_upper_label)

        self.bp_upper_spinbox = QDoubleSpinBox(self.emg_filter_page)
        self.bp_upper_spinbox.setObjectName("bp_upper_spinbox")
        sizePolicy1.setHeightForWidth(
            self.bp_upper_spinbox.sizePolicy().hasHeightForWidth()
        )
        self.bp_upper_spinbox.setSizePolicy(sizePolicy1)
        self.bp_upper_spinbox.setMaximum(999.990000000000009)
        self.bp_upper_spinbox.setValue(50.000000000000000)

        self.horizontalLayout_61.addWidget(self.bp_upper_spinbox)

        self.bp_upper_layout.addLayout(self.horizontalLayout_61)

        self.horizontalSpacer_71 = QSpacerItem(
            10, 10, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
        )

        self.bp_upper_layout.addItem(self.horizontalSpacer_71)

        self.verticalLayout_13.addLayout(self.bp_upper_layout)

        self.horizontalLayout_73 = QHBoxLayout()
        self.horizontalLayout_73.setObjectName("horizontalLayout_73")
        self.horizontalSpacer_85 = QSpacerItem(
            5, 5, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
        )

        self.horizontalLayout_73.addItem(self.horizontalSpacer_85)

        self.reset_emg_params_button = QPushButton(self.emg_filter_page)
        self.reset_emg_params_button.setObjectName("reset_emg_params_button")
        sizePolicy1.setHeightForWidth(
            self.reset_emg_params_button.sizePolicy().hasHeightForWidth()
        )
        self.reset_emg_params_button.setSizePolicy(sizePolicy1)

        self.horizontalLayout_73.addWidget(self.reset_emg_params_button)

        self.horizontalSpacer_84 = QSpacerItem(
            5, 5, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
        )

        self.horizontalLayout_73.addItem(self.horizontalSpacer_84)

        self.verticalLayout_13.addLayout(self.horizontalLayout_73)

        self.verticalSpacer_6 = QSpacerItem(
            5, 5, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding
        )

        self.verticalLayout_13.addItem(self.verticalSpacer_6)

        self.horizontalLayout_82.addLayout(self.verticalLayout_13)

        self.verticalLayout_12 = QVBoxLayout()
        self.verticalLayout_12.setObjectName("verticalLayout_12")
        self.emg_filter_description_label = QLabel(self.emg_filter_page)
        self.emg_filter_description_label.setObjectName("emg_filter_description_label")
        self.emg_filter_description_label.setStyleSheet("background-color: white;")
        self.emg_filter_description_label.setAlignment(
            Qt.AlignmentFlag.AlignLeading
            | Qt.AlignmentFlag.AlignLeft
            | Qt.AlignmentFlag.AlignTop
        )
        self.emg_filter_description_label.setWordWrap(True)
        self.emg_filter_description_label.setMargin(12)

        self.verticalLayout_12.addWidget(self.emg_filter_description_label)

        self.horizontalLayout_82.addLayout(self.verticalLayout_12)

        self.horizontalLayout_82.setStretch(0, 1)
        self.horizontalLayout_82.setStretch(1, 1)
        self.settings_stack.addWidget(self.emg_filter_page)
        self.hyperparameter_page = QWidget()
        self.hyperparameter_page.setObjectName("hyperparameter_page")
        self.horizontalLayout_83 = QHBoxLayout(self.hyperparameter_page)
        self.horizontalLayout_83.setSpacing(20)
        self.horizontalLayout_83.setObjectName("horizontalLayout_83")
        self.verticalLayout_14 = QVBoxLayout()
        self.verticalLayout_14.setSpacing(10)
        self.verticalLayout_14.setObjectName("verticalLayout_14")
        self.horizontalLayout_64 = QHBoxLayout()
        self.horizontalLayout_64.setObjectName("horizontalLayout_64")
        self.horizontalLayout_66 = QHBoxLayout()
        self.horizontalLayout_66.setObjectName("horizontalLayout_66")
        self.batch_size_label = QLabel(self.hyperparameter_page)
        self.batch_size_label.setObjectName("batch_size_label")
        sizePolicy1.setHeightForWidth(
            self.batch_size_label.sizePolicy().hasHeightForWidth()
        )
        self.batch_size_label.setSizePolicy(sizePolicy1)

        self.horizontalLayout_66.addWidget(self.batch_size_label)

        self.batch_size_spinbox = QSpinBox(self.hyperparameter_page)
        self.batch_size_spinbox.setObjectName("batch_size_spinbox")
        sizePolicy1.setHeightForWidth(
            self.batch_size_spinbox.sizePolicy().hasHeightForWidth()
        )
        self.batch_size_spinbox.setSizePolicy(sizePolicy1)
        self.batch_size_spinbox.setMinimum(1)
        self.batch_size_spinbox.setMaximum(9999)
        self.batch_size_spinbox.setValue(64)

        self.horizontalLayout_66.addWidget(self.batch_size_spinbox)

        self.horizontalLayout_64.addLayout(self.horizontalLayout_66)

        self.horizontalSpacer_80 = QSpacerItem(
            5, 5, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
        )

        self.horizontalLayout_64.addItem(self.horizontalSpacer_80)

        self.verticalLayout_14.addLayout(self.horizontalLayout_64)

        self.horizontalLayout_67 = QHBoxLayout()
        self.horizontalLayout_67.setObjectName("horizontalLayout_67")
        self.horizontalLayout_68 = QHBoxLayout()
        self.horizontalLayout_68.setObjectName("horizontalLayout_68")
        self.learning_rate_label = QLabel(self.hyperparameter_page)
        self.learning_rate_label.setObjectName("learning_rate_label")
        sizePolicy1.setHeightForWidth(
            self.learning_rate_label.sizePolicy().hasHeightForWidth()
        )
        self.learning_rate_label.setSizePolicy(sizePolicy1)

        self.horizontalLayout_68.addWidget(self.learning_rate_label)

        self.learning_rate_spinbox = QDoubleSpinBox(self.hyperparameter_page)
        self.learning_rate_spinbox.setObjectName("learning_rate_spinbox")
        sizePolicy1.setHeightForWidth(
            self.learning_rate_spinbox.sizePolicy().hasHeightForWidth()
        )
        self.learning_rate_spinbox.setSizePolicy(sizePolicy1)
        self.learning_rate_spinbox.setDecimals(5)
        self.learning_rate_spinbox.setMinimum(0.000010000000000)
        self.learning_rate_spinbox.setSingleStep(0.000100000000000)
        self.learning_rate_spinbox.setValue(0.001000000000000)

        self.horizontalLayout_68.addWidget(self.learning_rate_spinbox)

        self.horizontalLayout_67.addLayout(self.horizontalLayout_68)

        self.horizontalSpacer_81 = QSpacerItem(
            5, 5, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
        )

        self.horizontalLayout_67.addItem(self.horizontalSpacer_81)

        self.verticalLayout_14.addLayout(self.horizontalLayout_67)

        self.horizontalLayout_69 = QHBoxLayout()
        self.horizontalLayout_69.setObjectName("horizontalLayout_69")
        self.horizontalLayout_70 = QHBoxLayout()
        self.horizontalLayout_70.setObjectName("horizontalLayout_70")
        self.momentum_label = QLabel(self.hyperparameter_page)
        self.momentum_label.setObjectName("momentum_label")
        sizePolicy1.setHeightForWidth(
            self.momentum_label.sizePolicy().hasHeightForWidth()
        )
        self.momentum_label.setSizePolicy(sizePolicy1)

        self.horizontalLayout_70.addWidget(self.momentum_label)

        self.momentum_spinbox = QDoubleSpinBox(self.hyperparameter_page)
        self.momentum_spinbox.setObjectName("momentum_spinbox")
        sizePolicy1.setHeightForWidth(
            self.momentum_spinbox.sizePolicy().hasHeightForWidth()
        )
        self.momentum_spinbox.setSizePolicy(sizePolicy1)
        self.momentum_spinbox.setDecimals(3)
        self.momentum_spinbox.setMinimum(0.010000000000000)
        self.momentum_spinbox.setMaximum(0.999000000000000)
        self.momentum_spinbox.setSingleStep(0.010000000000000)
        self.momentum_spinbox.setValue(0.900000000000000)

        self.horizontalLayout_70.addWidget(self.momentum_spinbox)

        self.horizontalLayout_69.addLayout(self.horizontalLayout_70)

        self.horizontalSpacer_82 = QSpacerItem(
            5, 5, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
        )

        self.horizontalLayout_69.addItem(self.horizontalSpacer_82)

        self.verticalLayout_14.addLayout(self.horizontalLayout_69)

        self.horizontalLayout_71 = QHBoxLayout()
        self.horizontalLayout_71.setObjectName("horizontalLayout_71")
        self.horizontalLayout_72 = QHBoxLayout()
        self.horizontalLayout_72.setObjectName("horizontalLayout_72")
        self.training_epochs_label = QLabel(self.hyperparameter_page)
        self.training_epochs_label.setObjectName("training_epochs_label")
        sizePolicy1.setHeightForWidth(
            self.training_epochs_label.sizePolicy().hasHeightForWidth()
        )
        self.training_epochs_label.setSizePolicy(sizePolicy1)

        self.horizontalLayout_72.addWidget(self.training_epochs_label)

        self.training_epochs_spinbox = QSpinBox(self.hyperparameter_page)
        self.training_epochs_spinbox.setObjectName("training_epochs_spinbox")
        sizePolicy1.setHeightForWidth(
            self.training_epochs_spinbox.sizePolicy().hasHeightForWidth()
        )
        self.training_epochs_spinbox.setSizePolicy(sizePolicy1)
        self.training_epochs_spinbox.setMinimum(1)
        self.training_epochs_spinbox.setValue(6)

        self.horizontalLayout_72.addWidget(self.training_epochs_spinbox)

        self.horizontalLayout_71.addLayout(self.horizontalLayout_72)

        self.horizontalSpacer_83 = QSpacerItem(
            40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
        )

        self.horizontalLayout_71.addItem(self.horizontalSpacer_83)

        self.verticalLayout_14.addLayout(self.horizontalLayout_71)

        self.horizontalLayout_74 = QHBoxLayout()
        self.horizontalLayout_74.setObjectName("horizontalLayout_74")
        self.horizontalSpacer_86 = QSpacerItem(
            5, 5, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
        )

        self.horizontalLayout_74.addItem(self.horizontalSpacer_86)

        self.reset_hyperparams_button = QPushButton(self.hyperparameter_page)
        self.reset_hyperparams_button.setObjectName("reset_hyperparams_button")
        sizePolicy1.setHeightForWidth(
            self.reset_hyperparams_button.sizePolicy().hasHeightForWidth()
        )
        self.reset_hyperparams_button.setSizePolicy(sizePolicy1)

        self.horizontalLayout_74.addWidget(self.reset_hyperparams_button)

        self.horizontalSpacer_87 = QSpacerItem(
            5, 5, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
        )

        self.horizontalLayout_74.addItem(self.horizontalSpacer_87)

        self.verticalLayout_14.addLayout(self.horizontalLayout_74)

        self.verticalSpacer = QSpacerItem(
            5, 5, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding
        )

        self.verticalLayout_14.addItem(self.verticalSpacer)

        self.horizontalLayout_83.addLayout(self.verticalLayout_14)

        self.verticalLayout_15 = QVBoxLayout()
        self.verticalLayout_15.setObjectName("verticalLayout_15")
        self.hyperparameter_description_label = QLabel(self.hyperparameter_page)
        self.hyperparameter_description_label.setObjectName(
            "hyperparameter_description_label"
        )
        self.hyperparameter_description_label.setStyleSheet("background-color: white;")
        self.hyperparameter_description_label.setTextFormat(Qt.TextFormat.MarkdownText)
        self.hyperparameter_description_label.setAlignment(
            Qt.AlignmentFlag.AlignLeading
            | Qt.AlignmentFlag.AlignLeft
            | Qt.AlignmentFlag.AlignTop
        )
        self.hyperparameter_description_label.setWordWrap(True)
        self.hyperparameter_description_label.setMargin(12)

        self.verticalLayout_15.addWidget(self.hyperparameter_description_label)

        self.horizontalLayout_83.addLayout(self.verticalLayout_15)

        self.horizontalLayout_83.setStretch(0, 1)
        self.horizontalLayout_83.setStretch(1, 1)
        self.settings_stack.addWidget(self.hyperparameter_page)

        self.settings_layout.addWidget(self.settings_stack)

        self.settings_tab_layout.addLayout(self.settings_layout, 1, 0, 1, 1)

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
        QWidget.setTabOrder(self.score_all_button, self.user_manual_button)
        QWidget.setTabOrder(self.user_manual_button, self.recording_list_widget)
        QWidget.setTabOrder(self.recording_list_widget, self.message_area)

        self.retranslateUi(PrimaryWindow)
        self.settings_combobox.currentIndexChanged.connect(
            self.settings_stack.setCurrentIndex
        )

        self.upper_tab_widget.setCurrentIndex(0)
        self.lower_tab_widget.setCurrentIndex(0)
        self.settings_combobox.setCurrentIndex(0)
        self.settings_stack.setCurrentIndex(0)

        QMetaObject.connectSlotsByName(PrimaryWindow)

    # setupUi

    def retranslateUi(self, PrimaryWindow):
        PrimaryWindow.setWindowTitle(
            QCoreApplication.translate("PrimaryWindow", "MainWindow", None)
        )
        self.epochlengthlabel.setText(
            QCoreApplication.translate("PrimaryWindow", "Epoch length:", None)
        )
        self.epoch_length_input.setSuffix(
            QCoreApplication.translate("PrimaryWindow", " sec", None)
        )
        self.recordinglistgroupbox.setTitle(
            QCoreApplication.translate("PrimaryWindow", "Recording list", None)
        )
        self.add_button.setText(
            QCoreApplication.translate("PrimaryWindow", "add", None)
        )
        self.remove_button.setText(
            QCoreApplication.translate("PrimaryWindow", "remove", None)
        )
        # if QT_CONFIG(tooltip)
        self.export_button.setToolTip(
            QCoreApplication.translate(
                "PrimaryWindow", "Export recording list to file", None
            )
        )
        # endif // QT_CONFIG(tooltip)
        self.export_button.setText(
            QCoreApplication.translate("PrimaryWindow", "export", None)
        )
        # if QT_CONFIG(tooltip)
        self.import_button.setToolTip(
            QCoreApplication.translate(
                "PrimaryWindow", "Import recording list from file", None
            )
        )
        # endif // QT_CONFIG(tooltip)
        self.import_button.setText(
            QCoreApplication.translate("PrimaryWindow", "import", None)
        )
        self.accusleepy2.setText(
            QCoreApplication.translate("PrimaryWindow", "AccuSleePy", None)
        )
        self.accusleepy3.setText(
            QCoreApplication.translate("PrimaryWindow", "AccuSleePy", None)
        )
        self.accusleepy1.setText(
            QCoreApplication.translate("PrimaryWindow", "AccuSleePy", None)
        )
        self.version_label.setText(
            QCoreApplication.translate("PrimaryWindow", "TextLabel", None)
        )
        # if QT_CONFIG(tooltip)
        self.user_manual_button.setToolTip(
            QCoreApplication.translate("PrimaryWindow", "User manual", None)
        )
        # endif // QT_CONFIG(tooltip)
        self.user_manual_button.setText("")
        self.selected_recording_groupbox.setTitle(
            QCoreApplication.translate(
                "PrimaryWindow", "Data / actions for Recording 1", None
            )
        )
        self.samplingratelabel.setText(
            QCoreApplication.translate("PrimaryWindow", "Sampling rate:", None)
        )
        self.sampling_rate_input.setSuffix(
            QCoreApplication.translate("PrimaryWindow", " Hz", None)
        )
        # if QT_CONFIG(tooltip)
        self.recording_file_button.setToolTip(
            QCoreApplication.translate(
                "PrimaryWindow", "Select EEG+EMG recording", None
            )
        )
        # endif // QT_CONFIG(tooltip)
        self.recording_file_button.setText(
            QCoreApplication.translate("PrimaryWindow", "Select recording file", None)
        )
        self.recording_file_label.setText("")
        # if QT_CONFIG(tooltip)
        self.select_label_button.setToolTip(
            QCoreApplication.translate(
                "PrimaryWindow", "Select existing label file", None
            )
        )
        # endif // QT_CONFIG(tooltip)
        self.select_label_button.setText(
            QCoreApplication.translate("PrimaryWindow", "Select", None)
        )
        self.or_label.setText(QCoreApplication.translate("PrimaryWindow", "or", None))
        # if QT_CONFIG(tooltip)
        self.create_label_button.setToolTip(
            QCoreApplication.translate(
                "PrimaryWindow", "Choose filename for new label file", None
            )
        )
        # endif // QT_CONFIG(tooltip)
        self.create_label_button.setText(
            QCoreApplication.translate("PrimaryWindow", "create", None)
        )
        self.label_text.setText(
            QCoreApplication.translate("PrimaryWindow", "label file", None)
        )
        self.label_file_label.setText("")
        # if QT_CONFIG(tooltip)
        self.manual_scoring_button.setToolTip(
            QCoreApplication.translate(
                "PrimaryWindow",
                "View and edit brain state labels for this recording",
                None,
            )
        )
        # endif // QT_CONFIG(tooltip)
        self.manual_scoring_button.setText(
            QCoreApplication.translate("PrimaryWindow", "Score manually", None)
        )
        self.manual_scoring_status.setText("")
        # if QT_CONFIG(tooltip)
        self.create_calibration_button.setToolTip(
            QCoreApplication.translate(
                "PrimaryWindow", "Create calibration file for this subject", None
            )
        )
        # endif // QT_CONFIG(tooltip)
        self.create_calibration_button.setText(
            QCoreApplication.translate("PrimaryWindow", "Create calibration file", None)
        )
        self.calibration_status.setText("")
        # if QT_CONFIG(tooltip)
        self.select_calibration_button.setToolTip(
            QCoreApplication.translate(
                "PrimaryWindow", "Load calibration file for this recording", None
            )
        )
        # endif // QT_CONFIG(tooltip)
        self.select_calibration_button.setText(
            QCoreApplication.translate("PrimaryWindow", "Select calibration file", None)
        )
        self.calibration_file_label.setText("")
        # if QT_CONFIG(tooltip)
        self.load_model_button.setToolTip(
            QCoreApplication.translate(
                "PrimaryWindow", "Load a trained sleep scoring classifier", None
            )
        )
        # endif // QT_CONFIG(tooltip)
        self.load_model_button.setText(
            QCoreApplication.translate(
                "PrimaryWindow", "Load classification model", None
            )
        )
        self.model_label.setText("")
        # if QT_CONFIG(tooltip)
        self.score_all_button.setToolTip(
            QCoreApplication.translate(
                "PrimaryWindow",
                "Use classification model to score all recordings",
                None,
            )
        )
        # endif // QT_CONFIG(tooltip)
        self.score_all_button.setText(
            QCoreApplication.translate("PrimaryWindow", "Score all automatically", None)
        )
        self.score_all_status.setText("")
        self.overwritecheckbox.setText(
            QCoreApplication.translate(
                "PrimaryWindow", "Only overwrite undefined epochs", None
            )
        )
        self.save_confidence_checkbox.setText(
            QCoreApplication.translate("PrimaryWindow", "Save confidence scores", None)
        )
        self.boutlengthlabel.setText(
            QCoreApplication.translate("PrimaryWindow", "Minimum bout length:", None)
        )
        self.bout_length_input.setPrefix("")
        self.bout_length_input.setSuffix(
            QCoreApplication.translate("PrimaryWindow", " sec", None)
        )
        self.lower_tab_widget.setTabText(
            self.lower_tab_widget.indexOf(self.classification_tab),
            QCoreApplication.translate("PrimaryWindow", "Classification", None),
        )
        self.label.setText(
            QCoreApplication.translate("PrimaryWindow", "Epochs per image:", None)
        )
        self.delete_image_box.setText(
            QCoreApplication.translate(
                "PrimaryWindow", "Delete images after training", None
            )
        )
        self.label_2.setText(
            QCoreApplication.translate("PrimaryWindow", "Model type:", None)
        )
        self.default_type_button.setText(
            QCoreApplication.translate("PrimaryWindow", "Default", None)
        )
        self.real_time_button.setText(
            QCoreApplication.translate("PrimaryWindow", "Real-time", None)
        )
        # if QT_CONFIG(tooltip)
        self.train_model_button.setToolTip(
            QCoreApplication.translate(
                "PrimaryWindow", "Begin training the classification model", None
            )
        )
        # endif // QT_CONFIG(tooltip)
        self.train_model_button.setText(
            QCoreApplication.translate(
                "PrimaryWindow", "Train classification model", None
            )
        )
        self.calibrate_checkbox.setText(
            QCoreApplication.translate("PrimaryWindow", "Calibrate model", None)
        )
        self.calibrate_label.setText(
            QCoreApplication.translate("PrimaryWindow", "Calibration set size:", None)
        )
        self.calibration_spinbox.setSuffix(
            QCoreApplication.translate("PrimaryWindow", "%", None)
        )
        self.lower_tab_widget.setTabText(
            self.lower_tab_widget.indexOf(self.model_training_tab),
            QCoreApplication.translate("PrimaryWindow", "Model training", None),
        )
        self.messagesgroupbox.setTitle(
            QCoreApplication.translate("PrimaryWindow", "Messages", None)
        )
        self.upper_tab_widget.setTabText(
            self.upper_tab_widget.indexOf(self.scoring_tab),
            QCoreApplication.translate("PrimaryWindow", "Sleep scoring", None),
        )
        # if QT_CONFIG(tooltip)
        self.save_config_button.setToolTip(
            QCoreApplication.translate(
                "PrimaryWindow", "Save current configuration", None
            )
        )
        # endif // QT_CONFIG(tooltip)
        self.save_config_button.setText(
            QCoreApplication.translate("PrimaryWindow", "Save settings", None)
        )
        self.save_config_status.setText("")
        self.settings_combobox.setItemText(
            0, QCoreApplication.translate("PrimaryWindow", "Brain states", None)
        )
        self.settings_combobox.setItemText(
            1, QCoreApplication.translate("PrimaryWindow", "UI defaults", None)
        )
        self.settings_combobox.setItemText(
            2, QCoreApplication.translate("PrimaryWindow", "EMG filter", None)
        )
        self.settings_combobox.setItemText(
            3,
            QCoreApplication.translate(
                "PrimaryWindow", "Training hyperparameters", None
            ),
        )

        self.label_15.setText(
            QCoreApplication.translate("PrimaryWindow", "Digit", None)
        )
        self.label_14.setText(
            QCoreApplication.translate("PrimaryWindow", "Enabled", None)
        )
        self.label_16.setText(QCoreApplication.translate("PrimaryWindow", "Name", None))
        self.label_13.setText(
            QCoreApplication.translate("PrimaryWindow", "Scored", None)
        )
        self.label_18.setText(
            QCoreApplication.translate("PrimaryWindow", "Frequency", None)
        )
        self.label_12.setText(QCoreApplication.translate("PrimaryWindow", "1", None))
        self.enable_state_1.setText("")
        self.state_scored_1.setText("")
        self.label_11.setText(QCoreApplication.translate("PrimaryWindow", "2", None))
        self.enable_state_2.setText("")
        self.state_scored_2.setText("")
        self.label_10.setText(QCoreApplication.translate("PrimaryWindow", "3", None))
        self.enable_state_3.setText("")
        self.state_scored_3.setText("")
        self.label_9.setText(QCoreApplication.translate("PrimaryWindow", "4", None))
        self.enable_state_4.setText("")
        self.state_scored_4.setText("")
        self.label_8.setText(QCoreApplication.translate("PrimaryWindow", "5", None))
        self.enable_state_5.setText("")
        self.state_scored_5.setText("")
        self.label_7.setText(QCoreApplication.translate("PrimaryWindow", "6", None))
        self.enable_state_6.setText("")
        self.state_scored_6.setText("")
        self.label_6.setText(QCoreApplication.translate("PrimaryWindow", "7", None))
        self.enable_state_7.setText("")
        self.state_scored_7.setText("")
        self.label_5.setText(QCoreApplication.translate("PrimaryWindow", "8", None))
        self.enable_state_8.setText("")
        self.state_scored_8.setText("")
        self.label_4.setText(QCoreApplication.translate("PrimaryWindow", "9", None))
        self.enable_state_9.setText("")
        self.state_scored_9.setText("")
        self.label_3.setText(QCoreApplication.translate("PrimaryWindow", "0", None))
        self.enable_state_0.setText("")
        self.state_scored_0.setText("")
        self.settings_text.setText(
            QCoreApplication.translate(
                "PrimaryWindow",
                "This is the current set of brain states. Important notes:\n"
                "- You must click 'Save settings' for changes to take effect.\n"
                "- Changing these settings can prevent existing label files, calibration files, and trained models from working properly.\n"
                "- Reinstalling AccuSleePy will overwrite this configuration.\n"
                "\n"
                "Each brain state has several attributes:\n"
                "- Digit: the indicator for this state in label files, and the keyboard shortcut for this state in manual scoring.\n"
                "- Enabled: whether a brain state for this digit exists.\n"
                "- Name: unique name of the brain state (e.g., REM).\n"
                "- Scored: whether a classification model should output this brain state. If a state corresponds to missing/corrupted data, you probably want to uncheck this box.\n"
                "- Frequency: approximate relative frequency of this brain state. Does not need to be very accurate. Frequencies for scored states must sum to 1.",
                None,
            )
        )
        self.primary_defaults_groupbox.setTitle(
            QCoreApplication.translate("PrimaryWindow", "Primary window settings", None)
        )
        self.label_17.setText(
            QCoreApplication.translate("PrimaryWindow", "Epoch length:", None)
        )
        # if QT_CONFIG(tooltip)
        self.default_epoch_input.setToolTip(
            QCoreApplication.translate(
                "PrimaryWindow",
                "This sets the temporal resolution for sleep scoring",
                None,
            )
        )
        # endif // QT_CONFIG(tooltip)
        self.default_epoch_input.setSuffix(
            QCoreApplication.translate("PrimaryWindow", " sec", None)
        )
        # if QT_CONFIG(tooltip)
        self.overwrite_default_checkbox.setToolTip(
            QCoreApplication.translate(
                "PrimaryWindow",
                "During automated scoring, only overwrite epochs with the undefined brain state",
                None,
            )
        )
        # endif // QT_CONFIG(tooltip)
        self.overwrite_default_checkbox.setText(
            QCoreApplication.translate(
                "PrimaryWindow", "Only overwrite undefined epochs", None
            )
        )
        # if QT_CONFIG(tooltip)
        self.confidence_setting_checkbox.setToolTip(
            QCoreApplication.translate(
                "PrimaryWindow",
                "Save the model's confidence scores in label files",
                None,
            )
        )
        # endif // QT_CONFIG(tooltip)
        self.confidence_setting_checkbox.setText(
            QCoreApplication.translate("PrimaryWindow", "Save confidence scores", None)
        )
        self.label_19.setText(
            QCoreApplication.translate("PrimaryWindow", "Minimum bout length:", None)
        )
        # if QT_CONFIG(tooltip)
        self.default_min_bout_length_spinbox.setToolTip(
            QCoreApplication.translate(
                "PrimaryWindow",
                "Minimum allowed duration for a bout of any brain state",
                None,
            )
        )
        # endif // QT_CONFIG(tooltip)
        self.default_min_bout_length_spinbox.setSuffix(
            QCoreApplication.translate("PrimaryWindow", " sec", None)
        )
        self.manual_defaults_groupbox.setTitle(
            QCoreApplication.translate(
                "PrimaryWindow", "Manual scoring window settings", None
            )
        )
        self.label_20.setText(
            QCoreApplication.translate("PrimaryWindow", "Epochs to show", None)
        )
        # if QT_CONFIG(tooltip)
        self.epochs_to_show_spinbox.setToolTip(
            QCoreApplication.translate(
                "PrimaryWindow", "Number of epochs to display in the lower plots", None
            )
        )
        # endif // QT_CONFIG(tooltip)
        # if QT_CONFIG(tooltip)
        self.autoscroll_checkbox.setToolTip(
            QCoreApplication.translate(
                "PrimaryWindow",
                "After scoring each epoch, automatically advance to the next one",
                None,
            )
        )
        # endif // QT_CONFIG(tooltip)
        self.autoscroll_checkbox.setText(
            QCoreApplication.translate("PrimaryWindow", "Auto scroll", None)
        )
        self.ui_default_description_label.setText(
            QCoreApplication.translate(
                "PrimaryWindow",
                "<html><head/><body><p>These are the default values/settings that are shown in the primary window and manual scoring window when they start up.</p><p>Changes here will not affect the **current** state of the controls in the &quot;Sleep scoring&quot; tab.</p><p>You must click 'Save settings' for changes to take effect.</p></body></html>",
                None,
            )
        )
        self.filter_order_label.setText(
            QCoreApplication.translate("PrimaryWindow", "Filter order:", None)
        )
        self.bp_lower_label.setText(
            QCoreApplication.translate("PrimaryWindow", "Lower cutoff:", None)
        )
        self.bp_lower_spinbox.setSuffix(
            QCoreApplication.translate("PrimaryWindow", " Hz", None)
        )
        self.bp_upper_label.setText(
            QCoreApplication.translate("PrimaryWindow", "Upper cutoff:", None)
        )
        self.bp_upper_spinbox.setSuffix(
            QCoreApplication.translate("PrimaryWindow", " Hz", None)
        )
        self.reset_emg_params_button.setText(
            QCoreApplication.translate("PrimaryWindow", "reset to defaults", None)
        )
        self.emg_filter_description_label.setText(
            QCoreApplication.translate(
                "PrimaryWindow",
                "<html><head/><body><p>EMG power is calculated as the root-mean-square of the EMG signal in each epoch after a bandpass filter is applied. You can change the proprties of that filter here. If the manual scoring window displays the EMG signal but not the EMG power, you can try lowering the filter order to 4.</p><p>Changing these settings can prevent existing classification models and calibration files from working as expected. You will probably want to recreate them using your new settings.</p><p>You must click 'Save settings' for changes to take effect.</p></body></html>",
                None,
            )
        )
        self.batch_size_label.setText(
            QCoreApplication.translate("PrimaryWindow", "Batch size:", None)
        )
        self.learning_rate_label.setText(
            QCoreApplication.translate("PrimaryWindow", "Learning rate:", None)
        )
        self.momentum_label.setText(
            QCoreApplication.translate("PrimaryWindow", "Momentum:", None)
        )
        self.training_epochs_label.setText(
            QCoreApplication.translate("PrimaryWindow", "Epochs:", None)
        )
        self.reset_hyperparams_button.setText(
            QCoreApplication.translate("PrimaryWindow", "reset to defaults", None)
        )
        self.hyperparameter_description_label.setText(
            QCoreApplication.translate(
                "PrimaryWindow",
                "These are the hyperparameters used for training new classification models.\n"
                "- Batch size: number of examples in each training iteration\n"
                "- Learning rate: step size for adjusting model weights\n"
                "- Momentum: amount of past gradients to use in the current update. Typically between 0.6 and 0.99\n"
                "- Epochs: number of passes over the training data",
                None,
            )
        )
        self.upper_tab_widget.setTabText(
            self.upper_tab_widget.indexOf(self.settings_tab),
            QCoreApplication.translate("PrimaryWindow", "Settings", None),
        )

    # retranslateUi
