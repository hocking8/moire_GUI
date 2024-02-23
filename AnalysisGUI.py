"""

Filename: AnalysisGUI_v2.py
Author: Risa Hocking

Purpose: Create a GUI that will load the .txt files for a given 
sample name and then allow processing to find the strain and twist.

"""
import sys
import os
import glob
import pickle
import numpy as np
import re

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
matplotlib.use('Qt5Agg')

import scipy 
import scipy.ndimage
import scipy.fft
import scipy.signal
from scipy.optimize import curve_fit

from peakfindingtab import PeakFindingTab
from straintwisttab import StrainTwistTab
from infotab import InfoTab


class QHSeparationLine(QFrame):
	'''
	a horizontal separation line
	'''
	def __init__(self):
		super().__init__()
		self.setMinimumWidth(1)
		self.setFixedHeight(20)
		self.setFrameShape(QFrame.HLine)
		self.setFrameShadow(QFrame.Sunken)
		self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
		return

class QVSeparationLine(QFrame):
	'''
	a vertical separation line
	'''
	def __init__(self):
		super().__init__()
		self.setFixedWidth(20)
		self.setMinimumHeight(1)
		self.setFrameShape(QFrame.VLine)
		self.setFrameShadow(QFrame.Sunken)
		self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)
		return

class MainWindow(QMainWindow):
	def __init__(self): 
		super().__init__()

		self.setWindowTitle('AFM Analysis')
		layout = QGridLayout()

		# initialize tab screen
		self.tabs = QTabWidget()
		self.pftab = PeakFindingTab()
		self.sftab = StrainTwistTab()
		self.inftab = InfoTab()

		# add tabs
		self.tabs.addTab(self.pftab, 'Peak-finding')
		self.tabs.addTab(self.sftab, 'Strain and twist')
		self.tabs.addTab(self.inftab, 'Info')

		# update file saving information, scan info, FFT bounds, and kvecs on both sides
		self.pftab.fileimportbtn.clicked.connect(lambda: self.update_scan_info_in_tab(self.pftab, self.sftab))
		self.sftab.fileimportbtn.clicked.connect(lambda: self.update_scan_info_in_tab(self.sftab, self.pftab))

		# update peak position on both sides when data is either read from or saved to the pickle
		self.pftab.save_fit_btn.clicked.connect(lambda: self.update_kvecs(self.sftab))
		self.pftab.read_fit_btn.clicked.connect(lambda: self.update_kvecs(self.sftab))
		self.sftab.read_fit_btn.clicked.connect(lambda: self.update_kvecs(self.pftab))

		# start main window layout
		layout.addWidget(self.tabs, 0, 0)

		# finish creating tabs
		self.pftab.setLayout(self.pftab.layout)
		self.sftab.setLayout(self.sftab.layout)
		self.inftab.setLayout(self.inftab.layout)

		# add tabs to widget
		container = QWidget()
		container.setLayout(layout)

		self.setCentralWidget(container)

	def update_scan_info_in_tab(self, parenttab, childtab):
		"""
		As soon as scan info is uploaded in one tab, it writes it to the other.
		"""
		# update keys
		childtab.save_dir = parenttab.save_dir
		childtab.kn = parenttab.kn
		childtab.picklename = parenttab.picklename
		childtab.file = parenttab.file
		childtab.c = parenttab.c
		childtab.fft_bounds = parenttab.fft_bounds

		# update filename, directory, scan size, and FFT bound label displays on other tab
		# update filename
		head_tail = os.path.split(childtab.file)
		name = head_tail[1]
		childtab.filelbl.setText(f'Current file: {name}')

		# update directory text
		childtab.set_dir_txt.setText(childtab.save_dir)

		# update scan sizes, FFT bounds, and kvec parameters
		childtab.read_fit_params()


	def update_kvecs(self, childtab):
		"""
		Updates the peak positions in both tabs when saved or read from the moire_data pickle.
		"""

		childtab.read_fit_params()


app = QApplication(sys.argv)
app.setStyle('Fusion')
window = MainWindow()
window.show()

app.exec()
