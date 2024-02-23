"""

Filename: peakfindingtab.py
Author: Risa Hocking

Purpose: Populates the FFT peak-finding tab for 
fitting PFM/TFM/LFM data of moiré structures.

"""

import sys
import os
import pickle
import numpy as np


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

import time


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


class PeakFindingTab(QWidget):
	def __init__(self):
		super().__init__()

		# create tab layouts
		self.layout = QGridLayout()

		# initialize data to be shared between files
		self.file = ''
		self.kn = ''  # key name
		self.save_dir = ''
		self.picklename = f'{self.save_dir}{self.kn}_moire_data.pickle'
		self.c = ''  # channel type ('A', 'P', or 'T')
		self.fft_bounds = 0.05

		# set up file import
		self.fileimportbtn = QPushButton('Select file')
		self.fileimportbtn.clicked.connect(self.get_file)
		self.filelbl = QLabel(f'Current file: none')
		file_layout = QVBoxLayout()
		file_layout.addWidget(self.filelbl)
		file_layout.addWidget(self.fileimportbtn)
		self.layout.addLayout(file_layout, 0, 0, 1, 1)

		# set up file saving
		# set up file saving
		self.set_dir_lbl = QLabel('Set save directory: ')
		self.set_dir_txt = QLineEdit('')
		self.set_dir_btn = QPushButton('Set directory')
		self.set_dir_btn.clicked.connect(self.update_save_dir)
		dir_layout = QHBoxLayout()
		dir_layout.addWidget(self.set_dir_lbl)
		dir_layout.addWidget(self.set_dir_txt)
		dir_layout.addWidget(self.set_dir_btn)
		file_layout.addLayout(dir_layout)

		# set up scan size input
		self.scanlbl = QLabel('Scan size: ')
		scan_layout = QHBoxLayout()
		self.scanxlbl = QLabel('X (µm)')
		self.scanxinpt = QLineEdit('')
		self.scanylbl = QLabel('Y (µm)')
		self.scanyinpt = QLineEdit('')
		scan_layout.addWidget(self.scanlbl)
		scan_layout.addWidget(self.scanxlbl)
		scan_layout.addWidget(self.scanxinpt)
		scan_layout.addWidget(self.scanylbl)
		scan_layout.addWidget(self.scanyinpt)
		file_layout.addLayout(scan_layout)

		# set up file save for everything
		self.save_dict_btn = QPushButton('Save file information')
		self.save_dict_btn.clicked.connect(self.update_dictionaries)
		save_layout = QVBoxLayout()
		save_layout.addWidget(self.save_dict_btn)
		file_layout.addLayout(save_layout)
		self.horz_line1 = QHSeparationLine()
		self.layout.addWidget(self.horz_line1, 1, 0, 1, 1)

		# set up ability to display plots of file
		img_disp_layout = QVBoxLayout()
		fft_params_layout = QGridLayout()
		self.raw_data_btn = QPushButton('Show raw data')
		self.raw_data_btn.clicked.connect(self.display_raw_data)
		self.fft_bounds_lbl = QLabel('FFT crop bounds(nm⁻¹)')
		self.fft_bounds_txt = QLineEdit(str(self.fft_bounds))
		self.fft_center_lbl = QLabel('Center mask radius (nm⁻¹)')
		self.fft_center_txt = QLineEdit('0.01')
		self.fft_line_lbl = QLabel('Center line mask height (nm⁻¹)')
		self.fft_line_txt = QLineEdit('0.')
		self.fft_btn = QPushButton('Show FFT')
		self.fft_btn.clicked.connect(self.display_fft_data)
		img_disp_layout.addWidget(self.raw_data_btn)
		fft_params_layout.addWidget(self.fft_bounds_lbl, 0, 0, 1, 1)
		fft_params_layout.addWidget(self.fft_bounds_txt, 0, 1, 1, 1)
		fft_params_layout.addWidget(self.fft_center_lbl, 1, 0, 1, 1)
		fft_params_layout.addWidget(self.fft_center_txt, 1, 1, 1, 1)
		fft_params_layout.addWidget(self.fft_line_lbl, 2, 0, 1, 1)
		fft_params_layout.addWidget(self.fft_line_txt, 2, 1, 1, 1)
		img_disp_layout.addLayout(fft_params_layout)
		img_disp_layout.addWidget(self.fft_btn)
		self.layout.addLayout(img_disp_layout, 2, 0, 1, 1)
		self.horz_line2 = QHSeparationLine()
		self.layout.addWidget(self.horz_line2, 3, 0, 1, 1)

		# set up actual image display location
		img_layout = QVBoxLayout()
		self.canvas = FigureCanvas(Figure(figsize=(5,5)))
		self.canvas.setMinimumSize(self.canvas.size())
		self.ax = None
		self.toolbar = NavigationToolbar(self.canvas, self)
		img_layout.addWidget(self.toolbar)
		img_layout.addWidget(self.canvas)
		self.layout.addLayout(img_layout, 0, 2, -1, -1)
		self.vert_line1 = QVSeparationLine()
		self.layout.addWidget(self.vert_line1, 0, 1, -1, 1)

		# set up fit parameter area for 2D Gaussian
		fit_param_layout = QGridLayout()
		self.param_descr_lbl = QLabel('Note: Peak positions refer to the upper three peaks in FFT.')
		self.pk1_lbl = QLabel('Right peak')
		self.pk1_lbl.setFont(QFont('',weight=QFont.Bold))
		self.pk2_lbl = QLabel('Left peak')
		self.pk2_lbl.setFont(QFont('',weight=QFont.Bold))
		self.pk3_lbl = QLabel('Middle peak')
		self.pk3_lbl.setFont(QFont('',weight=QFont.Bold))
		fit_param_layout.addWidget(self.param_descr_lbl, 0, 0, 1, 4)
		fit_param_layout.addWidget(self.pk1_lbl, 1, 1, 1, 1)
		fit_param_layout.addWidget(self.pk2_lbl, 1, 2, 1, 1)
		fit_param_layout.addWidget(self.pk3_lbl, 1, 3, 1, 1)
		self.xo_lbl = QLabel('X (nm⁻¹)')
		self.xo_txt1 = QLineEdit('0.')
		self.xo_txt2 = QLineEdit('0.')
		self.xo_txt3 = QLineEdit('0.')
		fit_param_layout.addWidget(self.xo_lbl, 2, 0, 1, 1)
		fit_param_layout.addWidget(self.xo_txt1, 2, 1, 1, 1)
		fit_param_layout.addWidget(self.xo_txt2, 2, 2, 1, 1)
		fit_param_layout.addWidget(self.xo_txt3, 2, 3, 1, 1)
		self.yo_lbl = QLabel('Y (nm⁻¹)')
		self.yo_txt1 = QLineEdit('0.')
		self.yo_txt2 = QLineEdit('0.')
		self.yo_txt3 = QLineEdit('0.')
		fit_param_layout.addWidget(self.yo_lbl, 3, 0, 1, 1)
		fit_param_layout.addWidget(self.yo_txt1, 3, 1, 1, 1)
		fit_param_layout.addWidget(self.yo_txt2, 3, 2, 1, 1)
		fit_param_layout.addWidget(self.yo_txt3, 3, 3, 1, 1)
		self.rad_lbl = QLabel('Crop radius (nm⁻¹)')
		self.rad_txt1 = QLineEdit('0.003')
		self.rad_txt2 = QLineEdit('0.003')
		self.rad_txt3 = QLineEdit('0.003')
		fit_param_layout.addWidget(self.rad_lbl, 4, 0, 1, 1)
		fit_param_layout.addWidget(self.rad_txt1, 4, 1, 1, 1)
		fit_param_layout.addWidget(self.rad_txt2, 4, 2, 1, 1)
		fit_param_layout.addWidget(self.rad_txt3, 4, 3, 1, 1)
		self.amp_lbl = QLabel('Amplitude(a.u.)')
		self.amp_txt1 = QLineEdit('1')
		self.amp_txt2 = QLineEdit('1')
		self.amp_txt3 = QLineEdit('1')
		fit_param_layout.addWidget(self.amp_lbl, 5, 0, 1, 1)
		fit_param_layout.addWidget(self.amp_txt1, 5, 1, 1, 1)
		fit_param_layout.addWidget(self.amp_txt2, 5, 2, 1, 1)
		fit_param_layout.addWidget(self.amp_txt3, 5, 3, 1, 1)
		self.amp_lbl = QLabel('Amplitude(a.u.)')
		self.amp_txt1 = QLineEdit('10000')
		self.amp_txt2 = QLineEdit('10000')
		self.amp_txt3 = QLineEdit('10000')
		fit_param_layout.addWidget(self.amp_lbl, 5, 0, 1, 1)
		fit_param_layout.addWidget(self.amp_txt1, 5, 1, 1, 1)
		fit_param_layout.addWidget(self.amp_txt2, 5, 2, 1, 1)
		fit_param_layout.addWidget(self.amp_txt3, 5, 3, 1, 1)
		self.sigx_lbl = QLabel('σₓ (nm⁻¹)')
		self.sigx_txt1 = QLineEdit('0.003')
		self.sigx_txt2 = QLineEdit('0.003')
		self.sigx_txt3 = QLineEdit('0.003')
		fit_param_layout.addWidget(self.sigx_lbl, 6, 0, 1, 1)
		fit_param_layout.addWidget(self.sigx_txt1, 6, 1, 1, 1)
		fit_param_layout.addWidget(self.sigx_txt2, 6, 2, 1, 1)
		fit_param_layout.addWidget(self.sigx_txt3, 6, 3, 1, 1)
		self.sigy_lbl = QLabel('σᵧ (nm⁻¹)')
		self.sigy_txt1 = QLineEdit('0.003')
		self.sigy_txt2 = QLineEdit('0.003')
		self.sigy_txt3 = QLineEdit('0.003')
		fit_param_layout.addWidget(self.sigy_lbl, 7, 0, 1, 1)
		fit_param_layout.addWidget(self.sigy_txt1, 7, 1, 1, 1)
		fit_param_layout.addWidget(self.sigy_txt2, 7, 2, 1, 1)
		fit_param_layout.addWidget(self.sigy_txt3, 7, 3, 1, 1)
		self.theta_lbl = QLabel('θ (°)')
		self.theta_txt1 = QLineEdit('0.')
		self.theta_txt2 = QLineEdit('0.')
		self.theta_txt3 = QLineEdit('0.')
		fit_param_layout.addWidget(self.theta_lbl, 8, 0, 1, 1)
		fit_param_layout.addWidget(self.theta_txt1, 8, 1, 1, 1)
		fit_param_layout.addWidget(self.theta_txt2, 8, 2, 1, 1)
		fit_param_layout.addWidget(self.theta_txt3, 8, 3, 1, 1)
		self.testfit_btn = QPushButton('Test all peaks')
		self.testfit_btn1 = QPushButton('Test peak 1')
		self.testfit_btn2 = QPushButton('Test peak 2')
		self.testfit_btn3 = QPushButton('Test peak 3')
		self.testfit_btn.clicked.connect(lambda: self.update_FFT_fit(pk='all'))
		self.testfit_btn1.clicked.connect(lambda: self.update_FFT_fit(pk=1))
		self.testfit_btn2.clicked.connect(lambda: self.update_FFT_fit(pk=2))
		self.testfit_btn3.clicked.connect(lambda: self.update_FFT_fit(pk=3))
		fit_param_layout.addWidget(self.testfit_btn, 9, 0, 1, 1)
		fit_param_layout.addWidget(self.testfit_btn1, 9, 1, 1, 1)
		fit_param_layout.addWidget(self.testfit_btn2, 9, 2, 1, 1)
		fit_param_layout.addWidget(self.testfit_btn3, 9, 3, 1, 1)
		self.test_crop_rad_btn = QPushButton('Show crop radii of peaks')
		self.test_crop_rad_btn.clicked.connect(self.show_cropped_radii)
		fit_param_layout.addWidget(self.test_crop_rad_btn, 10, 0, 1, 4)
		self.runfit_btn = QPushButton('Run fit')
		self.runfit_btn.clicked.connect(self.run_2D_fitting)
		fit_param_layout.addWidget(self.runfit_btn, 11, 0, 1, 4)
		self.read_fit_btn = QPushButton('Read values')
		self.read_fit_btn.clicked.connect(self.read_fit_params)
		fit_param_layout.addWidget(self.read_fit_btn, 12, 0, 1, 4)
		self.save_fit_btn = QPushButton('Save values')
		self.save_fit_btn.clicked.connect(self.save_fit_params)
		fit_param_layout.addWidget(self.save_fit_btn, 13, 0, 1, 4)
		self.layout.addLayout(fit_param_layout, 4, 0, 1, 1)

		# finish creating tabs
		self.setLayout(self.layout)

		# set up error messages
		self.no_data_err = QMessageBox()
		self.no_data_err.setIcon(QMessageBox.Critical)
		self.no_data_err.setText("Error")
		self.no_data_err.setInformativeText('You must save the file information.')
		self.no_data_err.setWindowTitle("Error")

	def show_cropped_radii(self):
		"""
		Shows the FFT with the manually selected crop radii for each
		peak. Helps determine if the crop radii is accurate or not.
		"""

		# check to see if start of dictionary already exists
		if os.path.exists(self.picklename):
			# print(f'Loading {self.kn} moiré data from pickle.')
			with open(self.picklename, 'rb') as handle:
				moire_data = pickle.load(handle)

			fft_bounds = float(self.fft_bounds_txt.text())

			# make the FFT, adjust the bounds as necessary
			Kx, Ky, FFT = self.crop_data(data=moire_data)

			# read the input parameters in the boxes
			amp, xo, yo, sig_x, sig_y, theta, crad = self.get_peak_params(pk='all')

			# initialize array to hold data for all cropped radii
			cropped_peaks = np.zeros(FFT.shape)

			# loop through all the peaks
			for p in range(0, 3):
				# only show a small circle around the peak
				# center of the peak is chosen manually
				fft_cropped = self.crop_circle(fft=FFT, r=crad[p], c=(xo[p], yo[p]), x=Kx, y=Ky)
				# add the cropped section to the array holding other cropped sections
				cropped_peaks += fft_cropped

			# display the FFT

			# create axis
			if self.ax is None:
				self.ax = self.canvas.figure.subplots()

			# close axis
			self.ax.cla()
			# clear axis
			self.ax.clear()

			# plot stuff
			data = self.ax.pcolormesh(Kx, Ky, cropped_peaks/np.amax(cropped_peaks), vmax=1, shading = 'auto',cmap = 'inferno')
			self.ax.set_xlabel('1/$\lambda_X$ (nm$^{-1}$)')
			self.ax.set_ylabel('1/$\lambda_Y$ (nm$^{-1}$)')
			self.ax.set_title(self.kn)
			self.ax.set_aspect('equal')
			self.canvas.draw()

		else:
			print('Cannot find moiré data from pickle.')
			self.no_data_err.exec()



	def update_save_dir(self):
		self.save_dir = self.set_dir_btn.text()


	def get_peak_params(self, pk):
		"""
		Read all the values written in the input boxes for a given peak and
		returns them as a tuple of lists.

		Parameter order: ampl, xo, yo, sig_x, sig_y, theta, crad
		"""
		try:
			# look for peak number
			if pk == 1:
				amp = [round(float(self.amp_txt1.text()), 8)]
				xo = [round(float(self.xo_txt1.text()), 5)]
				yo = [round(float(self.yo_txt1.text()), 5)]
				crad = [round(float(self.rad_txt1.text()), 4)]
				sig_x = [round(float(self.sigx_txt1.text()), 5)]
				sig_y = [round(float(self.sigy_txt1.text()), 5)]
				theta = [round(float(self.theta_txt1.text()), 3) * np.pi/180]
			elif pk == 2:
				amp = [round(float(self.amp_txt2.text()), 8)]
				xo = [round(float(self.xo_txt2.text()), 5)]
				yo = [round(float(self.yo_txt2.text()), 5)]
				crad = [round(float(self.rad_txt2.text()), 4)]
				sig_x = [round(float(self.sigx_txt2.text()), 5)]
				sig_y = [round(float(self.sigy_txt2.text()), 5)]
				theta = [round(float(self.theta_txt2.text()), 3) * np.pi/180]
			elif pk == 3:
				amp = [round(float(self.amp_txt3.text()), 8)]
				xo = [round(float(self.xo_txt3.text()), 5)]
				yo = [round(float(self.yo_txt3.text()), 5)]
				crad = [round(float(self.rad_txt3.text()), 4)]
				sig_x = [round(float(self.sigx_txt3.text()), 5)]
				sig_y = [round(float(self.sigy_txt3.text()), 5)]
				theta = [round(float(self.theta_txt3.text()), 3) * np.pi/180]
			# case where it's all of them
			else:
				amp = [round(float(self.amp_txt1.text()), 8),
						round(float(self.amp_txt2.text()), 8),
						round(float(self.amp_txt3.text()), 8)]
				xo = [round(float(self.xo_txt1.text()), 5),
						round(float(self.xo_txt2.text()), 5),
						round(float(self.xo_txt3.text()), 5)]
				yo = [round(float(self.yo_txt1.text()), 5),
						round(float(self.yo_txt2.text()), 5),
						round(float(self.yo_txt3.text()), 5)]
				crad = [round(float(self.rad_txt1.text()), 4),
						round(float(self.rad_txt2.text()), 4),
						round(float(self.rad_txt3.text()), 4)]
				sig_x = [round(float(self.sigx_txt1.text()), 5),
						round(float(self.sigx_txt2.text()), 5),
						round(float(self.sigx_txt3.text()), 5)]
				sig_y = [round(float(self.sigy_txt1.text()), 5),
						round(float(self.sigy_txt2.text()), 5),
						round(float(self.sigy_txt3.text()), 5)]
				theta = [round(float(self.theta_txt1.text()), 3) * np.pi/180,
						round(float(self.theta_txt2.text()), 3) * np.pi/180,
						round(float(self.theta_txt3.text()), 3) * np.pi/180]

			return amp, xo, yo, sig_x, sig_y, theta, crad

		except:
			self.no_data_err.exec()

	def write_peak_params(self, pk, params):
		"""
		Takes parameters matching with the options for the input boxes
		in the Gaussian fitting section and writes them to the input boxes.

		Params is a tuple of lists.

		Parameter order: ampl, xo, yo, sig_x, sig_y, theta, crad
		"""

		amp, xo, yo, sig_x, sig_y, theta, crad = params

		if pk == 1:
			self.amp_txt1.setText(str(round(amp[0], 8)))
			self.xo_txt1.setText(str(round(xo[0], 5)))
			self.yo_txt1.setText(str(round(yo[0], 5)))
			self.sigx_txt1.setText(str(round(sig_x[0], 5)))
			self.sigy_txt1.setText(str(round(sig_y[0], 5)))
			self.theta_txt1.setText(str(round(theta[0]* 180/np.pi, 3) * 180/np.pi))
			self.rad_txt1.setText(str(round(crad[0], 4)))
		elif pk == 2:
			self.amp_txt2.setText(str(round(amp[1], 8)))
			self.xo_txt2.setText(str(round(xo[1], 5)))
			self.yo_txt2.setText(str(round(yo[1], 5)))
			self.sigx_txt2.setText(str(round(sig_x[1], 5)))
			self.sigy_txt2.setText(str(round(sig_y[1], 5)))
			self.theta_txt2.setText(str(round(theta[1]* 180/np.pi, 3)))
			self.rad_txt2.setText(str(round(crad[1], 4)))
		elif pk == 3:
			self.amp_txt3.setText(str(round(amp[2], 8)))
			self.xo_txt3.setText(str(round(xo[2], 5)))
			self.yo_txt3.setText(str(round(yo[2], 5)))
			self.sigx_txt3.setText(str(round(sig_x[2], 5)))
			self.sigy_txt3.setText(str(round(sig_y[2], 5)))
			self.theta_txt3.setText(str(round(theta[2]* 180/np.pi, 3)))
			self.rad_txt3.setText(str(round(crad[2], 4)))
		else:
			self.amp_txt1.setText(str(round(amp[0], 8)))
			self.xo_txt1.setText(str(round(xo[0], 5)))
			self.yo_txt1.setText(str(round(yo[0], 5)))
			self.sigx_txt1.setText(str(round(sig_x[0], 5)))
			self.sigy_txt1.setText(str(round(sig_y[0], 5)))
			self.theta_txt1.setText(str(round(theta[0]* 180/np.pi, 3)))
			self.rad_txt1.setText(str(round(crad[0], 4)))

			self.amp_txt2.setText(str(round(amp[1], 8)))
			self.xo_txt2.setText(str(round(xo[1], 5)))
			self.yo_txt2.setText(str(round(yo[1], 5)))
			self.sigx_txt2.setText(str(round(sig_x[1], 5)))
			self.sigy_txt2.setText(str(round(sig_y[1], 5)))
			self.theta_txt2.setText(str(round(theta[1]* 180/np.pi, 3)))
			self.rad_txt2.setText(str(round(crad[1], 4)))

			self.amp_txt3.setText(str(round(amp[2], 8)))
			self.xo_txt3.setText(str(round(xo[2], 5)))
			self.yo_txt3.setText(str(round(yo[2], 5)))
			self.sigx_txt3.setText(str(round(sig_x[2], 5)))
			self.sigy_txt3.setText(str(round(sig_y[2], 5)))
			self.theta_txt3.setText(str(round(theta[2]* 180/np.pi, 3)))
			self.rad_txt3.setText(str(round(crad[2], 4)))


	def read_fit_params(self):
		"""
		Checks to see if the moire_data dictionary is already stored.
		Writes the parameters to the input boxes.
		"""

		# check to see if start of dictionary already exists
		if os.path.exists(self.picklename):
			# print(f'Loading {self.kn} moiré data from pickle.')
			with open(self.picklename, 'rb') as handle:
				moire_data = pickle.load(handle)

			# pull out FFT bounds from dictionary
			fft_bounds = moire_data['fft_cropping'][0]
			fft_center = moire_data['fft_cropping'][1]
			# make sure that the recently added FFT line crop value exists, otherwise make it 0
			if len(moire_data['fft_cropping']) == 2:
				# add it to dictionary
				moire_data['fft_cropping'].append(0.)
				fft_line = 0.
			else:
				fft_line = moire_data['fft_cropping'][2]
			self.fft_bounds_txt.setText(str(fft_bounds))
			self.fft_center_txt.setText(str(fft_center))
			self.fft_line_txt.setText(str(fft_line))

			# pull out scan size from dictionary
			x_scan = moire_data['window_size'][0] * 1E6
			y_scan = moire_data['window_size'][1] * 1E6
			self.scanxinpt.setText(str(x_scan))
			self.scanyinpt.setText(str(y_scan))

			# check to see if fitting parameters have been saved
			if len(moire_data['xo']) > 0:
				# pull out data from dictionary
				xo = moire_data['xo']
				yo = moire_data['yo']
				crad = moire_data['crop_rad']
				amp = moire_data['gaus_amp']
				sig_x = moire_data['sig_x']
				sig_y = moire_data['sig_y']
				theta = moire_data['gaus_theta']

				# put it into a nice tuple of lists
				inputs = (amp, xo, yo, sig_x, sig_y, theta, crad)
				# write to the input fields
				self.write_peak_params(pk='all', params=inputs)

		else:
			print('Cannot find moiré data from pickle.')
			self.no_data_err.exec()



	def save_fit_params(self):
		"""
		Writes the contents of the input boxes to the moire_data dictionary.
		"""
		# check to see if start of dictionary already exists
		if os.path.exists(self.picklename):
			# print(f'Loading {self.kn} moiré data from pickle.')
			with open(self.picklename, 'rb') as handle:
				moire_data = pickle.load(handle)

			# get the parameters from the input fields and unzip them
			amp, xo, yo, sig_x, sig_y, theta, crad = self.get_peak_params(pk='all')
			moire_data.__setitem__('xo', xo)
			moire_data.__setitem__('yo', yo)
			moire_data.__setitem__('crop_rad', crad)
			moire_data.__setitem__('gaus_amp', amp)
			moire_data.__setitem__('sig_x', sig_x)
			moire_data.__setitem__('sig_y', sig_y)
			moire_data.__setitem__('gaus_theta', theta)

			# store dictionary 
			with open(self.picklename, 'wb') as handle:
				pickle.dump(moire_data, handle, protocol=4)

			print('Saved fit parameters in moire_data pickle!')

		else:
			print('Cannot find moiré data from pickle.')
			self.no_data_err.exec()



	def update_dictionaries(self):

		# check to see if start of dictionary already exists
		if os.path.exists(self.picklename):
			print(f'Loading {self.kn} moiré data from pickle.')
			with open(self.picklename, 'rb') as handle:
				moire_data = pickle.load(handle)
		else:
			print('Initializing moiré data.')
			moire_data = {'ampl_data' : [], # raw amplitude data
						'topo_data' : [], # raw topography data
						'phase_data' : [], # raw phase data
						'window_size' : [], # scan size, in m
						'l_pp' : [], # conversion factor for pixel to length
						'fft_cropping' : [], # FFT bounds and center mask redius
						'topo' : [], # processed topography data
						'ampl' : [], # processed amplitude data
						'phase' : [], # processed phase data
						'X' : [], # X vector for each scan
						'Y' : [], # Y vector for each scan
						'P_xs' : [], # phase cropped x
						'P_ys' : [], # phase cropped y
						'P_fft' : [], # phase cropped FFT
						'A_xs' : [], # amplitude cropped x
						'A_ys' : [], # amplitude cropped y
						'A_fft' : [], # amplitude cropped FFT
						'xo' : [], # kx from peaks in FFT
						'yo' : [], # ky from peaks in FFT
						'crop_rad' : [], # crop radius from peak-finding
						'gaus_amp' : [], # amplitude from Gaussian peak fit
						'sig_x' : [], # sigma_x from Gaussian peak fit
						'sig_y' : [], # sigma_y from Gaussian peak fit
						'gaus_theta' : [], # theta from Gaussian peak fit
						'twist' : [], # twist angle from strain/twist fitting
						'strain' : [], # strain from strain/twist fitting
						'global_ang' : [], # global angle from strain/twist fitting
						'shear_ang' : [] # shear angle from strain/twist fitting
						}

		# get key information
		descr = self.file.split('_') # parts will be NAME, channel.txt
		# save scan size written from window converted to meters
		moire_data['window_size'] = [float(self.scanxinpt.text()) * (10**(-6)), 
													float(self.scanyinpt.text()) * (10**(-6))]

		moire_data['fft_cropping'] = [float(self.fft_bounds_txt.text()), 
													float(self.fft_center_txt.text()),
													float(self.fft_line_txt.text())]

		# update amplitude, topography, and phase dictionaries
		if 'Ampl' in self.file:
			moire_data['ampl_data'] = self.file
			moire_data['ampl'] = np.flip(np.loadtxt(moire_data['ampl_data']), axis=0)
		elif 'Topo' in self.file:
			moire_data['topo_data'] = self.file
		else:
			moire_data['phase_data'] = self.file
			moire_data['phase'] = np.flip(np.loadtxt(moire_data['phase_data']), axis=0)

		# adjust X and Y to match scan size for amplitude
		if descr[-1] == 'Ampl.txt':
			x_pixels = np.shape(moire_data['ampl'])[1]
			y_pixels = np.shape(moire_data['ampl'])[0]
			lx_pp= moire_data['window_size'][0]/x_pixels
			ly_pp = moire_data['window_size'][1]/y_pixels
			moire_data['l_pp'] = (lx_pp, ly_pp)  # length per pixel
			moire_data['X'] = lx_pp*np.arange(0, np.shape(moire_data['ampl'])[1],1)*1E9 - moire_data['window_size'][1]*1E9/2  # nm
			moire_data['Y'] = ly_pp*np.arange(0, np.shape(moire_data['ampl'])[0],1)*1E9 - moire_data['window_size'][0]*1E9/2  # nm
		elif descr[-1] == 'Phase.txt':
			# adjust X and Y to match scan size for phase
			x_pixels = np.shape(moire_data['phase'])[1]
			y_pixels = np.shape(moire_data['phase'])[0]
			lx_pp= moire_data['window_size'][0]/x_pixels
			ly_pp = moire_data['window_size'][1]/y_pixels
			moire_data['l_pp'] = (lx_pp, ly_pp)  # length per pixel
			moire_data['X'] = lx_pp*np.arange(0, np.shape(moire_data['phase'])[1],1)*1E9 - moire_data['window_size'][1]*1E9/2  # nm
			moire_data['Y'] = ly_pp*np.arange(0, np.shape(moire_data['phase'])[0],1)*1E9 - moire_data['window_size'][0]*1E9/2  # nm
		else:
			pass

		# store dictionary 
		self.save_dir = self.set_dir_txt.text()
		with open(self.picklename, 'wb') as handle:
			pickle.dump(moire_data, handle, protocol=4)

		
	def get_file(self):
		filename, _ = QFileDialog.getOpenFileName(self, 'Open file', self.set_dir_txt.text(), 'Text files (*.txt)')
		# confirm that there is a file, preventing a crash if you cancel
		if filename:
			# split into filename and the .txt filter info
			# fname, flter = filename
			head_tail = os.path.split(filename)
			self.save_dir = f'{head_tail[0]}/'
			self.set_dir_txt.setText(self.save_dir)
			name = head_tail[1]
			# set text as the current filename
			self.filelbl.setText(f'Current file: {name}')
			# save the filename and key name to a class property
			self.file = filename
			descr = name[0:-4].split('_')
			self.kn = f'{descr[0]}'
			# get the type of channel
			if descr[-1].lower() == 'phase':
				self.c = 'P'
			elif descr[-1].lower() == 'ampl':
				self.c = 'A'
			else:
				self.c = 'T'
			# update pickle name based on 
			self.picklename = f'{self.save_dir}{self.kn}_moire_data.pickle'

			# see if moire_data.pickle is available
			if os.path.exists(self.picklename):
				# update the values that are already stored in the dictionary
				self.read_fit_params()
			else:
				pass


	def display_raw_data(self):
		try:

			# determine which channel is being shown
			# just the filename without the rest of the directory
			head_tail = os.path.split(self.file)
			# remove last 4 characters (.txt) from filename, split by _, and take last part of that in lowercase
			channel = (((head_tail[1])[0:-4].split('_'))[-1]).lower()

			# set up unit for displaying later
			unit = '°'
			# change if the channel is topo or ampl
			if not channel == 'phase':
				unit = 'pm'

			# create axis
			if self.ax is None:
				self.ax = self.canvas.figure.subplots()

			# clear axis
			self.ax.clear()

			# check to see if start of dictionary already exists
			if os.path.exists(self.picklename):
				print(f'Loading {self.kn} moiré data from pickle.')
				with open(self.picklename, 'rb') as handle:
					moire_data = pickle.load(handle)
			else:
				print('Cannot find moiré data from pickle.')

			# plot stuff
			data = self.ax.pcolormesh(moire_data['X'], moire_data['Y'],
							 moire_data[channel]*1E12, shading = 'auto',cmap = 'viridis')
			# self.canvas.figure.colorbar(data, ax=self.ax, orientation='vertical').set_label(label=f'{channel} ({unit})')
			self.ax.set_xlabel('X (nm)')
			self.ax.set_ylabel('Y (nm)')
			self.ax.set_title(self.kn)
			self.ax.set_aspect('equal')

			self.canvas.draw()
		except:
			self.no_data_err.exec()


	def display_fft_data(self):
		"""
		Pulls the filename, determines which data is being shown (A or P),
		crops the data, runs an FFT, then displays it.
		"""

		try:

			# check to see if start of dictionary already exists
			if os.path.exists(self.picklename):
				print(f'Loading {self.kn} moiré data from pickle.')
				with open(self.picklename, 'rb') as handle:
					moire_data = pickle.load(handle)
			else:
				print('Cannot find moiré data from pickle.')


			# set up unit for displaying later
			unit = '°'
			fft_channel = 'phase'
			# change if the channel is topo or ampl
			if not self.c == 'P':
				unit = 'pm'
				fft_channel = 'ampl'

			# write the correct key names based on channel
			c_xs = f'{self.c}_xs'
			c_ys = f'{self.c}_ys'
			c_fft = f'{self.c}_fft'

			moire_data[c_fft], moire_data[c_xs], moire_data[c_ys] = self.makefft(moire_data[fft_channel], 
																					moire_data['window_size'][0]*1E9, 
																					moire_data['window_size'][1]*1E9)

			# create axis
			if self.ax is None:
				self.ax = self.canvas.figure.subplots()

			# close axis
			self.ax.cla()
			# clear axis
			self.ax.clear()

			# make the FFT, cropping the bounds
			Kx, Ky, FFT = self.crop_data(data=moire_data)

			# remove the bright spot in the center to make scaling easier
			FFT = self.remove_fft_lowfreq(fft=FFT, x=Kx, y=Ky, bounds=float(self.fft_center_txt.text()))

			# plot stuff
			data = self.ax.pcolormesh(Kx, Ky, FFT/np.amax(FFT), vmax=1, shading = 'auto',cmap = 'inferno')
			self.ax.set_xlabel('1/$\lambda_X$ (nm$^{-1}$)')
			self.ax.set_ylabel('1/$\lambda_Y$ (nm$^{-1}$)')
			self.ax.set_title(self.kn)
			self.ax.set_aspect('equal')

			self.canvas.draw()

			# store dictionary 
			with open(self.picklename, 'wb') as handle:
				pickle.dump(moire_data, handle, protocol=4)

		except:
			self.no_data_err.exec()


	def update_FFT_fit(self, pk):
		"""
		Reads all the values written in the input boxes for a given peak and plots
		the corresponding 2D Gaussian function.
		"""

		# find the bounds of the FFT
		fft_bounds = float(self.fft_bounds_txt.text())

		# create an artificial kx and ky
		kx = np.linspace(-fft_bounds, fft_bounds, 201)
		ky = np.linspace(-fft_bounds, fft_bounds, 201)

		x, y = np.meshgrid(kx, ky)

		# get the input parameters
		amp, xo, yo, sig_x, sig_y, theta, crad = self.get_peak_params(pk=pk)

		# update fft to erase previous fit attempts
		self.display_fft_data()

		# run all of the peaks through the 2D Gaussian function and plot
		for p in range(len(amp)):
			fit = self.twoD_Gaussian((x, y), amp[p], xo[p], yo[p], 
									sig_x[p], sig_y[p], theta[p])
			self.ax.contour(x, y, fit.reshape(201, 201), 2, colors='w')

		self.canvas.draw()

		print('Done updating 2D Gaussian fits!')


	def makefft(self, data, lx, ly):
		"""
		Make FFT given a certain image.
		"""
		padding = 5
		sy, sx = data.shape
		window2d = np.sqrt(np.outer(scipy.signal.windows.hann(sx), scipy.signal.windows.hann(sy)))
		forfft = np.pad(np.multiply(data.T, window2d), [(sy * padding, sy * padding), (sx * padding, sx * padding)])
		scalex = lx * len(forfft.T) / sx # nm
		scaley = ly * len(forfft) / sy # nm
		xs = np.linspace(-len(forfft.T) / 2 / scalex, len(forfft.T) / 2 / scalex, len(forfft.T))
		ys = np.linspace(-len(forfft) / 2 / scaley, len(forfft) / 2 / scaley, len(forfft))

		fft = np.abs(scipy.fft.fftshift(scipy.fft.fft2(forfft)))

		return fft, xs, ys


	def crop_data(self, data):
		"""
		data = data dictionary (moire_data)
		Channel = 'A' or 'P' for amplitude or phase
		"""

		# read bounds from text entry
		limit = float(self.fft_bounds_txt.text())

		# write the correct key names based on channel
		c_xs = f'{self.c}_xs'
		c_ys = f'{self.c}_ys'
		c_fft = f'{self.c}_fft'

		# crop the data to fit within the bounds
		X_CR = np.where(np.abs(data[c_xs]) < limit)[0]
		Y_CR = np.where(np.abs(data[c_ys]) < limit)[0]
		return data[c_xs][X_CR[0]:X_CR[-1]], data[c_ys][Y_CR[0]:Y_CR[-1]], data[c_fft][X_CR[0]:X_CR[-1], Y_CR[0]:Y_CR[-1]]


	def remove_fft_lowfreq(self, fft, x, y, bounds=0.001):
		"""
		Removes the low freqency noise of an FFT from the center.

		Basically places a small, blurred black circle over the center of an FFT.

		Recent addition: also removes the line around y=0.

		Bounds default is 0.001 nm^-1. 
		"""
		# find indices where there are points within the bound box on both ends
		x_ind_max = np.where(x < bounds)
		x_ind_min = np.where(x > -1*bounds)
		y_ind_max = np.where(y < bounds)
		y_ind_min = np.where(y > -1*bounds )

		# find the points that fulfill both conditions
		x_ind = np.intersect1d(x_ind_min, x_ind_max)
		y_ind = np.intersect1d(y_ind_min, y_ind_max)

		# cut out the circle
		for xi in x_ind:
			for yi in y_ind:
				# make sure they're within a circle
				radius = np.sqrt(y[yi]**2 + x[xi]**2)
				if radius < bounds:
					fft[yi, xi] = 0
				else:
					pass

		# get the bounds for the box height
		box_height = float(self.fft_line_txt.text())

		# make sure that the box height is not zero
		if not box_height == 0.:
			# find indices where it is within the box around the y=0 line
			# yes i know i'm overwriting stuff and it's bad programming but idc
			y_ind_max = np.where(y < box_height)
			y_ind_min = np.where(y > -1*box_height)

			# find the intersections
			y_ind = np.intersect1d(y_ind_min, y_ind_max)

			# given that this is a box, the y's should be constant
			y_max = np.amax(y_ind)
			y_min = np.amin(y_ind)

			# cut out the box
			fft[y_min:y_max, :] = 0

		fft = scipy.ndimage.gaussian_filter(fft, 2)

		return fft


	def twoD_Gaussian(self, xy, amplitude, xo, yo, sigma_x, sigma_y, theta):
		"""
		Fits a 2D Gaussian. 
		"""

		x, y = xy
		xo = float(xo)
		yo = float(yo)
		a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
		b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
		c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
		g = amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
		return g.ravel()


	def crop_circle(self, fft, r, c, x, y):
		"""
		Takes an FFT and returns the image cropped in a circle around a given peak.

		r = radius
		c = (x, y) = center of circle
		x, y = coordinates of the image relative to the indices of the image
		"""
		# extract center coordinates
		x0, y0 = c

		# find indices where there are points within the bound box on both ends
		x_ind_max = np.where(x < (x0 + r))
		x_ind_min = np.where(x > (x0 - r))
		y_ind_max = np.where(y < (y0 + r))
		y_ind_min = np.where(y > (y0 - r))

		# find the points that fulfill both conditions
		x_ind = np.intersect1d(x_ind_min, x_ind_max)
		y_ind = np.intersect1d(y_ind_min, y_ind_max)

		# initialize a mask of zeros
		mask = np.zeros(fft.shape)

		# only write values that are within the circle
		for xi in x_ind:
			for yi in y_ind:
				# make sure they're within a circle
				radius = np.sqrt((y[yi] - y0)**2 + (x[xi] - x0)**2)
				if radius <= r:
					mask[yi, xi] = fft[yi, xi]
				else:
					pass

		return mask


	def run_2D_fitting(self):
		"""

		Returns fitted 2D Gaussian parameters (g_amp, xo, yo, sigma_x, sigma_y, theta)
		"""

		# start a measure to see how long it takes
		time_lst = [time.time()]

		# check to see if start of dictionary already exists
		if os.path.exists(self.picklename):
			# print(f'Loading {self.kn} moiré data from pickle.')
			with open(self.picklename, 'rb') as handle:
				moire_data = pickle.load(handle)
		else:
			print('Cannot find moiré data from pickle.')

		fft_bounds = float(self.fft_bounds_txt.text())

		# Added to manually take FFT rather than take FFT while loading
		if self.c == 'P':
			moire_data['P_fft'], moire_data['P_xs'], moire_data['P_ys'] = self.makefft(moire_data['phase'], moire_data['window_size'][0]*1E9, moire_data['window_size'][1]*1E9)
		else:
			moire_data['A_fft'], moire_data['A_xs'], moire_data['A_ys'] = self.makefft(moire_data['ampl'], moire_data['window_size'][0]*1E9, moire_data['window_size'][1]*1E9)

		# make the FFT, adjust the bounds as necessary
		Kx, Ky, FFT = self.crop_data(data=moire_data)

		# read the input parameters in the boxes
		amp, xo, yo, sig_x, sig_y, theta, crad = self.get_peak_params(pk='all')

		# initialize matrix for storing data
		fit_data = np.zeros((3, len(FFT.ravel())))
		fit_params = np.zeros((3, 6)) # store fit parameters

		print('Starting peak-finding.')

		# loop through all the peaks
		for p in range(0, 3):
			# only show a small circle around the peak
			# center of the peak is chosen manually
			fft_cropped = self.crop_circle(fft=FFT, r=crad[p], c=(xo[p], yo[p]), x=Kx, y=Ky)

			# create the xy data for fitting the 2D Gaussian
			x, y = np.meshgrid(Kx, Ky)

			# set up initial guess for curvefitting for a single peak
			# guess = xy, amplitude, x0, y0, sigma_x, sigma_y, theta
			# good initial guesses: (1, peak x, peak y, 0.003, 0.003, 0)
			initial_guess = (amp[p], xo[p], yo[p], sig_x[p], sig_y[p], theta[p])

			# set up the bounds
			bounds_min = [-np.inf, xo[p]-crad[p], yo[p]-crad[p], 0., 0., -np.pi]
			bounds_max = [np.inf, xo[p]+crad[p], yo[p]+crad[p], np.inf, np.inf, np.pi]

			# do curve fit
			popt, pcov = scipy.optimize.curve_fit(self.twoD_Gaussian, (x, y), fft_cropped.ravel(), p0=initial_guess, bounds=(bounds_min, bounds_max), maxfev=6000)

			# store parameters for later
			fit_params[p] = popt

			print(f'Fit parameters for peak {p} are: {popt}')

			# create a set of data with the correct parameters
			fit_data[p] = self.twoD_Gaussian((x, y), *popt)

			print(f'Finished finding peak {p+1}')

		# write_params = (fit_params[:][0], fit_params[:][1], fit_params[:][2], fit_params[:][3], 
		# 			fit_params[:][4], fit_params[:][5])

		# write the fit params to the input fields
		# self.write_peak_params(pk='all', params=write_params)

		# doing this for now because I don't want to figure out the error
		self.amp_txt1.setText(str(round(fit_params[0][0], 8)))
		self.xo_txt1.setText(str(round(fit_params[0][1], 5)))
		self.yo_txt1.setText(str(round(fit_params[0][2], 5)))
		self.sigx_txt1.setText(str(round(fit_params[0][3], 5)))
		self.sigy_txt1.setText(str(round(fit_params[0][4], 5)))
		self.theta_txt1.setText(str(round(fit_params[0][5], 3)))
		self.amp_txt2.setText(str(round(fit_params[1][0], 8)))
		self.xo_txt2.setText(str(round(fit_params[1][1], 5)))
		self.yo_txt2.setText(str(round(fit_params[1][2], 5)))
		self.sigx_txt2.setText(str(round(fit_params[1][3], 5)))
		self.sigy_txt2.setText(str(round(fit_params[1][4], 5)))
		self.theta_txt2.setText(str(round(fit_params[1][5], 3)))
		self.amp_txt3.setText(str(round(fit_params[2][0], 8)))
		self.xo_txt3.setText(str(round(fit_params[2][1], 5)))
		self.yo_txt3.setText(str(round(fit_params[2][2], 5)))
		self.sigx_txt3.setText(str(round(fit_params[2][3], 5)))
		self.sigy_txt3.setText(str(round(fit_params[2][4], 5)))
		self.theta_txt3.setText(str(round(fit_params[2][5], 3)))


		# plot the fit
		self.update_FFT_fit(pk='all')

		time_lst.append(time.time())
		print(f'It took {(time_lst[-1] - time_lst[-2])/60} min to fit peaks.')


