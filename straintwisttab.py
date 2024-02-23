"""

Filename: straintwisttab.py
Author: Risa Hocking
Purpose: Populates the twist and strain fit tab for 
fitting PFM/TFM/LFM data of moiré structures.

"""

import sys
import os
import pickle
import copy
import numpy as np


from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

import matplotlib
from matplotlib.patches import Ellipse
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
matplotlib.use('Qt5Agg')

import scipy 
import scipy.ndimage
import scipy.fft
import scipy.signal
from scipy.optimize import curve_fit

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

class StrainTwistTab(QWidget):
	def __init__(self):
		super().__init__()

		# create tab layouts
		self.layout = QGridLayout()

		# initialize data to be shared between files
		self.file = ''
		self.kn = ''  # key name
		self.save_dir = ''
		self.c = ''  # channel type ('A', 'P', or 'T')
		self.picklename = ''
		self.fft_bounds = 0.05
		self.lat_param1 = 0.2504

		# initialize other windows
		self.adv_param_w = None

		# set up file import
		self.fileimportbtn = QPushButton('Select file')
		self.fileimportbtn.clicked.connect(self.get_file)
		self.filelbl = QLabel(f'Current file: none')
		file_layout = QVBoxLayout()
		file_layout.addWidget(self.filelbl)
		file_layout.addWidget(self.fileimportbtn)
		self.layout.addLayout(file_layout, 0, 0, 1, 1)

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
		self.fft_center_txt = QLineEdit('0.005')
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

		# add in the peak finding information
		kvec_layout = QGridLayout()
		self.param_descr_lbl = QLabel('Note: Peak positions refer to the upper three peaks in FFT.')
		self.pk1_lbl = QLabel('Right peak')
		self.pk1_lbl.setFont(QFont('',weight=QFont.Bold))
		self.pk2_lbl = QLabel('Left peak')
		self.pk2_lbl.setFont(QFont('',weight=QFont.Bold))
		self.pk3_lbl = QLabel('Middle peak')
		self.pk3_lbl.setFont(QFont('',weight=QFont.Bold))
		kvec_layout.addWidget(self.param_descr_lbl, 0, 0, 1, 4)
		kvec_layout.addWidget(self.pk1_lbl, 1, 1, 1, 1)
		kvec_layout.addWidget(self.pk2_lbl, 1, 2, 1, 1)
		kvec_layout.addWidget(self.pk3_lbl, 1, 3, 1, 1)
		self.xo_lbl = QLabel('X (nm⁻¹)')
		self.xo_txt1 = QLabel('0.')
		self.xo_txt2 = QLabel('0.')
		self.xo_txt3 = QLabel('0.')
		kvec_layout.addWidget(self.xo_lbl, 2, 0, 1, 1)
		kvec_layout.addWidget(self.xo_txt1, 2, 1, 1, 1)
		kvec_layout.addWidget(self.xo_txt2, 2, 2, 1, 1)
		kvec_layout.addWidget(self.xo_txt3, 2, 3, 1, 1)
		self.yo_lbl = QLabel('Y (nm⁻¹)')
		self.yo_txt1 = QLabel('0.')
		self.yo_txt2 = QLabel('0.')
		self.yo_txt3 = QLabel('0.')
		kvec_layout.addWidget(self.yo_lbl, 3, 0, 1, 1)
		kvec_layout.addWidget(self.yo_txt1, 3, 1, 1, 1)
		kvec_layout.addWidget(self.yo_txt2, 3, 2, 1, 1)
		kvec_layout.addWidget(self.yo_txt3, 3, 3, 1, 1)
		self.layout.addLayout(kvec_layout, 4, 0, 1, 1)

		self.horz_line3 = QHSeparationLine()
		self.layout.addWidget(self.horz_line3, 5, 0, 1, 1)

		# set up fit parameter area for finding stress and strain
		fit_param_layout = QGridLayout()
		self.minval_lbl = QLabel('Minimum value')
		self.minval_lbl.setFont(QFont('',weight=QFont.Bold))
		self.pkval_lbl = QLabel('Peak value')
		self.pkval_lbl.setFont(QFont('',weight=QFont.Bold))
		self.maxval_lbl = QLabel('Maximum value')
		self.maxval_lbl.setFont(QFont('',weight=QFont.Bold))
		fit_param_layout.addWidget(self.minval_lbl, 0, 1, 1, 1)
		fit_param_layout.addWidget(self.pkval_lbl, 0, 2, 1, 1)
		fit_param_layout.addWidget(self.maxval_lbl, 0, 3, 1, 1)
		self.glob_ang_lbl = QLabel('Global angle (°)')
		self.twist_lbl = QLabel('Twist angle (°)')
		self.strain_lbl = QLabel('Strain (%)')
		self.shear_ang_lbl = QLabel('Shear angle (°)')
		fit_param_layout.addWidget(self.glob_ang_lbl, 1, 0, 1, 1)
		fit_param_layout.addWidget(self.twist_lbl, 2, 0, 1, 1)
		fit_param_layout.addWidget(self.strain_lbl, 3, 0, 1, 1)
		fit_param_layout.addWidget(self.shear_ang_lbl, 4, 0, 1, 1)
		self.glob_ang_txt2 = QLineEdit('30')
		fit_param_layout.addWidget(self.glob_ang_txt2, 1, 2, 1, 1)
		self.twist_txt1 = QLabel('0.2')
		self.twist_txt2 = QLineEdit('0.2')
		self.twist_txt3 = QLabel('0.2')
		self.twist_txt2.editingFinished.connect(lambda: self.hit_enter('twist'))
		fit_param_layout.addWidget(self.twist_txt1, 2, 1, 1, 1)
		fit_param_layout.addWidget(self.twist_txt2, 2, 2, 1, 1)
		fit_param_layout.addWidget(self.twist_txt3, 2, 3, 1, 1)
		self.strain_txt1 = QLabel('0.04')
		self.strain_txt2 = QLineEdit('0.04')
		self.strain_txt3 = QLabel('0.04')
		self.strain_txt2.editingFinished.connect(lambda: self.hit_enter('strain'))
		fit_param_layout.addWidget(self.strain_txt1, 3, 1, 1, 1)
		fit_param_layout.addWidget(self.strain_txt2, 3, 2, 1, 1)
		fit_param_layout.addWidget(self.strain_txt3, 3, 3, 1, 1)
		self.shear_ang_txt2 = QLineEdit('60.')

		fit_param_layout.addWidget(self.shear_ang_txt2, 4, 2, 1, 1)

		self.advfitbtn = QPushButton('Advanced Fit Parameters')
		self.advfitbtn.clicked.connect(self.show_advanced_param_window)
		fit_param_layout.addWidget(self.advfitbtn, 5, 0, 1, 4)

		# self.latparam_lbl = QLabel('Lattice parameter (nm)')
		# fit_param_layout.addWidget(self.latparam_lbl, 5, 0, 1, 1)
		# self.latparam_txt = QLineEdit('0.2504')
		# fit_param_layout.addWidget(self.latparam_txt, 5, 1, 1, 1)
		self.homostructure_warning_lbl = QLabel('Note: this fitting is only designed to function with hexagonal homostructures.')
		# fit_param_layout.addWidget(self.latparam_txt, 5, 2, 1, 2)
		self.runfit_btn = QPushButton('Run fit')
		self.runfit_btn.clicked.connect(self.fit_straintwist)
		fit_param_layout.addWidget(self.runfit_btn, 6, 0, 1, 4)
		self.read_fit_btn = QPushButton('Read values')
		self.read_fit_btn.clicked.connect(self.read_fit_params)
		fit_param_layout.addWidget(self.read_fit_btn, 7, 0, 1, 4)
		self.save_fit_btn = QPushButton('Save values')
		self.save_fit_btn.clicked.connect(self.save_fit_params)
		fit_param_layout.addWidget(self.save_fit_btn, 8, 0, 1, 4)

		self.layout.addLayout(fit_param_layout, 6, 0, 1, 1)

		# finish creating tabs
		self.setLayout(self.layout)


		# set up error messages

		# no file data is saved
		self.no_data_err = QMessageBox()
		self.no_data_err.setIcon(QMessageBox.Critical)
		self.no_data_err.setText("Error")
		self.no_data_err.setInformativeText('You must save the file information.')
		self.no_data_err.setWindowTitle("Error")

	def show_advanced_param_window(self):
		self.adv_param_w = AdvancedFitParametersWindow()
		self.adv_param_w.updateClicked.connect(self.on_advpw_confirm)
		self.adv_param_w.show()

	def on_advpw_confirm(self, lat_param):
		self.lat_param1 = float(lat_param)
		print(f'Lattice parameter changed to {float(self.lat_param1)}')


	def update_save_dir(self):
		self.save_dir = self.set_dir_btn.text()


	def hit_enter(self, btn):

		if btn == 'twist':
			self.twist_txt1.setText(self.twist_txt2.text())
			self.twist_txt3.setText(self.twist_txt2.text())
		elif btn == 'strain':
			self.strain_txt1.setText(self.strain_txt2.text())
			self.strain_txt3.setText(self.strain_txt2.text())
		else:
			pass


	def get_fit_params(self):
		"""
		Read all the values written in the input boxes for a given peak and
		returns them as a tuple of lists.

		Parameter order: global_ang, twist, strain, shear_ang
		"""

		twist = [float(self.twist_txt1.text()) * np.pi/180,
				float(self.twist_txt2.text()) * np.pi/180,
				float(self.twist_txt3.text()) * np.pi/180]

		strain = [float(self.strain_txt1.text()) / 100,
				float(self.strain_txt2.text()) / 100,
				float(self.strain_txt3.text()) / 100]

		shear_ang = [float(self.shear_ang_txt2.text()) * np.pi/180]

		global_ang = [float(self.glob_ang_txt2.text()) * np.pi/180]

		return global_ang, twist, strain, shear_ang

	def write_fit_params(self, params):
		"""
		Takes parameters matching with the options for the input boxes
		in the Gaussian fitting section and writes them to the input boxes.

		Params is a tuple of lists.

		Parameter order: global angle, twist, strain, shear angle
		"""

		global_ang, twist, strain, shear_ang = params

		self.glob_ang_txt2.setText(f'{global_ang[0]*180/np.pi:.3f}')

		self.twist_txt1.setText(f'{twist[0]*180/np.pi:.3f}')
		self.twist_txt2.setText(f'{twist[1]*180/np.pi:.3f}')
		self.twist_txt3.setText(f'{twist[2]*180/np.pi:.3f}')

		self.strain_txt1.setText(f'{strain[0]*100:.3f}')
		self.strain_txt2.setText(f'{strain[1]*100:.3f}')
		self.strain_txt3.setText(f'{strain[2]*100:.3f}')

		self.shear_ang_txt2.setText(f'{shear_ang[0]*180/np.pi:.3f}')


	def read_fit_params(self):
		"""
		Read the existing parameters that have been saved to the moire_data.pickle file
		and write them to the screen.
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

			# check to see if the peak-finding parameters exist
			if len(moire_data['xo']) > 0:

				# pull out data from dictionary
				xo = moire_data['xo']
				yo = moire_data['yo']

				# write to the labels
				self.xo_txt1.setText(str(round(xo[0], 4)))
				self.xo_txt2.setText(str(round(xo[1], 4)))
				self.xo_txt3.setText(str(round(xo[2], 4)))
				self.yo_txt1.setText(str(round(yo[0], 4)))
				self.yo_txt2.setText(str(round(yo[1], 4)))
				self.yo_txt3.setText(str(round(yo[2], 4)))

			# make sure that the strain/twist parameters exist in the pickle
			# and read them into the GUI
			if len(moire_data['strain']) > 0:

				global_ang = moire_data['global_ang']
				twist = moire_data['twist']
				strain = moire_data['strain']
				shear_ang = moire_data['shear_ang']

				self.write_fit_params(params=(global_ang, twist, strain, shear_ang))


		else:
			print('Cannot find moiré data from pickle.')


	def save_fit_params(self):
		"""
		Writes the contents of the input boxes to the moire_data dictionary.
		"""
		# check to see if start of dictionary already exists
		if os.path.exists(self.picklename):
			with open(self.picklename, 'rb') as handle:
				moire_data = pickle.load(handle)
		else:
			print('Cannot find moiré data from pickle.')

		# get the parameters from the input fields and unzip them
		global_ang, twist, strain, shear_ang = self.get_fit_params()

		moire_data['twist'] = twist
		moire_data['strain'] = strain
		moire_data['global_ang'] = global_ang
		moire_data['shear_ang'] = shear_ang
		# moire_data.__setitem__('twist', twist)
		# moire_data.__setitem__('strain', strain)
		# moire_data.__setitem__('global_ang', global_ang)
		# moire_data.__setitem__('shear_ang', shear_ang)

		# store dictionary 
		self.save_dir = self.set_dir_txt.text()
		with open(self.picklename, 'wb') as handle:
			pickle.dump(moire_data, handle, protocol=4)

		print('Saved fit parameters in moire_data pickle!')


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
		descr = self.file.split('_')# parts will be NAME, channel.txt
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
			# adjust X and Y to match scan size for amplitude
			x_pixels = np.shape(moire_data['ampl'])[1]
			y_pixels = np.shape(moire_data['ampl'])[0]
			lx_pp= moire_data['window_size'][0]/x_pixels
			ly_pp = moire_data['window_size'][1]/y_pixels
			moire_data['l_pp'] = (lx_pp, ly_pp)  # length per pixel
			moire_data['X'] = lx_pp*np.arange(0, np.shape(moire_data['ampl'])[1],1)*1E9 - moire_data['window_size'][1]*1E9/2  # nm
			moire_data['Y'] = ly_pp*np.arange(0, np.shape(moire_data['ampl'])[0],1)*1E9 - moire_data['window_size'][0]*1E9/2  # nm

		elif 'Topo' in self.file:
			moire_data['topo_data'] = self.file 
		else:
			moire_data['phase_data'] = self.file 
			moire_data['phase'] = np.flip(np.loadtxt(moire_data['phase_data']), axis=0)
			# adjust X and Y to match scan size for amplitude
			x_pixels = np.shape(moire_data['phase'])[1]
			y_pixels = np.shape(moire_data['phase'])[0]
			lx_pp= moire_data['window_size'][0]/x_pixels
			ly_pp = moire_data['window_size'][1]/y_pixels
			moire_data['l_pp'] = (lx_pp, ly_pp)  # length per pixel
			moire_data['X'] = lx_pp*np.arange(0, np.shape(moire_data['phase'])[1],1)*1E9 - moire_data['window_size'][1]*1E9/2  # nm
			moire_data['Y'] = ly_pp*np.arange(0, np.shape(moire_data['phase'])[0],1)*1E9 - moire_data['window_size'][0]*1E9/2  # nm

		# store dictionary 
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

			self.read_fit_params()



	def display_raw_data(self):

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


	def display_fft_data(self):
		"""
		Pulls the filename, determines which data is being shown (A or P),
		crops the data, runs an FFT, then displays it.
		"""

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



	def makefft(self, data, lx, ly):
		"""
		Make FFT given a certain image.
		"""
		padding = 3
		sy, sx = data.shape
		window2d = np.sqrt(np.outer(scipy.signal.windows.hann(sx), scipy.signal.windows.hann(sy)))
		forfft = np.pad(np.multiply(data.T, window2d), [(sy * padding, sy * padding), (sx * padding, sx * padding)])
		scalex = lx * len(forfft.T) / sx # nm
		scaley = ly * len(forfft) / sy # nm
		xs = np.linspace(-len(forfft.T) / 2 / scalex, len(forfft.T) / 2 / scalex, len(forfft.T))
		ys = np.linspace(-len(forfft) / 2 / scaley, len(forfft) / 2 / scaley, len(forfft))

		fft = np.abs(scipy.fft.fftshift(scipy.fft.fft2(forfft)))

		return fft, xs, ys


	def remove_fft_lowfreq(self, fft, x, y, bounds=0.001):
		"""
		Removes the low freqency noise of an FFT from the center.

		Basically places a small, blurred black circle over the center of an FFT.

		Bounds default is 0.02 nm^-1. 
		"""
		# find indices where there are points within the bound box on both ends
		x_ind_max = np.where(x < bounds)
		x_ind_min = np.where(x > -1*bounds)
		y_ind_max = np.where(y < bounds)
		y_ind_min = np.where(y > -1*bounds)

		# find the points that fulfill both conditions
		x_ind = np.intersect1d(x_ind_min, x_ind_max)
		y_ind = np.intersect1d(y_ind_min, y_ind_max)

		for xi in x_ind:
			for yi in y_ind:
				# make sure they're within a circle
				radius = np.sqrt(y[yi]**2 + x[xi]**2)
				if radius < bounds:
					fft[yi, xi] = 0
				else:
					pass

		fft = scipy.ndimage.gaussian_filter(fft, 3)

		return fft


	def is_in_ellipse(self, xy, xo, yo, sig_x, sig_y, theta):
		"""
		Determines if a point is in the ellipse or not.
		xo, yo = floats, center of ellipse
		sig_x, sig_x = floats, semi-axes for ellipse
		theta = float, angle of rotation for ellipse
		xy = (x, y) = np.array, point to be considered
		"""

		# unpack reference point
		xy = np.ravel(xy)
		po, qo = xy[0], xy[1]


		# set up ellipse coefficients
		s, c = np.sin(theta), np.cos(theta)
		a, b = sig_x, sig_y
		A = (-a**2 * s**2 - b**2 * c**2 + (yo - qo)**2)
		B = 2 * (c * s * (a**2 - b**2) - (xo - po)*(yo - qo))
		C = (-a**2 * c**2 - b**2 * s**2 + (xo - po)**2)

		# check if it's within the ellipse
		if B**2 - 4*A*C < 0:
			return True
		else:
			return False


	def brute_force_uncertainties(self, epss, twis, moire_data, sigma=2):
	    
		"""
		Takes in the potential range of twists and strains that may
		be found from the 2D Gaussian peak fitting and returns the
		minimum and maximum values of both strain and twist.

		epss: np.array, goes from minimum to maximum strain to be tested
		twis: np.array, goes from minimum to maximum twist to be tested
		moire_data: dictionary containing peak and previous fit information
		sigma: number of standard deviations to be fit
		"""
		# set up stuff for pulling out reciprocal lattice vectors 
		aa = self.lat_param1 # lattice parameter of hBN, nm
		G1 = 4 * np.pi / np.sqrt(3) / aa * np.array([[0], [-1]]) # reciprocal lattice vector of hBN
		G2 = 4 * np.pi / np.sqrt(3) / aa * np.array([[np.sqrt(3) / 2], [1 / 2]])  # reciprocal lattice vector of hBN
		theta_0 = np.array([[1, 0], [0, 1]]) # identity matrix of 2x2
		def R(theta):
			return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
		def S(eps, phi):
			return eps * R(-phi) @ np.array([[-1, 0], [0, 0.21]]) @ R(phi)
		def g1(t, theta, eps, phi):
			return R(t) @ (R(theta / 2) @ (theta_0 - S(eps / 2, phi)) - R(-theta / 2) @ (theta_0 - S(-eps / 2, phi))) @ G1
		def g2(t, theta, eps, phi):
			return R(t) @ (R(theta / 2) @ (theta_0 - S(eps / 2, phi)) - R(-theta / 2) @ (theta_0 - S(-eps / 2, phi))) @ G2
		def g3(t, theta, eps, phi):
			return g1(t, theta, eps, phi) + g2(t, theta, eps, phi)

		# read the 2D Gaussian fit parameters from the dictionary
		xo = moire_data['xo']
		yo = moire_data['yo']
		# sig_x = moire_data['sig_x']
		# sig_y = moire_data['sig_y']
		sig_x = [x*(sigma/2) for x in moire_data['sig_x']] # scale sig_x 
		sig_y = [y*(sigma/2) for y in moire_data['sig_y']] # scale sig_y
		theta = moire_data['gaus_theta']

		# pull out other constants from fit
		shear_ang = moire_data['shear_ang'][0]
		glob_ang = moire_data['global_ang'][0]

		# create something to store values
		str_mat = np.zeros((len(epss), len(twis)))
		tw_mat = np.zeros((len(epss), len(twis)))
		uncert_points = np.zeros((4, 2))

		# raster through the potential range of strain and twist
		for i, s in enumerate(epss):
			for j, tw in enumerate(twis):
				# go through all the peaks
				gg1 = g1(t=glob_ang, theta=tw, eps=s, phi=shear_ang)
				gg2 = g2(t=glob_ang, theta=tw, eps=s, phi=shear_ang)
				gg3 = g3(t=glob_ang, theta=tw, eps=s, phi=shear_ang)

				# create an out-of-bounds test, initializing all variables as True
				oob_test = np.full((3, 1), True)

				# go through all the ellipse and determine if the point is inside the ellipse
				for p, g in enumerate([gg1, gg2, gg3]):
					if self.is_in_ellipse(g, xo[p]*2*np.pi, yo[p]*2*np.pi, sig_x[p]*2*np.pi, sig_y[p]*2*np.pi, theta[p]):
						oob_test[p][0] = False
					else:
						pass
	
				# see if all peaks are false
				if not any(oob_test):
					# write the current strain and twist values to the matrix
					str_mat[i, j] = s
					tw_mat[i, j] = tw


		# pull out the nonzero elements from each array
		pot_strains = str_mat[np.nonzero(str_mat)]
		pot_twists = tw_mat[np.nonzero(tw_mat)]

		print('Potential strains:')
		print(pot_strains)
		print('Potential twists:')
		print(pot_twists)
		try:
			# pull out the min and max values for strain and twist
			strain_max = np.amax(pot_strains) 
			strain_min = np.amin(pot_strains)

			twist_max = np.amax(pot_twists)
			twist_min = np.amin(pot_twists)

			# find the indices of the min and max points
			ind_smaxy, ind_smaxx = np.where(str_mat == strain_max)
			ind_sminy, ind_sminx = np.where(str_mat == strain_min)
			ind_tmaxy, ind_tmaxx = np.where(tw_mat == twist_max)
			ind_tminy, ind_tminx = np.where(tw_mat == twist_min)
		except:
			print('Not able to find min/max strain and twist. Adjust initial parameters.')

		return strain_min, strain_max, twist_min, twist_max

	def find_moire_fit_uncertainties(self, moire_data, sigma=2):
		"""
		Creates a set of vectors for each set of FFT peaks that vary twist and strain
		from 1% to 1000% of their current values. Determines which of these twist and
		strain values allow all reciprocal space vectors to land within the ellipse
		mapped out by sigma/2 standard deviations in x and y from the center.

		Takes in the moire_data dictionary loaded from another file.

		"""

		# pull out only the fits from the peak values
		strain = moire_data['strain'][1]
		twist = moire_data['twist'][1]

		print(f'Initial strain is {strain*100:.3f}% and initial twist is {twist*180/np.pi:.3f}°')

		# establish starting conditions for twist/strain brute forcing
		# start with 10% of peak strain as min and 1000% of peak strain as max
		smin = strain*0.01
		smax = strain*100
		sn = 1000
		# start with 50% of peak twist as min and 150% of peak twist as max
		tmin = twist*0.1
		tmax = twist*10
		tn = 1000

		# initialize conditions for a while loop
		twist_okay = False
		strain_okay = False
		scounter = 0
		tcounter = 0
		smax_prev = smax
		smin_prev = smin
		tmax_prev = tmax
		tmin_prev = tmin

		while twist_okay is False or strain_okay is False:

			print(f'Exploratory strain range is {smin*100:.3f}% to {smax*100:.3f}%, ' +
				f'exploratory twist range is {tmin*180/np.pi:.3f}° to {tmax*180/np.pi:.3f}°')

			# create a varied grid to test values
			epss = np.linspace(smin, smax, sn)
			twis = np.linspace(tmin, tmax, tn)

			# iterate through all the values and check if the potential reciprocal lattice vectors are in there
			# return the max/min strain and twist values
			strain_min, strain_max, twist_min, twist_max = self.brute_force_uncertainties(epss=epss, twis=twis, 
																					moire_data=moire_data, sigma=sigma)


			print(f'Strain minimum is {strain_min*100:.3f}%, and strain maximum is {strain_max*100:.3f}%.' + '\n'
				+ f'Twist minimum is {twist_min*180/np.pi:.3f}°, and twist maximum is {twist_max*180/np.pi:.3f}°.')
			# check to make sure neither of these is the boundary values for the system
			# go down another order of magnitude

			# hitting bounds on both sides, make sure it's continuously expanding range
			if strain_min == smin and strain_max == smax and strain_min <= smin_prev and strain_max >= smax_prev:
				smax_prev = smax
				smin_prev = smin
				smin = strain_min*0.1 
				smax = strain_max*10
				sn = 1000 # increase number of items for an increased range
				scounter += 1
				print('Strain min and max are at the bounds of the range. Expanding range.')
			# hitting bounds on min and outside of original bounds
			elif strain_min == smin and strain_min <= smin_prev:
				# make sure that we're not getting into too much precision for our FFT
				if round(strain_min, 3) == 0.000:
					smax_prev = smax
					smin_prev = smin
					# set min value to zero
					smin = 0.
					smax = strain_max
					sn = 1000
					print('Strain min is getting very small. Trying minimum value of 0.')
				# make sure this isn't coming back around a second time because we can't go lower than zero
				elif strain_min == 0.:
					strain_okay = True
					print('')
				# all other cases
				else:
					smax_prev = smax
					smin_prev = smin
					smin = strain_min*0.1
					smax = strain_max
					print('Strain min is at the bounds of the range. Expanding range.')
				scounter += 1
			# hitting bounds on max and outside of original bounds
			elif strain_max == smax and strain_max >= smax_prev:
				smax_prev = smax
				smin_prev = smin
				smax = strain_max*10
				smin = strain_min
				scounter += 1
				print('Strain max is at the bounds of the range. Expanding range.')
			# everything is within the bounds
			else:
				strain_okay = True

			# hitting bounds on both sides, make sure it's continuously expanding range
			if twist_min == tmin and twist_max == tmax and twist_min <= tmin_prev and twist_max >= tmax_prev:
				tmax_prev = tmax
				tmin_prev = tmin
				tmin = twist_min*0.1 
				tmax = twist_max*10
				tn = 1000 # increase number of items for an increased range
				tcounter += 1
				print('Twist min and max are at the bounds of the range. Expanding range.')
			# hitting bounds on min and outside of original bounds
			elif twist_min == tmin and twist_min <= tmin_prev:
				# make sure that we're not getting into too much precision for our FFT
				if round(twist_min, 3) == 0.000:
					tmax_prev = tmax
					tmin_prev = tmin
					# set min value to zero
					tmin = 0.
					tmax = twist_max
					tn = 1000
					print('Twist min is getting very small. Trying minimum value of 0.')
				# make sure this isn't coming back around a second time because we can't go lower than zero
				elif twist_min == 0.:
					twist_okay = True
					print('')
				# all other cases
				else:
					tmax_prev = tmax
					tmin_prev = tmin
					tmin = twist_min*0.1
					tmax = twist_max
					print('Twist min is at the bounds of the range. Expanding range.')
				tcounter += 1
			# hitting bounds on max and outside of original bounds
			elif twist_max == tmax and twist_max >= tmax_prev:
				tmax_prev = tmax
				tmin_prev = tmin
				tmax = twist_max*10
				tmin = twist_min
				tcounter += 1
				print('Twist max is at the bounds of the range. Expanding range.')
			# everything is within the bounds
			else:
				twist_okay = True

		# write them into the dictionary
		moire_data['strain'][0] = round(strain_min, 3)
		moire_data['strain'][2] = round(strain_max, 3)
		moire_data['twist'][0] = round(twist_min, 3)
		moire_data['twist'][2] = round(twist_max, 3)

		# store in dictionary for future pulls
		with open(self.picklename, 'wb') as handle:
			pickle.dump(moire_data, handle, protocol=4)

		return strain_min, strain_max, twist_min, twist_max


	def compute_straintwist(self, gg1, gg2, gg3, guess):
		"""
		Takes in the found peaks and determines the global angle, twist, and strain.
		"""
		aa = self.lat_param1
		G1 = 4 * np.pi / np.sqrt(3) / aa * np.array([[0], [-1]])
		G2 = 4 * np.pi / np.sqrt(3) / aa * np.array([[np.sqrt(3) / 2], [1 / 2]])
		theta_0 = np.array([[1, 0], [0, 1]])
		def R(theta):
			return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
		def S(eps, phi):
			return eps * R(-phi) @ np.array([[-1, 0], [0, 0.21]]) @ R(phi)
		def g1(t, theta, eps, phi):
			return R(t) @ (R(theta / 2) @ (theta_0 - S(eps / 2, phi)) - R(-theta / 2) @ (theta_0 - S(-eps / 2, phi))) @ G1
		def g2(t, theta, eps, phi):
			return R(t) @ (R(theta / 2) @ (theta_0 - S(eps / 2, phi)) - R(-theta / 2) @ (theta_0 - S(-eps / 2, phi))) @ G2
		def g3(t, theta, eps, phi):
			return g1(t, theta, eps, phi) + g2(t, theta, eps, phi)
		def res(x):
			geez1 = g1(x[0], x[1], x[2], x[3]) - gg1
			geez2 = g2(x[0], x[1], x[2], x[3]) - gg2
			geez3 = g3(x[0], x[1], x[2], x[3]) - gg3
			return np.array([geez1[0][0], geez1[1][0], geez2[0][0], geez2[1][0], geez3[0][0], geez3[1][0]])
		result = scipy.optimize.least_squares(res, guess, bounds=([0, 0, -np.inf, 0], [np.pi, 2 * np.pi, np.inf, np.pi]))
		if not result.success:
			raise Exception(result)
		return result.x

	def fit_straintwist1(self):
		"""
		Looks up the peaks from the input fields, puts them through the equations,
		and returns the global angle, twist, strain, and shear angle PLUS the min/max values
		of them based on the peaks + sig_x/sig_y. 
		"""
		# initialize functons for plotting reciprocal lattice vectors
		aa = self.lat_param1
		G1 = 4 * np.pi / np.sqrt(3) / aa * np.array([[0], [-1]])
		G2 = 4 * np.pi / np.sqrt(3) / aa * np.array([[np.sqrt(3) / 2], [1 / 2]])
		theta_0 = np.array([[1, 0], [0, 1]])
		def R(theta):
			return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
		def S(eps, phi):
			return eps * R(-phi) @ np.array([[-1, 0], [0, 0.21]]) @ R(phi)
		def g1(t, theta, eps, phi):
			return R(t) @ (R(theta / 2) @ (theta_0 - S(eps / 2, phi)) - R(-theta / 2) @ (theta_0 - S(-eps / 2, phi))) @ G1
		def g2(t, theta, eps, phi):
			return R(t) @ (R(theta / 2) @ (theta_0 - S(eps / 2, phi)) - R(-theta / 2) @ (theta_0 - S(-eps / 2, phi))) @ G2
		def g3(t, theta, eps, phi):
			return g1(t, theta, eps, phi) + g2(t, theta, eps, phi)

		# check to see if start of dictionary already exists
		if os.path.exists(self.picklename):
			print(f'Loading {self.kn} moiré data from pickle.')
			with open(self.picklename, 'rb') as handle:
				moire_data = pickle.load(handle)
		else:
			print('Cannot find moiré data from pickle.')

		# write in the lattice parameter used
		moire_data['lat_param1'] = self.lat_param1

		# pull out data from dictionary, scaling it for reciprocal space
		xo = moire_data['xo']
		yo = moire_data['yo']
		sig_x = moire_data['sig_x']
		sig_y = moire_data['sig_y']
		theta = moire_data['gaus_theta']

		gg1 = 2 * np.pi * np.array([[xo[0]], [yo[0]]])
		gg2 = 2 * np.pi * np.array([[xo[1]], [yo[1]]])
		gg3 = 2 * np.pi * np.array([[xo[2]], [yo[2]]])

		# get the parameters from the input fields
		glob_ang_g, twist_ang_g, strain_g, shear_ang_g = self.get_fit_params() 

		guess_p_peak = (glob_ang_g[0], twist_ang_g[1], strain_g[1], shear_ang_g[0])

		# raise error if 

		glob_peak, tw_peak, str_peak, sh_peak = self.compute_straintwist(gg1, gg2, gg3, guess=guess_p_peak)

		# write these into the dictionary, simulutaneously padding with zeros on either side for min/max
		moire_data['global_ang'] = np.array([glob_peak])
		moire_data['shear_ang'] = np.array([sh_peak])
		moire_data['twist'] = np.array([0, tw_peak, 0])
		moire_data['strain'] = np.array([0, str_peak, 0])

		# write these to the GUI
		self.glob_ang_txt2.setText(f'{glob_peak*180/np.pi:.3f}')
		self.twist_txt2.setText(f'{tw_peak*180/np.pi:.3f}')
		self.strain_txt2.setText(f'{str_peak*100:.3f}')
		self.shear_ang_txt2.setText(f'{sh_peak*180/np.pi:.3f}')

		# find the uncertainties in twist and strain
		str_min, str_max, tw_min, tw_max = self.find_moire_fit_uncertainties(moire_data)

		# create the lists of these parameters
		glob_ang = [glob_peak]
		twist = [tw_min, tw_peak, tw_max]
		strain =  [str_min, str_peak, str_max]
		shear_ang = [sh_peak]

		# write them to the main GUI
		self.write_fit_params((glob_ang, twist, strain, shear_ang))

		# plot the data
		# create axis
		if self.ax is None:
			self.ax = self.canvas.figure.subplots()

		# clear axis
		self.ax.clear()

		# plot one standard deviation of the peaks as a semiaxis for a transparent gray ellipse
		for p in range(0, 3):
			ell = Ellipse(xy=(2*np.pi*xo[p], 2*np.pi*yo[p]), width=4*np.pi*sig_x[p], height=4*np.pi*sig_y[p], angle=theta[p]*180/np.pi, alpha=0.5, color='lightgray')
			self.ax.add_artist(ell)

		# plot the peaks from the real FFT
		for g in [gg1, gg2, gg3]:
			self.ax.plot(g[0][0], g[1][0], 'o', color='mediumorchid', markersize=3)
			self.ax.plot(-g[0][0], -g[1][0], 'o', color='mediumorchid', markersize=3)

		# plot the best fit parameters
		for g in [g1, g2, g3]:
			gg = g(glob_peak, tw_peak, str_peak, sh_peak)
			self.ax.plot(gg[0][0], gg[1][0], 'x', color='dimgray')
			self.ax.plot(-gg[0][0], -gg[1][0], 'x', color='dimgray')

		# plot min and max values
		for g in [g1, g2, g3]:
			gg = g(glob_peak, tw_min, str_min, sh_peak)
			self.ax.plot(gg[0][0], gg[1][0], 'x', color='aquamarine')
			self.ax.plot(-gg[0][0], -gg[1][0], 'x', color='aquamarine')
			gg = g(glob_peak, tw_max, str_max, sh_peak)
			self.ax.plot(gg[0][0], gg[1][0], 'x', color='teal')
			self.ax.plot(-gg[0][0], -gg[1][0], 'x', color='teal')


		self.ax.set_ylabel('$K_Y$ (nm$^{-1}$)')
		self.ax.set_xlabel('$K_X$ (nm$^{-1}$)')
		self.ax.set_title('FFT Fits')
		self.ax.set_aspect('equal')

		self.canvas.draw()

	def fit_straintwist(self):
		"""
		Looks up the peaks from the input fields, puts them through the equations,
		and returns the global angle, twist, strain, and shear angle PLUS the min/max values
		of them based on the peaks + sig_x/sig_y. 
		"""
		# initialize functons for plotting reciprocal lattice vectors
		aa = self.lat_param1
		G1 = 4 * np.pi / np.sqrt(3) / aa * np.array([[0], [-1]])
		G2 = 4 * np.pi / np.sqrt(3) / aa * np.array([[np.sqrt(3) / 2], [1 / 2]])
		theta_0 = np.array([[1, 0], [0, 1]])
		def R(theta):
			return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
		def S(eps, phi):
			return eps * R(-phi) @ np.array([[-1, 0], [0, 0.21]]) @ R(phi)
		def g1(t, theta, eps, phi):
			return R(t) @ (R(theta / 2) @ (theta_0 - S(eps / 2, phi)) - R(-theta / 2) @ (theta_0 - S(-eps / 2, phi))) @ G1
		def g2(t, theta, eps, phi):
			return R(t) @ (R(theta / 2) @ (theta_0 - S(eps / 2, phi)) - R(-theta / 2) @ (theta_0 - S(-eps / 2, phi))) @ G2
		def g3(t, theta, eps, phi):
			return g1(t, theta, eps, phi) + g2(t, theta, eps, phi)

		# check to see if start of dictionary already exists
		if os.path.exists(self.picklename):
			print(f'Loading {self.kn} moiré data from pickle.')
			with open(self.picklename, 'rb') as handle:
				moire_data = pickle.load(handle)
		else:
			print('Cannot find moiré data from pickle.')

		# write in the lattice parameter used
		moire_data['lat_param1'] = self.lat_param1

		# pull out data from dictionary, scaling it for reciprocal space
		xo = moire_data['xo']
		yo = moire_data['yo']
		sig_x = moire_data['sig_x']
		sig_y = moire_data['sig_y']
		theta = moire_data['gaus_theta']

		gg1 = 2 * np.pi * np.array([[xo[0]], [yo[0]]])
		gg2 = 2 * np.pi * np.array([[xo[1]], [yo[1]]])
		gg3 = 2 * np.pi * np.array([[xo[2]], [yo[2]]])

		# get the parameters from the input fields
		glob_ang_g, twist_ang_g, strain_g, shear_ang_g = self.get_fit_params() 

		guess_p_peak = (glob_ang_g[0], twist_ang_g[1], strain_g[1], shear_ang_g[0])

		# raise error if 

		glob_peak, tw_peak, str_peak, sh_peak = self.compute_straintwist(gg1, gg2, gg3, guess=guess_p_peak)

		# write these into the dictionary, simulutaneously padding with zeros on either side for min/max
		moire_data['global_ang'] = np.array([glob_peak])
		moire_data['shear_ang'] = np.array([sh_peak])
		moire_data['twist'] = np.array([0, tw_peak, 0])
		moire_data['strain'] = np.array([0, str_peak, 0])

		# write these to the GUI
		self.glob_ang_txt2.setText(f'{glob_peak*180/np.pi:.3f}')
		self.twist_txt2.setText(f'{tw_peak*180/np.pi:.3f}')
		self.strain_txt2.setText(f'{str_peak*100:.3f}')
		self.shear_ang_txt2.setText(f'{sh_peak*180/np.pi:.3f}')

		# create the lists of these parameters
		glob_ang = [glob_peak]
		twist = [tw_peak, tw_peak, tw_peak]
		strain =  [str_peak, str_peak, str_peak]
		shear_ang = [sh_peak]

		# write them to the main GUI
		self.write_fit_params((glob_ang, twist, strain, shear_ang))

		# plot the data
		# create axis
		if self.ax is None:
			self.ax = self.canvas.figure.subplots()

		# clear axis
		self.ax.clear()

		# plot one standard deviation of the peaks as a semiaxis for a transparent gray ellipse
		for p in range(0, 3):
			ell = Ellipse(xy=(2*np.pi*xo[p], 2*np.pi*yo[p]), width=4*np.pi*sig_x[p], height=4*np.pi*sig_y[p], angle=theta[p]*180/np.pi, alpha=0.5, color='lightgray')
			self.ax.add_artist(ell)
			ell = Ellipse(xy=(-2*np.pi*xo[p], -2*np.pi*yo[p]), width=4*np.pi*sig_x[p], height=4*np.pi*sig_y[p], angle=theta[p]*180/np.pi, alpha=0.5, color='lightgray')
			self.ax.add_artist(ell)
			
		# plot the peaks from the real FFT
		for g in [gg1, gg2, gg3]:
			self.ax.plot(g[0][0], g[1][0], 'o', color='mediumorchid', markersize=3)
			self.ax.plot(-g[0][0], -g[1][0], 'o', color='mediumorchid', markersize=3)

		# plot the best fit parameters
		for g in [g1, g2, g3]:
			gg = g(glob_peak, tw_peak, str_peak, sh_peak)
			self.ax.plot(gg[0][0], gg[1][0], 'x', color='dimgray')
			self.ax.plot(-gg[0][0], -gg[1][0], 'x', color='dimgray')

		self.ax.set_ylabel('$K_Y$ (nm$^{-1}$)')
		self.ax.set_xlabel('$K_X$ (nm$^{-1}$)')
		self.ax.set_title('FFT Fits')
		self.ax.set_aspect('equal')

		self.canvas.draw()


class AdvancedFitParametersWindow(QWidget):

	updateClicked = pyqtSignal(str)
	
	def __init__(self):
		super().__init__()

		# create tab layouts
		mainlayout = QGridLayout()

		self.lat_param1_lbl = QLabel('Lattice parameter (nm):')
		self.lat_param1_txt = QLineEdit('0.2504')
		self.lat_param1_btn = QPushButton('Update')

		self.lat_param1_btn.clicked.connect(self.update_latparam1)

		mainlayout.addWidget(self.lat_param1_lbl, 0, 0, 1, 1)
		mainlayout.addWidget(self.lat_param1_txt, 0, 1, 1, 1)
		mainlayout.addWidget(self.lat_param1_btn, 0, 2, 1, 1)

		# finish creating layout
		self.setLayout(mainlayout)


		# call out specific variables
		self.lat_param1 = self.lat_param1_txt.text()


	def update_latparam1(self):
		"""
		Emit the signal to update the lattice parameter in the main window.
		"""

		self.updateClicked.emit(self.lat_param1_txt.text())






		

	
