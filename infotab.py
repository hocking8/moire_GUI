"""

Filename: infotab.py
Author: Risa Hocking
Purpose: Provides information on how to run the program and how 
the data is saved in the moire_data.pickle file.

"""


import sys
import os
import pickle
import numpy as np


from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *


class InfoTab(QWidget):
	def __init__(self):
		super().__init__()

		# create tab layouts
		self.layout = QVBoxLayout()

		# write the description for everything
		pf_descr = "Before using the program, save the phase and amplitude channels" \
					"as .txt files with the following format: NAME_Ampl.txt or " \
					"NAME_Phase.txt. Do not include more underscores. Open the desired file in the peak-finding tab. " \
					"Enter the correct scan size and save the file information. This starts a " \
					"new key in the data dictionary. Now adjust the FFT crop bounds either smaller" \
					" or larger depending on your expected twist angle. You can always run it and" \
					" adjust, though FFTs take ~20 s to run. Increase the center mask radius to " \
					"remove the low-frequency signals from the center of the FFT, improving clarity" \
					" of the actual peaks. You can also use the center line height mask to block out" \
					" intense signals from the fast-scan direction lines. Hover over the peaks in the FFT " \
					"and look at the upper right " \
					"corner to see their position on the figure. Type them into the X and Y boxes for " \
					"the correct peak in the lower left section. Test all peaks to confirm " \
					"that you picked decent locations. Reduce the crop radius for specific peaks if " \
					"they are close to other noise. Once you like the location of the peaks, save the " \
					"values. This writes them to the dictionary, and then you can recall them later. " \
					"Run fit. This will take several minutes, which is why you saved " \
					"the peaks earlier. Once the program finishes and you like the results, hit save " \
					"values to keep them in the dictionary."

		self.info_lbl = QLabel('How to use peak-finding:')
		self.info_lbl.setFont(QFont('',weight=QFont.Bold))
		self.info_lbl.setWordWrap(True)
		self.info_txt = QLabel(pf_descr)
		self.info_txt.setWordWrap(True)
		self.layout.addWidget(self.info_lbl)
		self.layout.addWidget(self.info_txt)

		sttw_descr = "You must have used the peak-finding tab before using this tab. Open the same " \
					"file in the strain and twist tab. No need to save the file information again. " \
					"Enter estimated values into the lower left fit fields. Note: the fitting currently " \
					"only works for moiré made of the same material with the same lattice parameter, " \
					"which you can fill in. The default is the lattice constant of hBN. " \
					"Run fit. This will show the main peak values (in purple) as " \
					"well an expanded and retracted set (in gray). The peaks from peak-finding are shown as " \
					"circles, whereas the fits from this tab are shown as X's. The expanded and retracted " \
					"points show the general error from the peak-fitting, providing an estimated range for some " \
					"parameters of sorts. Save the values when you like them."

		self.sttw_lbl = QLabel('How to find strain and twist:')
		self.sttw_lbl.setFont(QFont('',weight=QFont.Bold))
		self.sttw_lbl.setWordWrap(True)
		self.sttw_txt = QLabel(sttw_descr)
		self.sttw_txt.setWordWrap(True)
		self.layout.addWidget(self.sttw_lbl)
		self.layout.addWidget(self.sttw_txt)


		data_descr = "The data is stored in [FILENAME]_moire_data.pickle, which is found in the save directory that you " \
					"indicated on each page. The dictionary has the following keys that might be useful: the 2D Gaussian " \
					"fit parameters (xo, yo, gaus_amp, sig_x, sig_y, gaus_theta) and the strain/twist fitting parameters " \
					"(global_angle, shear_angle, twist, strain). The Gaussian fitting parameters are lists of " \
					"three items, with item0 = peak1 (right), item1 = peak2 (left), item2 = peak3 (right). " \
					"The strain/twist fitting parameters are also lists of three, but their items correspond to " \
					"expected error (i.e. item0 = contracted peaks, item1 = peaks, item2 = expanded peaks). Expansion " \
					"and contraction are done using the angle and the sigmas for each peak."

		self.data_lbl = QLabel('Data organization:')
		self.data_lbl.setWordWrap(True)
		self.data_lbl.setFont(QFont('',weight=QFont.Bold))
		self.data_txt = QLabel(data_descr)
		self.data_txt.setWordWrap(True)
		self.layout.addWidget(self.data_lbl)
		self.layout.addWidget(self.data_txt)






