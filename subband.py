"""subband.py defines the class to select a subband of the reading data.
"""

import numpy as np
from scintillometry.base import TaskBase
from astropy import log
from copy import deepcopy


class SubBand(TaskBase):
    """SubBand class is designed to read a sub frequency band from the readin
       data.

    Parameter
    ---------
    ih : task or `baseband` stream reader
        Input data stream.
    freq_axis : int
        The index of frequency axis in `ih`.
    upper_cut : `astropy.Quantity` in the frequency unit, optional
        The upper cut of the frequency, default is None
    lower_cut : `astropy.Quantity` in the frequency unit, optional
        The lower cut of the frequency, default is None
    """
    def __init__(self, ih, freq_axis, lower_cut=None, upper_cut=None):
        self.ih = ih
        self.freq_axis = freq_axis
        if upper_cut is not None:
            if upper_cut > ih.frequency.max():
                log.warn("The upper cut exceed this maximum frequency of data. "
                         " Using the maximum frequency from the data.")
                self.upper_cut = ih.frequency.max()
            else:
                self.upper_cut = upper_cut
        else:
            self.upper_cut = ih.frequency.max()
        if lower_cut is not None:
            if lower_cut < ih.frequency.min():
                log.warn("The lower cut exceed this minimum frequency of data. "
                         " Using the minimum frequency from the data.")
                self.lower_cut = ih.frequency.min()
            else:
                self.lower_cut = lower_cut

        else:
            self.lower_cut = ih.frequency.min()

        self.upper_ind = self._search_frequency(self.upper_cut)
        self.lower_ind = self._search_frequency(self.lower_cut)
        frequency = ih.frequency[self.lower_ind : self.upper_ind + 1]
        raw_shape = list(ih.shape[:])
        raw_shape[self.freq_axis] = len(frequency)
        shape = tuple(raw_shape)
        super().__init__(ih, shape=shape, frequency=frequency)

    def _search_frequency(self, freq):
        abs_diff = np.abs(self.ih.frequency - freq)
        return np.argmin(abs_diff)

    def task(self, data):
        return data.take(np.arange(self.lower_ind, self.upper_ind+1),
        axis=self.freq_axis)
