"""incohr_dedisps.py defines the scintillometry style class for incoherent
de-dispersion.
"""

from scintillometry.dispersion import Disperse
from scintillometry.dm import DispersionMeasure
from scintillometry.base import PaddedTaskBase
import numpy as np
import astropy.units as u

class IncoherentDedisperse(PaddedTaskBase):
    """Incoherently disperse a time stream

    Parameters
    ----------
    ih : task or `baseband` stream reader
        Input data stream, with time as the first axis.
    dm : float or `~scintillometry.dm.DispersionMeasure` quantity
        Dispersion measure.  If negative, will dedisperse correctly, but
        clearer to use the `~scintillometry.dispersion.Dedisperse` class.
    reference_frequency : `~astropy.units.Quantity`, optional
        Frequency to which the data should be dispersed.  Can be an array.
        By default, the mean frequency.
    samples_per_frame : int, optional
        Number of samples which should be dispersed in one go. The number of
        output dispersed samples per frame will be smaller to avoid wrapping.
        If not given, the minimum power of 2 needed to get at least 75%
        efficiency.
    frequency : `~astropy.units.Quantity`, optional
        Frequencies for each channel in ``ih`` (channelized frequencies will
        be calculated).  Default: taken from ``ih`` (if available).

    Note
    ----
    The reference frequency is highest frequency.
    """

    def __init__(self, ih, dm, samples_per_frame=None, frequency=None,
                 frequency_axis=None):
        dm = DispersionMeasure(dm)
        if frequency is None:
            frequency = ih.frequency

        if frequency_axis is None:
            self._freq_axis = ih.freq_axis
        else:
            self._freq_axis = frequency_axis

        # Calculate frequencies at the top and bottom of each band.
        freq_rate = (frequency.max() -
                     frequency.min()) / (len(frequency) - 1)
        freq_low = frequency
        freq_high = frequency + freq_rate

        reference_frequency = freq_high.max()

        self.delay_low = dm.time_delay(freq_low, reference_frequency)
        self.delay_high = dm.time_delay(freq_high, reference_frequency)
        delay_max = max(self.delay_low.max(), self.delay_high.max())
        delay_min = min(self.delay_low.min(), self.delay_high.min())
        # Calculate the padding needed to avoid wrapping in what we extract.
        pad_start = int(np.ceil((delay_max * ih.sample_rate).to_value(u.one)))
        pad_end = int(np.ceil((-delay_min * ih.sample_rate).to_value(u.one)))

        # Generally, the padding will be on both sides.  If either is negative,
        # that indicates that the reference frequency is outside of the band,
        # and we can do part of the work with a simple sample shift.
        if pad_start < 0:
            # Both delays less than 0; do not need start, so shift by
            # that number of samples, reducing the padding at the end.
            assert pad_end > 0
            sample_offset = pad_start
            pad_end += pad_start
            pad_start = 0
        elif pad_end < 0:
            # Both delays greater than 0; do not need end, so shift by
            # that number of samples, reducing the padding at the start.
            sample_offset = -pad_end
            pad_start += pad_end
            pad_end = 0
        else:
            # Default case: passing on both sides; not useful to offset.
            sample_offset = 0

        if samples_per_frame is None:
            samples_per_frame = pad_start + pad_end + ih.samples_per_frame
        else:
            samples_per_frame = samples_per_frame + pad_start + pad_end

        self.pad_start = pad_start
        self.pad_end = pad_end
        super().__init__(ih, pad_start=pad_start, pad_end=pad_end,
                         samples_per_frame=samples_per_frame,
                         frequency=frequency, sideband=1)

        self.dm = dm
        self.reference_frequency = reference_frequency
        self._sample_offset = sample_offset
        self._start_time += sample_offset / ih.sample_rate
        self._pad_slice = slice(self._pad_start,
                                self._padded_samples_per_frame - self._pad_end)
        self.delay_low_index = ((self.delay_low * self.sample_rate).decompose().astype(int))
        self.delay_high_index = ((self.delay_high * self.sample_rate).decompose().astype(int))
        # super(IncoherentDisperse, self).__init__(ih, dm,
        #                                          reference_frequency=reference_frequency,
        #                                          samples_per_frame=samples_per_frame,
        #                                          frequency=frequency,
        #                                          sideband=1)

    def task(self, data):
        sum_index = self.delay_low_index - self.delay_high_index
        result = []
        for ii in range(len(sum_index)):
            sub_channel = np.take(data, [ii], axis=self.freq_axis)[self.delay_high[ii]:,...]
            sum_res = np.apply_along_axis(lambda m: np.convolve(m,
                                          np.ones(sum_index), mode='valid'),
                                          axis=0, arr=sub_channel)
            result.append(sum_res[0:samples_per_frame])
        result = np.stack(result, axis=self.freq_axis)
        return result.reshape((samples_per_frame, ) + ih.shape[1:]
