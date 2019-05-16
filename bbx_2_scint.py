"""bbx_2_scint.py defines the converting function from lofasm bbx file to the
format that scintillometry pipeline accepts.
"""
from lofasm.bbx.bbx import LofasmFile
from scintillometry.generators import StreamGenerator
from baseband.helpers.sequentialfile import SequentialFileReader
from collections import namedtuple
import astropy.units as u
from astropy.time import Time
import numpy as np
import re


__all__ = ['BbxReader']
unit_pt = re.compile(r'\((.*?)\)')

class Bbx2Scint(LofasmFile):
    """Bbx2Scint is designed for converting
    """
    _properties = ('start_time', 'sample_rate', 'shape', 'samples_per_frame',
                   'frequency')
    def __init__(self, file_name, mode='rb'):
        mode_map = {'rb': 'read',}
        super().__init__(file_name, mode=mode_map[mode])
        self.dim_units, self.label = self.get_dim_units()

    def get_dim_units(self, total_dim=2):
        units = []
        names = []
        for ii in range(total_dim):
            dim_label = self.header['dim{}_label'.format(ii + 1)]
            res_unit = unit_pt.search(dim_label).group(1)
            units.append(u.Unit(res_unit))
            dim_name = dim_label.split(" (")[0]
            names.append(dim_name)
        return tuple(units), tuple(names)

    @property
    def shape(self):
        shape_name = ['n' + l for l in self.label]
        s = {}
        for ii, sn in enumerate(shape_name):
            s[sn] = self.header['metadata']['dim{}_len'.format(ii + 1)]
        st = namedtuple('shape', shape_name)
        return st(**s)

    @property
    def time_span(self):
        return u.Quantity(float(self.header['dim1_span']), self.dim_units[0])

    @property
    def sample_rate(self):
        sample_time = self.time_span / self.shape[0]
        return 1.0 / sample_time

    @property
    def start_time(self):
        basetime = Time('J2000.0', format='jyear_str')
        dt = float(self.header['dim1_start']) * u.s
        return basetime + dt

    @property
    def samples_per_frame(self):
        return self.shape[0]

    @property
    def frequency(self):
        nchan = self.header['metadata']['dim2_len']
        freq_span = float(self.header['dim2_span'])
        # Need a better way to handle it.
        offset_unit = unit_pt.search(self.header['frequency_offset_DC']).group(1)
        offset_unit = u.Unit(offset_unit)
        off_DC = (float(self.header['frequency_offset_DC'].split(' (')[0]) *
                  offset_unit)
        # Needs check in the future.
        freq_start = (float(self.header['dim2_start']) * self.dim_units[1] +
                      off_DC )
        return (freq_start + np.arange(nchan) * freq_span /
                nchan * self.dim_units[1])

    # @property
    # def dtype(self):
    #     return bool(float(self.header['metadata']['complex']) - 1)


class BbxReader(StreamGenerator):
    """BbxReader defines the stream reading for Bbx files.

    Parameter
    ---------
    bbx_file : Bbx2Scint object

    """
    def __init__(self, bbx_file, **kwargs):
        self.source = Bbx2Scint(bbx_file, **kwargs)
        self.input_args = kwargs
        # The required argument will come from the source.
        self.req_args = {'shape': None, 'start_time': None,
                         'sample_rate': None}
        self.opt_args = {'samples_per_frame': 1, 'frequency': None,
                         'sideband': None, 'polarization': None, 'dtype':None}

        self._prepare_args()
        self._setup_args()
        function = None
        super(BbxReader, self).__init__(*(function, self.req_args['shape'],
                                       self.req_args['start_time'],
                                       self.req_args['sample_rate']),
                                     **self.opt_args)

    def _prepare_args(self):
        """This setup function setups up the argrument for initializing the
        StreamGenerator.
        """
        input_args_keys = self.input_args.keys()
        source_properties = self.source._properties
        for rg in self.req_args.keys():
            if rg not in source_properties:
                raise ValueError("'{}' is required.".format(rg))

            self.req_args[rg] = getattr(self.source, rg)

        for og in self.opt_args.keys():
            if og in source_properties:
                self.opt_args[og] = getattr(self.source, og)
            elif og in input_args_keys:
                self.opt_args[og] = self.input_args[og]
            else:
                continue
        self._setup_args()
        return

    def _read_frame(self, frame_index):
        self.source.read_data()
        return self.source.data.T

    def _setup_args(self):
        pass

    @property
    def header(self):
        return self.source.header

class BbxStreamReader(SequentialFileReader):
    """The BBX stream reader reads a sequential of files.
    """
    def __init__(self, files):
        super().__init__(files, mode='rb', opener=BbxReader)
        self.fh0 = self.opener(self.files[0], mode='rb')

    @property
    def header0(self):
        """Return the first header of the sequential files.
        """
        return self.fh0.header

    @property
    def shape(self):
        shape0 = self.fh0.shape
        shape = shape0._replace(ntime = self.size)
        return shape

    @property
    def start_time(self):
        return self.fh0.start_time

    @property
    def sample_rate(self):
        return self.fh0.sample_rate

    @property
    def samples_per_frame(self):
        return self.fh0.samples_per_frame

    @property
    def frequency(self):
        return self.fh0.frequency

    def tell(self, unit=None):
        # it need gap filling.
        offset = super().tell()
        if unit is None:
            return offset

        # "isinstance" avoids costly comparisons of an actual unit with 'time'.
        if not isinstance(unit, u.UnitBase) and unit == 'time':
            return self.start_time + self.tell(unit=u.s)

        return (offset / self.sample_rate).to(unit)

    def read(self, count=None):
        if self.closed:
            raise ValueError('read of closed file.')

        if count is None or count < 0:
            count = max(self.size - self.tell(), 0)

        data = None
        while count > 0:
            avil = self.fh.shape[0] - self.fh.tell()
            request = min(avil, count)
            extra = self.fh.read(request)
            count -= len(extra)
            if data is None:  # avoid copies for first read.
                data = extra
            else:
                data = np.vstack((data, extra))
            # Go to current offset, possibly opening new file.
            self.seek(0, 1)

        return data
