"""
QICK on an AD9361 front end (AntSDR E200, Zynq-7020).

Everything QICK normally learns from the RF data converter has to be supplied
another way here, because there isn't one:

  sample rate     the AD9361's, read over IIO (or passed in explicitly)
  converter port  stated, not discovered: sg0/m_axis goes to a TX mux and
                  ro0/s_axis is driven from concatenated ADC buses, so there is
                  no AXI-Stream path to trace to a converter
  datapath clock  util_ad9361_divclk/clk_out, a runtime-selectable divide of the
                  AD9361 LVDS DATA_CLK. QickMetadata.trace_clk_back deliberately
                  refuses to resolve this: it only accepts a PS or RFDC source.

Three differences from an RFSoC worth keeping in mind:

  * There is no digital mixer. AxisSignalGen already declares HAS_MIXER = False,
    so QICK never asks for one; frequency placement is the AD9361's analog LO,
    set through IIO independently of QICK.
  * There is no Nyquist-zone switch.
  * One complex sample per clock (N_DDS=1), not 16. That comes from the hwh via
    the N_DDS parameter, so envelope lengths are in units of 1 sample.

fs_mult and fs_div are both 1 and refclk_freq is set to fs, which makes QICK's
frequency grid refclk*fs_mult/fdds_div/2**32 = fs/2**32 -- exactly the step of
the 32-bit DDS phase accumulator, so requested frequencies land where asked.
"""

import glob
import os

from .qick import QickSoc

# Data register offset in an AXI GPIO.
_GPIO_DATA = 0x0

# TX mux select values; must match qick_tx_mux.v in antsdr-pynq.
TX_SRC = {'dma': 0, 'qick': 1}


def read_ad9361_fs():
    """Read the AD9361 sample rate, in MHz, from IIO sysfs.

    Uses sysfs rather than pyadi-iio so that loading the overlay does not
    depend on a libiio Python binding being installed.
    """
    for devdir in sorted(glob.glob('/sys/bus/iio/devices/iio:device*')):
        try:
            with open(os.path.join(devdir, 'name')) as f:
                name = f.read().strip()
        except OSError:
            continue
        if not name.startswith('cf-ad9361'):
            continue
        for attr in ('in_voltage_sampling_frequency', 'out_voltage_sampling_frequency'):
            path = os.path.join(devdir, attr)
            if os.path.exists(path):
                with open(path) as f:
                    return float(f.read().strip()) / 1e6
    raise RuntimeError(
        "could not read the AD9361 sample rate from IIO sysfs; "
        "pass fs=<MHz> to QickSocE200 instead")


class AD9361RF:
    """Stands in for the RF data converter, describing an AD9361 chain.

    Only the parts QICK's generator and readout drivers actually read are
    implemented: the ['dacs'] / ['adcs'] config dicts, plus set_nyquist. QICK
    never calls set_mixer_freq on this board because AxisSignalGen declares
    HAS_MIXER = False.
    """

    def __init__(self, fs):
        # One TX chain and one RX chain, both at the same rate. Keys are strings
        # because the RFDC path derives them from port names ('m02_axis' -> '02')
        # and the drivers index with whatever find_rf_port returned.
        chan = {
            'fs': fs,
            # fs expressed against refclk_freq as an exact rational; with
            # refclk_freq == fs this is simply 1/1.
            'fs_mult': 1,
            'fs_div': 1,
            # The AD9361's own decimation and interpolation filters are internal
            # to the chip and invisible to QICK, which only needs to know the
            # rate at the fabric boundary.
            'f_fabric': fs,
        }
        self.cfg = {
            'dacs': {'00': dict(chan, interpolation=1)},
            'adcs': {'00': dict(chan, decimation=1)},
        }

    def __getitem__(self, key):
        return self.cfg[key]

    def set_nyquist(self, dac, nqz):
        """No Nyquist-zone control on an AD9361; zone 1 is the only valid ask."""
        if int(nqz) != 1:
            raise NotImplementedError(
                "the AD9361 has no Nyquist zone switch; use the analog LO to "
                "place the signal instead (requested zone %s)" % (nqz,))


class QickSocE200(QickSoc):
    """QickSoc for the AntSDR E200 overlay in antsdr-pynq/boards/e200/qick.

    Parameters
    ----------
    bitfile : str
        Path to the .bit; the matching .hwh must sit beside it.
    fs : float
        Sample rate in MHz. Read from the AD9361 over IIO if omitted. Note this
        is the rate at the fabric boundary, which is what the QICK datapath is
        clocked at, and is what the base design's util_ad9361_divclk produces.
    """

    def __init__(self, bitfile=None, fs=None, **kwargs):
        self._fs_arg = fs
        if kwargs.pop('no_rf', False) is not False:
            raise ValueError("QickSocE200 always runs with no_rf=True")
        super().__init__(bitfile=bitfile, no_rf=True, **kwargs)

    # -- hooks that QickSoc leaves for boards without an RF data converter ----

    def config_rf_alt(self):
        fs = self._fs_arg if self._fs_arg is not None else read_ad9361_fs()
        self.rf = AD9361RF(fs)
        self['rf'] = self.rf.cfg
        self['refclk_freq'] = fs
        self['extra_description'].append(
            "AD9361 front end, fs = %.6f MHz (no RF data converter)" % (fs,))

    def clk_src(self, fullpath, port):
        """The whole QICK datapath runs on the AD9361 sample clock.

        Reported as an 'adc' source so that QickConfig's clock grouping treats
        the generator and readout as sharing one domain, which on this board
        they genuinely do.
        """
        return {'source': ('adc', 0),
                'f_clk': float(self['refclk_freq']),
                'src_range': None}

    def find_rf_port(self, block, kind, port):
        """One TX chain and one RX chain, both port '00'.

        Stated rather than traced: sg0/m_axis feeds qick_tx_mux and ro0/s_axis
        is driven from an xlconcat of the ADC buses, so neither has an
        AXI-Stream path to a converter for trace_forward/trace_back to follow.
        """
        return self.rf, '00'

    # -- board-specific control ----------------------------------------------

    def tx_source(self, src):
        """Choose what drives the AD9361 transmit port.

        'dma'  the stock ADI DMA/DDS path (the reset default, so the board's
               existing loopback behaviour survives a QICK build that produces
               nothing)
        'qick' the QICK signal generator
        """
        if src not in TX_SRC:
            raise ValueError("tx_source must be one of %s" % (sorted(TX_SRC),))
        self.qick_gpio.mmio.write(_GPIO_DATA, TX_SRC[src])

    def get_tx_source(self):
        val = self.qick_gpio.mmio.read(_GPIO_DATA) & 1
        return {v: k for k, v in TX_SRC.items()}[val]
