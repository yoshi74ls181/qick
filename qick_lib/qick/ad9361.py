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
import subprocess
import time

from .qick import QickSoc

# Programming the PL resets axi_ad9361 underneath the bound AD9361 driver stack.
# Passing a device tree overlay makes the kernel tear those drivers down and
# re-probe them against the freshly programmed core, which is the difference
# between a working radio and a chip left asleep: without it the AD9361 goes to
# sleep, its sample rate halves on every load, ensm_mode writes are silently
# ignored, and further access eventually wedges the driver. Recovering then needs
# a reboot -- unbinding and rebinding the SPI device is not enough and hangs on a
# calibration timeout.
#
# This is the same overlay the board's own /boot/boot.py uses to bring up the
# base design, and it describes the AD9361 peripherals, which the QICK overlay
# keeps at their original addresses. The QICK IP themselves need no kernel
# drivers -- PYNQ reaches them through /dev/mem.
DEFAULT_DTBO = '/home/xilinx/jupyter_notebooks/base/pl.dtbo'

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
            'dacs': {'00': dict(chan, interpolation=1, label='AD9361 TX1')},
            'adcs': {'00': dict(chan, decimation=1, label='AD9361 RX1')},
            # QickConfig.description() reports which clocks are related. On this
            # board the transmit and receive chains are not merely related, they
            # are the same clock: both are the AD9361's, divided down by
            # util_ad9361_divclk. So there is exactly one group.
            'clk_groups': [[('dac', 0), ('adc', 0)]],
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

    def __init__(self, bitfile=None, fs=None, dtbo=DEFAULT_DTBO,
                 restart_iiod=True, **kwargs):
        self._fs_arg = fs
        if kwargs.pop('no_rf', False) is not False:
            raise ValueError("QickSocE200 always runs with no_rf=True")
        if dtbo is not None and not os.path.exists(dtbo):
            raise FileNotFoundError(
                "device tree overlay %s not found. It is required: without it "
                "the AD9361 drivers are not re-probed after the PL is "
                "reprogrammed and the radio is left asleep. Pass dtbo=None only "
                "if the PL is already programmed and you are loading with "
                "download=False." % (dtbo,))
        self._restart_iiod = restart_iiod
        super().__init__(bitfile=bitfile, no_rf=True, dtbo=dtbo, **kwargs)

    # -- hooks that QickSoc leaves for boards without an RF data converter ----

    def config_rf_alt(self):
        # iiod holds its own handles on the IIO devices, which the dtbo re-probe
        # has just replaced. Restarting it is what /boot/boot.py does after
        # loading the base design, for the same reason.
        if self._restart_iiod:
            subprocess.run(['systemctl', 'restart', 'iiod'], check=False)
            time.sleep(2.0)

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

    def start_tproc(self):
        """Enable the timing and processor cores, then start.

        QickSoc.start_tproc() only issues PROCESSOR_START (tproc_ctrl bit 2).
        On this build that is not enough: after a program is loaded the status
        register reports Core_EN = 0 and Time_EN = 0, the timing counter
        (time_usr) does not advance, and the shot counter never increments -- so
        acquire() spins forever waiting for a shot that cannot happen. Nothing in
        QICK's Python ever calls core_start(), so the cores must come up enabled
        on the firmware their projects ship; they do not here.

        Issuing TIME_UPDATE and CORE_START first fixes it, and is idempotent, so
        this is safe to do on every round. Note start_src() calls stop() just
        before start_tproc() is reached, which is why the enables have to be
        re-asserted here rather than once at load time.

        Verified on hardware: time_usr then advances at 6.16e6 ticks per 200 ms,
        i.e. 30.8 MHz, matching the 30.72 MHz sample clock.
        """
        if self.TPROC_VERSION == 2 and self.tproc.get_start_src() == 'internal':
            self.tproc.time_update()
            self.tproc.core_start()
        super().start_tproc()

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
