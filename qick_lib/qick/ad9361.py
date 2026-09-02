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

  * The generator drives the AD9361 transmit port directly -- there is no mux and
    no select, so loading this overlay stops the board's own loopback demo from
    transmitting. The AD9361 still has to have its transmit channel in DMA mode
    for any fabric data to be forwarded; see docs/loopback.md in pluto-qick.
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
import logging
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

logger = logging.getLogger(__name__)



def radio_is_awake():
    """True if the AD9361 is in a state where it drives the LVDS DATA_CLK.

    This is a safety check, not a convenience. The AD9361 sources the clock that
    l_clk -- and therefore the divided clock the whole QICK datapath runs on --
    is derived from. If the chip is asleep that clock is stopped, and an AXI
    access into axi_ad9361 then has no clock to complete against. On a Zynq that
    is not a failed read, it is a hung interconnect: the CPU blocks forever and
    stops servicing interrupts, so even magic SysRq over the serial console gets
    no response and only a power cycle recovers the board.

    That is the most likely explanation for the hard hang seen during the first
    bring-up: the transmit path had been corrupted, the AD9361 driver's digital
    calibration failed, the chip was left asleep, and the next thing to touch its
    registers took the machine down with it.
    """
    for devdir in glob.glob('/sys/bus/iio/devices/iio:device*'):
        try:
            if open(os.path.join(devdir, 'name')).read().strip() != 'ad9361-phy':
                continue
            return open(os.path.join(devdir, 'ensm_mode')).read().strip() in ('fdd', 'tdd', 'alert')
        except OSError:
            return False
    return False

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


def quiesce_axi_masters(verbose=False):
    """Idle the AXI masters in the PL before it is reprogrammed.

    Reprogramming the PL while a master has a burst outstanding can lock the AXI
    interconnect. The CPU then blocks forever on its next bus access -- hard
    enough that magic SysRq over the serial console gets no response either, and
    only a power cycle recovers. /boot/boot.py never hits this because at boot
    nothing is running yet; by the time an overlay is loaded from a shell, the
    ADI ADC and DAC DMAs and iiod are all live masters.

    So: stop iiod, then disable any active IIO buffer, which is what makes the
    ADI DMAs issue transactions in the first place.
    """
    subprocess.run(['systemctl', 'stop', 'iiod'], check=False)

    stopped = []
    for devdir in sorted(glob.glob('/sys/bus/iio/devices/iio:device*')):
        enable = os.path.join(devdir, 'buffer', 'enable')
        if not os.path.exists(enable):
            continue
        try:
            if open(enable).read().strip() not in ('0', ''):
                with open(enable, 'w') as f:
                    f.write('0')
                stopped.append(devdir)
        except OSError:
            # A wedged device can block here; nothing useful to do but move on.
            pass
    time.sleep(0.5)
    if verbose:
        print("quiesce: iiod stopped, buffers disabled on %s"
              % (stopped if stopped else 'none'))
    return stopped

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
    # How many times to try bringing the tProc cores up before giving up. The
    # control handshake swallows writes unpredictably; see start_tproc().
    START_ATTEMPTS = 12

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
        # Must happen before Overlay.__init__ programs the PL.
        if kwargs.get('download', True):
            quiesce_axi_masters()
        super().__init__(bitfile=bitfile, no_rf=True, dtbo=dtbo, **kwargs)

    # -- hooks that QickSoc leaves for boards without an RF data converter ----

    def config_rf_alt(self):
        # iiod holds its own handles on the IIO devices, which the dtbo re-probe
        # has just replaced. Restarting it is what /boot/boot.py does after
        # loading the base design, for the same reason.
        if self._restart_iiod:
            subprocess.run(['systemctl', 'restart', 'iiod'], check=False)
            time.sleep(2.0)

        # Refuse to go further if the radio did not survive the PL load. Reading
        # the sample rate below is itself an access into that clock domain, so
        # this check has to come first -- and stopping here with an exception is
        # very much better than the alternative, which is hanging the CPU hard
        # enough to need a power cycle.
        if not radio_is_awake():
            raise RuntimeError(
                "the AD9361 is not awake after programming the PL, so it is not "
                "driving the LVDS DATA_CLK that the QICK datapath is clocked "
                "from. Touching axi_ad9361 registers in this state can hang the "
                "CPU hard enough to require a power cycle, so stopping here. "
                "Check ensm_mode, and that a dtbo was passed so the AD9361 "
                "drivers were re-probed against the freshly programmed PL.")

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
        """Take the tProc out of reset, then start.

        QickSoc.start_tproc() issues only PROCESSOR_START. On this build that is
        inert, because acquire() calls clear_tproc_counter() immediately before,
        and on tProc v2 that means tproc.reset() -- PROCESSOR_RESET. After a
        reset the only control command this firmware responds to is
        PROCESSOR_RUN; time_reset, time_update, core_start and start all leave
        the status register untouched. Measured directly:

            after reset  Core_EN=0 Time_EN=0 time_usr=0
            + time_reset / time_update / core_start / start  -> no change
            + run        Core_EN=1 Time_EN=1 time_usr advancing at 30.76 MHz

        Without it the processor core can be coaxed into C_RUN while the timing
        core stays disabled and time_usr sits frozen. The program then blocks on
        its first timed instruction, never reaches inc_ext_counter, and acquire()
        polls a shot counter that can never advance -- which presents as a hang
        with no error anywhere.

        run() is issued unconditionally because it is idempotent, and start_src()
        calls stop() just before this point, so the state has to be re-established
        on every round rather than once at load time.

        The delay below is load-bearing, not defensive: the first control write
        after a reset is swallowed, so run() alone leaves the core stopped and
        run() immediately followed by start() is no better. Only two writes
        separated in time bring both cores up.
        """
        if self.TPROC_VERSION != 2:
            super().start_tproc()
            return

        # The control handshake is not reliable enough to issue blind. After a
        # reset the first write is swallowed, and whether a given write takes
        # appears to be timing dependent -- the same sequence that brings both
        # cores up one run leaves Core_EN=0 the next. So issue and verify,
        # rather than assume.
        for attempt in range(self.START_ATTEMPTS):
            self.tproc.run()
            time.sleep(0.01)
            super().start_tproc()
            time.sleep(0.01)
            status = self.tproc.tproc_status
            core_en = (status >> 3) & 1
            time_en = (status >> 7) & 1
            if core_en and time_en:
                if attempt:
                    logger.info("tProc cores came up on attempt %d", attempt + 1)
                return
        raise RuntimeError(
            "the tProc cores did not enable after %d attempts (status=0x%08x). "
            "Core_EN and Time_EN must both be set or the program blocks on its "
            "first timed instruction and the shot counter never advances."
            % (self.START_ATTEMPTS, self.tproc.tproc_status))
