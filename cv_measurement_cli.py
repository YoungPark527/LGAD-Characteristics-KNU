import argparse
import os
import sys
import signal
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pyvisa


# ------------------------- Configuration -------------------------
SMU_ADDRESS = "GPIB1::18::INSTR"  # Keithley 2470
LCR_ADDRESS = "GPIB1::6::INSTR"   # Wayne Kerr 4300
# -----------------------------------------------------------------


def getdate():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def mkdir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


class CVMeasurement:
    def __init__(self):
        self.output_dir = Path.home() / "LGAD_test" / "C-V_test"
        self.voltage_start = 0
        self.voltage_end = -50
        self.voltage_steps = 101
        self.frequency = 1000  # Hz
        self.iteration_count = 1
        self.sensor_name = "SENSOR"
        self.pad_number = "1_1"
        self.return_sweep = True

    # --------------------------------------------------------------
    # Connection & Configuration
    # --------------------------------------------------------------
    def connect_instruments(self):
        """Connect Keithley 2470 and Wayne Kerr 4300 via GPIB."""
        rm = pyvisa.ResourceManager()

        print("\n🔌 Scanning VISA GPIB devices...")
        resources = rm.list_resources()
        for i, r in enumerate(resources):
            print(f"{i+1}: {r}")

        try:
            print(f"\nConnecting to Keithley 2470 SMU at {SMU_ADDRESS} ...")
            self.smu = rm.open_resource(SMU_ADDRESS)
            self.smu.read_termination = "\n"
            self.smu.write_termination = "\n"

            print(f"Connecting to Wayne Kerr 4300 LCR at {LCR_ADDRESS} ...")
            self.lcr = rm.open_resource(LCR_ADDRESS)
            self.lcr.read_termination = "\n"
            self.lcr.write_termination = "\n"

            print("\n✅ Connected instruments:")
            print(f"SMU: {self.smu.query('*IDN?').strip()}")
            print(f"LCR: {self.lcr.query('*IDN?').strip()}")
            return True

        except Exception as e:
            print(f"\n❌ Connection error: {e}")
            return False

    # --------------------------------------------------------------
    def configure_instruments(self):
        """Configure Keithley 2470 SMU and Wayne Kerr 4300."""
        try:
            print("\n⚙️ Configuring Keithley 2470 SMU...")
            self.smu.write("*RST")
            self.smu.write(":SOUR:FUNC VOLT")
            self.smu.write(":SOUR:VOLT:MODE FIXED")
            self.smu.write(":SENS:FUNC 'CURR'")
            self.smu.write(":SOUR:VOLT:RANG 200")
            self.smu.write(":SENS:CURR:RANG 1e-3")
            self.smu.write(":SOUR:VOLT:ILIM 1e-4")  # 100 µA limit
            self.smu.write(":OUTP OFF")

            print("⚙️ Configuring Wayne Kerr 4300 LCR meter...")
            self.lcr.write("*RST")
            self.lcr.write("FUNC:IMP CPD")              # C and D (parallel)
            self.lcr.write(f"FREQ {self.frequency}")    # Hz
            self.lcr.write("VOLT 0.1")                  # 100 mVrms test signal
            self.lcr.write("BIAS:STAT OFF")             # disable internal DC bias
            self.lcr.write("TRIG:SOUR BUS")             # trigger via GPIB
            self.lcr.write("DISP:PAGE MEAS")
            time.sleep(0.5)
            return True

        except Exception as e:
            print(f"❌ Configuration error: {e}")
            return False

    # --------------------------------------------------------------
    # Measurement Loop
    # --------------------------------------------------------------
    def perform_measurement(self):
        """Perform full C–V measurement."""
        voltages = np.linspace(self.voltage_start, self.voltage_end, self.voltage_steps)
        if self.return_sweep:
            voltages = np.concatenate([voltages, voltages[::-1]])

        print(f"\nStarting voltage sweep ({len(voltages)} points)")

        data = {"voltage": [], "capacitance": [], "resistance": [], "current": []}

        def safe_off():
            try:
                self.smu.write(":SOUR:VOLT 0")
                self.smu.write(":OUTP OFF")
                self.lcr.write("BIAS:STAT OFF")
            except Exception:
                pass

        def sigint_handler(sig, frame):
            print("\n⚠️ Interrupted by user. Turning off safely...")
            safe_off()
            sys.exit(0)

        signal.signal(signal.SIGINT, sigint_handler)

        # Enable voltage source
        self.smu.write(":SOUR:VOLT 0")
        self.smu.write(":OUTP ON")
        time.sleep(1)

        for V in voltages:
            if V > 0:
                print("⚠️ Positive bias not allowed — forcing 0 V.")
                V = 0

            self.smu.write(f":SOUR:VOLT {V}")
            time.sleep(0.2)

            # Measure current and voltage
            try:
                raw = self.smu.query(":READ?")
                vals = [float(x) for x in raw.split(",") if self._isfloat(x)]
                if len(vals) >= 2:
                    I_meas, V_meas = vals[0], vals[1]
                else:
                    I_meas, V_meas = np.nan, V
            except Exception:
                I_meas, V_meas = np.nan, V

            # Trigger LCR measurement
            try:
                self.lcr.write("TRIG")
                time.sleep(0.05)
                res = self.lcr.query("FETC?").strip()
                vals = [float(x) for x in res.split(",") if self._isfloat(x)]
                if len(vals) >= 2:
                    C_meas, D_meas = vals[0], vals[1]
                    R_meas = 1 / (2 * np.pi * self.frequency * C_meas * D_meas) if (C_meas > 0 and D_meas > 0) else np.nan
                else:
                    C_meas, R_meas = np.nan, np.nan
            except Exception:
                C_meas, R_meas = np.nan, np.nan

            data["voltage"].append(V)
            data["capacitance"].append(C_meas)
            data["resistance"].append(R_meas)
            data["current"].append(I_meas)

            print(f"V={V:6.1f} V  C={C_meas:8.2e} F  R={R_meas:8.2e} Ω  I={I_meas:8.2e} A")

        safe_off()
        return data

    @staticmethod
    def _isfloat(x):
        try:
            float(x)
            return True
        except Exception:
            return False

    # --------------------------------------------------------------
    def save_data(self, data):
        if data is None:
            return None

        timestamp = getdate()
        outdir = self.output_dir / self.sensor_name
        mkdir(outdir)

        fname = f"CV_{self.sensor_name}_pad{self.pad_number}_{timestamp}_{self.frequency:.0f}Hz.csv"
        fpath = outdir / fname

        np.savetxt(
            fpath,
            np.column_stack(
                (data["voltage"], data["capacitance"], data["resistance"], data["current"])
            ),
            delimiter=",",
            header="Voltage(V),Capacitance(F),Resistance(Ohm),Current(A)",
            comments="",
        )

        print(f"\n💾 Data saved: {fpath}")
        return fpath

    # --------------------------------------------------------------
    def plot_results(self, data, save_path=None):
        if data is None:
            return

        V = np.array(data["voltage"])
        C = np.array(data["capacitance"])
        R = np.array(data["resistance"])
        I = np.array(data["current"])

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9))

        ax1.plot(abs(V), C * 1e12, "b.-", label="C (pF)")
        ax1.set_xlabel("|Vbias| (V)")
        ax1.set_ylabel("Capacitance (pF)", color="b")
        ax1.grid(True)

        mask = (C > 0) & np.isfinite(C)
        if np.any(mask):
            ax1_twin = ax1.twinx()
            ax1_twin.plot(abs(V[mask]), 1 / C[mask] ** 2, "r.-", label="1/C²")
            ax1_twin.set_ylabel("1/C² (F⁻²)", color="r")

        ax2.plot(abs(V), R, "g.-", label="R (Ω)")
        ax2_twin = ax2.twinx()
        ax2_twin.plot(abs(V), np.abs(I), "m.-", label="|I| (A)")
        ax2.set_xlabel("|Vbias| (V)")
        ax2.set_ylabel("Resistance (Ω)", color="g")
        ax2_twin.set_ylabel("Current (A)", color="m")
        ax2.grid(True)

        plt.tight_layout()
        if save_path:
            plotfile = Path(save_path).with_suffix(".png")
            plt.savefig(plotfile)
            print(f"📊 Plot saved: {plotfile}")
        plt.show()

    # --------------------------------------------------------------
    def cleanup(self):
        try:
            self.smu.close()
            self.lcr.close()
        except Exception:
            pass


# -----------------------------------------------------------------
# CLI
# -----------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Automated C–V measurement (Keithley 2470 + Wayne Kerr 4300 via GPIB)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--iteration", "-n", type=int)
    parser.add_argument("--V0", type=float)
    parser.add_argument("--V1", type=float)
    parser.add_argument("--npts", type=int)
    parser.add_argument("--freq", type=float)
    parser.add_argument("--no-return", dest="return_sweep", action="store_false")
    parser.add_argument("--outdir", type=Path)
    parser.add_argument("--sensorname", type=str)
    parser.add_argument("--npad", type=str)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    start = time.time()

    cv = CVMeasurement()
    cv.iteration_count = args.iteration or 1
    cv.voltage_start = args.V0 or 0
    cv.voltage_end = args.V1 or -40
    cv.voltage_steps = args.npts or 81
    cv.frequency = args.freq or 1000
    cv.return_sweep = args.return_sweep if args.return_sweep is not None else True
    cv.output_dir = args.outdir or cv.output_dir
    cv.sensor_name = args.sensorname or "SENSOR"
    cv.pad_number = args.npad or "1_1"

    print("\n=== LGAD C–V Measurement (Keithley 2470 + Wayne Kerr 4300) ===")

    if not cv.connect_instruments():
        print("Exiting — instruments not found.")
        return
    if not cv.configure_instruments():
        print("Exiting — configuration failed.")
        cv.cleanup()
        return

    data = cv.perform_measurement()
    if data:
        path = cv.save_data(data)
        if path:
            cv.plot_results(data, path)

    cv.cleanup()
    print(f"\n✅ Measurement complete in {time.time() - start:.1f} s.")


if __name__ == "__main__":
    main()
