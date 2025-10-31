import argparse
import os
import sys
import signal
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import pyvisa


# ------------------------- Configuration -------------------------
PAU_ADDRESS = "GPIB1::22::INSTR"  # Keithley 6487 (PAU)
# -----------------------------------------------------------------


def getdate():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def mkdir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


class CVMeasurement:
    def __init__(self):
        self.output_dir = Path.home() / "LGAD_test" / "C-V_test"
        self.voltage_start = 0
        self.voltage_end = -40
        self.voltage_steps = 81
        self.frequency = 1000  # Hz
        self.iteration_count = 1
        self.sensor_name = "SENSOR"
        self.pad_number = "1_1"
        self.return_sweep = True

    # --------------------------------------------------------------
    # Connection & Configuration
    # --------------------------------------------------------------
    def connect_instruments(self):
        """Connect Keithley 6487 (GPIB) and LCR meter (USB)."""
        rm = pyvisa.ResourceManager()

        print("\n🔌 Scanning VISA devices...")
        resources = rm.list_resources()
        for i, r in enumerate(resources):
            print(f"{i+1}: {r}")

        try:
            print(f"\nConnecting to Keithley 6487 at {PAU_ADDRESS} ...")
            self.pau = rm.open_resource(PAU_ADDRESS)

            lcr_idx = int(input("\nSelect LCR meter number from the list above: ")) - 1
            if lcr_idx < 0 or lcr_idx >= len(resources):
                raise ValueError("Invalid LCR selection")

            self.lcr = rm.open_resource(resources[lcr_idx])
            self.lcr.read_termination = "\n"
            self.lcr.write_termination = "\n"

            print("\n✅ Connected instruments:")
            print(f"PAU: {self.pau.query('*IDN?').strip()}")
            print(f"LCR: {self.lcr.query('*IDN?').strip()}")
            return True

        except Exception as e:
            print(f"\n❌ Connection error: {e}")
            return False

    def configure_instruments(self):
        """Configure both instruments."""
        try:
            print("\n⚙️ Configuring Keithley 6487...")
            self.pau.write("*RST")
            self.pau.write(":SYST:ZCH OFF")
            self.pau.write(":SOUR:VOLT:RANG 500")
            self.pau.write(":SOUR:VOLT:ILIM 1e-4")
            self.pau.write(":SOUR:VOLT:STAT OFF")
            self.pau.write(":FORM:ELEM CURR,VOLT")

            print("⚙️ Configuring LCR meter...")
            self.lcr.write("*RST")
            self.lcr.write(":MEAS:FUNC1 C")
            self.lcr.write(":MEAS:FUNC2 R")
            self.lcr.write(":MEAS:EQU-CCT PAR")
            self.lcr.write(":MEAS:SPEED FAST")
            self.lcr.write(":MEAS:LEV 0.1")
            self.lcr.write(":MEAS:BIAS OFF")
            self.lcr.write(f":MEAS:FREQ {self.frequency}")
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
                self.pau.write(":SOUR:VOLT 0")
                self.pau.write(":SOUR:VOLT:STAT OFF")
                self.lcr.write(":MEAS:BIAS OFF")
            except Exception:
                pass

        def sigint_handler(sig, frame):
            print("\n⚠️ Interrupted by user. Turning off safely...")
            safe_off()
            sys.exit(0)

        signal.signal(signal.SIGINT, sigint_handler)

        self.pau.write(":SOUR:VOLT 0")
        self.pau.write(":SOUR:VOLT:STAT ON")
        time.sleep(1)

        for V in voltages:
            if V > 0:
                print("Warning: positive bias not allowed. Forcing 0 V.")
                V = 0

            self.pau.write(f":SOUR:VOLT {V}")
            time.sleep(0.1)

            # Measure current and voltage
            try:
                raw = self.pau.query(":READ?").strip()
                vals = [float(x) for x in raw.split(",") if self._isfloat(x)]
                Ipau, Vpau = vals[0], vals[1]
            except Exception:
                Ipau, Vpau = np.nan, V

            # Measure capacitance/resistance
            cap_vals, res_vals = [], []
            for _ in range(self.iteration_count):
                try:
                    res = self.lcr.query(":MEAS:TRIG?").strip()
                    vals = [float(x) for x in res.split(",") if self._isfloat(x)]
                    if len(vals) >= 2:
                        cap_vals.append(vals[0])
                        res_vals.append(vals[1])
                except Exception:
                    continue
                time.sleep(0.05)

            C_avg = np.mean(cap_vals) if cap_vals else np.nan
            R_avg = np.mean(res_vals) if res_vals else np.nan

            data["voltage"].append(V)
            data["capacitance"].append(C_avg)
            data["resistance"].append(R_avg)
            data["current"].append(Ipau)

            print(f"V={V:6.1f}V  C={C_avg:8.2e}F  R={R_avg:8.2e}Ω  I={Ipau:8.2e}A")

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
    # Data Handling
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

        # Save metadata
        meta = outdir / (fname.replace(".csv", "_meta.txt"))
        with open(meta, "w") as f:
            f.write(f"Date: {timestamp}\n")
            f.write(f"Sensor: {self.sensor_name}\n")
            f.write(f"Pad: {self.pad_number}\n")
            f.write(f"Frequency: {self.frequency} Hz\n")
            f.write(f"Voltage range: {self.voltage_start} to {self.voltage_end} V\n")
            f.write(f"Steps: {self.voltage_steps}\n")

        print(f"\n💾 Data saved: {fpath}")
        return fpath

    # --------------------------------------------------------------
    # Plotting
    # --------------------------------------------------------------
    def plot_results(self, data, save_path=None):
        if data is None:
            return

        V = np.array(data["voltage"])
        C = np.array(data["capacitance"])
        R = np.array(data["resistance"])
        I = np.array(data["current"])

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9))

        # --- Capacitance plot ---
        ax1.plot(abs(V), C * 1e12, "b.-", label="C (pF)")
        ax1.set_xlabel("|Vbias| (V)")
        ax1.set_ylabel("Capacitance (pF)", color="b")
        ax1.grid(True)

        mask = (C > 0) & np.isfinite(C)
        if np.any(mask):
            ax1_twin = ax1.twinx()
            ax1_twin.plot(abs(V[mask]), 1 / C[mask] ** 2, "r.-", label="1/C²")
            ax1_twin.set_ylabel("1/C² (F⁻²)", color="r")

        # --- Resistance & Current plot ---
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
            self.pau.close()
            self.lcr.close()
        except Exception:
            pass


# -----------------------------------------------------------------
# CLI argument parsing
# -----------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Automated C–V measurement with Keithley 6487 and LCR meter",
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

    # Interactive fallback
    if len(sys.argv) == 1:
        print("\nNo command-line args detected — switching to interactive mode.\n")
        def ask(prompt, default, t):
            v = input(f"{prompt} [{default}]: ").strip()
            return t(v) if v else default

        args.iteration = ask("Number of measurements per voltage", 1, int)
        args.V0 = ask("Start voltage (V)", 0.0, float)
        args.V1 = ask("End voltage (V)", -40.0, float)
        args.npts = ask("Number of voltage points", 81, int)
        args.freq = ask("Measurement frequency (Hz)", 1000.0, float)
        args.return_sweep = ask("Include return sweep? (y/n)", "y", str).lower().startswith("y")
        args.outdir = Path(ask("Output directory", str(Path.home() / "LGAD_test" / "C-V_test"), str))
        args.sensorname = ask("Sensor name", "SENSOR", str)
        args.npad = ask("Pad number", "1_1", str)

    return args


# -----------------------------------------------------------------
# Main function
# -----------------------------------------------------------------
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

    print("\n=== LGAD C–V Measurement ===")

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
