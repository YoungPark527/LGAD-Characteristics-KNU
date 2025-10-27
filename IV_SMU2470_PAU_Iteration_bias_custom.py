import argparse
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pyvisa

# GPIB instrument addresses
SMU_ADDRESS = "GPIB0::18::INSTR"  # 2400=24 // 2470=18
PAU_ADDRESS = "GPIB0::22::INSTR"  # 6487
PAU2_ADDRESS = "GPIB0::14::INSTR"  # Second PAU


def getdate():
    return datetime.now().strftime("%Y%m%d")


def mkdir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def init_instruments(smu_addr: str, pau_addr: str, pau2_addr: str | None = None, verbose: bool = True):
    """Open and configure the SMU and up to two PAU instruments.

    Returns (smu, pau, pau2, rm) where pau2 may be None.
    """
    rm = pyvisa.ResourceManager()
    if verbose:
        try:
            print("Found resources:", rm.list_resources())
        except Exception:
            # Some backends may raise if no resources are present
            pass

    smu = rm.open_resource(smu_addr)
    pau = rm.open_resource(pau_addr)
    pau2 = None
    if pau2_addr:
        pau2 = rm.open_resource(pau2_addr)

    # Configure SMU
    smu.read_termination = "\n"
    smu.write_termination = "\n"
    smu.write(":SOUR:FUNC VOLT")
    smu.write("SOUR:VOLT:LEV 0")
    smu.write("SOUR:VOLT:RANG 1000")
    smu.write(":SOUR:VOLT:ILIMIT 300e-6")
    smu.write(":SENS:CURR:RANG 300e-6")
    smu.write(":SENS:FUNC \"VOLT\"")
    smu.write(":SENS:FUNC \"CURR\"")

    # Configure PAU(s)
    pau.write("*RST")
    pau.write("curr:range auto")
    pau.write("INIT")
    try:
        pau.write("syst:zcor:stat off")
        pau.write("syst:zch off")
    except Exception:
        # Some models may not support these commands
        pass

    if pau2 is not None:
        try:
            pau2.write("*RST")
            pau2.write("curr:range auto")
            pau2.write("INIT")
            pau2.write("syst:zcor:stat off")
            pau2.write("syst:zch off")
        except Exception:
            pass

    if verbose:
        try:
            print("SMU:", smu.query("*IDN?"))
            print("PAU:", pau.query("*IDN?"))
            if pau2 is not None:
                print("PAU2:", pau2.query("*IDN?"))
        except Exception:
            pass

    return smu, pau, pau2, rm


def iv_smu_pau(
    smu,
    pau,
    pau2,
    outdir: Path,
    sensorname: str,
    nmeas: str,
    npad: str,
    V0: float,
    V1: float,
    npts: int,
    V2: float | None,
    V3: float | None,
    npts1: int,
    iteration: int,
    return_sweep: bool = True,
):
    """Run the IV sweep using SMU and PAU and save results + plot.
    """
    running = True

    def _safe_off_and_close():
        try:
            if smu:
                smu.write(":SOUR:VOLT:LEV 0")
                smu.write("OUTP OFF")
                smu.close()
        except Exception:
            pass
        try:
            if pau:
                pau.close()
        except Exception:
            pass
        try:
            if pau2:
                pau2.close()
        except Exception:
            pass

    def handler(signum, frame):
        print("User interrupt. Turning off the output and closing instruments...")
        _safe_off_and_close()
        sys.exit(1)

    signal.signal(signal.SIGINT, handler)

    # build voltage array
    Varr = np.linspace(V0, V1, npts)
    if V2 is not None and V3 is not None and (V2 > V1) and (V3 > V1):
        VarrL = Varr[Varr > V2]
        VarrH = Varr[Varr < V3]
        VarrM = np.linspace(V2, V3, npts1)
        Varr = np.concatenate([VarrL, VarrM, VarrH])

    if return_sweep:
        Varr_return = np.linspace(V1, V0, int(abs(V0 - V1) / 5 + 1))
        Varr = np.concatenate([Varr, Varr_return])

    print("Voltage sweep ({} pts):".format(len(Varr)))

    smu.write(":SOUR:VOLT:LEV 0")
    smu.write("OUTP ON")
    time.sleep(0.5)

    data = []
    for V in Varr:
        smu.write(f":SOUR:VOLT:LEV {V}")
        time.sleep(0.5)

        if iteration > 0:
            Ismu_values = []
            Ipau_values = []
            for _ in range(iteration):
                Ismu_raw = smu.query(":MEAS:CURR?")
                pau_raw = pau.query("READ?")
                pau2_raw = None
                if pau2 is not None:
                    pau2_raw = pau2.query("READ?")
                try:
                    Ismu = float(Ismu_raw.strip())
                except Exception:
                    Ismu = np.nan
                # PAU returns CSV-like: value,range,status
                try:
                    Ipau_str = pau_raw.split(',')[0].strip()
                    Ipau = float(Ipau_str)
                except Exception:
                    Ipau = np.nan
                try:
                    if pau2_raw is not None:
                        Ipau2_str = pau2_raw.split(',')[0].strip()
                        Ipau2 = float(Ipau2_str)
                    else:
                        Ipau2 = np.nan
                except Exception:
                    Ipau2 = np.nan

                Ismu_values.append(Ismu)
                Ipau_values.append(Ipau)
                # store pau2 values per-iteration in local list; will average later
                if 'Ipau2_values' not in locals():
                    Ipau2_values = []
                Ipau2_values.append(Ipau2)
                time.sleep(0.1)

            Ismu_avg = float(np.nanmean(Ismu_values))
            Ipau_avg = float(np.nanmean(Ipau_values))
            Ipau2_avg = float(np.nanmean(Ipau2_values)) if 'Ipau2_values' in locals() else float(np.nan)
        else:
            print("Iteration must be > 0")
            Ismu_avg = np.nan
            Ipau_avg = np.nan
            Ipau2_avg = np.nan

        Vsmu_raw = smu.query(":MEAS:VOLT?")
        try:
            Vsmu = float(Vsmu_raw.strip())
        except Exception:
            Vsmu = float(V)

        print(f"Vset={V:.3f} Vmeas={Vsmu:.3f} I_pad={Ipau_avg:.3e} I_other={Ipau2_avg:.3e} I_back={Ismu_avg:.3e}")
        # Order: bias voltage, I_pad (6487), I_other (PAU2), I_back (2470)
        data.append([V, Ipau_avg, Ipau2_avg, Ismu_avg])

    _safe_off_and_close()

    date = getdate()
    outpath = outdir / nmeas / f"{date}_{sensorname}"
    mkdir(outpath)
    base = outpath / f"IV_SMU+PAU_{sensorname}_{date}_{V0}_{V1}_pad{npad}"
    ofname = str(base)
    k = 0
    while (Path(ofname + ".csv")).exists():
        ofname = f"{base}_{k}"
        k += 1

    arr = np.array(data, dtype=float)
    header = "bias_voltage,I_pad,I_other,I_back"
    fmt = "%.6e,%.6e,%.6e,%.6e"
    np.savetxt(ofname + ".csv", arr, delimiter=",", header=header, fmt=fmt, comments="")    # Plot all currents with updated labels
    plt.figure()
    Vabs = np.abs(arr[:, 0])  # bias voltage
    plt.plot(Vabs, np.abs(arr[:, 1]), marker=".", label="I_pad")
    plt.plot(Vabs, np.abs(arr[:, 2]), marker=".", label="I_other")
    plt.plot(Vabs, np.abs(arr[:, 3]), marker=".", label="I_back")
    plt.yscale("log")
    plt.xlabel("|Bias Voltage| (V)")
    plt.ylabel("|Current| (A)")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(ofname + ".png")


def parse_args():
    parser = argparse.ArgumentParser(description="Run IV sweep with SMU and PAU (simplified)")
    parser.add_argument("--iteration", "-n", type=int, default=1, help="Number of measurements per voltage")
    parser.add_argument("--V0", type=float, default=0.0, help="Start voltage")
    parser.add_argument("--V1", type=float, default=-250.0, help="End voltage")
    parser.add_argument("--npts", type=int, default=251, help="Number of points in main sweep")
    parser.add_argument("--V2", type=float, default=-50.0, help="Optional mid low for custom sweep")
    parser.add_argument("--V3", type=float, default=-150.0, help="Optional mid high for custom sweep")
    parser.add_argument("--npts1", type=int, default=51, help="Points in mid sweep")
    parser.add_argument("--no-return", dest="return_sweep", action="store_false", help="Disable return sweep")
    parser.add_argument("--outdir", type=Path, default=Path.home() / "LGAD_test" / "I-V_test", help="Output root directory")
    parser.add_argument("--sensorname", type=str, default="sensor", help="Sensor name for filenames")
    parser.add_argument("--nmeas", type=str, default="LGADtest", help="Measurement name folder")
    parser.add_argument("--npad", type=str, default="1", help="Pad number")
    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()

    try:
        smu, pau, pau2, rm = init_instruments(SMU_ADDRESS, PAU_ADDRESS, PAU2_ADDRESS)
    except Exception as e:
        print("Failed to open instruments:", e)
        return

    try:
        iv_smu_pau(
            smu=smu,
            pau=pau,
            pau2=pau2,
            outdir=args.outdir,
            sensorname=args.sensorname,
            nmeas=args.nmeas,
            npad=args.npad,
            V0=args.V0,
            V1=args.V1,
            npts=args.npts,
            V2=args.V2,
            V3=args.V3,
            npts1=args.npts1,
            iteration=args.iteration,
            return_sweep=args.return_sweep,
        )
    finally:
        # ensure resources closed if not already
        try:
            smu.write(":SOUR:VOLT:LEV 0")
            smu.write("OUTP OFF")
        except Exception:
            pass
        try:
            smu.close()
        except Exception:
            pass

    end_time = time.time()
    print(f"Elapsed: {end_time - start_time:.2f} s")


if __name__ == "__main__":
    main()