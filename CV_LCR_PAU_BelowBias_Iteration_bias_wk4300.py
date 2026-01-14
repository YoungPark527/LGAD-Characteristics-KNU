import os
import numpy as np
import matplotlib.pyplot as plt
import pyvisa
import time
import signal
import matplotlib as mp
from datetime import datetime
from pathlib import Path

# --- Setup Plotting Style ---
mp.rcParams.update({'font.size': 15})

# --- Configuration & Paths ---
opathroot = r'C:\LGAD_test\C-V_test' 
sensorname = r'UFSD-K2_2x2'
Nmeas = r'test20250731'
Npad = '1_1'
Ntimes = True

# --- Utility Functions (Replacing 'util' library) ---
def getdate():
    return datetime.now().strftime("%Y%m%d")

def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

# --- Measurement parameters ---
Iteration = int(input("Enter a positive integer for Iteration (Averaging): "))
V0 = 0 
V1 = -40
npts = 81

# Optional fine-sweep range
V2 = -15
V3 = -25
npts1 = 51

date = getdate()

def CVmeasurement(freq, return_sweep=False):
    rm = pyvisa.ResourceManager()
    rlist = rm.list_resources()
    print("Resources found:", rlist)

    # Initialize Instruments
    try:
        pau = rm.open_resource('GPIB0::22::INSTR')
        lcr = rm.open_resource('GPIB0::6::INSTR')
        lcr.read_termination = '\n'
        lcr.write_termination = '\n'
        pau.read_termination = '\n' # Added for stability
    except Exception as e:
        print(f"Error connecting to instruments: {e}")
        return

    print("PAU IDN:", pau.query("*IDN?"))
    print("LCR IDN:", lcr.query("*IDN?"))

    # Configure Picoammeter (PAU)
    pau.write("*RST")
    pau.write("curr:range 2e-4")
    pau.write("INIT")
    pau.write("syst:zch off")
    pau.write("SOUR:VOLT:STAT off")
    pau.write("SOUR:VOLT:RANG 500")
    pau.write("SOUR:VOLT:ILIM 1e-4")
    pau.write("FORM:ELEM READ,UNIT,STAT,VSO")

    # Configure LCR Meter
    lcr.write(":MEAS:NUM-OF-TESTS 1")
    lcr.write(":MEAS:FUNC1 C")
    lcr.write(":MEAS:FUNC2 Z")
    lcr.write(":meas:equ-cct par")
    lcr.write(":MEAS:SPEED med") # Changed from fast to med for better SNR on LGADs
    lcr.write(":MEAS:LEV 0.1")
    lcr.write(":MEAS:V-BIAS 0V")
    lcr.write(f":MEAS:FREQ {freq}")

    # Interrupt Handler for Safety
    def handler(signum, frame):
        print("\nUser interrupt! Safely ramping down...")
        pau.write(":sour:volt:lev 0")
        pau.write(":sour:volt:stat off")
        lcr.write(":MEAS:BIAS OFF")
        lcr.close()
        pau.close()
        exit(1)

    signal.signal(signal.SIGINT, handler)

    # --- Generate Voltage Array ---
    Varr = np.linspace(V0, V1, npts)
    if (V2 is not None):
        if (V2 > V1) and (V3 > V1):
            VarrL = Varr[Varr > V2]
            VarrH = Varr[Varr < V3]
            VarrM = np.linspace(V2, V3, npts1)
            Varr = np.concatenate([VarrL, VarrM, VarrH])

    if return_sweep:
        Varr_return = np.linspace(V1, V0, int(abs(V0-V1)/10 + 1))
        Varr = np.concatenate([Varr, Varr_return])

    # --- Measurement Loop ---
    Vpau_arr, Ipau_arr, CV_arr, RV_arr = [], [], [], []

    pau.write(":sour:volt 0")
    pau.write(":sour:volt:stat on")
    lcr.write("meas:bias OFF")
    time.sleep(1)

    print("\nStarting Sweep (V_target | V_actual | I_pau | Cap | Res)")
    t0 = time.time()
    
    for Vdc in Varr:
        if Vdc > 0:
            Vdc = 0 # Safety: no forward bias

        pau.write(f':sour:volt {Vdc}')
        time.sleep(0.05) 
        
        # Read Picoammeter
        raw_pau = pau.query(":READ?").split(',')
        Ipau = float(raw_pau[0])
        Vpau = float(raw_pau[3])

        # Read LCR Meter with Averaging
        cap_samples = []
        res_samples = []
        
        for _ in range(max(1, Iteration)):
            res = lcr.query('meas:trig?')
            try:
                c_val, r_val = map(float, res.split(','))
                cap_samples.append(c_val)
                res_samples.append(r_val)
            except ValueError:
                continue
            time.sleep(0.02)

        Cavg = np.mean(cap_samples) if cap_samples else 0
        Ravg = np.mean(res_samples) if res_samples else 0

        Vpau_arr.append(Vpau)
        Ipau_arr.append(Ipau)
        CV_arr.append(Cavg)
        RV_arr.append(Ravg)

        print(f"{Vdc:.1f}V | {Vpau:.2f}V | {Ipau:.2e}A | {Cavg:.3e}F | {Ravg:.2e}R")

    t1 = time.time()
    
    # --- Shutdown ---
    pau.write(":sour:volt 0")
    pau.write(":sour:volt:stat off")
    pau.close()
    lcr.write(":MEAS:BIAS OFF")
    lcr.close()

    # --- Save Data ---
    opath = os.path.join(opathroot, Nmeas, f"{date}_{sensorname}")
    mkdir(opath)
    
    base_fname = f'CV_LCR+PAU_{sensorname}_{date}_{freq}Hz_pad{Npad}'
    ofname = os.path.join(opath, base_fname)

    i = 0
    while os.path.isfile(ofname + '.txt'):
        ofname = os.path.join(opath, f"{base_fname}_{i}")
        i += 1

    header = 'Vpau(V)\tC(F)\tR(Ohm)\tIpau(A)'
    data_to_save = np.array([Vpau_arr, CV_arr, RV_arr, Ipau_arr]).T
    np.savetxt(ofname + '.txt', data_to_save, header=header, delimiter='\t')
    
    print(f"\nMeasurement complete. Elapsed: {t1-t0:.1f}s")
    print(f"Data saved to: {ofname}.txt")

    plot_cv(ofname + '.txt', freq)
    plt.savefig(ofname + '.png')

def plot_cv(fname, freq=None):
    # Load data, skipping header
    data = np.genfromtxt(fname)
    V = data[:, 0]
    C = data[:, 1]
    R = data[:, 2]

    # Use absolute value of Voltage for standard C-V plotting
    V_plot = np.abs(V)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Axis 1: Capacitance (nF)
    ax1.plot(V_plot, C * 1e9, 'o-', color='tab:blue', label="Capacitance")
    ax1.set_xlabel('Bias Voltage |V| (V)')
    ax1.set_ylabel('C (nF)', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True, alpha=0.3)

    # Axis 2: 1/C^2 (Standard for Doping/Depletion)
    ax2 = ax1.twinx()
    inv_c2 = 1.0 / (C**2)
    ax2.plot(V_plot, inv_c2, 's-', color='tab:green', alpha=0.6, label="$1/C^2$")
    ax2.set_ylabel('$1/C^2$ ($F^{-2}$)', color='tab:green')
    ax2.tick_params(axis='y', labelcolor='tab:green')

    plt.title(f"C-V Test: {sensorname} ({freq} Hz)")
    fig.tight_layout()

def cvtest():
    freq = int(1e3)
    print("Frequency [Hz] :", freq)
    CVmeasurement(freq, return_sweep=True)
    plt.show()

if __name__ == '__main__':
    cvtest()