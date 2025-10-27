import os
import sys
import time
import pathlib
import signal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyvisa
from datetime import datetime

start_time = time.time()
pau = None

opathroot = r'C:\LGAD_test\I-V_test'
sensorname = 'UFSD-K1_W5_R(4_1)_T10_GR3_0_5x5'

def getdate():
    return datetime.now().strftime("%Y%m%d")

def mkdir(path):
    os.makedirs(path, exist_ok=True)

date = getdate()
Nmeas = 'LFtest20250226'
Npad = '1'

# Input validation for Iteration
try:
    Iteration = int(input("Enter a positive integer for Iteration: "))
    if Iteration <= 0:
        raise ValueError("Iteration must be a positive integer.")
except ValueError as e:
    print(f"Invalid input: {e}")
    sys.exit(1)

# Input for return sweep
return_sweep_input = input("Include return sweep? (y/n): ").strip().lower()
while return_sweep_input not in ['y', 'n']:
    print("Invalid input. Please enter 'y' or 'n'.")
    return_sweep_input = input("Include return sweep? (y/n): ").strip().lower()
return_sweep = return_sweep_input == 'y'

V0 = 0
V1 = -250
step = -1  # Voltage step size for main sweep (V0 to V1)
V2 = None  # Default to None (no refined range unless specified)
V3 = None  # Default to None (no refined range unless specified)
step1 = -0.5  # Voltage step size for refined range (if V2, V3 are set)
return_step = 5  # Voltage step size for return sweep (V1 to V0)

def init():
    global pau
    rm = pyvisa.ResourceManager()
    try:
        print(rm.list_resources())
        pau = rm.open_resource('GPIB0::22::INSTR')  # Keithley 6487
    except pyvisa.VisaIOError as e:
        print(f"Error connecting to picoammeter: {e}")
        sys.exit(1)

    try:
        pau.write("*RST")
        pau.write("curr:range auto")
        pau.write("INIT")
        pau.write("syst:zcor:stat off")
        pau.write("syst:zch off")
        print(pau.query("*IDN?"))
    except pyvisa.VisaIOError as e:
        print(f"Error configuring picoammeter: {e}")
        sys.exit(1)

def iv_pau_only():
    def handler(signum, frame):
        print("User interrupt. Turning off the picoammeter ...")
        try:
            pau.close()
        except:
            pass
        print("WARNING: Please ensure the external voltage source is turned off!")
        sys.exit(1)

    signal.signal(signal.SIGINT, handler)

    # Generate voltage array with step size
    Varr = np.arange(V0, V1 + step/2, step)  # Include V1 by adjusting endpoint

    # Refine voltage range if V2 and V3 are not None and within bounds
    if V2 is not None and V3 is not None:
        if V0 >= V2 >= V1 and V0 >= V3 >= V1:
            VarrL = Varr[Varr > V2]
            VarrH = Varr[Varr < V3]
            VarrM = np.arange(V2, V3 + step1/2, step1)  # Include V3
            Varr = np.concatenate([VarrL, VarrM, VarrH])

    print(Varr)

    if return_sweep:
        Varr_return = np.arange(V1, V0 + return_step/2, return_step)  # Include V0
        Varr = np.concatenate([Varr, Varr_return])

    try:
        print("Please ensure the external voltage source is set to 0V before starting.")
        input("Press Enter to begin measurements...")

        arr = []
        for V in Varr:
            print(f"Please set the external voltage source to {V}V.")
            input("Press Enter when ready...")
            time.sleep(0.5)  # Wait for voltage stabilization

            if Iteration > 0:
                Ipau_values = []

                for _ in range(Iteration):
                    try:
                        Ipau, _, _ = pau.query("READ?").split(',')
                        Ipau = float(Ipau[:-1])
                    except (ValueError, pyvisa.VisaIOError) as e:
                        print(f"Error reading measurement at V={V}: {e}")
                        continue
                    Ipau_values.append(Ipau)
                    time.sleep(0.1)  # Wait between iterations

                if Ipau_values:  # Ensure valid measurements
                    Ipau_avg = np.mean(Ipau_values)
                else:
                    print(f"No valid measurements at V={V}. Skipping.")
                    continue
            else:
                print("Invalid iteration count.")
                continue

            print(f"V_set={V}, I_pau={Ipau_avg}")
            arr.append([V, Ipau_avg])

    finally:
        # Ensure picoammeter is closed
        try:
            pau.close()
        except:
            pass
        print("WARNING: Please ensure the external voltage source is turned off!")

    # Save data as CSV
    opath = os.path.join(opathroot, Nmeas, f'{date}_{sensorname}')
    mkdir(opath)

    fname = f'IV_PAU_{sensorname}_{date}_{V0}_{V1}_pad{Npad}'
    ofname = os.path.join(opath, fname)
    k = 0
    while os.path.isfile(ofname + '.csv'):
        ofname = os.path.join(opath, f'{fname}_{k}')
        k += 1

    arr = np.array(arr, dtype=float)
    df = pd.DataFrame(arr, columns=['V_set(V)', 'I_pau(A)'])
    df.to_csv(ofname + '.csv', index=False, float_format='%.6e')
    
    ivplot(arr)
    plt.savefig(ofname + '.png')
    plt.close()

def ivplot(arr, yrange=None):
    arr = np.array(arr).T
    V = arr[0]  # Set voltage
    I = arr[1]  # PAU current
    I[I > 1e37] = np.min(I[np.isfinite(I)])  # Replace invalid values
    plt.plot(np.abs(V), np.abs(I))
    plt.yscale('log')
    plt.xlabel('Voltage (V)')
    plt.ylabel('Current (A)')
    plt.title(f'I-V Curve for {sensorname} (PAU)')
    plt.grid(True)
    if yrange:
        plt.ylim(yrange)

if __name__ == '__main__':
    init()
    iv_pau_only()
    plt.show()

end_time = time.time()
print(f"Execution time: {end_time - start_time:.2f} seconds")