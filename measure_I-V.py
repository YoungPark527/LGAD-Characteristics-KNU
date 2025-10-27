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
smu = None
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
V2 = None  # Optional voltage for refined range
V3 = None  # Optional voltage for refined range
step1 = None # Voltage step size for refined range (V2 to V3)
return_step = 5  # Voltage step size for return sweep (V1 to V0)

def init():
    global smu, pau
    rm = pyvisa.ResourceManager()
    try:
        print(rm.list_resources())
        smu = rm.open_resource('GPIB0::18::INSTR')  # 2400=24 // 2470=18
        pau = rm.open_resource('GPIB0::22::INSTR')  # 6487
    except pyvisa.VisaIOError as e:
        print(f"Error connecting to instruments: {e}")
        sys.exit(1)

    smu.read_termination = '\n'
    smu.write_termination = '\n'

    try:
        smu.write(":SOUR:FUNC VOLT")
        smu.write("SOUR:VOLT:LEV 0")
        smu.write("SOUR:VOLT:RANG 1000")
        smu.write(":SOUR:VOLT:ILIMIT 100e-6")  # 100 µA limit
        smu.write(":SENS:CURR:RANG 100e-6")
        smu.write(":SENS:FUNC \"VOLT\"")
        smu.write(":SENS:FUNC \"CURR\"")

        pau.write("*RST")
        pau.write("curr:range auto")
        pau.write("INIT")
        pau.write("syst:zcor:stat off")
        pau.write("syst:zch off")

        print(smu.query("*IDN?"))
        print(pau.query("*IDN?"))
    except pyvisa.VisaIOError as e:
        print(f"Error configuring instruments: {e}")
        sys.exit(1)

def iv_smu_pau():
    def handler(signum, frame):
        print("User interrupt. Turning off the output ...")
        try:
            smu.write(':sour:volt:lev 0')
            smu.write('outp off')
            smu.close()
            pau.close()
        except:
            pass
        print("WARNING: Please make sure the output is turned off!")
        sys.exit(1)

    signal.signal(signal.SIGINT, handler)

    # Generate voltage array with step size
    Varr = np.arange(V0, V1 + step/2, step)  # Include V1 by adjusting endpoint

    # Refine voltage range if V2 and V3 are within bounds
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
        smu.write(':sour:volt:lev 0')
        smu.write('outp on')
        time.sleep(0.5)  # Wait for instrument stabilization

        arr = []
        for V in Varr:
            smu.write(f':sour:volt:lev {V}')
            time.sleep(0.5)  # Wait for voltage settling

            if Iteration > 0:
                Ismu_values = []
                Ipau_values = []

                for _ in range(Iteration):
                    try:
                        Ismu = float(smu.query(":MEAS:CURR?"))
                        Ipau, _, _ = pau.query("READ?").split(',')
                        Ipau = float(Ipau[:-1])
                    except (ValueError, pyvisa.VisaIOError) as e:
                        print(f"Error reading measurement at V={V}: {e}")
                        continue
                    Ismu_values.append(Ismu)
                    Ipau_values.append(Ipau)
                    time.sleep(0.1)  # Wait between iterations

                if Ismu_values and Ipau_values:  # Ensure valid measurements
                    Ismu_avg = np.mean(Ismu_values)
                    Ipau_avg = np.mean(Ipau_values)
                else:
                    print(f"No valid measurements at V={V}. Skipping.")
                    continue
            else:
                print("Invalid iteration count.")
                continue

            try:
                Vsmu = float(smu.query(":MEAS:VOLT?"))
            except (ValueError, pyvisa.VisaIOError) as e:
                print(f"Error reading voltage at V={V}: {e}")
                continue

            print(V, Vsmu, Ismu_avg, Ipau_avg)
            arr.append([V, Vsmu, Ismu_avg, Ipau_avg])

    finally:
        # Ensure instruments are safely turned off
        try:
            smu.write(':sour:volt:lev 0')
            smu.write('outp off')
            smu.close()
            pau.close()
        except:
            pass

    # Save data as CSV
    opath = os.path.join(opathroot, Nmeas, f'{date}_{sensorname}')
    mkdir(opath)

    fname = f'IV_SMU+PAU_{sensorname}_{date}_{V0}_{V1}_pad{Npad}'
    ofname = os.path.join(opath, fname)
    k = 0
    while os.path.isfile(ofname + '.csv'):
        ofname = os.path.join(opath, f'{fname}_{k}')
        k += 1

    arr = np.array(arr, dtype=float)
    df = pd.DataFrame(arr, columns=['V_set(V)', 'V_meas(V)', 'I_smu(A)', 'I_pau(A)'])
    df.to_csv(ofname + '.csv', index=False, float_format='%.6e')
    
    ivplot(arr)
    plt.savefig(ofname + '.png')
    plt.close()

def ivplot(arr, yrange=None):
    arr = np.array(arr).T
    V = arr[0]  # Set voltage
    I = arr[2]  # SMU current (use arr[3] for PAU current if needed)
    I[I > 1e37] = np.min(I[np.isfinite(I)])  # Replace invalid values
    plt.plot(np.abs(V), np.abs(I))
    plt.yscale('log')
    plt.xlabel('Voltage (V)')
    plt.ylabel('Current (A)')
    plt.title(f'I-V Curve for {sensorname}')
    plt.grid(True)
    if yrange:
        plt.ylim(yrange)

if __name__ == '__main__':
    init()
    iv_smu_pau()
    plt.show()

end_time = time.time()
print(f"Execution time: {end_time - start_time:.2f} seconds")