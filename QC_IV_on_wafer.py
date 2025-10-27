import numpy as np
import os
import time
import signal
import logging
import matplotlib.pyplot as plt
import pyvisa
from datetime import date

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
opathroot = r'C:\LGAD_test\I-V_test'
sensorname = 'UFSD-K2_W11_irr50_2x2_T9_GR3_0'
Nmeas = 'test20250313'
Npad = '2_2'
Iteration = int(input("Enter a positive integer for Iteration: "))

# Voltage settings
Vfd_NOM = -25  # Example value, replace with actual nominal full depletion voltage
Vbd_NOM = -200  # Example value, replace with actual nominal breakdown voltage
VMaxOp = Vbd_NOM + 10  # Maximum voltage of operation, at least 10 V lower than Vbd

V0 = 0
V1 = Vfd_NOM - 5
V2 = VMaxOp

step1 = -2
step2 = -10

# Initialize SMU
def init_smu():
    global smu
    try:
        rm = pyvisa.ResourceManager()
        logging.info(f"Available resources: {rm.list_resources()}")
        smu = rm.open_resource('GPIB0::18::INSTR')  # 2470 SMU

        smu.read_termination = '\n'
        smu.write_termination = '\n'

        smu.write(":SOUR:FUNC VOLT")
        smu.write("SOUR:VOLT:LEV 0")
        smu.write("SOUR:VOLT:RANG 1000")
        smu.write(":SOUR:VOLT:ILIMIT 300e-6")  # 300 µA current limit for full sensors
        smu.write(":SENS:CURR:RANG 20e-6")
        smu.write(":SENS:FUNC \"VOLT\"")
        smu.write(":SENS:FUNC \"CURR\"")

        logging.info(f"SMU ID: {smu.query('*IDN?')}")
    except Exception as e:
        logging.error(f"Failed to initialize SMU: {e}")
        raise

# Signal handler for safe exit
def handler(signum, frame):
    logging.warning("User interrupt. Turning off the output ...")
    smu.write(':sour:volt:lev 0')
    smu.write('outp off')
    smu.close()
    logging.warning("Output turned off. Exiting.")
    exit(1)

signal.signal(signal.SIGINT, handler)

# IV measurement function
def iv_smu():
    start_time = time.time()  # Define start time

    Varr1 = np.arange(V0, V1 + step1, step1)
    Varr2 = np.arange(V1 + step1, V2, step2)
    Varr = np.concatenate((Varr1, Varr2))

    arr = []
    reverse_sweep_started = False

    try:
        for V in Varr:
            smu.write(f':sour:volt:lev {V}')
            time.sleep(0.5 if V >= V1 else 2)

            Ismu_values = [float(smu.query(":MEAS:CURR?")) for _ in range(Iteration)]
            Ismu_avg = np.mean(Ismu_values)
            Ismu_std = np.std(Ismu_values)
            Vsmu = float(smu.query(":MEAS:VOLT?"))

            logging.info(f"Measured Voltage: {Vsmu} V, Measured Current: {Ismu_avg} A")

            if Ismu_avg > 300e-6 and not reverse_sweep_started:
                logging.warning(f"Current exceeds 300 µA. Starting reverse sweep at {Vsmu} V.")
                reverse_sweep_started = True
                reverse_Varr = np.linspace(V, V0, len(Varr))
                smu.write(':sour:volt:lev 0')
                time.sleep(0.5)
                for V_rev in reverse_Varr:
                    smu.write(f':sour:volt:lev {V_rev}')
                    time.sleep(0.5)
                    Ismu = float(smu.query(":MEAS:CURR?"))
                    Vsmu = float(smu.query(":MEAS:VOLT?"))
                    arr.append([V_rev, Vsmu, Ismu, 0])
                break

            arr.append([V, Vsmu, Ismu_avg, Ismu_std])

    except Exception as e:
        logging.error(f"Error during IV measurement: {e}")
    finally:
        smu.write(':sour:volt:lev 0')
        smu.write('outp off')
        smu.close()

    save_results(arr)

# Save results to file
def save_results(arr):
    opath = os.path.join(opathroot, Nmeas, f'{date.today()}_{sensorname}')
    os.makedirs(opath, exist_ok=True)

    fname = f'IV_SMU_{sensorname}_{date.today()}_{V0}_{V1}_pad{Npad}'
    ofname = os.path.join(opath, fname)
    k = 0
    while os.path.isfile(ofname + '.txt'):
        ofname = f'{ofname}_{k}'
        k += 1

    arr = np.array(arr, dtype=float)
    np.savetxt(ofname + '.txt', arr)
    ivplot(arr)
    plt.savefig(ofname + '.png')

# Plot IV characteristics
def ivplot(arr, yrange=None):
    arr = np.array(arr).T
    plt.figure()
    plt.plot(arr[0], arr[2], label='Current (A)')
    plt.xlabel('Voltage (V)')
    plt.ylabel('Current (A)')
    plt.title('IV Characteristics')
    plt.legend()
    if yrange:
        plt.ylim(yrange)
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    init_smu()
    iv_smu()
