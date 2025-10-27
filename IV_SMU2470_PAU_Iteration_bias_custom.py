import os
import sys
import time
import pathlib
import signal
import numpy as np
import pylab as plt
import pyvisa
from datetime import datetime
import matplotlib.pyplot as plt
sys.path.append(pathlib.Path(__file__).parent.resolve())

start_time = time.time()
smu = []
smuI = []
pau = []

opathroot = r'C:\LGAD_test\I-V_test'
sensorname = 'UFSD-K1_W5_R(4_1)_T10_GR3_0_5x5'

def getdate():
    return datetime.now().strftime("%Y%m%d")

date = getdate()
Nmeas = 'LFtest20250226'
Npad = '1'

Ntimes = True
Iteration = int(input("Enter a positive integer for Iteration: " ) )
#Iteration = 1
V0 = 0
V1 = -250
npts = 251

V2 = -50
V3 = -150
npts1 = 51

return_sweep = True

def init():
    global smu, pau
    ## initialize lcr and smu
    rm = pyvisa.ResourceManager()
    print (rm.list_resources())
    smu = rm.open_resource('GPIB0::18::INSTR') # 2400=24 // 2470=18
    pau = rm.open_resource('GPIB0::22::INSTR') # 6487

    smu.read_termination = '\n'
    smu.write_termination = '\n'

    smu.write(":SOUR:FUNC VOLT")
    smu.write("SOUR:VOLT:LEV 0")
    smu.write("SOUR:VOLT:RANG 1000")
    smu.write(":SOUR:VOLT:ILIMIT 100e-6") #1mA for full sensors and 100mA for other pads
#   smu.write(":SOUR:VOLT:ILIMIT 100e-6") #1mA for full sensors and 100mA for other pads
    smu.write(":SENS:CURR:RANG 100e-6")
    smu.write(":SENS:FUNC \"VOLT\"")
    smu.write(":SENS:FUNC \"CURR\"")

    pau.write("*RST")
    pau.write("curr:range auto")
    pau.write("INIT")
    pau.write("syst:zcor:stat off")
    pau.write("syst:zch off")

    print (smu.query("*IDN?"))
    print (pau.query("*IDN?"))

init()

def iv_smu_pau():

    def handler(signum, frame):
        print ("User interrupt. Turning off the output ...")
        smu.write(':sour:volt:lev 0')
        smu.write('outp off')
        smu.close()
        pau.close()
        print ("WARNING: Please make sure the output is turned off!")

        exit(1)

    signal.signal(signal.SIGINT, handler)

    # voltages start/end
    Varr = np.linspace(V0, V1, npts)

    if (V2 is not None):
        if (V2 > V1) and (V3 > V1):
            VarrL = Varr[Varr > V2]
            VarrH = Varr[Varr < V3]
            VarrM = np.linspace(V2, V3, npts1)
            Varr = np.concatenate([VarrL, VarrM, VarrH])

    print (Varr)

    if return_sweep:
        Varr_return = np.linspace(V1,V0,int(abs(V0-V1)/5 + 1))
        Varr = np.concatenate([Varr, Varr_return ])

    smu.write(':sour:volt:lev 0')
    smu.write('outp on')

    time.sleep(0.5) # 이 값을 바꿔서 충분한 시간을 sleep해줘요! Taiwoo Kim
    print ("\n")

    arr = []
    for V in Varr:
        smu.write(f':sour:volt:lev {V}')
        #time.sleep(0.1)
        time.sleep(0.5) ## 이 값을 바꿔서 충분한 시간을 sleep해줘요! Taiwoo Kim

        if Ntimes and Iteration > 0:
            Ismu_values = []
            Ipau_values = []

            for n in range(Iteration):
                Ismu = smu.query(":MEAS:CURR?")
                Ipau, _, _ = pau.query("READ?").split(',')

                Ismu = float(Ismu)
                Ipau = float(Ipau[:-1])

                Ipau_values.append(Ipau)
                Ismu_values.append(Ismu)
                time.sleep(0.1)  # Pauses for 0.1 seconds between iterations        이 값을 바꿔서 충분한 시간을 sleep해줘요! Taiwoo Kim

            Ismu_avg = np.mean(Ismu_values)
            Ipau_avg = np.mean(Ipau_values)
        
        else:
            print("Invalid input.")

        Vsmu = smu.query(":MEAS:VOLT?")
        Vsmu = float(Vsmu)

        print (V, Vsmu, Ismu_avg, Ipau_avg)
        arr.append([V, Vsmu, Ismu_avg, Ipau_avg])
        #print (V, Vsmu, Ipau_avg, Ismu_avg)
        #arr.append([V, Vsmu, Ipau_avg, Ismu_avg])

    smu.write(':sour:volt:lev 0')
    smu.write('outp off')
    smu.close()
    pau.close()

    opath = os.path.join(opathroot, Nmeas, f'{date}_{sensorname}')
    mkdir(opath)

    fname = f'IV_SMU+PAU_{sensorname}_{date}_{V0}_{V1}_pad{Npad}'
    ofname = os.path.join(opath, f'{fname}')
    k=0
    while (os.path.isfile(ofname)):
        ofname = f'{ofname}_{k}'
        k += 1

    arr = np.array(arr, dtype=float)
    np.savetxt(ofname+'.txt', arr)
    ivplot(arr)
    plt.savefig(ofname+'.png')

def ivplot(arr, yrange=None):
    arr = np.array(arr).T
    V = arr[0]
    I = arr[2] # 2인 경우, total current 측정 , 3인 경우 단일 pad current 측정
    I[I>1e37] = min(I)
    plt.plot(abs(arr[0]), abs(arr[2]))
    plt.yscale('log')
    if yrange:
        plt.ylim(yrange)	
if __name__=='__main__':
    iv_smu_pau()
    plt.show()
end_time = time.time()
print(end_time - start_time)

def mkdir(path):
    os.makedirs(path, exist_ok=True)

def getdate():
    return datetime.now().strftime("%Y%m%d")