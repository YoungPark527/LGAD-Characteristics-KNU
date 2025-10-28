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

# GPIB/USB addresses
PAU_ADDRESS = "GPIB1::22::INSTR"  # 6487 PAU

def getdate():
    return datetime.now().strftime("%Y%m%d")

def mkdir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

class CVMeasurement:
    def __init__(self):
        # Default settings
        self.output_dir = Path.home() / "LGAD_test" / "C-V_test"
        self.voltage_start = 0
        self.voltage_end = -40
        self.voltage_steps = 81
        self.frequency = 1000  # Hz
        self.iteration_count = 1
        self.sensor_name = 'SENSOR'
        self.pad_number = '1_1'
        self.return_sweep = True

    def connect_instruments(self):
        """Connect to instruments - PAU via GPIB and LCR via USB"""
        # Initialize resource managers for both GPIB and USB
        rm_gpib = pyvisa.ResourceManager()  # Default backend for GPIB
        rm_usb = pyvisa.ResourceManager('@py')  # py backend for USB
        
        # List available devices
        gpib_resources = rm_gpib.list_resources()
        usb_resources = rm_usb.list_resources()
        
        print("\nAvailable GPIB devices (for PAU):")
        for idx, resource in enumerate(gpib_resources):
            print(f"{idx + 1}: {resource}")
            
        print("\nAvailable USB devices (for LCR):")
        for idx, resource in enumerate(usb_resources):
            print(f"{idx + 1}: {resource}")

        try:
            print(f"\nConnecting to PAU at {PAU_ADDRESS}")
            self.pau = rm_gpib.open_resource(PAU_ADDRESS)
            
            # Let user select LCR meter from USB devices
            lcr_idx = int(input("\nSelect LCR meter number from USB devices: ")) - 1
            if lcr_idx < 0 or lcr_idx >= len(usb_resources):
                raise ValueError("Invalid LCR meter selection")
                
            self.lcr = rm_usb.open_resource(usb_resources[lcr_idx])

            # Configure termination characters for LCR
            self.lcr.read_termination = '\n'
            self.lcr.write_termination = '\n'

            # Test connections
            print("\nConnected devices:")
            print(f"PAU (GPIB): {self.pau.query('*IDN?')}")
            print(f"LCR (USB): {self.lcr.query('*IDN?')}")
            return True

        except Exception as e:
            print(f"\nError connecting to instruments: {str(e)}")
            print("Make sure PAU is connected via GPIB and LCR meter is connected via USB")
            return False

    def configure_instruments(self):
        """Configure PAU and LCR meter with default settings"""
        try:
            # Configure PAU
            self.pau.write("*RST")
            self.pau.write("curr:range 2e-4")
            self.pau.write("INIT")
            self.pau.write("syst:zch off")
            self.pau.write("SOUR:VOLT:STAT off")
            self.pau.write("SOUR:VOLT:RANG 500")
            self.pau.write("SOUR:VOLT:ILIM 1e-4")
            self.pau.write("FORM:ELEM READ,UNIT,STAT,VSO")

            # Configure LCR meter
            self.lcr.write(":MEAS:NUM-OF-TESTS 1")
            self.lcr.write(":MEAS:FUNC1 C")
            self.lcr.write(":MEAS:FUNC2 Z")
            self.lcr.write(":meas:equ-cct par")
            self.lcr.write(":MEAS:SPEED fast")
            self.lcr.write(":MEAS:LEV 0.1")
            self.lcr.write(":MEAS:V-BIAS 0V")
            self.lcr.write(f":MEAS:FREQ {self.frequency}")
            return True

        except Exception as e:
            print(f"Error configuring instruments: {str(e)}")
            return False

    def perform_measurement(self):
        """Perform the CV measurement"""
        # Prepare voltage array
        voltages = np.linspace(self.voltage_start, self.voltage_end, self.voltage_steps)
        if self.return_sweep:
            voltages_return = np.linspace(self.voltage_end, self.voltage_start, int(abs(self.voltage_start - self.voltage_end) / 5 + 1))
            voltages = np.concatenate([voltages, voltages_return])
        
        print(f"Voltage sweep ({len(voltages)} pts):")
        
        # Initialize arrays for measurements
        measured_data = {
            'voltage': [],
            'capacitance': [],
            'resistance': [],
            'current': []
        }

        def safe_off_and_cleanup():
            try:
                self.pau.write(":sour:volt 0")
                self.pau.write(":sour:volt:stat off")
                self.lcr.write(":MEAS:BIAS OFF")
                self.lcr.write(":MEAS:V-BIAS 0V")
            except:
                pass

        def handler(signum, frame):
            print("\nUser interrupt. Turning off instruments safely...")
            safe_off_and_cleanup()
            sys.exit(1)

        signal.signal(signal.SIGINT, handler)

        # Start measurement
        self.pau.write(":sour:volt 0")
        self.pau.write(":sour:volt:stat on")
        self.lcr.write(":MEAS:V-BIAS 0V")
        self.lcr.write("meas:bias OFF")
        time.sleep(1)

        try:
            for voltage in voltages:
                # Set voltage
                if voltage > 0:
                    print("Warning: positive bias not allowed. Setting to 0V.")
                    voltage = 0

                self.pau.write(f':sour:volt {voltage}')
                time.sleep(0.1)
                
                # Measure current
                Ipau, stat_pau, Vpau = self.pau.query(":READ?").split(',')
                Vpau = float(Vpau)
                Ipau = float(Ipau[:-1])

                # Measure capacitance and resistance (with iterations)
                cap_values = []
                res_values = []
                
                for _ in range(self.iteration_count):
                    try:
                        res = self.lcr.query('meas:trig?')
                        C0, R0 = map(float, res.split(','))
                        cap_values.append(C0)
                        res_values.append(R0)
                        time.sleep(0.05)
                    except Exception as e:
                        print(f"Error in iteration: {str(e)}")
                        continue

                # Calculate averages
                C_avg = np.mean(cap_values) if cap_values else float('nan')
                R_avg = np.mean(res_values) if res_values else float('nan')

                # Store measurements
                measured_data['voltage'].append(voltage)
                measured_data['capacitance'].append(C_avg)
                measured_data['resistance'].append(R_avg)
                measured_data['current'].append(Ipau)

                # Print progress
                print(f"V={voltage:.1f}V, C={C_avg:.2e}F, R={R_avg:.2e}Ω, I={Ipau:.2e}A")

            return measured_data

        except Exception as e:
            print(f"Error during measurement: {str(e)}")
            return None
        finally:
            safe_off_and_cleanup()

    def save_data(self, data):
        """Save measurement data to CSV file"""
        if data is None:
            return None

        # Create output directory
        date = getdate()
        outpath = self.output_dir / self.sensor_name / f"{date}_{self.sensor_name}"
        mkdir(outpath)

        # Create filename base
        base = outpath / f"CV_{self.sensor_name}_{date}_{self.frequency}Hz_pad{self.pad_number}"
        outfile = str(base)
        
        # Add index if file exists
        k = 0
        while (Path(outfile + ".csv")).exists():
            outfile = f"{base}_{k}"
            k += 1

        # Save to CSV
        try:
            header = "Voltage(V),Capacitance(F),Resistance(Ohm),Current(A)"
            np.savetxt(outfile + ".csv", 
                      np.column_stack((data['voltage'], 
                                     data['capacitance'],
                                     data['resistance'],
                                     data['current'])),
                      delimiter=',',
                      header=header,
                      comments='')  # No # in header
            print(f"\nData saved to: {outfile}.csv")
            return outfile + ".csv"
        except Exception as e:
            print(f"Error saving data: {str(e)}")
            return None

    def plot_results(self, data, save_path=None):
        """Plot measurement results"""
        if data is None:
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        # Plot C-V and 1/C² curves
        voltage = np.array(data['voltage'])
        capacitance = np.array(data['capacitance'])
        
        ax1.plot(abs(voltage), capacitance * 1e12, 'b.-', label='C')
        ax1.set_xlabel('|Bias Voltage| (V)')
        ax1.set_ylabel('Capacitance (pF)')
        ax1.grid(True)
        
        # Add 1/C² curve
        ax1_twin = ax1.twinx()
        ax1_twin.plot(abs(voltage), 1/capacitance**2, 'r.-', label='1/C²')
        ax1_twin.set_ylabel('1/C² (F⁻²)')
        
        # Plot R and I curves
        ax2.plot(abs(voltage), data['resistance'], 'g.-', label='R')
        ax2.set_xlabel('|Bias Voltage| (V)')
        ax2.set_ylabel('Resistance (Ω)')
        ax2_twin = ax2.twinx()
        ax2_twin.plot(abs(voltage), np.abs(data['current']), 'm.-', label='|I|')
        ax2_twin.set_ylabel('Current (A)')
        ax2.grid(True)

        # Add legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_twin.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        lines3, labels3 = ax2.get_legend_handles_labels()
        lines4, labels4 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines3 + lines4, labels3 + labels4, loc='upper left')

        plt.tight_layout()
        
        if save_path:
            plot_path = save_path.rsplit('.', 1)[0] + '.png'
            plt.savefig(plot_path)
            print(f"Plot saved to: {plot_path}")
        
        plt.show()

    def cleanup(self):
        """Safely close instrument connections"""
        try:
            self.pau.close()
            self.lcr.close()
        except:
            pass

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run CV measurement with PAU and LCR meter",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Define optional arguments
    parser.add_argument("--iteration", "-n", type=int, help="Number of measurements per voltage")
    parser.add_argument("--V0", type=float, help="Start voltage")
    parser.add_argument("--V1", type=float, help="End voltage")
    parser.add_argument("--npts", type=int, help="Number of points in voltage sweep")
    parser.add_argument("--freq", type=float, help="Measurement frequency (Hz)")
    parser.add_argument("--no-return", dest="return_sweep", action="store_false", help="Disable return sweep")
    parser.add_argument("--outdir", type=Path, help="Output root directory")
    parser.add_argument("--sensorname", type=str, help="Sensor name for filenames")
    parser.add_argument("--nmeas", type=str, help="Measurement name folder")
    parser.add_argument("--npad", type=str, help="Pad number")

    args = parser.parse_args()

    # If no arguments provided → prompt interactively
    if len(sys.argv) == 1:
        print("\n🔧 No command-line arguments detected. Enter parameters manually:\n")

        def ask(prompt, default=None, type_=str):
            val = input(f"{prompt} [{default}]: ") if default is not None else input(f"{prompt}: ")
            return type_(val) if val.strip() != "" else default

        args.iteration = ask("Number of measurements per voltage", 1, int)
        args.V0 = ask("Start voltage (V0)", 0.0, float)
        args.V1 = ask("End voltage (V1)", -40.0, float)
        args.npts = ask("Number of points in sweep", 81, int)
        args.freq = ask("Measurement frequency (Hz)", 1000.0, float)
        args.return_sweep = ask("Include return sweep? (y/n)", "y", str).lower().startswith("y")
        args.outdir = Path(ask("Output directory", str(Path.home() / "LGAD_test" / "C-V_test")))
        args.sensorname = ask("Sensor name", "SENSOR")
        args.nmeas = ask("Measurement name folder", "LGADtest")
        args.npad = ask("Pad number", "1_1")

    return args

def main():
    # Parse arguments
    args = parse_args()
    start_time = time.time()

    # Create measurement instance
    cv = CVMeasurement()
    
    # Configure settings from arguments
    if args.iteration is not None:
        cv.iteration_count = args.iteration
    if args.V0 is not None:
        cv.voltage_start = args.V0
    if args.V1 is not None:
        cv.voltage_end = args.V1
    if args.npts is not None:
        cv.voltage_steps = args.npts
    if args.freq is not None:
        cv.frequency = args.freq
    if args.outdir is not None:
        cv.output_dir = args.outdir
    if args.sensorname is not None:
        cv.sensor_name = args.sensorname
    if args.npad is not None:
        cv.pad_number = args.npad
    cv.return_sweep = args.return_sweep if args.return_sweep is not None else True
    
    print("\n=== CV Measurement System ===")
    print(f"Using Python {sys.version.split()[0]}")
    
    # Connect and configure instruments
    if not cv.connect_instruments():
        print("Failed to connect to instruments. Exiting...")
        return
    
    if not cv.configure_instruments():
        print("Failed to configure instruments. Exiting...")
        cv.cleanup()
        return
    
    # Perform measurement
    print("\nStarting measurement...")
    data = cv.perform_measurement()
    
    if data:
        csv_path = cv.save_data(data)
        if csv_path:
            cv.plot_results(data, csv_path)
    
    cv.cleanup()
    
    end_time = time.time()
    print(f"\nMeasurement completed. Elapsed time: {end_time - start_time:.2f} s")

if __name__ == '__main__':
    main()