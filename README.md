# LGAD Characteristics Measurement System

Automated measurement system for characterizing Low Gain Avalanche Detectors (LGADs) developed at Kyungpook National University (KNU).

## Overview

This project provides Python scripts for automated I-V (current-voltage) and C-V (capacitance-voltage) characterization of LGAD sensors using Keithley instruments. The measurement setup supports both on-wafer probing and packaged sensor testing.

![Measurement Setup](LGAD%20Measurement%20Setup.png)

## Measurement Configurations

The system supports two measurement modes:

### I-V Measurement Setup
- **Keithley 6487 Picoammeter**: Measures pad current (I_pad)
- **Keithley 6485 Picoammeter**: Measures guard ring + other current (I_gr+other)
- **Keithley 2470 SourceMeter**: Applies bias voltage and measures total current (I_total)
- **Configuration**: I_total = I_pad + I_gr + I_other

### C-V Measurement Setup
- **WK 4300 LCR Meter**: Measures capacitance at high and low force
- **Keithley 2470 SourceMeter**: Applies bias voltage
- **Configuration**: Simplified setup for capacitance measurements

## Features

- **Automated I-V measurements** with customizable voltage sweep parameters
- **Iterative measurements** with averaging for improved accuracy
- **Return sweep support** for hysteresis analysis
- **Safety features** including current compliance and automatic shutdown
- **Data export** to CSV format with automatic file naming
- **Real-time plotting** of I-V characteristics
- **On-wafer QC measurements** with adaptive voltage stepping
- **GPIB communication** via PyVISA

## Requirements

### Hardware
- Keithley 2470 or 2400 SourceMeter
- Keithley 6487 Picoammeter
- Keithley 6485 Picoammeter (for I-V measurements)
- WK 4300 LCR Meter (for C-V measurements)
- GPIB interface
- Probe station or test fixture

### Software
```bash
pip install pyvisa
pip install numpy pandas matplotlib
```

### GPIB Configuration
Ensure the GPIB addresses match your hardware setup:
- SMU (2470): `GPIB0::18::INSTR`
- PAU (6487): `GPIB0::22::INSTR`

## Usage

### Check GPIB Connection
```bash
python list_visa_devices.py
```

### Basic I-V Measurement
```bash
python measure_I-V.py
```
You will be prompted to enter:
- Number of iterations for averaging
- Whether to include return sweep (y/n)

### I-V Measurement with PAU Integration
```bash
python measure_I-V_PAU.py
```

### On-Wafer QC Testing
```bash
python QC_IV_on_wafer.py
```
This script includes:
- Adaptive voltage stepping
- Current compliance checking
- Automatic reverse sweep on overcurrent

### Custom Bias Iteration
```bash
python IV_SMU2470_PAU_Iteration_bias_custom.py
```

## Configuration

Edit the following parameters in each script:

```python
opathroot = r'C:\LGAD_test\I-V_test'  # Output directory
sensorname = 'UFSD-K1_W5_R(4_1)_T10_GR3_0_5x5'  # Sensor identifier
Nmeas = 'LFtest20250226'  # Measurement campaign name
Npad = '1'  # Pad number

V0 = 0        # Start voltage
V1 = -250     # End voltage
step = -1     # Voltage step size
```

## Output Format

Data is saved in CSV format with the following columns:
- `V_set(V)`: Set voltage
- `V_meas(V)`: Measured voltage
- `I_smu(A)`: Current from SourceMeter
- `I_pau(A)`: Current from Picoammeter

Each measurement generates:
- CSV file with raw data
- PNG plot of I-V curve

## Safety Features

- Current limit protection (default: 100 µA for single pads, 300 µA for full sensors)
- Graceful interrupt handling (Ctrl+C)
- Automatic output shutdown on errors
- Voltage ramping to prevent device damage

## Project Structure

```
.
├── measure_I-V.py                                    # Standard I-V measurement
├── measure_I-V_PAU.py                               # I-V with PAU integration
├── IV_SMU2470_PAU_Iteration_bias_custom.py          # Custom iteration script
├── QC_IV_on_wafer.py                                # On-wafer QC measurements
├── GPIB_check.py                                    # GPIB connection test
├── list_visa_devices.py                             # List available VISA devices
├── I-V_output_ex.json                               # Example output format
└── LGAD Measurement Setup.png                       # System diagram
```

## Authors

Young Park - Kyungpook National University (KNU)

## License

This project is intended for research use in LGAD characterization.
