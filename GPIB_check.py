import pyvisa as visa

try:
    # Create a VISA resource manager
    rm = visa.ResourceManager()
    
    # List all VISA resources
    resources = rm.list_resources()

    if resources:
        print("Detected resources:")
        for resource in resources:
            print(resource)
    else:
        print("No VISA resources detected.")
    
    # Check if there is a GPIB device by looking for 'GPIB' in the resource names
    gpib_resources = [r for r in resources if "GPIB" in r]
    if gpib_resources:
        print("\nDetected GPIB resources:")
        for gpib in gpib_resources:
            print(gpib)
    else:
        print("\nNo GPIB resources detected.")

except Exception as e:
    print(f"Error: {e}")
