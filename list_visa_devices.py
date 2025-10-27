import pyvisa

# Create a resource manager instance
rm = pyvisa.ResourceManager()

# List available VISA resources (connected devices)
resources = rm.list_resources()

if not resources:
    print("No devices found.")
else:
    print(f"Available resources: {resources}")

    # Flag to check if any device responded
    any_device_responded = False

    print("Connected devices:")
    for resource in resources:
        try:
            # Open the resource to get more details
            with rm.open_resource(resource) as instrument:
                idn = instrument.query('*IDN?')  # Query the device
                print(f"{resource}: {idn}")
                any_device_responded = True
        except pyvisa.errors.VisaIOError as e:
            print(f"{resource}: Could not query device details. VISA IO Error: {e}")

    if not any_device_responded:
        print("No responding devices found.")
