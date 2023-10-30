import serial
import csv

# Set the serial port and baud rate to match your Arduino
ser = serial.Serial('COM5', 9600)  # Replace 'COM4' with your Arduino's serial port

# Open a CSV file for writing
with open('arduino_data_2.csv', 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)

    # Write the header row if needed
    csv_writer.writerow(['Temperature', 'Humidity'])

    try:
        while True:
            # Read a line of data from the Arduino
            data = ser.readline().decode().strip()
            if data:
                temperature, humidity = data.split(',')  # Assuming data is formatted as "value,timestamp"
                csv_writer.writerow([temperature, humidity])
    except KeyboardInterrupt:
        # Close the serial port
        ser.close()

