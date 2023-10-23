import serial
import pandas as pd
import threading
import keyboard
from datetime import datetime
import plotly.express as px


def read_serial(data_list):
    with serial.Serial('COM5', 9600) as ser:
        while not stop_event.is_set():
            try:
                data = float(ser.readline().decode().strip())
                timestamp = datetime.now().strftime("%H:%M:%S")
                data_list.append({'Timestamp': timestamp, 'Data': data})
                print(timestamp, data)
            except ValueError:
                print("Invalid data received")


def listen_for_input():
    print("Press RETURN to stop logging and save to CSV.")
    keyboard.wait('enter')  # Wait for the 'enter' key to be pressed
    stop_event.set()  # Signal to the serial reading thread to stop
    print("RETURN key pressed. Saving to CSV...")


if __name__ == '__main__':
    stop_event = threading.Event()  # A flag used to signal the thread to stop
    data_list = []  # A list to store the data dicts before converting them to a DataFrame

    try:
        # Start serial reading thread
        serial_thread = threading.Thread(target=read_serial, args=(data_list,))
        serial_thread.start()

        # Start keyboard listening thread
        input_thread = threading.Thread(target=listen_for_input)
        input_thread.start()

        # Wait for the input thread to finish (i.e., for the RETURN key to be pressed)
        input_thread.join()

        # Wait for the serial thread to finish
        serial_thread.join()

        # Convert the list of data dicts to a DataFrame and save to CSV
        df = pd.DataFrame(data_list)
        df.to_csv('log.csv', index=False)
        print("Data saved to log.csv")


    except serial.SerialException:
        print("COM5 is not available")
