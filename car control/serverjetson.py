import socket
from controls import Controls
import time
import signal
import sys


rover = Controls()

def signal_handler(sig, frame):
    print('Exiting...')
    if 'rover' in globals():
        rover.cleanup()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def start_server():
    # Create a TCP/IP socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Set the SO_REUSEADDR option
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    # Bind the socket to all available interfaces on port 5000
    server_socket.bind(('0.0.0.0', ))
    
    # Listen for incoming connections
    server_socket.listen(1)
    print("Server is listening...")
    
    while True:
        # Wait for a connection
        client_socket, client_address = server_socket.accept()
        try:
            time.sleep(1)
            print(f"Connected to {client_address}")
            
            # Receive the data
            data = client_socket.recv(1024).decode('utf-8')
            print(f"Received: {data}")
            
            # Send acknowledgment
            client_socket.sendall("Message received".encode('utf-8'))
            if data == 'center left':
                rover.turn_left(30)
                time.sleep(1)
            elif data == 'center right':
                rover.turn_right(30)
                time.sleep(1)
        

            
        finally:
            client_socket.close()

if __name__ == '__main__':
    start_server()
    