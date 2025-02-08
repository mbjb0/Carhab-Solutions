import socket
import time
import msvcrt

def send_message_nonblocking(client_socket, message, server_address):
    try:
        client_socket.sendto(message.encode('utf-8'), server_address)
    except socket.error as e:
        print(f"Send error: {e}")

def start_client(server_ip):
    # Set up the client's network interface
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    client_socket.setblocking(False)
    
    # Bind the client to a specific IP address and port
    client_ip = '192.168.1.1'  # Static IP for client
    client_socket.bind((client_ip, 0))  # 0 means any available port
    
    server_address = (server_ip, 5000)

    try:
        while True:
            if msvcrt.kbhit():
                key = msvcrt.getch().decode('utf-8').lower()
                
                if key in ['w', 'a', 's', 'd', 'q']:
                    send_message_nonblocking(client_socket, key, server_address)
                    print(f"Sent: {key}")
            
            try:
                data, _ = client_socket.recvfrom(1024)
                print(f"Got response: {data.decode('utf-8')}")
            except socket.error:
                pass
                
    except KeyboardInterrupt:
        print("\nStopping client...")
    finally:
        client_socket.close()

if __name__ == '__main__':
    server_ip = '192.168.1.2'  # Static IP for server
    start_client(server_ip)



'''import socket
import time
import msvcrt  # For Windows keyboard input
# Note: For Linux/Mac, you would need to use a different approach like 'keyboard' library

def send_message_nonblocking(client_socket, message, server_address):
    try:
        client_socket.sendto(message.encode('utf-8'), server_address)
    except socket.error as e:
        print(f"Send error: {e}")

def start_client(server_ip):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    client_socket.setblocking(False)
    server_address = (server_ip, 5000)

    try:
        while True:
            # Check if a key is pressed
            if msvcrt.kbhit():
                key = msvcrt.getch().decode('utf-8').lower()
                
                # Check for WASD keys
                if key in ['w', 'a', 's', 'd', 'q']:
                    send_message_nonblocking(client_socket, key, server_address)
                    print(f"Sent: {key}")
            
            # Try to receive response without blocking
            try:
                data, _ = client_socket.recvfrom(1024)
                print(f"Got response: {data.decode('utf-8')}")
            except socket.error:
                pass  # No data available, continue
                
    except KeyboardInterrupt:
        print("\nStopping client...")
    finally:
        client_socket.close()

if __name__ == '__main__':
    server_ip = '192.168.1.103'  # Replace with your server's IP address
    start_client(server_ip)
'''