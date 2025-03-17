import socket

def start_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_socket.bind(('0.0.0.0', 5000))
    
    # Make server non-blocking too
    server_socket.setblocking(False)
    
    print("Server is listening...")
    
    while True:
        try:
            data, client_address = server_socket.recvfrom(1024)
            message = data.decode('utf-8')
            print(f"Received: {message}")
            
            # Optional: Send response without blocking
            try:
                server_socket.sendto("OK".encode('utf-8'), client_address)
            except socket.error:
                pass
                
        except socket.error:
            continue  # No data available, continue loop
            
        except KeyboardInterrupt:
            print("\nStopping server...")
            break

if __name__ == '__main__':
    start_server()