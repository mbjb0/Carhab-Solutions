import socket

def send_message(message, server_ip):
    # Create a TCP/IP socket
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    try:
        # Connect to the server
        client_socket.connect((server_ip, 5000))
        
        # Send the message
        client_socket.sendall(message.encode('utf-8'))
        
        # Receive the response
        response = client_socket.recv(1024).decode('utf-8')
        print(f"Server response: {response}")
        
    finally:
        client_socket.close()

if __name__ == '__main__':
    server_ip = '168.150.14.211' # Replace with your server's IP address
    message = "1, 2, 3, 4"
    send_message(message, server_ip)