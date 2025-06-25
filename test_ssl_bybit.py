import socket
import ssl
import sys
import time

# Bybit API host and port
HOST = 'api.bybit.com'
PORT = 443

print(f"Attempting direct SSL connection to {HOST}:{PORT}")

try:
    # Create a raw socket
    sock = socket.create_connection((HOST, PORT), timeout=10) # Added timeout here too

    # Wrap the socket with SSL/TLS
    # Use default SSL context for best compatibility
    context = ssl.create_default_context()
    # Optional: If the default context still fails, you might try to relax verification
    # context.check_hostname = False
    # context.verify_mode = ssl.CERT_NONE
    # WARNING: Disabling certificate verification is INSECURE for production.
    # It's for debugging only if nothing else works.

    ssl_sock = context.wrap_socket(sock, server_hostname=HOST)

    print("SSL handshake initiated...")
    ssl_sock.do_handshake() # Perform the handshake
    print("SSL handshake successful!")

    # If handshake is successful, try sending a simple HTTP GET request (for /v5/market/time)
    # This is a public endpoint that doesn't require API keys
    request_line = b"GET /v5/market/time HTTP/1.1\r\n"
    host_header = f"Host: {HOST}\r\n".encode('utf-8')
    connection_header = b"Connection: close\r\n\r\n"
    
    full_request = request_line + host_header + connection_header
    
    print(f"Sending request:\n{full_request.decode('utf-8')}")
    ssl_sock.sendall(full_request)

    # Receive response
    response_data = b""
    while True:
        chunk = ssl_sock.recv(4096)
        if not chunk:
            break
        response_data += chunk
    
    print("\nReceived response:")
    print(response_data.decode('utf-8', errors='ignore')) # Decode, ignoring errors for raw HTTP data

except socket.timeout:
    print(f"Error: Socket connection timed out when connecting to {HOST}:{PORT}")
    print("This might indicate network issues or a very slow server response.")
except ConnectionRefusedError:
    print(f"Error: Connection refused by {HOST}:{PORT}")
    print("This usually means the server is not running or a firewall is blocking the connection.")
except ConnectionResetError as e:
    print(f"Error: Connection reset by remote host: {e}")
    print("This means Bybit's server actively closed the connection during the SSL handshake.")
    print("Possible causes: IP restrictions (even if you think none), outdated SSL certificates on your system, or an aggressive network proxy/antivirus.")
except ssl.SSLError as e:
    print(f"SSL Error during handshake: {e}")
    print("This indicates a problem with the SSL certificate validation or protocol negotiation.")
    print("Possible causes: Outdated root certificates, system clock issues, or a proxy/antivirus interfering with SSL.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    import traceback
    traceback.print_exc()
finally:
    if 'ssl_sock' in locals() and ssl_sock:
        ssl_sock.close()
    elif 'sock' in locals() and sock:
        sock.close()
    print("\nConnection test complete.")
