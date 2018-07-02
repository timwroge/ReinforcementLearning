import socket

TCP_IP= '172.17.25.46'
TCP_PORT = 5000
BUFFER_SIZE = 1024
MESSAGE = "Ava is dumb for not going to hong kong"

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    s.connect((TCP_IP, TCP_PORT))
    s.send(MESSAGE)
    data = s.recv(BUFFER_SIZE)
    s.close()

    print "received data:", data
except Exception as e: print(e)
