import socket
import threading

bind_ip= '192.168.0.255'
bind_port = 5000

server=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(( bind_ip, bind_port))

server.listen(5)

print("[+] Listening on %s:%d "% (bind_ip, bind_port)  )

def handle_client(client_socket):
    request= client_socket.recv(1024)
    print("[+] Client: %s "%request )
    client_socket.send("[!] We have a problem'  " )
    client_socket.close()
while True:
    client, address=server.accept()
    print("[+] Accepting Connection From: %s:%s "% (address[0], address[1]  ))
    print("[+] Establishing a connection from: %s:%s "% (address[0],address[1]  )  )

    client_handler= threading.Thread(target=handle_client,args=( client ,))
    client_handler.start()
