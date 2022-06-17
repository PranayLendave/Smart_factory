import socket

from time import sleep
# host='192.168.0.106'

host = "192.168.43.151" #ESP32 IP in local network
port = 5050             #ESP32 Server Port
print("[Starting]")
def socket_calling(message):
    try:
        sock = socket.socket()
        sock.connect((host, port))
        message_1=message.encode("utf-8")
        sock.send(bytes(message_1))
        print("MSG sent")
        data = ""
        while len(data) < len(message):
            data += sock.recv(1).decode()
        print(data)
        sock.close()
    except TimeoutError:
        print("[!Connection lost] socket client disconnected")
        pass

while True:
    socket_calling("a")
    sleep(3)
    socket_calling("b")
    sleep(3)
    socket_calling("c")
    sleep(3)
    socket_calling("d")
    sleep(3)
    socket_calling("e")
    sleep(3)
    socket_calling("d")
    sleep(3)
