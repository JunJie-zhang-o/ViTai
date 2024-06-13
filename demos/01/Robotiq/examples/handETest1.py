import time, os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from Robotiq import HandEForRtu

# you need to change serial port for your device
port = "/dev/tty.usbserial-2120"
hande = HandEForRtu(port=port)
time.sleep(1)
while True:
    hande.move(0,0,0,True)
    hande.move(50,1,3,False)
    # time.sleep(1)
    hande.stop()
    time.sleep(1)
    hande.move(50,1,3,True)