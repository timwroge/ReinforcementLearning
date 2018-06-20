import serial
import time

arduinoUno=serial.Serial('/dev/ttyACM0', 9600)
print arduinoUno.readline()

while True:
    print("Enter 1 or 0 to turn the LED on or off respectively" )
    vari=raw_input()
    if vari =="1" :
        arduinoUno.write('1' )

    if vari == "0" :
        arduinoUno.write('0' )

