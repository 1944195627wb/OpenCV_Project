import serial


def Connect():
    global ser
    ser= serial.Serial(port="/dev/ttyTHS0",baudrate=9600,timeout=0.5)
    if ser.isOpen():
        return True
    else:
        print("STM32 Serial Open Error!")
        return False


def DisConnect():
    ser.close()


def SendCommand(command):
    meg_tranform = messgae_tranform(command)
    ser.write(meg_tranform.encode('utf-8'))


def messgae_tranform(meg):
    if meg[1]=="0":
        meg_tranform="a"
    elif meg=="a1":
        meg_tranform = "b"
    elif meg=="a2":
        meg_tranform = "c"
    elif meg == "a3":
        meg_tranform = "d"
    elif meg == "b1":
        meg_tranform = "e"
    elif meg == "b2":
        meg_tranform = "f"
    elif meg == "b3":
        meg_tranform = "g"
    elif meg == "c1":
        meg_tranform = "h"
    elif meg == "c2":
        meg_tranform = "i"
    elif meg == "c3":
        meg_tranform = "j"
    return meg_tranform