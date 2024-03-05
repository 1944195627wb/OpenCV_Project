import time
import ser
import numpy as np
import get_command

def send_command(command,cap):
    counter = 1
    command_ = 'a1'
    if command[0] =='a':
        ser.SendCommand(command)
        time.sleep(0.1)
    elif command[0] in ['b','c']:
        if command in ['b3','c3']:
            ser.SendCommand(command)
            time.sleep(0.1)
        else:
            while True:
                if counter % 2 == 0:
                    ser.SendCommand(command_)
                    time.sleep(0.1)
                else:
                    ser.SendCommand(command)
                    time.sleep(0.1)
                command_ = get_command.get_command(cap)
                if command_[0] in ['a']:
                    break
                counter += 1







