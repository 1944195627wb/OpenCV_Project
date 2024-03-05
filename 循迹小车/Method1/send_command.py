import time
import ser
import get_command


def send_command(command,cap):
    command1 = 'a1'
    command2 = 'a2'
    counter = 1
    if command[0] =='a':
        ser.SendCommand(command1)
        time.sleep(0.1)
    elif command[0] in ['b','c']:
        if command in ['b3','c3']:
            #for i in range(2):
                #ser.SendCommand(command2)
                #time.sleep(0.1)
            for i in range(2):
                ser.SendCommand(command1)
                time.sleep(0.1)
            ser.SendCommand(command)
            time.sleep(0.1)
        else:
            while True:
                if counter % 2 == 0:
                    ser.SendCommand(command1)
                    time.sleep(0.1)
                else:
                    ser.SendCommand(command)
                    time.sleep(0.1)
                command_ = get_command.get_command(cap)
                if command_[0] in ['a']:
                    break
                counter += 1