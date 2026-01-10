from pynput.mouse import Controller
import time


f = open('mouse_data.txt', 'w')

# learning how mouse works
mouse = Controller()

while True:
    x, y = mouse.position
    thing = '{0} {1}\n'.format(x, y)
    f.write(thing)
    f.flush()
    print(thing)
    time.sleep(0.01)

f.close()
