from pynput.mouse import Controller
import time

COLLECTION_DURATION = 60 # seconds
COLLECTION_INTERVAL = 0.01 # seconds

f = open('mouse_data.txt', 'w')

mouse = Controller()

start = time.time()

# collect positions
while time.time() - start < COLLECTION_DURATION:
    x, y = mouse.position
    thing = '{0} {1}\n'.format(x, y)
    f.write(thing)
    f.flush()
    print(thing)
    time.sleep(COLLECTION_INTERVAL)

f.close()

# load data and train network
import mousenet
import mouse_predictor
import mouse_data_loader
net = mousenet.Network([200,30,2])
training_data, validation_data = mouse_data_loader.load_mouse_data()
net.SGD(training_data, 50, 10, 0.1, evaluation_data=validation_data, monitor_evaluation_cost=True, monitor_training_cost=True)

# display predictions
import mousenet
import mouse_predictor
net = mousenet.load('mousenet_test')
mouse_predictor.run_display(net)
