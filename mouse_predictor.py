import mousenet
from pynput.mouse import Controller
import numpy as np
import time
import threading
import tkinter as tk

DOT_RADIUS = 10
UPDATE_INTERVAL = 50 # milliseconds
CANVAS_SIZE = 500  # width and height of the window
SCALE = 5 # multiply the distance predicted by the model

def inverse_sigmoid(y):
    return np.nan_to_num(np.log(y) - np.log(1-y))


# displays prediction
def run_display(net):

    root = tk.Tk()
    root.overrideredirect(True)          # no window frame
    root.attributes("-topmost", True)    # always on top
    root.attributes("-transparent", True)

    canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE,
                       bg="white", highlightthickness=0)
    canvas.pack()
    
    dot = canvas.create_oval(
        CANVAS_SIZE//2 - DOT_RADIUS,
        CANVAS_SIZE//2 - DOT_RADIUS,
        CANVAS_SIZE//2 + DOT_RADIUS,
        CANVAS_SIZE//2 + DOT_RADIUS,
        fill="red"
    )

    center_dot = canvas.create_oval(
        CANVAS_SIZE//2 - DOT_RADIUS,
        CANVAS_SIZE//2 - DOT_RADIUS,
        CANVAS_SIZE//2 + DOT_RADIUS,
        CANVAS_SIZE//2 + DOT_RADIUS,
        fill="black"
    )

    points = np.zeros([100, 2])
    i = 0
    mouse = Controller()

    def update():
        nonlocal i
        
        # log new mouse position
        x, y = mouse.position
        points[i] = [x, y]
        i += 1
        if i>99: i = 0

        # get prediction from recent mouse positions
        ordered_points = np.concatenate((points[i:], points[:i]), axis=0)

        a = net.feedforward(ordered_points)
        dx = inverse_sigmoid(a[0][0])
        dy = inverse_sigmoid(a[1][0])
        print('prediction: ({0}, {1})'.format(dx, dy))

        # display
        # move the dot inside the canvas relative to the center
        center = CANVAS_SIZE // 2
        canvas.coords(
            dot,
            center - DOT_RADIUS + (dx * SCALE),
            center - DOT_RADIUS + (dy * SCALE),
            center + DOT_RADIUS + (dx * SCALE),
            center + DOT_RADIUS + (dy * SCALE)
        )


        # pause
        root.after(UPDATE_INTERVAL, update)

    update()       # start the update loop
    root.mainloop() # Tkinter GUI


# moves the mouse for you
def run_move(net):
    points = np.zeros([100, 2])
    i = 0
    mouse = Controller()

    while True:
        
        # log new mouse position
        x, y = mouse.position
        points[i] = [x, y]
        i += 1
        if i>99: i = 0

        # get prediction from recent mouse positions
        ordered_points = np.concatenate((points[i:], points[:i]), axis=0)

        a = net.feedforward(ordered_points)
        dx = inverse_sigmoid(a[0][0])
        dy = inverse_sigmoid(a[1][0])
        print('prediction: ({0}, {1})'.format(dx, dy))

        # move the mouse based on prediction
        vx = 0
        vy = 0
        alpha = 0.2  # smoothing

        vx = (1 - alpha) * vx + alpha * dx
        vy = (1 - alpha) * vy + alpha * dy
        
        mouse.move(int(vx), int(vy))

        # eepy time
        time.sleep(UPDATE_INTERVAL)
        


'''
import mousenet
import mouse_predictor
net = mousenet.load('mousenet_test')
mouse_predictor.run(net)




import mousenet
import mouse_predictor
import mouse_data_loader
net = mousenet.Network([200,30,10])
training_data, validation_data = mouse_data.loader.load_mouse_data()
net.SGD(training_data, 50, 10, 0.1, evaluation_data=validation_data, monitor_evaluation_cost=True, monitor_training_cost=True)
'''



