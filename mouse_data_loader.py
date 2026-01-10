import numpy as np

# load mouse data
# must have >100 points
def load_mouse_data():
    
    points = np.loadtxt('mouse_data.txt') # numpy array of shape (m, 2)
    m = points.shape[0]

    differences = np.empty(points.shape)
    
    for i in range(1, m):
        differences[i][0] = sigmoid(points[i][0] - points[i-1][0])
        differences[i][1] = sigmoid(points[i][1] - points[i-1][1])
        
    
    data = [(differences[i-100:i, :].copy().reshape(-1,1), differences[i:i+1, :].reshape(-1,1)) for i in range(100, m - 1)]

    '''
    data is a list containing 2-tuples ``(p, q)``.
    
    ``p`` is a 200-dimensional numpy.ndarray containing differences in
    x and y-coordinate across previous positions.
    
    ``q`` is a 2-dimensional numpy.ndarray representing the expected change in position
    '''

    n = int(0.8*len(data))
                  
    training_data = data[:n]
    test_data = data[n:]
    return training_data, test_data

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))
