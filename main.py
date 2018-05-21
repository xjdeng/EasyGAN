import numpy as np

def random_image(height, width):
    return np.random.randint(0,255,(height, width,3))