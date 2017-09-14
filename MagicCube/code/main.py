import cube_JP
from cube_JP import Cube
import keras
from keras.models import load_model
import pycuber as pc
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import widgets
import numpy as np

from    random import randint
if __name__ == '__main__':

    possible_moves = ["R","R'","R2","U","U'","U2","F","F'","F2","D","D'","D2","B","B'","B2","L","L'","L2"]
    faces = ['L','U','R','D','F','B'] # for pycuber
    colors = ['[r]','[y]','[o]','[w]','[g]','[b]'] #for pycuber

    max_moves  = 6

    model = load_model('/home/jerpint/Dropbox/rubiks/rubiks_model_new.h5')

    try:
        N = int(sys.argv[1])
    except:
        N = 3

    c = Cube(N)


    c.draw_interactive()
    #c.rotate_face('U')

    plt.show()
