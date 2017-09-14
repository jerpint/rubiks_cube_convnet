import pycuber as pc
from random import randint
import numpy as np
import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import load_model

np.random.seed(1337)
max_moves =  10


mycube = pc.Cube()
faces = ['L','U','R','D','F','B']
colors = ['[r]','[y]','[o]','[w]','[g]','[b]']
possible_moves = ["R","R'","R2","U","U'","U2","F","F'","F2","D","D'","D2","B","B'","B2","L","L'","L2"]




def sol2cat(solution):
    # transform solution to one hot vector encoding
    # first map move to number, then genereate one hot encoding
    # using keras utils
    
    global possible_moves
    sol_tmp = []
    for j in range(len(solution)):
        sol_tmp.append(possible_moves.index(solution[j]))
        
    sol_cat = to_categorical(sol_tmp)
    
    return sol_cat
    
    

def cube2np(mycube):
    # transform cube object to np array
    # works around the weird data type used
    global faces
    global colors
    cube_np = np.zeros((6,3,3))
    for i,face in enumerate(faces):
        face_tmp = mycube.get_face(face)
        for j in range(3):
            for k in range(len(face_tmp[j])):
                caca = face_tmp[j][k]
                cube_np[i,j,k] = colors.index(str(caca))
    return cube_np

def generate_game(max_moves = max_moves):
    
    # generate a single game with max number of permutations number_moves
    
    mycube = pc.Cube()

    global possible_moves
    formula = []
    cube_original = cube2np(mycube)
    number_moves = max_moves#randint(3,max_moves)
    for j in range(number_moves):
        formula.append(possible_moves[randint(0,len(possible_moves)-1)])
        
    #my_formula = pc.Formula("R U R' U' D' R' F R2 U' D D R' U' R U R' D' F'")

    my_formula = pc.Formula(formula)


    mycube = mycube((my_formula))
    # use this instead if you want it in OG data type

    cube_scrambled = mycube.copy()
    
    solution = my_formula.reverse()

    #print(mycube)


    return cube_scrambled,solution

def generate_N_games(N=10,max_moves=max_moves):
    
    
    scrambled_cubes = []
    solutions = []
    for j in range(N):
        cube_scrambled,solution = generate_game(max_moves = max_moves)
        scrambled_cubes.append(cube_scrambled)
        solutions.append(solution)
        
    return scrambled_cubes,solutions

def generate_action_space(number_games=100):
    D = [] # action space
    states_hist = []
    game_count = 0
    play_game = True
    global max_moves
    while play_game:


        scrambled_cube,solutions = generate_game(max_moves = max_moves)




        state = scrambled_cube.copy()
        for j in range(len(solutions)):
            states_hist.append(state.copy())
            action = solutions[j]
            current_state = state.copy()
            state_next = state(action)

            state_next = state_next.copy()

            reward  = j+1

            D.append([current_state,action,reward,state_next])

            state = state_next.copy()

        states_hist.append(state.copy())


        game_count+=1

        if game_count>=number_games:
            break
            
    return D

def generate_data(N=32):

    while True:
        x  = []
        y = []

        D = generate_action_space(N)
        for d in D:
            x.append(cube2np(d[0]))
            y.append(to_categorical(possible_moves.index((str(d[1]))),len(possible_moves)))

        x = np.asarray(x)
        x = x.reshape(x.shape[0],18, 3, 1)
        x = x.astype('float32')

        y = np.asarray(y)
        y = y.reshape(y.shape[0],y.shape[2] )

        
        yield (x,y)




batch_size = 256
num_classes = len(possible_moves)
num_epochs = 150
input_shape = (18, 3, 1)

model = Sequential()
model.add(Conv2D(256, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
# model.add(Conv2D(128, kernel_size=(3, 3),
#                  activation='relu',
#                  input_shape=input_shape))
#model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.summary()

tbCallBack = keras.callbacks.TensorBoard(log_dir='/home/jerpint/Dropbox/rubiks/', histogram_freq=0, write_graph=True, write_images=True)


model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

for j in range(num_epochs): 
    
    if (j%10 == 0):
        print ('epoch #',j)
    model.fit_generator(generator= generate_data(64),steps_per_epoch=50,
                                  epochs=1,verbose=2,validation_data=None,max_queue_size=1,use_multiprocessing=True,workers=6,initial_epoch =0)#generate_data(8)
model.save('rubiks_model_wtvr.h5')  # creates a HDF5 file 'my_model.h5'




