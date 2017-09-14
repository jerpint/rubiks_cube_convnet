# rubiks_cube_convnet

How to train a simple convnet to solve a rubiks cube.  

Plenty of efficient algorithms exist to solve a rubik's cube. This is to see if a neural net could learn how to solve a cube in the most "efficient" way, by solving the cube in less than 20 moves, i.e [god's number](http://www.cube20.org/).

This is a very naive solution, to start as a proof of concept. I used a 2 layer neural net: 1 convnet layer and 1 feedforward layer.  The input is the state of the cube to be solved. The output is the next predicted move until solved. For the training set, I generated games at random during training of 10 moves or less from solved with the corresponding solutions as label. At each step of the solution, I made the network make a guess for its next move and used SGD for training.  I trained this over many epochs until the loss was relatively steady.

Surprisingly, the network works decently well for any position less than 6 moves away from solved. I found that by shuffling up to 6 moves, sometimes more, it is able to correctly solve it. It doesn't always work and can definitely use improvement, but its a good place to start.

I used Keras with tensorflow for training. You will need to install Keras to be able to run the network to make predictions. I provide the weights so you don't have to train from scratch.

I used the pycuber library for writing the code to train, which also needs to be installed. I used the [MagicCube](https://github.com/davidwhogg/MagicCube) library for the cube simulation/visualization. I only found MagicCube after using pycuber, so you need to have both installed for now :

```
pip install pycuber 
```

For MagicCuber, I provide the .zip of their github repo I used at the time. I suggest using this one to avoid conflicts. I provide the training weights for my ConvNet. 

To run and play with the cube/shuffle it yourself, clone this repository, then in the commandline

```
python rubiks_cube_convnet/MagicCube/code/cube_convnet_solver.py
```
You can shuffle using the keyboard and have it solve your own cube. There is a hard-coded reset if you've gone too far and the network can't solve it. This simply retraces back the steps to the initial solved position.

For training of the cube :
 
```
python rubiks_cube_convnet/train_cube.py
```
There is plenty of exploring to do! Bigger networks might be one solution, fancier networks would likely be more appropriate. I thought of reinforcement learning, but decided to use the simpler supervised-learning approach to begin.
