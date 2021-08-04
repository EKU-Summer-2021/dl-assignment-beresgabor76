"""
Module for function minimization with automatic differentiation and SGD
"""
import os
import tensorflow as tf
import matplotlib.pyplot as plt


def function_minimization():
    """
    Function minimization with automatic differentiation and SGD
    """
    # Initialize a random value for our initial x
    x = tf.Variable([tf.random.normal([1])])
    print("Initializing x={}".format(x.numpy()))

    learning_rate = 1e-2 # learning rate for SGD
    history = []
    # Define the target value
    x_f = 4

    # We will run SGD for a number of iterations. At each iteration, we compute the loss,
    # compute the derivative of the loss with respect to x, and perform the SGD update.
    for _ in range(500):
        with tf.GradientTape() as tape:
            # define the loss as described above
            loss = tf.pow((x - x_f), 2)
        # loss minimization using gradient tape
        grad = tape.gradient(loss, x) # compute the derivative of the loss with respect to x
        new_x = x - learning_rate * grad # sgd update
        x.assign(new_x) # update the value of x
        history.append(x.numpy()[0])

    # Plot the evolution of x as we optimize towards x_f!
    plt.plot(history)
    plt.plot([0, 500], [x_f, x_f])
    plt.legend(('Predicted', 'True'))
    plt.xlabel('Iteration')
    plt.ylabel('x value')
    plt.savefig(os.path.join(os.path.dirname(__file__), '../results/tutorial', 'minimization.png'))
