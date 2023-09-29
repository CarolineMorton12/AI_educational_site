###################################################################################
#
#   Core Artificial Neural Network (ANN) algorithms:
#   - forward propagation
#   - backwards propagation (including error calculation and gradient descent)
#
#   called from:
#         train_and_test.py
#         realtime_query.py
#
###################################################################################

# numpy for various mathematical operations (e.g. multiplication of matrices)
import numpy
# scipy.special for the sigmoid function expit()
import scipy.special
# access to main.py needed for global constants
import main

# activation function is the sigmoid function
activation_function = lambda x: scipy.special.expit(x)

# forward_prop is the "forward propagation" part of
# the overall "back propagation" algorithm
def forward_prop(inputs, wih, who):
    #
    #  inputs = inputs to ANN
    #  wih = weights used between input and hidden layers
    #  who = weights used between hidden and output
    #
    ######  feed forward - input layer to hidden layer
    # calculate inputs into hidden layer
    # (by multiplying together the inputs to ANN -
    # i.e. the outputs from the input layer -
    # and the weights used between input and hidden layers)
    hidden_layer_inputs = numpy.dot(wih, inputs)
    # calculate the outputs from hidden layer (by applying the activation function)

    hidden_layer_outputs = activation_function(hidden_layer_inputs)

    ######  feed forward - hidden layer to output layer
    # calculate inputs into output layer
    # (by multiplying together the outputs to hidden layer and
    # the weights used between hidden and output layers)
    output_layer_inputs = numpy.dot(who, hidden_layer_outputs)
    # calculate the outputs from output layer (by applying the activation function)
    output_layer_outputs = activation_function(output_layer_inputs)

    return hidden_layer_outputs, output_layer_outputs


# backward_prop is the "backward propagation"
# (including error calculation and gradient descent) part of
# the overall "back propagation" algorithm
def backward_prop(inputs, hidden_layer_outputs, output_layer_outputs, targets,
                  wih, who):
    #
    #  inputs = inputs to ANN
    #  hidden_layer_outputs =
    #         outputs from hidden layer generated during forward propagation
    #  output_layer_outputs =
    #         outputs from output layer generated during forward propagation
    #  targets = expected values for each output (based on label in MNIST data)
    #  wih = weights used between input and hidden layers
    #  who = weights used between hidden and output layers
    #

    ####### calculate errors
    # output layer errors are calculated by taking differences between
    # the expected (as derived from label) values & the predicted (by ANN) values
    output_errors = targets - output_layer_outputs
    # hidden layer errors are calculated from
    # the output_errors & the weights used between hidden and output layers
    # (i.e. split errors across hidden nodes based on the weights)
    hidden_errors = numpy.dot(who.T, output_errors)

    ####### update weights (backward propagation using gradient descent)
    # update the weights for the links between the input and hidden layers
    wih += main.configuration["learning_rate"] * numpy.dot(
        (hidden_errors * hidden_layer_outputs *
         (1.0 - hidden_layer_outputs)), numpy.transpose(inputs))
    # update the weights for the links between the hidden and output layers
    who += main.configuration["learning_rate"] * numpy.dot(
        (output_errors * output_layer_outputs *
         (1.0 - output_layer_outputs)), numpy.transpose(hidden_layer_outputs))

    return wih, who