###################################################################################
#
#   Train and test artificial neural network
#
#   Reads from MNIST handwritten digit training and test datasets and
#   generates weights for 4 scenarios:
#
#         scenario 0  =  600 training records, 1 epoch
#         scenario 1  =  6,000 training records, 1 epoch
#         scenario 2  =  60,000 training records, 1 epoch
#         scenario 3  =  60,000 training records, 5 epochs
#
#   Also generates accuracy for each scenario based on MNIST test dataset
#
#   Note: this script is run separately and in advance of the realtime application
#   which is started by running main.py
#
#   calls:
#         ANN.py
#
#   files read in:
#         mnist_train.csv - MNIST handwritten digit training dataset
#         mnist_test.csv - MNIST handwritten digit test dataset
#
#   files read from:
#         weights_wih_0.csv - weights for input to hidden layers in scenario 0
#         weights_wih_1.csv - weights for input to hidden layers in scenario 1
#         weights_wih_2.csv - weights for input to hidden layers in scenario 2
#         weights_wih_3.csv - weights for input to hidden layers in scenario 3
#         weights_who_0.csv - weights for hidden to output layers in scenario 0
#         weights_who_1.csv - weights for hidden to output layers in scenario 1
#         weights_who_2.csv - weights for hidden to output layers in scenario 2
#         weights_who_3.csv - weights for hidden to output layers in scenario 3
#         test_scores.csv - accuracy for each scenario based on MNIST test dataset
#
###################################################################################

# numpy for various mathematical actions (e.g. use of matrices)
import numpy
# savetxt is used to write to file
from numpy import savetxt
# time for measuring time taken to load MNIST files
import time
# access to main.py needed for global constants
import main
# access to ANN.py needed
import ANN


# utility function to save test scores
# (% accuracy for each scenario based on testing with MNIST test data)
def save_test_scores(test_scores):
    savetxt('test_scores.csv', test_scores, delimiter=',', fmt="%s")


# utility function to load training data
def load_training_data(num_lines):

    ##### start clock for reading training data
    tic = time.perf_counter()

    # load the MNIST training data CSV file into a list
    training_data_file = open("mnist_train.csv", 'r')
    training_data_list = training_data_file.readlines()[0:num_lines]
    training_data_file.close()

    ##### stop clock for reading training data
    toc = time.perf_counter()
    print(f"Reading training data took {toc - tic:0.4f} seconds")

    return training_data_list


# utility function to save weights to file
def save_weights(wih, who, wih_file, who_file):

    savetxt(wih_file, wih, delimiter=',')
    savetxt(who_file, who, delimiter=',')


# load and prepare training data, then iterate through epochs
# and records performing forward and backward propagation
def training(i, wih_init, who_init):
    #
    #  i = scenario index
    #  wih_init = initial weights used between input and hidden layers
    #  who_init = initial weights used between hidden and output layers
    #

    ##### start clock for training
    tic = time.perf_counter()

    # set up file name for weights for input to hidden layers for current scenario
    wih_file = "weights_wih_" + str(i) + ".csv"
    # set up file name for weights for hidden to output layers for current scenario
    who_file = "weights_who_" + str(i) + ".csv"
    # number of records (i.e. "samples") to be read in from MNIST training dataset
    num_lines = main.scenarios[i]["num_lines"]
    # number of epochs (i.e. number of iterations through training dataset)
    num_epochs = main.scenarios[i]["num_epochs"]

    # load training data
    training_data_list = load_training_data(num_lines)

    # initialise weights used between input and hidden layers
    wih = wih_init
    # initialise weights used between hidden and output layers
    who = who_init

    # iterate through specified number of epochs
    for epoch in range(num_epochs):
        # iterate through specified number of records from the training data set
        for record in training_data_list:
            # split the record by the ',' commas
            all_values = record.split(',')
            # normalise (so in range 0 - 1) and
            # shift the inputs (ANN works best if avoid 0)
            inputs_list = (numpy.asfarray(all_values[1:]) / 255.0 *
                           0.99) + 0.01
            # set up target output values based on label in MNIST data
            # the output corresponding to the label is set to 0.99
            # the other outputs are set to 0.01
            # (shifted from 1 and 0 respectively to avoid causing problems for ANN)
            targets_list = numpy.zeros(
                main.configuration["output_nodes"]) + 0.01
            targets_list[int(all_values[0])] = 0.99
            # convert inputs list to 2d array
            inputs = numpy.array(inputs_list, ndmin=2).T
            # convert targets list to 2d array
            targets = numpy.array(targets_list, ndmin=2).T
            # forward propagation
            hidden_outputs, final_outputs = ANN.forward_prop(inputs, wih, who)
            # backwards propagation (incl. error calculation & gradient descent)
            wih, who = ANN.backward_prop(inputs, hidden_outputs, final_outputs,
                                         targets, wih, who)

    ##### stop clock for training
    toc = time.perf_counter()
    print(f"Training took {toc - tic:0.4f} seconds")

    # save weights to file
    save_weights(wih, who, wih_file, who_file)

    return wih, who


def load_test_data():

    ##### start clock for reading test data
    tic = time.perf_counter()

    # load the MNIST test data CSV file into a list
    test_data_file = open("mnist_test.csv", 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    ##### stop clock for reading test data
    toc = time.perf_counter()
    print(f"Reading test data took {toc - tic:0.4f} seconds")

    return test_data_list


# load and prepare test data,
# then iterate through records performing forward propagation
def testing(test_data_list, wih, who):
    ##### start clock for testing
    tic = time.perf_counter()

    # count how many predictions are correct
    count = 0

    # go through all the records in the test data set
    for record in test_data_list:
        # split the record by the ',' commas
        all_values = record.split(',')
        # get label i.e. "correct" value
        # according to person who labelled the test data
        label = int(all_values[0])
        # normalise (so in range 0 - 1) and
        # shift the inputs (ANN works best if avoid 0)
        inputs_list = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        # forward propagation
        temp, outputs = ANN.forward_prop(inputs, wih, who)
        # get ANN's top prediction (by taking the index of the highest value)
        top_prediction = numpy.argmax(outputs)
        # append correct or incorrect to list
        if (top_prediction == label):
            # ANN's prediction matches MNIST label
            count = count + 1

    ##### stop clock for testing
    toc = time.perf_counter()
    print(f"Testing took {toc - tic:0.4f} seconds")

    # calculate the test score, the fraction of correct predictions
    score = count / len(test_data_list)
    test_score = str(round(score * 100, 2)) + "%"

    return test_score


# initialise ANN configuration, scenarios datablock and weights
def initialise_for_train_and_test():

    input_nodes = main.configuration["input_nodes"]
    hidden_nodes = main.configuration["hidden_nodes"]
    output_nodes = main.configuration["output_nodes"]

    # initialise weights

    wih_init = numpy.random.normal(0.0, pow(input_nodes, -0.5),
                                   (hidden_nodes, input_nodes))
    who_init = numpy.random.normal(0.0, pow(hidden_nodes, -0.5),
                                   (output_nodes, hidden_nodes))

    return wih_init, who_init


# this if statement stops the code below being executed when this file is imported
if __name__ == "__main__":

    # initialisation
    wih_init, who_init = initialise_for_train_and_test()

    test_scores = []

    # load test data (same for all scenarios, so only do once)
    test_data_list = load_test_data()

    # loop through each of 4 scenarios:
    for i in range(0, 4):

        # train the neural network
        wih, who = training(i, wih_init, who_init)

        # test the neural network
        test_score = testing(test_data_list, wih, who)

        test_scores.append(test_score)

    save_test_scores(test_scores)