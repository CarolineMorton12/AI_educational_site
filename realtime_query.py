###################################################################################
#
#   Handles real time processing of queries to Artificial Neural Network (ANN)
#   and associated input and output processing, i.e.
#   - journey from input of a digit at sketchpad to submission to ANN
#   - journey from output of prediction/confidence data from ANN to display to user
#
#   called from:
#         ANN route in main.py
#
#   calls:
#         ANN.py
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
# loadtxt is used to read from file
from numpy import loadtxt
# access to ANN.py needed
import ANN
# access to main.py needed for global constants
import main


# utility function to load test scores
# (% accuracy for each scenario based on testing with MNIST test data)
def load_test_scores():

    # load the MNIST test data file into a list
    test_scores_file = open("test_scores.csv", 'r')
    test_scores = test_scores_file.readlines()
    test_scores_file.close()

    return test_scores


# initialisation
def initialise_parameters():

    test_scores = load_test_scores()

    parameters = {
        "predictions": ['-', '-', '-', '-'],
        "confidences": ['-', '-', '-', '-'],
        "test_scores": test_scores,
        "confidence_band": "",
        "indices": [],
        "values": []
    }

    return parameters


# reformat and compress sketchpad image received from browser,
# reformat again for use by ANN
def compress_image(input_string):

    # split string into list sepated by ","
    # scale and shift the inputs
    inputs = (numpy.asfarray(input_string.split(',')) / 255.0 * 0.99) + 0.01

    # compress image from 224x224 obtained from browser to 28x28
    # as expected by MNIST-trained ANN
    new_arr = numpy.zeros((main.pixel_width, main.pixel_height))
    arr = numpy.array(inputs).reshape(224, 224)
    for i in range(0, 224, 8):
        for j in range(0, 224, 8):
            temp_array = arr[i:i + 8, j:j + 8]
            new_arr[i // 8, j // 8] = numpy.mean(temp_array)

    inputs_list = numpy.array(new_arr).reshape(784, )

    return inputs_list


# process sketchpad image received from browser, including querying ANN and
# preparing prediction/confidence data, for all scenarios
def process_image(input_string):

    inputs_list = compress_image(input_string)

    # initialise other lists
    outputs = []
    sum_of_outputs = []
    new_outputs = []
    ranked_outputs = []
    index = []
    predicted_labels = []
    confidence_levels = []

    # run query based on input from sketchpad
    # looping through 4 scenarios
    for i in range(0, 4):

        # read weights from file
        wih = loadtxt('weights_wih_' + str(i) + '.csv', delimiter=',')
        who = loadtxt('weights_who_' + str(i) + '.csv', delimiter=',')

        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T

        # query the neural network: inputs = image originating from sketchpad,
        # outputs give confidence levels for each possible digit (0 - 9)
        not_used, myoutput = ANN.forward_prop(inputs, wih, who)
        outputs.append(myoutput)
        sum_of_outputs.append(sum(outputs[i]))

        temp_outputs = []
        for output in outputs[i]:
            output = numpy.round((output[0] / sum_of_outputs[i]) * 100, 2)
            temp_outputs.append(output[0])
        new_outputs.append(temp_outputs)

        # sort in descending order while retaining index
        ranked_outputs.append(
            sorted(enumerate(new_outputs[i]), key=lambda x: x[1],
                   reverse=True))

        # grab prediction by getting index of largest output
        index.append(numpy.argmax(outputs[i]))

        # return as string
        predicted_labels.append(str(index[0]))

        confidence_levels.append(str((ranked_outputs[i][0][1])) + "%")

    # ranked list (for main scenario only)
    ranked_index_list = []
    ranked_value_list = []

    for item in ranked_outputs[3]:
        ranked_index_list.append(item[0])
        ranked_value_list.append(str(item[1]) + "%")

    # set confidence band for headline scenario
    if ranked_outputs[3][0][1] >= main.high_threshold:
        confidence_band = "high"
    elif ranked_outputs[3][0][1] >= main.medium_threshold:
        confidence_band = "medium"
    elif ranked_outputs[3][0][1] >= main.low_threshold:
        confidence_band = "low"
    else:
        confidence_band = "no"

    # prepare parameters for sending to browser
    parameters = initialise_parameters()
    parameters["predictions"] = predicted_labels
    parameters["confidences"] = confidence_levels
    parameters["confidence_band"] = confidence_band
    parameters["indices"] = ranked_index_list
    parameters["values"] = ranked_value_list

    return parameters