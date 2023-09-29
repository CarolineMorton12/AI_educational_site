###################################################################################
#
#   The starting point for the realtime application
#
#   Launches the website and through that allows interaction with the artificial
#   neural network which has already been trained/tested
#
#   Note: training/testing is done separately by calling train_and_test.py
#
###################################################################################

# flask is used to allow Python application on server to communicate with
# webpages on browser (including passing parameters both ways)
from flask import Flask, request, render_template

import realtime_query

# global constants

# ANN configuration data
configuration = {
    "input_nodes": 784,  # number of nodes in input layer
    "hidden_nodes": 200,  # number of nodes in hidden layer
    "output_nodes": 10,  # number of nodes in output layer
    "learning_rate": 0.1  # step size for gradient descent
}

# ANN training scenarios
# "num_lines" = number of records (samples) to be read from MNIST training dataset
# "num_epochs" = number of epochs (i.e. iterations through training datatset)
scenarios = [{
    "num_lines": 600,
    "num_epochs": 1
}, {
    "num_lines": 6000,
    "num_epochs": 1
}, {
    "num_lines": 60000,
    "num_epochs": 1
}, {
    "num_lines": 60000,
    "num_epochs": 5
}]

# confidence band thresholds
high_threshold = 90.00
medium_threshold = 60.00
low_threshold = 20.00

# pixel dimensions
pixel_width = 28
pixel_height = 28

# set up Flask app including where to find items in subdirectories
app = Flask(__name__,
            static_url_path='',
            static_folder='static',
            template_folder='templates')


# Home route for Flask (corresponds to Home.html)
@app.route("/Home")
def Home():
    # any application code needed to support Home.html goes here
    return render_template('Home.html')


# Info route for Flask (corresponds to Info.html)
@app.route("/Info")
def Info():
    # any application code needed to support Info.html goes here
    return render_template('Info.html')


# Quiz route for Flask (corresponds to Quiz.html)
@app.route("/Quiz")
def Quiz():
    # any application code needed to support Info.html goes here
    return render_template('Quiz.html')


# ANN route for Flask (corresponds to ANN.html)
@app.route('/ANN', methods=['GET', 'POST'])
def index():

    # initialise parameters that are going to be sent to browser
    parameters = realtime_query.initialise_parameters()

    if request.method == 'POST':
        # get image captured by sketchpad at browser (as a string)
        input_string = request.form.get('sketchpad_image')
        # process image (including submission to ANN) and get back
        # prediction/confidence data as "parameters"
        parameters = realtime_query.process_image(input_string)
        # return used by "POST" part of Flask to send parameters to ANN.html
        return render_template('ANN.html', parameters=parameters)

    # return used by "GET" part of flask
    return render_template('ANN.html', parameters=parameters)


# open browser and run Flask app
if __name__ == '__main__':
    import webbrowser

    # open web browser on localserver and use Home.html
    webbrowser.open_new('http://127.0.0.1:5000/Home')

    # run via call to Flask app
    app.run()