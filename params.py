import os
import datetime

output_directory = ""
image_directory = ""

batch_size = 1024
correlation_threshold = 0.8

def setup(model_name=None, existingModel=False):
    global output_directory
    global image_directory

    if not os.path.exists('output'):
        os.makedirs('output')

    if model_name is None:
        dt = datetime.datetime.now()
        date_string = dt.strftime('%Y-%m-%d_%H-%M-%S')
        output_directory = 'output/' + date_string
    else:
        output_directory = 'output/' + model_name

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    elif not existingModel:
        dt = datetime.datetime.now()
        date_string = dt.strftime('%Y-%m-%d_%H-%M-%S')
        output_directory += "_" + date_string
        os.makedirs(output_directory)

    image_directory = output_directory + '/figures'
    if not os.path.exists(image_directory):
        os.makedirs(image_directory)
    image_directory = image_directory + '/'
