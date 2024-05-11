import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from datetime import datetime
tf.get_logger().setLevel('ERROR')


class my_training_callback(tf.keras.callbacks.Callback):
    """
    This callback class exists to collect information during training and validation.
    It gathers loss, accuracy and runtime information and produces plots based on such values.
    """
    def __init__(self):
        """
        This method instantiates the class and initialises all of the attributes to be collected during training.
        It tracks the start and end time of the training with "time_started" and "time_finished"
        It tracks the duration of the current epoch using "time_curr_epoch"
        It tracks the number of elapsed epochs with "num_epochs"
        Then there is an array for each of the following: Run time (_times), training loss (_loss), training accuracy (_acc),
        validation loss (_val_loss) and validation accuracy (_val_acc)
        """
        self.time_started=None
        self.time_finished=None
        self.time_curr_epoch=None
        self.num_epochs = 0
        self._times,self._loss, self._acc,self._val_loss,self._val_acc = [],[],[],[],[]
        
    def _plot_model_performance(self):
        """
        This function produces 3 plots.
        1. A plot of run time per epoch to evaluate linear runtime
        2. A plot of training and validation loss vs epochs
        3. 
        """

        # This is a plot of Runtime against epoch count.
        plt.xlabel('Epoch')
        plt.ylabel('Total time taken until an epoch in seconds')
        plt.plot(self._times)
        plt.show()

        # This is a plot of training and validation loss vs epoch count.
        fig,(ax1,ax2)=plt.subplots(1,2,figsize=(20,10))
        fig.suptitle('Model Performance',size=20)
        ax1.plot(range(self.num_epochs), self._loss, label='Training loss')
        ax1.plot(range(self.num_epochs), self._val_loss, label='Validation loss')
        ax1.set_xlabel('Epoch', size=14)
        ax1.set_ylabel('Loss', size=14)
        ax1.legend()
        
        # This is a plot of training and validation accuracy vs epoch count.
        ax2.plot(range(self.num_epochs), self._acc, label='Training accuracy')
        ax2.plot(range(self.num_epochs), self._val_acc, label='Validation Accuracy')
        ax2.set_xlabel('Epoch', size=14)
        ax2.set_ylabel('Accuracy', size=14)
        ax2.legend()
        plt.show()

    def on_epoch_begin(self,epoch,logs=None):
        """
        This function which runs at the beginning of an epoch captures updates the current timestamp stored. 
        This timestamp is used to track duration of epochs and training.
        """
        self.time_curr_epoch=datetime.now()
        
    def on_epoch_end(self,epoch,logs={}):
        """
        This function which runs at the end of an epoch captures the various metrics and timestamps that need to be collected for review.
        """
        # Update the number of epochs tracked.
        self.num_epochs+=1

        # Calculate the amount of time elapsed for this epoch and store it
        epoch_dur = (datetime.now()-self.time_curr_epoch).total_seconds()
        self._times.append(epoch_dur)

        # Capture other metrics calculated on per epoch basis.
        tl = logs['loss']; self._loss.append(tl)
        ta = logs['accuracy']; self._acc.append(ta)
        vl = logs['val_loss']; self._val_loss.append(vl)
        va = logs['val_accuracy']; self._val_acc.append(va)
        
        # User output - uncessary since TF already gives these during training but just good practice for using callbacks.
        train_metrics = f"train_loss: {tl:.4f}, train_accuracy: {ta:.4f}"
        valid_metrics = f"validation_loss: {vl:.4f}, validation_accuracy: {va:.4f}"
        print(f"\n\nEpoch: {epoch+1:4} | Runtime {epoch_dur:.3f}seconds\n{train_metrics}\n{valid_metrics}")

    def on_train_begin(self,logs=None):
        """
        This function which runs at the begining of training captures the start time and displays it to the user.
        """
        # Capture start time.
        self.time_started=datetime.now()    

        # Display start time to user.
        print(f'TRAINING STARTED | {self.time_started}\n')   

    def on_train_end(self,logs={}):
        """
        This function which runs at the end of training collects the final time stamp and evaluates the total runtime.
        In addition this function outputs the final performance of the model on the training and validation sets.
        """
        # Collect timestamp
        self.time_finished = datetime.now()

        # Calculate total duration of training.
        train_duration = str(self.time_finished - self.time_started)

        # Output duration to user.
        print(f'\nTRAINING FINISHED | {self.time_finished} | Duration: {train_duration}')
        
        # Capture final values for training and validation metrics.
        tl = f"Training loss:       {logs['loss']:.5f}"
        ta = f"Training accuracy:   {logs['accuracy']:.5f}"
        vl = f"Validation loss:     {logs['val_loss']:.5f}"
        va = f"Validation accuracy: {logs['val_accuracy']:.5f}"
        
        # Output for user review.
        print('\n'.join([tl, vl, ta, va]))

        # Output performance graphs for the user to review.
        ########################### WARNING ##############################################
        #       Comment out this line if you want to run a multiple models like in task 7.
        self._plot_model_performance()
        #       If you don't comment this out the run will pause until you close all the graphics
        #       This will mean you can't start a long run and walk away from your computer for an hour to see how it goes with all the learning rates.
        ########################### WARNING ##############################################

def cls():
    """
    A convenience function to clear vs code/codium's annoying terminal clutter.
    """
    # Cls is the command to clear the terminal in windows prompt.
    os.system("cls")

def Open_Data():
    """
    This function sets the current working directory to the location of this file and then searchs in that directory for a zip file.
    If the zip file is present it is unzipped and it's path is captured and returned.
    """

    # Name of dataset zip file downloaded from QUT Blackboard.
    name_zip_file = "small_flower_dataset.zip"    

    # The directory path of the file is captured.
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # We set the directory to the location of this file.
    os.chdir(dir_path)

    # We use the keras.utils module to find the data.
    path_to_zip = tf.keras.utils.get_file(fname=name_zip_file,origin=os.getcwd(),extract=True,cache_subdir=os.getcwd())

    # We report to the user that the file has been found.
    print(f"The original file is in {path_to_zip}\n")
    return path_to_zip

def Get_Path():
    """
    this function is a convenience function that sets the current directory to the location of this file and returns the path of the full unzipped data.
    """
    # Get the directory of this file.
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # Set the directory to the location of this file
    os.chdir(dir_path)

    # Assuming that the data has already been unzipped into this folder we attempt to create a string with the path of the dataset.
    path = os.path.join(os.getcwd(),"small_flower_dataset")

    # We just make sure that we have correctly found the location of the data, if it doesn't exist we create it now.
    if os.path.exists(path):
        print(f"Data successfully unpacked to {path}\n")
    return path

def Get_Model(IMG_SHAPE):
    """
    This model takes in the the "Shape" of the input images as a tuple with length, width and channels.
    Using the keras.applications module we download the MobileNetV2 model and instantiate it with our image shape.
    We do not download it with it's classification layer as this is not needed - denoted by include_top = False.
    The weights used are the pretrained 'imagenet' weights.
    """

    # Call the MobileNetV2 model.
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,include_top=False,weights='imagenet')
    return base_model

def Prepare_Data(path,BATCH_SIZE,IMG_SIZE):
    """
    This function creates the datasets need for testing and training.
    It also sets up dataset.prefetch() to more efficiently supply data to the model during training.
    """

    # Create the training and validation datasets.
    # We have possibly made a huge error here as both testing and training draw from the same source 
    # That is instead of one input dataset being created and partioned into test,train and validation.
    # We blame the tensorflow transfer learning tutorial for misleading us, this is exactly how they do it.
    train_dataset = tf.keras.utils.image_dataset_from_directory(path,seed=seed,shuffle=True,batch_size=BATCH_SIZE,image_size=IMG_SIZE)
    validation_dataset = tf.keras.utils.image_dataset_from_directory(path,seed=seed,shuffle=True,batch_size=BATCH_SIZE,image_size=IMG_SIZE)
    
    # Split the validation dataset into testing and validation sets. 
    # The Test set is 20% of the validation set.
    val_batches = tf.data.experimental.cardinality(validation_dataset)
    test_dataset = validation_dataset.take(val_batches//5)
    validation_dataset=validation_dataset.skip(val_batches//5)
    
    # Here we set up the prefectch to have an autotuned buffer size so it can dynmaically prepare an
    # appropriately sized batch of samples for the model to consume in the background during training.
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

    # Return the desire datasets.
    return train_dataset,validation_dataset,test_dataset

def Prepare_Augmentation():
    """
    This function creates an augmentation layer for a keras sequential model.
    The purpose of this layer is to introduce more data to the training model that is the same original information but oriented in all kinds of different ways
    This is to help it understand the same objects perceived from different orientations in future input data.
    """
    # Create a keras sequential model of 2 layers
    data_augmentation = tf.keras.Sequential([
        # Create a random horizontal flip layer to mirror around the y-axis
        tf.keras.layers.RandomFlip('horizontal'),
        # Create a random rotation layer to orient pictures at a random angle offset.
        tf.keras.layers.RandomRotation(0.2),
    ])
    return data_augmentation

def Make_Summary_Plot(train_dataset):
    """
    This convenience function exists for one to peruse the available images in the dataset.
    It takes in training dataset.
    """
    # Generate the class labels of the dataset.
    class_names = train_dataset.class_names

    # Create a figure
    plt.figure(figsize=(10,10))

    # Get 9 images and plot them in a matrix - assign their labels to each subplot.
    for images,labels in train_dataset.take(1):
        for i in range(9):
            ax=plt.subplot(3,3,i+1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")        
            i+=1

def Make_Augmentation_plot(train_dataset,data_augmentation):
    """
    This convenience function allows one to observe the kinds of mirroring and rotation operations that will be applied to the dataset.
    This function takes in a training dataset and a data_augmentation object configured with flips and rotations.
    """

    # Create the plot which displays the same single image oriented in 9 different ways.
    for image, _ in train_dataset.take(1):
        plt.figure(figsize=(10, 10))
        first_image = image[0]
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            # Apply the augmentation to each image.
            augmented_image = data_augmentation(tf.expand_dims(first_image, 0))

            # Plot the augmented image.
            plt.imshow(augmented_image[0] / 255)
            plt.axis('off')

def Prepare_Model(train_dataset,IMG_SHAPE):
    """
    This function prepares the full model for training by creating layers and accumulating them into a single model object.
    It does this in two stages:
    1. Firstly it takes in the base mode freezes it and then applies some preprocessing
    2. Secondly it adds a new, dense output, layer for classification.
    Inputs to this function are the training data set and the shape of the image as a tuple of length,width and number of colour channels.
    """

    # Setup the input layer to be sized as our input images.
    inputs = tf.keras.Input(shape=IMG_SHAPE)
    
    # Instatiate the augmentation object to perform flips and rotations on our images.
    data_augmentation = Prepare_Augmentation()   

    # Begin model assembly by having the augmentation applied to the inputs.
    x = data_augmentation(inputs)
    # Make_Augmentation_Plot(train_dataset, data_augmentation)

    # Next we add a preprocessing layer that rescales the RGB values back from [0,255] to [0,1].
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    # Apply the preprocessing layer.
    x = preprocess_input(x)

    # Next we instantiate the MobileNetV2 model and ensure that it's parameters are not trainable.
    base_model = Get_Model(IMG_SHAPE)
    base_model.trainable=False

    # Next we apply MobileNetV2 layer
    x = base_model(x,training=False)

    # Next we instantiate a global averaging layer to transform an image representation into a feature vector.    
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

    # Apply the pooling layer
    x = global_average_layer(x)

    # Next we instantiate a dropout layer which will guard against overfitting by randomly omitting features at a rate of 20%.
    # This will prevent the dependencies on neighbouring neurons to be created.
    dropout_layer = tf.keras.layers.Dropout(rate = 0.2)
    
    # Apply the dropout layer.
    x = dropout_layer(x)
    
    # Next we create the dense output layer with 5 nodes corresponding to our 5 output classes of flowers. We use the softmax activation as it is suitable for non-binary class problems such as this.
    classification_layer = tf.keras.layers.Dense(5,activation='softmax')
    # Apply the classification layer.
    outputs = classification_layer(x)
    
    # Finalise the assembly of the model
    model = tf.keras.Model(inputs=inputs,outputs=outputs)

    # Return the model
    return model

def Compile_Model(model,train_dataset,validation_dataset,initial_epochs=10,learning_rate=0.01,momentum=0,nesterov=False):
    """
    This function compiles and trains the input model.
    Inputs: Model - a fully assembled model.
    Inputs: training and validation datasets
    Inputs: The number of Epochs to train for, defaulting to 10
    Inputs: The learning rate to control how reactive the training is to new data, defaulting to 0.01
    Inputs: Momentum which allows the model to become more reactive over time once a good direction has been found. Defaults to off.
    Inputs Nesterov a modification to gradient descrent that prefers to overshoot but introduce a correction immediately aftewards using new information.
    """

    # Instantiate the SGD optimizer with the input parameters.
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate,momentum=momentum,nesterov=nesterov)

    # Compile the model with all the elements it needs for training. We select the Sparse Categorical Loss function as it is appropriate for problems with more than 2 classes.
    model.compile(optimizer=optimizer,loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),metrics=["accuracy"])

    # Instatiate a callback class that will dutifully collect metrics for us, the user, to use conveniently later on.
    timetaken=my_training_callback()

    # Train the model.
    history = model.fit(train_dataset,epochs=initial_epochs,validation_data=validation_dataset,callbacks=[timetaken])
    
    return model,history,timetaken

def Show_Results(model,history,test_dataset,timetaken,batch_size):
    """
    This function outputs some nice results for the user so they can get a feel for the performance of the trained model.
    Inputs: A trained model
    Inputs: The history object output by a trained model.
    Inputs: Timetaken for training
    Inputs: Batch size how many images to look at before updating weights.
    """

    print("\n\n###########################################################################################################################################################\n")
    # Repor the total runtime to the user.
    print(f"The model took {sum(timetaken._times)} seconds to train")

    # Evaluate the model against the test data set.
    results = model.evaluate(test_dataset,batch_size = batch_size)

    # Report the results of the evaluation to the user.
    print(f"Test loss and test accuracy: {results}")

    # Make predictions on the test dataset.
    prediction = model.predict(test_dataset)

    # Ensure the shape of the outputs is correct.
    print(f"Prediction Shape: {prediction.shape}")
    print("\n\n###########################################################################################################################################################\n")

def Task6_Graphics(history):
    """
    This function produces two simple plots as required by task 6.
    1. A plot of training and validation accuracy against the number of elapsed epochs
    2. A plot of training and validation loss against the number of elapsed epochs
    Inputs: A history object output from a trained model
    """
    
    # Setup a dictionary that where each key pulls a list from the history object.
    figure_labels = {"accuracy":0,"val_accuracy":0,"loss":0,"val_loss":0}
    for key,value in figure_labels.items():
        figure_labels[key]=history.history[key]
        
    # This is a plot of the training and validation accuracy against the number of epochs.
    plt.figure(figsize=(8,8))
    plt.subplot(2,1,1)
    plt.plot(figure_labels['accuracy'],label='Training Accuracy')
    plt.plot(figure_labels['val_accuracy'],label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()),1])
    plt.title('Training and Validation Accuracy')

    # This is a plot of the training and validation loss against the number of epochs.
    plt.subplot(2,1,2)
    plt.plot(figure_labels['loss'],label="Training Loss")
    plt.plot(figure_labels['val_loss'],label="Validation Loss")
    plt.legend(loc="upper right")
    plt.ylabel('Cross Entropy')
    plt.ylim([0,1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()

def learning_rate_graphics(rate_results):
    """
    This function creates the graphics and other results needed to address task 7 concerning learning rates.
    Inputs: A "rate_results" object which is a dictionary of tuples
    Inputs: The keys of "rate_results" are the various learning rates explored in task 7
    Inputs: The values of "rate_results" are tuples where each tuple contains: 
    1. Results - An output of the model.evaluate that has the test loss and accuracy outputs.
    2. History - A history object from a trained model which contains all of loss and accuracy values for training and validation runs.
    3. Timetaken - A callback object that has information on the time elapsed and other metrics (perhaps some overlap with what history does but I was teaching myself so i don't mind)
    """

    # Initialise the labels to be used in our plots.
    figure_labels = ["accuracy","val_accuracy","loss","val_loss"]
    
    # Loop over the labels - each label will spawn a graphic.
    for label in figure_labels:
        plt.figure(figsize=(16,10))
        # Loop over the learning rates, each learning rate will spawn a line graph on the current figure for the particular label of this iteration.
        for rate, out_tuple in rate_results.items():
            plt.plot(np.cumsum(out_tuple[2]._times),out_tuple[1].history[label],label=f"Learning Rate = {rate}")

        # Apply labels and titles to the plot containing all the line graphs.
        plt.xlabel("Runtime (Seconds)")
        plt.ylabel(f"Prediction {label}")
        plt.title(f"Learning Rate Evaluation - {label} vs runtime")
        plt.rcParams["figure.figsize"] = (16,10)
        plt.legend(loc='lower left', bbox_to_anchor=(1.04, 0.8))
        plt.tight_layout(rect=[0, 0, 0.95, 1])
        plt.savefig(f"Learning Rate Evaluation -  {label} vs runtime.png")
        # plt.show()


    # Next we loop over the test loss and accuracy and see how the different learning rates influence the min,max and prediction values
    for i in range(2):
        # Instantiate the arrays that will contain the min, max and prediction values of the current metric.
        min_val=[]
        max_val=[]
        predict_val=[]
        plt.figure(figsize=(16,10))

        # Loop over the learning rates and evaluate the min, max and prediction value of the metric.
        for rate, out_tuple in rate_results.items():
            min_val.append(min(out_tuple[1].history[figure_labels[2*i]]))
            max_val.append(max(out_tuple[1].history[figure_labels[2*i]]))
            predict_val.append(out_tuple[0][-(i+1)])
        
        # Set the x-axis as log10(learning_rate) for a fairer comparison.
        log_rates = [np.log10(i) for i in list(rate_results.keys())]
        # Plot 3 line graphs.
        plt.plot(log_rates,min_val,label=f"Minimum {figure_labels[2*i]}")    
        plt.plot(log_rates,max_val,label=f"Maximum {figure_labels[2*i]}") 
        plt.plot(log_rates,predict_val,label=f"Prediction {figure_labels[2*i]}") 

        # Apply labels, titles and legends then save the file
        plt.xlabel("log10(learning rate)")
        plt.ylabel(f"{figure_labels[2*i]}")
        plt.title(f"Min and Max Training {figure_labels[2*i]} and Prediction {figure_labels[2*i]}")
        plt.legend(loc='lower left', bbox_to_anchor=(1.04, 0.8))
        plt.tight_layout(rect=[0, 0, 0.95, 1])
        plt.rcParams["figure.figsize"] = (16,10)
        plt.savefig(f"Min and Max Training {figure_labels[2*i]} and Prediction {figure_labels[2*i]}.png")
        # plt.show()

def Task1():
    '''
    Download the small flower dataset and get it into python
    Place the Zip file containing the dataset in the same folder as this file and this function will find it.

    '''  
    # The data was downloaded from blackboard as a zip file and placed in the same folder as this script.
    
    # Clean up the console - vs code is a mess.
    cls()

    # Get the zip path and unzipped data path.
    path_to_zip = Open_Data()
    path = Get_Path()  
    
def Task2():
    """
    Using the tf.keras.applications module download a pretrained MobileNetV2 network.
    """
    # Clean up the console.
    cls()

    # Initialise hyper parameters such as image size and number of colour channels.
    IMG_SIZE=(160,160)
    IMG_SHAPE = IMG_SIZE + (3,)
    
    # Instantiate the MobileNetV2 object with the chosen image shape.
    model = Get_Model(IMG_SHAPE=IMG_SHAPE)
    print(f"The name of the model is: {model}")
    model.summary()

def Task3():
    """
    Replace the last layer of the downloaded neural network with a Dense layer of the appropriate shape for the 5 classes of the small flower dataset.
    """

    # Clean the console and initialise parameters
    cls()
    BATCH_SIZE = 32
    IMG_SIZE=(160,160)
    IMG_SHAPE = IMG_SIZE + (3,)

    # Get path to the data.
    path = Get_Path()

    # Get the dataset to allow for some preprocessing of the model.
    train_dataset,validation_dataset,test_dataset = Prepare_Data(path, BATCH_SIZE, IMG_SIZE)

    # Download the model and output it to model variable.
    model = Prepare_Model(train_dataset,IMG_SHAPE)

    # Confirm that the model is present.
    model.summary()

def Task4():
    '''
    Prepare your training, validation and test sets for the non-accelerated version of transfer learning.
    '''

    # Clean console up and initialise parameters.
    cls()
    path = Get_Path()
    BATCH_SIZE = 32
    IMG_SIZE=(160,160)
    IMG_SHAPE = IMG_SIZE + (3,)

    # Prepare the data as in previous questions.
    train_dataset,validation_dataset,test_dataset = Prepare_Data(path,BATCH_SIZE,IMG_SIZE)

    # Confirme the shape of the data.
    print(f"Number of training batches: {tf.data.experimental.cardinality(train_dataset)}")
    print(f"Number of validation batches: {tf.data.experimental.cardinality(validation_dataset)}")
    print(f"Number of test batches: {tf.data.experimental.cardinality(test_dataset)}")

def Task5():
    """
    Compile and train your model with an SGD optimizer using learning_rate=0.01, momentum=0.0, nesterov=False
    """

    # Clean the console and initalise parameters
    cls()
    path = Get_Path()
    BATCH_SIZE = 32
    IMG_SIZE=(160,160)
    IMG_SHAPE = IMG_SIZE + (3,)

    # New parameters are initialised that are required to train the model.
    initial_epochs=100
    learning_rate=0.01
    momentum=0.0
    nesterov=False

    # The dataset is prepare as per previous questions.
    train_dataset,validation_dataset,test_dataset = Prepare_Data(path,BATCH_SIZE,IMG_SIZE)

    # Download the base model
    model = Prepare_Model(train_dataset,IMG_SHAPE)
    # The model is preprocessed, frozen and extended with a dense output layer of 5 neurons. Then it is trained with the training dataset using an SGD optimizer.
    model,history,timetaken = Compile_Model(model,train_dataset,validation_dataset,initial_epochs=initial_epochs,learning_rate=learning_rate,momentum=momentum,nesterov=nesterov)

    # Show results shows some results - mainly to do with the test loss and test accuracy.
    Show_Results(model,history,test_dataset,timetaken,BATCH_SIZE)
     
def Task6():
    """
    Plot the training and validation errors vs time as well as the training and validation accuracies.
    """

    # Clear console and initialise all parameters needed to plot graphs.
    cls()
    path = Get_Path()
    BATCH_SIZE = 32
    IMG_SIZE=(160,160)
    IMG_SHAPE = IMG_SIZE + (3,)

    initial_epochs=10
    learning_rate=0.01
    momentum=0.0
    nesterov=False

    # Gather the datasets as in previous questions
    train_dataset,validation_dataset,test_dataset = Prepare_Data(path,BATCH_SIZE,IMG_SIZE)

    # Download the model
    model = Prepare_Model(train_dataset,IMG_SHAPE)

    # Assemble, preprocess, freeze and extend the base model with a dense output layer with 5 neurons, then train the model.
    model,history,timetaken = Compile_Model(model,train_dataset,validation_dataset,initial_epochs=initial_epochs,learning_rate=learning_rate,momentum=momentum,nesterov=nesterov)

    # Pass the output of the trained model to the graphics generator for task 6.
    Task6_Graphics(history)

def Task7():
    """
    Experiment with 3 different orders of magnitude for the learning rate - I did 5 because, why not. Plot the results and draw conclusions.
    """
    # Clear consol and intialise parameters.
    cls()
    path = Get_Path()
    BATCH_SIZE = 32
    IMG_SIZE=(160,160)
    IMG_SHAPE = IMG_SIZE + (3,)

    # Epochs set to 50 will take a good amount of time to run, ~ 700 seconds per learning rate. For a quicker option use the commented line of only two learning rates.
    initial_epochs=50
    learning_rates = [0.0001,0.001,0.01,0.1,1]
    # learning_rates = [0.01,0.1]
    momentum=0.0
    nesterov=False

    # Initialise dictionary of tuples that will contain model.evluate results, history objects and timetaken callback objects.
    rate_results = {}

    # Prepare the dataset as in previous questions.
    train_dataset,validation_dataset,test_dataset = Prepare_Data(path,BATCH_SIZE,IMG_SIZE)

    # Download the dataset as in previous questions.
    model = Prepare_Model(train_dataset,IMG_SHAPE)

    # Loop over all learning rates, assemble, preprocess, freeze and extend all models - then train them. 
    for learning_rate in learning_rates:
        print("\n\n###########################################################################################################################################################")
        print(f"###################################################################### The learning rate is: {learning_rate} #########################################################")
        print("###########################################################################################################################################################\n")

        # Compile as in previous questions.
        model,history,timetaken = Compile_Model(model,train_dataset,validation_dataset,initial_epochs=initial_epochs,learning_rate=learning_rate,momentum=momentum,nesterov=nesterov)

        # Collect nice results pertaining to runtime and test accuracy and test loss.
        Show_Results(model,history,test_dataset,timetaken,BATCH_SIZE)

        # Capture the results of model.evaluate in rate_results to produce graphics
        results = model.evaluate(test_dataset,batch_size = BATCH_SIZE)
        rate_results[learning_rate]=(results,history,timetaken)
    # After all learning rates have had models generated we pass all the results to the graphics function to produce graphcis that explore how learning rate affects accuracy and loss.
    learning_rate_graphics(rate_results)

def Task8():
    """
    With the best learning rate that you found in the previous task (0.1), add a non-zero momentum to the training with the SGD optimizer (consdier 3 values for momentum)
    Report how the results change.

    """
    # As usual clear the console and initialise parameters.
    cls()
    path = Get_Path()
    BATCH_SIZE = 32
    IMG_SIZE=(160,160)
    IMG_SHAPE = IMG_SIZE + (3,)

    initial_epochs=10    
    learning_rates = 0.1
    # Initialise array of learning rates to loop over.
    momentums = [0.1,0.5,0.9]
    nesterov = False
    train_dataset,validation_dataset,test_dataset = Prepare_Data(path,BATCH_SIZE,IMG_SIZE)
    # Loop over momentum values.
    for momentum in momentums:
        # download a fresh model
        model = Prepare_Model(train_dataset,IMG_SHAPE)
        
        # compile model as done in previous questions
        model,history,timetaken = Compile_Model(model,train_dataset,validation_dataset,initial_epochs=initial_epochs,learning_rate=learning_rate,momentum=momentum,nesterov=nesterov)

        # Show results to do with runtime, test loss and test accuracy
        Show_Results(model,history,test_dataset,timetaken,BATCH_SIZE)

def Task9():
    """
    Prepare you training, validation and test sets. Those are based on {F(x),t}
    """

    # As usual prepare parameters
    cls()
    path = Get_Path()
    BATCH_SIZE = 32
    IMG_SIZE=(160,160)
    IMG_SHAPE = IMG_SIZE + (3,)

    initial_epochs=10
    learning_rate = 0.01
    
    momentum=0.0
    nesterov=False
    
    rate_results = {}
    # train_dataset,validation_dataset,test_dataset = Prepare_Data(path,BATCH_SIZE,IMG_SIZE)

    train_dataset = tf.keras.utils.image_dataset_from_directory(path,seed=seed,shuffle=True,batch_size=BATCH_SIZE,image_size=IMG_SIZE)
    validation_dataset = tf.keras.utils.image_dataset_from_directory(path,seed=seed,shuffle=True,batch_size=BATCH_SIZE,image_size=IMG_SIZE)
    
    target_labels=[label for image, batch in train_dataset for label in batch]

    val_batches = tf.data.experimental.cardinality(validation_dataset)
    test_dataset = validation_dataset.take(val_batches//5)
    validation_dataset=validation_dataset.skip(val_batches//5)
    
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)


 
    #######################################
    # Begin building new model from scratch
    
    # Get a freshly downloaded model.
    model = Get_Model(IMG_SHAPE)

    # Instantiate the layers of the base model
    inputs = tf.keras.Input(shape=IMG_SHAPE)
    data_augmentation = Prepare_Augmentation()   
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    dropout_layer = tf.keras.layers.Dropout(rate = 0.2)
        

    # Preprocess and freeze the base model
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    model.trainable=False
    x = model(x,training=False)
    x = global_average_layer(x)
    outputs = dropout_layer(x)
    
    # Create an intermediate model that will output new datasets F(x)
    intermediate_model = tf.keras.Model(inputs=inputs,outputs=outputs)
    intermediate_model.summary()

    # Output predictions from the intermediate model as tensors to be turned into datasets.
    F_x_train = intermediate_model.predict(train_dataset)
    print(F_x_train.shape)
    F_x_validation = intermediate_model.predict(validation_dataset)
    F_x_test = intermediate_model.predict(test_dataset)
    # print(f"\n\n\nThe shape of the data is: {F_x_train.shape} and the shape of the validation set is now {F_x_validation.shape}\n")
    
    
    # Create the new model inputs from the output of the previous model.
    # F_x_train = tf.data.Dataset.from_tensor_slices(F_x_train[1:])
    Ninputs = tf.keras.Input(shape=F_x_train.shape)
    
    # An experiment to make a dataset that model.fit would accept - it was unsuccessful.
    # Even attempting to combine this with class labels didn't yield any fruit

    # Outputs classification layer with 5 neurons and softmax activation for non-binary class problems.    
    classification_layer = tf.keras.layers.Dense(5,activation='softmax')
    
    # Assemble model.
    Noutputs = classification_layer(Ninputs)

    new_model = tf.keras.Model(inputs=Ninputs,outputs=Noutputs)
    

    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate,momentum=momentum,nesterov=nesterov)
    new_model.compile(optimizer=optimizer,loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),metrics=["accuracy"])
    new_model.summary()
    timetaken=my_training_callback()    
    history = new_model.fit(x=F_x_train,epochs=initial_epochs,callbacks=[timetaken],use_multiprocessing=True)
    ##########################################################################################################################################   

def Task10():
    """
    Perform Task 8 with the new datasets/model.
    Task 8:
    With the best learning rate that you found in the previous task (0.1), add a non-zero momentum to the training with the SGD optimizer (consdier 3 values for momentum)
    Report how the results change.
    """
    # As usual clear the console and initialise parameters.
    cls()
    path = Get_Path()
    BATCH_SIZE = 32
    IMG_SIZE=(160,160)
    IMG_SHAPE = IMG_SIZE + (3,)

    initial_epochs=10    
    learning_rates = [0.1]
    # Initialise array of learning rates to loop over.
    momentums = [0.1,0.5,0.9]
    nesterov = False
    train_dataset,validation_dataset,test_dataset = Prepare_Data(path,BATCH_SIZE,IMG_SIZE)
    # Loop over momentum values.
    for momentum in momentums:
        # download a fresh model
            #######################################
        # Begin building new model from scratch
        
        # Get a freshly downloaded model.
        model = Get_Model(IMG_SHAPE)

        # Instantiate the layers of the base model
        inputs = tf.keras.Input(shape=IMG_SHAPE)
        data_augmentation = Prepare_Augmentation()   
        preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        dropout_layer = tf.keras.layers.Dropout(rate = 0.2)
            

        # Preprocess and freeze the base model
        x = data_augmentation(inputs)
        x = preprocess_input(x)
        model.trainable=False
        x = model(x,training=False)
        x = global_average_layer(x)
        outputs = dropout_layer(x)
        
        # Create an intermediate model that will output new datasets F(x)
        intermediate_model = tf.keras.Model(inputs=inputs,outputs=outputs)
        
        # Output predictions from the intermediate model as tensors to be turned into datasets.
        F_x_train = intermediate_model.predict(train_dataset)
        F_x_validation = intermediate_model.predict(validation_dataset)
        F_x_test = intermediate_model.predict(test_dataset)
                    
        # Create the new model
        Ninputs = tf.keras.Input(shape=F_x_train.shape)
        F_x_train = tf.data.Dataset.from_tensor_slices(F_x_train)
        # Outputs classification layer with 5 neurons and softmax activation for non-binary class problems.    
        classification_layer = tf.keras.layers.Dense(5,activation='softmax')
        
        # Assemble model.
        Noutputs = classification_layer(Ninputs)
        new_model = tf.keras.Model(inputs=Ninputs,outputs=Noutputs)
        
        # Compile the model with loss function and optimizer.
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate,momentum=momentum,nesterov=nesterov)
        new_model.compile(optimizer=optimizer,loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),metrics=["accuracy"])
        
        # Create objects to harvest metrics during training
        timetaken=my_training_callback()    

        # Train the model
        history = new_model.fit(x=F_x_train,epochs=initial_epochs,callbacks=[timetaken],use_multiprocessing=True)

        # Show results to do with runtime, test loss and test accuracy
        Show_Results(model,history,test_dataset,timetaken,BATCH_SIZE)
 
if __name__ == "__main__":
    global seed
    seed = 123
    
    ## Run only 1 function at a time - each one is self contained. The start of each function initialises any parameters needed - configure them as you will.

    # Task1()
    # Task2()
    # Task3()
    # Task4()
    # Task5()
    # Task6()
    # Task7()
    # Task8()

    # Incomplete - could not figure out how to make a dataset that model.fit would accept.
    # Can't understand why if I set my inputs to have the same shape as the outputs of the previous model why those same inputs are not accepted by new_model.fit()
    # ValueError: Input 0 of layer "model_1" is incompatible with the layer: expected shape=(None, 1000, 1280), found shape=(None, 1280)
    Task9()

    # Incomplete - just needs results from 9.
    # Task10()