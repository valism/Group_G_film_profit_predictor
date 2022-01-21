# Film profit classification - by GroupG from INST0060

The purpose of this topic is to evaluate the use of 
classification models to predict if movie x will
make more money than movie y. Four models will be compared:

* Logistic Regression
* Fishers Linear Discriminant
* Naive Bayes
* Random Forest

## Setup
* You will need a python runtime environment and a package manager such as pip.
* It is strongly recommended that you also
create a virtual environment for this project.

* You will also need to install all the dependencies that can be found in `requirements.txt`
This can be done using command line or an IDE such as PyCharm.

* Make sure you have the file named `profit_x_y.csv` in the same folder as 
  `evaluate_classifiers.py` and `data_preprocessing.py`.
  * If the `profit_x_y.csv` file is missing, simply run `data_preprocessing.py`.


## Running the application

* Ensure all dependencies have been installed correctly.
* Open a command prompt or your IDE's terminal and activate the runtime
environment that has the installed dependencies (a virtual or system environment).

* Run the file named `evaluate_classifiers.py` to run the experiments.
* You need to parse the appropriate parameters for the code to run.
The format is as follows `python evaluate_classifiers.py <DATA FILE> <COMMAND LINE OPTIONS>.`
This will load the data from `<DATA FILE>` and run the code with
the options selected in `<COMMAND LINE OPTIONS>`.
* Unless you have changed the name, `<DATA FILE>` is .csv file called `profit_x_y.csv`
* In `<COMMAND LINE OPTIONS>` you can select which of the four model you would like to use.
This is done using the `model_names` argument which takes in the initials of the model names,
separated by single spaces. The format is as follows
  * `lr` - Logistic Regression
  * `fld` - Fishers Linear Discriminant
  * `nb` - Naive Bayes
  * `rf` - Random Forest
* The `model_names` argument is optional, excluding it means
all the models will be used.

* For example, if you want to use Logistic Regression 
 and Fishers Linear Discriminant, type the following in the command line:
  * `python evaluate_classifiers.py profit_x_y.csv --model_names lr fld`
* If you want to use all the models, simply use:
  * `python evaluate_classifiers.py profit_x_y.csv`
  
* This should:
  * Run the code with the selected models
  * Print a list of cross validation scores for each model
  * Print the average cross validation score for each model
  * Print the area below ROC curve (AUC) for each model.
  * Plot the ROC curve for each model and save it to a file named `ROC Curves for Classifiers.png`


* The random seed in this project is set to '100' to ensure 
* the results are identical between runs.

