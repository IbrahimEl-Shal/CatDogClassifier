### CatDogClassifier using CNN, Pytorch, and Docker

Building a simple image recognition tool using CNN and the Pytorch framework that classifies whether the image is of a dog or a cat. The main objective is to create a simple cat or dog image classifier using CNN, then Dockerize the code and environment.

## Repository Structure

```
CatDogClassifier
│
├── Data
│   ├── test_set
│   │      └── cats
│   │      └── dogs
│   ├── training_set
│   │      └── cats
│   │      └── dogs
│   └── TestOne
│   │      └── single
│
├── models
│   └── model_no.py
│
├── Output
│   └── trained_saved_model.pth
│
├── utils
│   └── DataLoader.py
│   └── GetDataSet.py
│   └── Test.py
│   └── Train.py
│  
├── main.py
├── Dockerfile
├── docker-up.sh
├── docker-down.sh
├── requirements.txt
└── readme.md
```

- `Data/`: directory containing the data used in this project.
- `models/`: directory to build the architecture of CNN models.
- `Output/`: directory to save the latest weights of the trained model.
- `utils`: If the dataset does not already exist, download it, load it as a tensor, test, and train the CatDogClassifier model.
- `main.py`: The main python file that takes the main arguments to run the project, whether the mode is to train or test,
- `docker-up.sh`: Build the Docker image for the project by installing the main requirements and running the Docker container that mounts a system directory containing a dataset to /Data and/Outputs to get the checkpoint.
- `docker-down.sh`: Stop and Remove docker container.       
- `readme.md`: Instructions and links related to this Repository.

[//]: # (Image References)

[image1]: ./Examples/cat.1.jpg "cat1"
[image3]: ./Examples/dog.23.jpg "dog1"

### Plan of Project

1. Obtaining the dataset.
2. Importing libraries and DL framework.
3. Constructing CNN.
4. Full Connection.
5. Network Training and Evaluation: Steps of construction and train the model.
6. Testing: Test a random image.

### Data Set Summary & Exploration
The data can be downloaded in the kaggle website using this link https://www.kaggle.com/tongpython/cat-and-dog. 

#### 1. The statistical information about the dataset was calculated using numpy
* The size of the training set is 8005 images.
* The size of the testing set is 2023 images.
* The size of the TestOne set is 1 image, "used to test your single image." 
* The shape of the images is not the same size.
* The number of unique classes/labels in the data set is 2 (cats and dogs).
#### 2. Include an exploratory visualization of the dataset.

At first, I plotted some random images from the training set and here it is:

![alt text][image1]
![alt text][image3]
### The Model Architecture

#### 1. Preprocessing
* All images were rezied to 244x244.
* All the images were normalized using the mean = [0.485, 0.456, 0.406] and standard deviation = [0.229, 0.224, 0.225].

#### 2. Models
The recommended model in the model directory called model_2 is structured by four hidden layers, with a kernel size of 5, and a feature map expanded to 128 followed by two fully connected layers. The rectified-linear activation function (relu) was used over data. The last layer for predictions is log_softmax. 

[image5]: ./Model_1.png "model"
![alt text][image5]

### Installations

There are two types of run the project.

#### 1. Locally

Create a new environemt and run the following command in your terminal
```
virtualenv --python=python3.7 envcatdog
source envcatdog/bin/activate
pip install -r requirement.txt
```
#### Usage
To execute the project, firstly the arguments are 
* -d: (optional) if you need download the dataset, set for example "-d yes", default is "no".
* -b: (optional) if you need set the number of batch size, set for example "-b 128", default is "64".
* -n: (optional) if you need set the number of workers, set for example "-n 3", default is "0".
* -m: (required) The CNN Model Architecture in models directory.
* -t: (required) The mode of operation (Test or Train).

To Train the model run
```
python main.py -m model_2 -t train
```
To Test the model on single image run
```
python main.py -m model_2 -t test
```
#### 2. Docker
Using docker, just you neet to change the access permissions of files
```
sudo chmod+x docker-up.sh docker-down.sh 
```
then run the docker up to build the image and run the project container
```
./docker-up.sh 
```
### 
