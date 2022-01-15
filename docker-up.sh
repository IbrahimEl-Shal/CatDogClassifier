#!/bin/bash

echo "Run The Docker Image to Train, and Test CNN CatDogClassifier"


sudo docker build --tag docker-mlops .

sudo docker run -it --rm --name CatDogClassifier \
--mount type=bind,source=/home/ibrahimml/CatDogClassifier/Data,target=/src/Data \
--mount type=bind,source=/home/ibrahimml/CatDogClassifier/Output,target=/src/Output,bind-propagation=rslave \
docker-mlops
