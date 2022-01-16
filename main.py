from utils.GetDataSet import GetDataSet
from utils.DataLoader import DataLoader
from utils.Train import Train
from utils.Test import Test_Eval

from models.model_1 import CNN_1
from models.model_2 import CNN_2

import torch
import torch.optim as optim
import sys
import argparse

if __name__ == "__main__":

    input_size  = 224*224*3   # images are 224*224 pixels and has 3 channels because of RGB color
    output_size = 2      # there are 2 classes---Cat and dog

    global device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Device: {}'.format(device))

    if torch.cuda.is_available():
        print('GPU Model: {}'.format(torch.cuda.get_device_name(0)))
        torch.cuda.device_count()

    parser = argparse.ArgumentParser(description='Cat-Dog Classifier Using CNN')

    parser.add_argument('-d', '--download_dataset', nargs='?', default='no', help='Download Dataset to Path Data') 
    parser.add_argument('-b', '--batch_size', nargs='?', default='64', help='The Value of Batch Size') 
    parser.add_argument('-n', '--num_workers', nargs='?', default='0', help='The Value of Workers') 
    parser.add_argument('-m', '--model', help='The CNN Model Architecture', required=True) 
    parser.add_argument('-t', '--mode', help='Train or Test', required=True) 
    parser.add_argument('-p', '--img_name', help='Name of test image in Data/TestOne') 

    args = parser.parse_args()

    if (args.download_dataset == 'yes'):
        url = 'https://bit.ly/3GuMVI9'
        folder_name = 'Data'
        Get_Dataset= GetDataSet(url, folder_name)
        Get_Dataset.run()

    batch_size= int(args.batch_size)
    num_workers= int(args.num_workers)

    Get_Train_Test= DataLoader(batch_size, num_workers)
    train_loader, test_loader = Get_Train_Test.run()

    if(args.model == 'model_1'):
        # Training settings  for model 1 
        n_features = 2 # hyperparameter
        model_cnn = CNN_1(input_size, n_features, output_size)
        optimizer = optim.SGD(model_cnn.parameters(), lr=0.01, momentum=0.5)
    elif(args.model == 'model_2'):
        n_features = 6 # hyperparameter
        model_cnn = CNN_2(input_size, n_features, output_size)
        optimizer = optim.SGD(model_cnn.parameters(), lr=0.01, momentum=0.5)
    else:
        print("The model not defined")
        exit(-1)

    if(args.mode == 'train'):
        t = Train(train_loader, optimizer, model_cnn)
        e = Test_Eval(model_cnn)
        for epoch in range(0, 1):
            t.train(epoch)
            t.plot_loss()
            print("Traning Done .... \n")
            print("Testing Result is: ")
            e.test(test_loader)
    elif(args.mode == 'test'):
        e = Test_Eval(model_cnn)
        if(args.img_name):
            path_img_= 'Data/TestOne/'+str(args.img_name)
            e.test_single_image(path_img_)
        else:
            print("No Image Name Exists ..")
