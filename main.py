import torch.optim as optim
from get_dataset import GetDataSet
from DataLoader import DataLoader
from model_1 import CNN_1
from Train import Train
from Test import Test_Eval

input_size  = 224*224*3   # images are 224*224 pixels and has 3 channels because of RGB color
output_size = 2      # there are 2 classes---Cat and dog
n_features = 2 # hyperparameter


url = 'https://bit.ly/3GuMVI9'
folder_name = 'Main_Folder_Dataset'
batch_size= 64
num_workers= 0

#Get_Dataset= GetDataSet(url, folder_name)
#Get_Dataset.run()

Get_Train_Test= DataLoader(batch_size, num_workers)
train_loader, test_loader = Get_Train_Test.run()


# Training settings  for model 1
model_cnn1 = CNN_1(input_size, n_features, output_size)
optimizer = optim.SGD(model_cnn1.parameters(), lr=0.01, momentum=0.5)

t = Train(train_loader, optimizer, model_cnn1)
e = Test_Eval(test_loader, model_cnn1)

for epoch in range(0, 1):
    t.train(epoch)
    print("Traning Done .... \n")
    print("Testing Result is: ")
    e.test()

'''
# Training settings for model 2
n_features = 6 # hyperparameter
model_cnn2 = CNN_2(input_size, n_features, output_size)
optimizer = optim.SGD(model_cnn2.parameters(), lr=0.01, momentum=0.5)


for epoch in range(0, 1):
    train(epoch, model_cnn2)
    test(model_cnn2)
'''