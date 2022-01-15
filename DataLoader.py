import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

class DataLoader():
    def __init__(self ,batch_size:int, num_workers:int):
        ''' define training and test data directories '''
        self._train_dir = 'Main_Folder_Dataset/Dataset/training_set/'
        self._test_dir = 'Main_Folder_Dataset/Dataset/test_set/'
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._image_size = (224, 224)
        self._mean = [0.485, 0.456, 0.406]
        self._std  = [0.229, 0.224, 0.225]
        
    def create_transformers(self):
        train_transform = transforms.Compose([transforms.Resize(self._image_size), 
                                              transforms.ToTensor(), 
                                              transforms.Normalize(self._mean, self._std)])
        test_transforms = transforms.Compose([transforms.Resize(self._image_size), 
                                              transforms.ToTensor(), 
                                              transforms.Normalize(self._mean, self._std)])
        return train_transform, test_transforms

    def data_loader(self, train_transform, test_transforms):
        train_dataset = datasets.ImageFolder(root=self._train_dir, transform=train_transform)
        dev_dataset = datasets.ImageFolder(root=self._test_dir, transform=test_transforms)
        train_loader = torch.utils.data.DataLoader(train_dataset, self._batch_size, shuffle=True, num_workers=self._num_workers)
        test_loader = torch.utils.data.DataLoader(dev_dataset, 1, shuffle=False)
        return train_loader, test_loader
    
    def run(self):
        train_transform, test_transforms = self.create_transformers()
        train_loader, test_loader = self.data_loader(train_transform, test_transforms)
        return train_loader, test_loader

if __name__=="__main__":
    Get_Dataset= DataLoader(batch_size= 64, num_workers=0)
    train_loader, test_loader = Get_Dataset.run()
