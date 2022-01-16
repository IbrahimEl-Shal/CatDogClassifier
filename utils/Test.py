import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image

class Test_Eval():
    def __init__(self, model):
        self.accuracy_list = []
        self._model = model
        self._image_size = (224, 224)
        self._mean = [0.485, 0.456, 0.406]
        self._std  = [0.229, 0.224, 0.225]

    def test(self, test_loader):
        self._model .eval()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            
            output = self._model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss                                                               
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability                                                                 
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        self.accuracy_list.append(accuracy)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            accuracy))

    def old_test_single_image(self):
        single_image = datasets.ImageFolder(root="Data/TestOne",
                                            transform=transforms.Compose([transforms.Resize(self._image_size),
                                                                        transforms.ToTensor(),
                                                                        transforms.Normalize(self._mean, self._std),
                                                                        ]))
        dataloader = torch.utils.data.DataLoader(single_image,
                                                 batch_size=1,
                                                 shuffle=False,
                                                 num_workers=0)

        self._model.load_state_dict(torch.load('Output/dogcatwights.pth'))
        self._model.eval()
        output = self._model(list(dataloader)[0][0])
        _, predicted = torch.max(output, 1)
        if predicted[0].numpy() == 1:
            print("dog")
        else:
            print("cat")

    def image_transform(self, imagepath):
        test_transforms = transforms.Compose([
                                    transforms.Resize(self._image_size), 
                                    transforms.ToTensor(), 
                                    transforms.Normalize(self._mean, self._std)])

        
        image = Image.open(imagepath)
        imagetensor = test_transforms(image)
        return imagetensor

    def test_single_image(self, imagepath):

        self._model.load_state_dict(torch.load('Output/dogcatwights.pth'))
        self._model.eval()
        image = self.image_transform(imagepath)
        image1 = image[None,:,:,:]
        ps=torch.exp(self._model(image1))
        topconf, topclass = ps.topk(1, dim=1)
        if topclass.item() == 1:
            return {'class':'dog','confidence':str(topconf.item())}
        else:
            return {'class':'cat','confidence':str(topconf.item())}
