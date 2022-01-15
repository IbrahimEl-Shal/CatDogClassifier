import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets

class Test_Eval():
    def __init__(self, test_loader, model):
        self._model = model
        self._test_loader = test_loader
        self.accuracy_list = []

    def test(self):
        self._model .eval()
        test_loss = 0
        correct = 0
        for data, target in self._test_loader:
            
            output = self._model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss                                                               
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability                                                                 
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

        test_loss /= len(self._test_loader.dataset)
        accuracy = 100. * correct / len(self._test_loader.dataset)
        self.accuracy_list.append(accuracy)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(self._test_loader.dataset),
            accuracy))

    def test_single_image(self):
        single_image = datasets.ImageFolder(root="TestOne",
        transform=transforms.Compose([transforms.Resize([224, 224]),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                      ]))
        dataloader = torch.utils.data.DataLoader(single_image,
                                                 batch_size=1,
                                                 shuffle=False,
                                                 num_workers=0)

        self._model.load_state_dict(torch.load('dogcatwights.pth'))
        self._model.eval()
        output = self._model(list(dataloader)[0][0])
        _, predicted = torch.max(output, 1)
        if predicted[0].numpy() == 1:
            print("dog")
        else:
            print("cat")
