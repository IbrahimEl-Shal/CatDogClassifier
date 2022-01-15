import torch
import torch.nn.functional as F

class Train():
    def __init__(self, train_loader, optimizer, model):
        self._train_loader = train_loader
        self._model = model
        self._optimizer = optimizer

    def train(self, epoch):
        self._model.train()
        for batch_idx, (data, target) in enumerate(self._train_loader):
            self._optimizer.zero_grad()
            output = self._model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            self._optimizer.step()
            if (batch_idx % 10 and  batch_idx >5):
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self._train_loader.dataset),
                    100. * batch_idx / len(self._train_loader), loss.item()))
                
                torch.save(self._model.state_dict(), 'Output/dogcatwights.pth')
