import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn as nn

class Train():
    def __init__(self, train_loader, test_loader, optimizer, model):
        self._train_loader = train_loader
        self._test_loader = test_loader
        self._model = model
        self._optimizer = optimizer
        self._losses = []

    def train(self, epoch, no):
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
            self._losses.append(float(loss.item()))    
            torch.save(self._model.state_dict(), 'Output/dogcatwights'+no+'.pth')

    def train_new(self, epochs, device):
        traininglosses = []
        testinglosses = []
        testaccuracy = []
        totalsteps = []
        steps = 0
        running_loss = 0
        print_every = 5
        criterion = nn.NLLLoss()
        for epoch in range(epochs):
            for inputs, labels in self._train_loader:
                steps += 1
                inputs, labels = inputs.to(device), labels.to(device)
                
                self._optimizer.zero_grad()
                
                logps = self._model.forward(inputs)
                loss = criterion(logps, labels)
                loss.backward()
                self._optimizer.step()

                running_loss += loss.item()
                
                if steps % print_every == 0:
                    test_loss = 0
                    accuracy = 0
                    self._model.eval()
                    with torch.no_grad():
                        for inputs, labels in self._test_loader:
                            inputs, labels = inputs.to(device), labels.to(device)
                            logps = self._model.forward(inputs)
                            batch_loss = criterion(logps, labels)
                            
                            test_loss += batch_loss.item()
                            
                            # Calculate accuracy
                            ps = torch.exp(logps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
                    traininglosses.append(running_loss/print_every)
                    testinglosses.append(test_loss/len(self._test_loader))
                    testaccuracy.append(accuracy/len(self._test_loader))
                    totalsteps.append(steps)
                    print(f"Device {device}.."
                        f"Epoch {epoch+1}/{epochs}.. "
                        f"Step {steps}.. "
                        f"Train loss: {running_loss/print_every:.5f}.. "
                        f"Test loss: {test_loss/len(self._test_loader):.5f}.. "
                        f"Test accuracy: {accuracy/len(self._test_loader):.5f}")
                    running_loss = 0
                    self._model.train()
            torch.save(self._model.state_dict(), 'Output/dogcatwights_model32.pth')
                
    def plot_loss(self):
        
        plt.plot(self._losses)
        plt.legend(['loss'])
        plt.savefig('Model.png')