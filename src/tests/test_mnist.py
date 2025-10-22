import ctypes
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from torchvision import datasets, transforms
import logging
import torchvision.models as models

if not torch.cuda.is_available():
    logging.error("CUDA environment unavailable.")
    exit(1)
BATCH_SIZE=512
EPOCHS=20
device = torch.device("cuda")

training_data = datasets.MNIST('mnist_data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]))
test_data = datasets.MNIST('mnist_data', train=False, download=True, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X.to(device))
        loss = loss_fn(pred, y.to(device))

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    b = True
    with torch.no_grad():
        for X, y in dataloader:
            if b:
                b = False
            pred = model(X.to(device))
            cuda_y = y.to(device)
            test_loss += loss_fn(pred, cuda_y).item()
            correct += (pred.argmax(1) == cuda_y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
# traning
learning_rate = 1e-3
batch_size = 64
epochs = 5
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 10
if not os.path.exists('mnist_model_weights.pth'):
    print("mnist_model_weights.pth not found, start training...")
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)
    print("Done training! saving to mnist_model_weights.pth...")
    torch.save(model.state_dict(), 'mnist_model_weights.pth')
else:
    print("mnist_model_weights.pth founded, loading...")
    model.load_state_dict(torch.load('mnist_model_weights.pth', weights_only=True))

X = torch.rand(1, 28, 28, device=device)
cuda_pointer = X.data_ptr()

print("Cuda pointer: " + str(cuda_pointer))
default_stream = torch.cuda.default_stream()
print(f"Default CUDA Stream: {default_stream}")

test_mnist = ctypes.CDLL(Path(__file__).parent / "test_mnist.pyd")
test_mnist.init(__file__, "vk")
while not test_mnist.should_close():
    if test_mnist.update_frame(ctypes.c_uint64(cuda_pointer)):
        with torch.no_grad():
            print(X)
            logits = model(X)
            pred_probab = nn.Softmax(dim=1)(logits)
            y_pred = pred_probab.argmax(1)
            print(f"Predicted class: {y_pred}")
        
test_mnist.dispose()
