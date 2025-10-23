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
import sys

if not torch.cuda.is_available():
    logging.error("CUDA environment unavailable.")
    exit(1)
device = torch.device("cuda")

training_data = datasets.MNIST(
    "mnist_data",
    train=True,
    download=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    ),
)
test_data = datasets.MNIST(
    "mnist_data",
    train=False,
    download=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    ),
)
batch_size = 64
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

normalize_func = transforms.Normalize((0.1307,), (0.3081,))


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5) # Input channel 1 (grayscale), 10 output channels, 5x5 kernel
        self.pool = nn.MaxPool2d(2, 2) # 2x2 max pooling
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5) # 10 input channels, 20 output channels, 5x5 kernel
        self.fc1 = nn.Linear(320, 50) # Adjust input size based on output of conv/pool layers
        self.fc2 = nn.Linear(50, 10) # 10 output classes for digits 0-9

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 320) # Flatten the tensor for the fully connected layer
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = NeuralNetwork().to(device)


def train_loop(dataloader, model: NeuralNetwork, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        inputs = X.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, y.to(device))

        # Backpropagation
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model: NeuralNetwork, loss_fn):
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
            correct += (pred.argmax(1) == cuda_y).float().sum().item()

    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )

def get_first_test_data(dataloader):
    with torch.no_grad():
        for X, y in dataloader:
            return X.to(device)

# traning
learning_rate = 1e-4
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

epochs = 5
if not os.path.exists("mnist_model_weights.pth"):
    print("mnist_model_weights.pth not found, start training...")
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)
    print("Done training! saving to mnist_model_weights.pth...")
    torch.save(model.state_dict(), "mnist_model_weights.pth")
else:
    print("mnist_model_weights.pth founded, loading...")
    model.load_state_dict(torch.load("mnist_model_weights.pth", weights_only=True))

X = torch.rand(1, 28, 28, device=device)
cuda_pointer = X.data_ptr()
test_data = get_first_test_data(test_dataloader)
test_data_ptr = test_data.data_ptr()

print("Cuda pointer: " + str(cuda_pointer))
default_stream = torch.cuda.default_stream()
print(f"Default CUDA Stream: {default_stream}")

test_mnist = ctypes.CDLL(Path(__file__).parent / "test_mnist.pyd")
backend_name = "vk"
if len(sys.argv) > 1:
    backend_name = sys.argv[1]
test_mnist.init(__file__, backend_name)

while not test_mnist.should_close():
    if test_mnist.update_frame(ctypes.c_uint64(cuda_pointer), ctypes.c_uint64(test_data_ptr)):
        with torch.no_grad():
            S = normalize_func(X)
            logits = model(S)
            y_pred = logits.cpu().numpy()[-1].argmax()
            print(f"Predicted class: {y_pred}")
    test_data_ptr = 0
test_mnist.dispose()
