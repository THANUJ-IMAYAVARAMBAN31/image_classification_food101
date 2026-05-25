import torch
device = torch.device("cpu")
print(f"Using {device} device")

# IMPORTING 
import requests
import zipfile
from pathlib import Path
import os
import random
from PIL import Image
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import timeit
import torch.nn as nn

###HELPER-FUNCTIONS 

#TESTING THE TRANSFORMED IMAGE BY PLOTTING IT
def plot_transformed_images(image_paths, transform, n=3, seed=100):
    """Plots a series of random images from image_paths.

    Will open n image paths from image_paths, transform them
    with transform and plot them side by side.

    Args:
        image_paths (list): List of target image paths.
        transform (PyTorch Transforms): Transforms to apply to images.
        n (int, optional): Number of images to plot. Defaults to 3.
        seed (int, optional): Random seed for the random generator. Defaults to 42.
    """
    random.seed(seed)
    random_image_paths = random.sample(image_paths, k=n)
    for image_path in random_image_paths:
        with Image.open(image_path) as f:
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(f)
            ax[0].set_title(f"Original \nSize: {f.size}")
            ax[0].axis("off")

            # Transform and plot image
            # Note: permute() will change shape of image to suit matplotlib
            # (PyTorch default is [C, H, W] but Matplotlib is [H, W, C])
            transformed_image = transform(f).permute(1, 2, 0)
            ax[1].imshow(transformed_image)
            ax[1].set_title(f"Transformed \nSize: {transformed_image.shape}")
            ax[1].axis("off")

            fig.suptitle(f"Class: {image_path.parent.stem}", fontsize=16)

# writing a function to use training and test loop :
from sklearn.metrics import accuracy_score
def training_loop(model, optimizer, loss_fn):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch in train_data_set:
        X, y = batch
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        preds = y_pred.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_acc = correct / total
    train_loss /= len(train_data_set)
    print("TRAIN ACC :", correct / total)
    print(f"TRAIN LOSS : {train_loss/len(train_data_set)}")
    return train_acc
def testing_loop(model, loss_fn):
    model.eval()
    test_acc_list = []
    test_loss = 0
    correct = 0
    total = 0

    with torch.inference_mode():
        for batch in test_data_set:
          X, y = batch
          X, y = X.to(device), y.to(device)
          y_pred = model(X)
          loss = loss_fn(y_pred, y)
          test_loss += loss.item()
          preds = y_pred.argmax(dim=1)
          correct += (preds == y).sum().item()
          total += y.size(0)
    test_acc = correct / total
    test_loss /= len(test_data_set)
    print("TEST ACC :", correct / total)
    print(f"TEST LOSS : {test_loss/len(test_data_set)}")
    return test_acc
# CALCULATING TIME TAKEN

from timeit import default_timer as timer
def print_train_time(start: float, end: float, device: torch.device = None):
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time

#PLOT ACCURACY CURVES 
def plot_acc_curves(train_acc_list, test_acc_list):
    """
    Plot training and test accuracy curves.

    Args:
        train_acc_list (list): Training accuracy values per epoch
        test_acc_list (list): Test/validation accuracy values per epoch
    """

    epochs = range(1, len(train_acc_list) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_acc_list, label="Train Accuracy")
    plt.plot(epochs, test_acc_list, label="Test Accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Test Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()

### TRANSFORMS 
train_transform = transforms.Compose(
                         [ transforms.Resize((129,128)),
                          transforms.RandomHorizontalFlip(p=0.5),
                          transforms.ToTensor(),
                           transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )]
)

test_transform = transforms.Compose(
                         [ transforms.Resize((128,128)),
                          transforms.ToTensor(),
                           transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )]
)


#DOWNLOADING THE CONTENT 
data_path = Path("data/")
img_path = data_path / "pizza_steak_sushi"
if img_path.is_dir():
  print("THIS DIRECTORY ALREADY EXISTS")
else:
  print("CREATING THIS DIRECTORY")
  img_path.mkdir(parents=True, exist_ok=True)

  # Download pizza, steak, sushi data
  with open(data_path / "pizza_steak_sushi.zip", "wb") as f:
    request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
    print("Downloading pizza, steak, sushi data...")
    f.write(request.content)

    # Unzip pizza, steak, sushi data
  with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref:
    print("Unzipping pizza, steak, sushi data...")
    zip_ref.extractall(img_path)

image_path = list(img_path.glob("*/*/*.jpg"))

# Setup train and testing paths
train_dir = img_path / "train"
test_dir = img_path / "test"

print(f"Train directory: {train_dir}")
print(f"Test directory: {test_dir}\n")

random_image_path = random.choice(image_path)
#  Get image class from path name (the image class is the name of the directory where the image is stored)
image_class = random_image_path.parent.stem
print(f"\nRandom image path: {random_image_path}")
print(f"Random image class: {image_class}")
Image.open(random_image_path)


# PLOTTING RANDOM IMAGES AFTER TRANSFORMING
plot_transformed_images(image_path,
                        transform=train_transform,
                           n=3)


#CONVERTING THE DATA INTO DATASETS AND DATALOADER TO MAKE IT PYTHIN ITER
train_data_set = datasets.ImageFolder(train_dir, transform=train_transform)
test_data_set = datasets.ImageFolder(test_dir, transform=test_transform)

img, label = train_data_set[0]

class_names = train_data_set.classes

# convert into python iter by dataLoader
train_data_set = DataLoader(dataset = train_data_set, batch_size = 32, shuffle = True)
test_data_set = DataLoader(dataset = test_data_set, batch_size = 32, shuffle = False)


#CREATING A MODEL
torch.manual_seed(1)
class Model_0(nn.Module):
  def __init__(self,n):
    super().__init__()
    self.layer_1 = nn.Sequential(
        nn.Conv2d(in_channels = 3, out_channels = n, kernel_size = 3, stride = 1, padding = 1),
        nn.ReLU(),
        nn.Conv2d(in_channels=n, out_channels=n, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )
    self.layer_2 = nn.Sequential(
        nn.Conv2d(in_channels=n, out_channels=n, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=n, out_channels=n, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )
    self.layer_3 = nn.Sequential(
        nn.Conv2d(in_channels=n, out_channels=n, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=n, out_channels=n, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )
    self.layer_4 = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features = n*16*16, out_features = 3)
    )

  def forward(self,x):
    x = self.layer_1(x)
    #print(x.shape)  #- used for testing with dummy variable , dont uncomment in traing or testing phase
    x = self.layer_2(x)
    #print(x.shape)
    x = self.layer_3(x)
    #print(x.shape)
    x = self.layer_4(x)
    #print(x.shape)
    return x

# CREATING A OBJECT FOR MODEL_0
model_0 = Model_0(32).to(device)

#SETTING OPTIMIZER AND LOSS_FUNCTIONS
epochs = 20
train_acc_list = []
test_acc_list = []
optimizer = torch.optim.Adam(params = model_0.parameters(), lr = 0.0001)
loss_fn = nn.CrossEntropyLoss()
start_time = timer()
for epoch in range(epochs):
  if epoch == 17:
    break
  print(f"EPOCH : {epoch}")
  train_acc = training_loop(
        model_0,
        optimizer,
        loss_fn,
    )

  test_acc = testing_loop(
      model_0,
      loss_fn
  )

  train_acc_list.append(train_acc)
  test_acc_list.append(test_acc)

end_time = timer()
print_train_time(start_time, end_time, device)

# Save model weights
torch.save(model_0.state_dict(),"model.pth")

# Save class names
with open("class_names.txt","w") as f:
    for item in class_names:
        f.write(item+"\n")

print("Model saved successfully")