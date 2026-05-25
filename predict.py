import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn

device = torch.device("cpu")

# SAME MODEL ARCHITECTURE
class Model_0(nn.Module):
    def __init__(self,n):
        super().__init__()

        self.layer_1=nn.Sequential(
            nn.Conv2d(3,n,3,1,1),
            nn.ReLU(),
            nn.Conv2d(n,n,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.layer_2=nn.Sequential(
            nn.Conv2d(n,n,3,1,1),
            nn.ReLU(),
            nn.Conv2d(n,n,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.layer_3=nn.Sequential(
            nn.Conv2d(n,n,3,1,1),
            nn.ReLU(),
            nn.Conv2d(n,n,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.layer_4=nn.Sequential(
            nn.Flatten(),
            nn.Linear(n*16*16,3)
        )

    def forward(self,x):

        x=self.layer_1(x)
        x=self.layer_2(x)
        x=self.layer_3(x)
        x=self.layer_4(x)

        return x


# Load class names
class_names = ["pizza","steak","sushi"]

# Transform
transform=transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485,0.456,0.406],
            std=[0.229,0.224,0.225]
        )
])


# Load model
model=Model_0(32)

model.load_state_dict(
    torch.load(
        "model.pth",
        map_location=torch.device("cpu")
    )
)

model=model.to(device)

model.eval()


def predict_image(image_path):

    img=Image.open(image_path).convert("RGB")

    img=transform(img)

    img=img.unsqueeze(0).to(device)

    with torch.inference_mode():

        output=model(img)

        prediction=torch.softmax(
            output,
            dim=1
        )

        confidence=torch.max(
            prediction
        ).item()

        predicted=prediction.argmax(
            dim=1
        )

    return {
        "class":class_names[
            predicted.item()
        ],
        "confidence":round(
            confidence*100,
            2
        )
    }
