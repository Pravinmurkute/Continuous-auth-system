from facenet_pytorch import InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Load combined dataset
data_transforms = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder("combined_face_dataset", transform=data_transforms)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Load FaceNet model
model = InceptionResnetV1(pretrained='vggface2').train()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train model
for epoch in range(10):
    for images, labels in dataloader:
        embeddings = model(images)
        loss = torch.nn.CrossEntropyLoss()(embeddings, labels)
        loss.backward()
        optimizer.step()

print("Model retrained successfully!")
