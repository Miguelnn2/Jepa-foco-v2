import torch, os
from engine import JEPAEngine
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# Configuración
transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
train_data = datasets.ImageFolder('data/train', transform=transform)
loader = DataLoader(train_data, batch_size=1, shuffle=True)
model = JEPAEngine(num_clases=len(train_data.classes))

opt = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

print("[INFO] Iniciando entrenamiento...")
for epoch in range(50):
    loop = tqdm(loader, leave=False)
    for img, label in loop:
        opt.zero_grad()
        out, _ = model(img)
        loss = criterion(out, label)
        loss.backward()
        opt.step()
        loop.set_description(f"Época {epoch+1}")
        loop.set_postfix(loss=loss.item())

torch.save(model.state_dict(), "modelo.pth")
print("\n[ÉXITO] Modelo guardado como 'modelo.pth'")
