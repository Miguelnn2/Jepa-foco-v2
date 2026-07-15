import torch, time, os
from engine import JEPAEngine
from PIL import Image
from torchvision import transforms

# Cargar configuración
model = JEPAEngine(num_clases=2) # Ajustar según tus clases
model.load_state_dict(torch.load("modelo.pth"))
model.eval()
transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])

print("\n=== LABORATORIO DE PREDICCIÓN SACÁDICA ===")
for img_name in os.listdir('data/test'):
    path = f"data/test/{img_name}"
    img = transform(Image.open(path).convert('RGB')).unsqueeze(0)
    
    t0 = time.perf_counter()
    with torch.no_grad():
        out, idxs = model(img)
        prob = torch.softmax(out, dim=1)
        clase = torch.argmax(out, dim=1).item()
    t1 = time.perf_counter()
    
    print(f"\nImagen: {img_name}")
    print(f" -> Predicción: {['GATO', 'PERRO'][clase]} ({prob[0][clase]*100:.2f}%)")
    print(f" -> Foco: {[f'({i//14},{i%14})' for i in idxs.squeeze().tolist()]}")
    print(f" -> Latencia Inferencia: {(t1-t0)*1000:.2f} ms")
