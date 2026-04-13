import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import time

# =====================================================================
# 1. USTAWIENIA TWOJEGO EKSPERYMENTU
# =====================================================================
DATA_DIR = './testy_nzal/FunnyBirds/train' # Ścieżka do Twoich wygenerowanych zdjęć
NUM_CLASSES = 4                                # Ilość klas z Twojego frameworka
BATCH_SIZE = 8                                 # Po ile zdjęć naraz model ma analizować (8 to bezpieczne dla 8GB RAM)
EPOCHS = 5                                     # Ile razy model ma obejrzeć cały zbiór

# Sprawdzamy, czy masz dostępną kartę graficzną (GPU), czy liczymy na procesorze (CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Rozpoczynam pracę na: {device}")

# =====================================================================
# 2. PRZYGOTOWANIE ZDJĘĆ DLA SIECI NEURONOWEJ
# =====================================================================
# Model ResNet wymaga zdjęć w rozmiarze 224x224 i specyficznej normalizacji kolorów
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# PyTorch to magia: sam widzi foldery "0", "1", "2" i traktuje je jako etykiety!
image_dataset = datasets.ImageFolder(root=DATA_DIR, transform=data_transforms)
dataloader = DataLoader(image_dataset, batch_size=BATCH_SIZE, shuffle=True)

dataset_size = len(image_dataset)
print(f"Znaleziono {dataset_size} ptaków do treningu.")

# =====================================================================
# 3. BUDOWA MODELU RESNET
# =====================================================================
print("Pobieranie mózgu sztucznej inteligencji (ResNet-18)...")
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# ResNet domyślnie rozpoznaje 1000 klas. Odcinamy mu starą "głowę" i przypinamy nową, na Twoje 3 klasy:
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, NUM_CLASSES)

model = model.to(device)

# Ustawiamy funkcję błędu i optymalizator (algorytm, który uczy sieć)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# =====================================================================
# 4. GŁÓWNA PĘTLA TRENINGOWA
# =====================================================================
print("Rozpoczynamy trening...")

for epoch in range(EPOCHS):
    start_time = time.time()
    model.train() # Ustawiamy model w tryb uczenia
    
    running_loss = 0.0
    running_corrects = 0
    
    # Przechodzimy przez wszystkie zdjęcia w paczkach (batchach)
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()      # Zerujemy pamięć błędów
        
        outputs = model(inputs)    # Model zgaduje klasę ptaka
        _, preds = torch.max(outputs, 1) # Wyciągamy ostateczny werdykt (0, 1 lub 2)
        loss = criterion(outputs, labels) # Obliczamy, jak bardzo się pomylił
        
        loss.backward()            # Model analizuje swój błąd (wsteczna propagacja)
        optimizer.step()           # Model poprawia swoje "zwoje mózgowe" (aktualizacja wag)
        
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        
    epoch_loss = running_loss / dataset_size
    epoch_acc = running_corrects.double() / dataset_size
    end_time = time.time()
    
    print(f'Epoka {epoch+1}/{EPOCHS} | '
          f'Czas: {end_time - start_time:.0f}s | '
          f'Błąd (Loss): {epoch_loss:.4f} | '
          f'Skuteczność (Acc): {epoch_acc:.4f}')

print("Trening zakończony!")

# Zapisujemy wytrenowany "mózg" do pliku, żeby użyć go później do testów!
torch.save(model.state_dict(), 'wytrenowany_resnet.pth')
print("Model zapisany jako 'wytrenowany_resnet.pth'")