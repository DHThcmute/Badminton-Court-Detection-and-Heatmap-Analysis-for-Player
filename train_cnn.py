import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from PIL import Image

class BadmintonFeetDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_name = self.annotations.iloc[index, 0]
        img_path = os.path.join(self.root_dir, img_name)
        
        image = Image.open(img_path).convert("RGB")
        
        x_coord = float(self.annotations.iloc[index, 1])
        y_coord = float(self.annotations.iloc[index, 2])
        labels = torch.tensor([x_coord, y_coord], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, labels

CSV_FILE = 'dataset/labels.csv'
IMG_DIR = 'dataset/images'
CHECKPOINT_FILE = 'feet_detector_checkpoint.pth'
LOSS_LOG_FILE = 'loss_history.txt'

BATCH_SIZE = 16
LEARNING_RATE = 0.001
TARGET_EPOCHS = 500 

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])

    dataset = BadmintonFeetDataset(csv_file=CSV_FILE, root_dir=IMG_DIR, transform=transform)
    
    # --- PHÂN CHIA DỮ LIỆU ---
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42) # Cố định seed để chia giống nhau ở các lần chạy
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    start_epoch = 0
    if os.path.exists(CHECKPOINT_FILE):
        checkpoint = torch.load(CHECKPOINT_FILE, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1 
    
    if start_epoch >= TARGET_EPOCHS:
        return

    print(f"Bắt đầu Training. Kích thước dữ liệu: Train={train_size}, Val={val_size}, Test={test_size}")
    
    for epoch in range(start_epoch, TARGET_EPOCHS):
        #training phase
        model.train()
        running_train_loss = 0.0
        
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)

            predictions = model(images)
            loss = criterion(predictions, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            
        epoch_train_loss = running_train_loss / len(train_loader)
        
        #validation
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                predictions = model(images)
                loss = criterion(predictions, targets)
                running_val_loss += loss.item()
                
        epoch_val_loss = running_val_loss / len(val_loader)
        
        formatted_train = f"{epoch_train_loss:.10f}".rstrip('0').rstrip('.') if '.' in f"{epoch_train_loss:.10f}" else f"{epoch_train_loss:.10f}"
        formatted_val = f"{epoch_val_loss:.10f}".rstrip('0').rstrip('.') if '.' in f"{epoch_val_loss:.10f}" else f"{epoch_val_loss:.10f}"
        
        print(f"Epoch [{epoch+1}/{TARGET_EPOCHS}] - Train Loss: {formatted_train} | Val Loss: {formatted_val}")

        with open(LOSS_LOG_FILE, "a") as f:
            f.write(f"Train:{formatted_train} | Val:{formatted_val}\n")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': epoch_train_loss,
            'val_loss': epoch_val_loss
        }, CHECKPOINT_FILE)

if __name__ == "__main__":
    main()