if __name__ == "__main__":
    import torch
    import torch.nn as nn
    import torchvision
    from torchvision import transforms, datasets, models
    from torch.utils.data import DataLoader
    from sklearn.metrics import precision_score, recall_score, f1_score
    import time
    from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
    from tqdm import tqdm

    # ✅ Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🖥️ Device:", device)

    # ✅ Parameters
    BATCH_SIZE = 64 # Adjust as needed
    EPOCHS = 5 # Adjust as needed
    IMG_SIZE = 224
    NUM_WORKERS = 8 # Adjust as needed for your system

    # ✅ Transforms
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
    ])

    # ✅ Datasets
    train_data = datasets.ImageFolder(r"~path/working/data/train", transform=transform)
    val_data = datasets.ImageFolder(r"~path/working/data/val" , transform=transform)
    test_data = datasets.ImageFolder(r"~path/working/data/test" , transform=transform)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # ✅ Load pretrained MobileNetV2
    weights = MobileNet_V2_Weights.DEFAULT  # Use latest pretrained weights
    model = mobilenet_v2(weights=weights)
    model.classifier[1] = nn.Linear(model.last_channel, len(train_data.classes))
    model.to(device)

    # ✅ Load previously trained weights if any
    #model.load_state_dict(torch.load("model-name.pth"))

    # ✅ Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # ✅ Training loop with progress bar
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        # Wrap the DataLoader with tqdm
        train_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]", unit="batch")

        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

            # Optional: update tqdm description dynamically
            train_bar.set_postfix(loss=running_loss / (train_bar.n + 1))

        print(f"📘 Epoch [{epoch+1}/{EPOCHS}] Training Loss: {running_loss / len(train_loader):.4f}")

    # ✅ Evaluation
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())

    # ✅ Metrics
    print("Precision:", precision_score(y_true, y_pred, average='macro'))
    print("Recall:", recall_score(y_true, y_pred, average='macro'))
    print("F1 Score:", f1_score(y_true, y_pred, average='macro'))

    # ✅ Save model
    torch.save(model.state_dict(), "mobilenetv2_plant_disease.pth")#Save in different name if using already trained model
    print("✅ Model saved: mobilenetv2_plant_disease.pth")
