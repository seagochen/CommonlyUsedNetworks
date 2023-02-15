import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from yolo import YOLOv3
from yolo_loss import YOLOv3Loss
from yolo_dataset import YOLODataset
from utils import *

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model parameters
    img_size = 416
    num_classes = 80
    anchors = [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45), (59, 119), (116, 90), (156, 198), (373, 326)]
    model = YOLOv3(num_classes=num_classes, anchors=anchors).to(device)
    criterion = YOLOv3Loss(anchors=anchors, num_classes=num_classes, img_size=img_size)

    # Data parameters
    data_path = '/path/to/data'
    batch_size = 16
    train_dataset = YOLODataset(data_path, set_type='train', img_size=img_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)

    # Training parameters
    lr = 0.001
    momentum = 0.9
    weight_decay = 0.0005
    num_epochs = 50
    steps_per_epoch = len(train_loader)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for step, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            targets = [target.to(device) for target in targets]
            optimizer.zero_grad()
            predictions = model(images)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{step+1}/{steps_per_epoch}], Loss: {loss.item():.4f}')
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Epoch loss: {epoch_loss / steps_per_epoch:.4f}')

if __name__ == '__main__':
    main()
