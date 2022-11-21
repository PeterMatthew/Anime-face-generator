import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

def get_loader(image_size, batch_size, num_workers, path):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = datasets.ImageFolder(path, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    return dataloader, len(dataset)
