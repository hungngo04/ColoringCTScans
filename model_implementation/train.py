import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader
import pydicom
import numpy as np
from PIL import Image
import torchvision.transforms.functional as F
import random
import torch.nn as nn
import torch.optim as optim
from model import UNet
import matplotlib.pyplot as plt
from tqdm import tqdm

class DICOMDataset(Dataset):
    def __init__(self, root_dirs, transform=None):
        self.transform = transform
        self.image_paths = []
        self.mask_paths = []

        for root_dir in root_dirs:
            original_dir = os.path.join(root_dir, 'original')
            bone_dir = os.path.join(root_dir, 'bone')
            air_dir = os.path.join(root_dir, 'air')
            soft_tissue_dir = os.path.join(root_dir, 'soft_tissue')

            if not os.path.exists(original_dir):
                print(f"Original directory does not exist: {original_dir}")
                continue

            image_files = [f for f in sorted(os.listdir(original_dir)) if not f.startswith('.') and f.endswith(('.png', '.jpg'))]
            for img_file in image_files:
                image_path = os.path.join(original_dir, img_file)

                bone_img_file = img_file.replace('original', 'bone')
                air_img_file = img_file.replace('original', 'air')
                soft_tissue_img_file = img_file.replace('original', 'soft_tissue')

                bone_mask_path = os.path.join(bone_dir, bone_img_file)
                air_mask_path = os.path.join(air_dir, air_img_file)
                soft_tissue_mask_path = os.path.join(soft_tissue_dir, soft_tissue_img_file)

                if os.path.exists(bone_mask_path) and os.path.exists(air_mask_path) and os.path.exists(soft_tissue_mask_path):
                    self.image_paths.append(image_path)
                    self.mask_paths.append({
                        'bone': bone_mask_path,
                        'air': air_mask_path,
                        'soft_tissue': soft_tissue_mask_path
                    })
                else:
                    print(f"Mask files missing for image {img_file}")
                    if not os.path.exists(bone_mask_path):
                        print(f"Missing bone mask: {bone_mask_path}")
                    if not os.path.exists(air_mask_path):
                        print(f"Missing air mask: {air_mask_path}")
                    if not os.path.exists(soft_tissue_mask_path):
                        print(f"Missing soft tissue mask: {soft_tissue_mask_path}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('L')

        mask_paths = self.mask_paths[idx]
        mask_size = (image.size[1], image.size[0])
        num_classes = 4 # bone, air, soft tissue, background

        class_indices = {
            'bone': 1,
            'air': 2,
            'soft_tissue': 3
        }

        masks = np.zeros((num_classes, *mask_size), dtype=np.float32)
        foreground_sum = np.zeros(mask_size, dtype=np.float32)

        for label_name, mask_path in mask_paths.items():
            component_image = Image.open(mask_path).convert('L')
            component_array = np.array(component_image).astype(np.float32) / 255.0

            if label_name == 'bone':
                scale_factor = 5.0
                component_array *= scale_factor
                component_array = np.clip(component_array, 0.0, 1.0)

            class_idx = class_indices[label_name]
            masks[class_idx] = component_array
            foreground_sum += component_array

        background_probability = 1.0 - np.clip(foreground_sum, 0.0, 1.0)
        masks[0] = background_probability

        sum_probs = np.clip(np.sum(masks, axis=0), 1e-6, None)
        masks /= sum_probs

        mask_indices = np.argmax(masks, axis=0).astype(np.int64).copy()

        if self.transform is not None:
            image, mask_indices = self.transform(image, mask_indices)

        image = F.to_tensor(image)
        mask_indices = torch.from_numpy(mask_indices.copy()).long()

        return image, mask_indices

class JointTransform:
    def __init__(self, resize=None, horizontal_flip=False, vertical_flip=False):
        self.resize = resize
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip

    def __call__(self, image, mask):
        if self.resize is not None:
            image = F.resize(image, self.resize)
            mask = Image.fromarray(mask.astype(np.uint8))
            mask = F.resize(mask, self.resize, interpolation=Image.NEAREST)
            mask = np.array(mask).astype(np.int64)
        if self.horizontal_flip and random.random() > 0.5:
            image = F.hflip(image)
            mask = np.fliplr(mask)
        if self.vertical_flip and random.random() > 0.5:
            image = F.vflip(image)
            mask = np.flipud(mask)

        return image, mask

def dice_loss(pred, target, smooth=1e-6):
    pred = torch.softmax(pred, dim=1)
    target = nn.functional.one_hot(target, num_classes=pred.shape[1])
    target = target.permute(0, 3, 1, 2).float()

    intersection = torch.sum(pred * target, dim=(2, 3))
    union = torch.sum(pred + target, dim=(2, 3))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    dice_loss = 1 - dice.mean()
    return dice_loss

class CombinedLoss(nn.Module):
    def __init__(self, weight_ce=1.0, weight_dice=1.0):
        super(CombinedLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice

    def forward(self, pred, target):
        ce_loss = self.cross_entropy(pred, target)
        dice = dice_loss(pred, target)
        combined_loss = self.weight_ce * ce_loss + self.weight_dice * dice
        return combined_loss

if __name__ == "__main__":
    num_angles_config = sys.argv[1]
    cuda_num = sys.argv[2]
    print(f"Training with a dataset of {num_angles_config} angles")

    # Device configuration
    device = torch.device(f'cuda:{cuda_num}' if torch.cuda.is_available() else 'cpu')
    print("device: ", device)
    print("torch.cuda.is_available():", torch.cuda.is_available())

    # Initialize model, loss, optimizer
    model = UNet(in_channels=1, out_channels=4).to(device)  # Note: out_channels=4 (background + 3 classes)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Paths to data directories
    train_root_dirs = [
        f'/data/projection_output/{num_angles_config}_angles/num_distances_400/num_noises_0',
        f'/data/projection_output/{num_angles_config}_angles/num_distances_400/num_noises_2000',
        f'/data/projection_output/{num_angles_config}_angles/num_distances_400/num_noises_6000',
        f'/data/projection_output/{num_angles_config}_angles/num_distances_450/num_noises_2000',
        f'/data/projection_output/{num_angles_config}_angles/num_distances_450/num_noises_6000',
        f'/data/projection_output/{num_angles_config}_angles/num_distances_450/num_noises_10000',
        f'/data/projection_output/{num_angles_config}_angles/num_distances_500/num_noises_0',
        f'/data/projection_output/{num_angles_config}_angles/num_distances_500/num_noises_2000',
        f'/data/projection_output/{num_angles_config}_angles/num_distances_500/num_noises_10000',
    ]

    # Test dataset
    test_root_dirs = [
        f'/data/projection_output/{num_angles_config}_angles/num_distances_500/num_noises_6000',
        f'/data/projection_output/{num_angles_config}_angles/num_distances_400/num_noises_10000',
        f'/data/projection_output/{num_angles_config}_angles/num_distances_450/num_noises_0',
    ]

    # Transformations
    train_transform = JointTransform(resize=(256, 256), horizontal_flip=True, vertical_flip=True)
    test_transform = JointTransform(resize=(256, 256), horizontal_flip=False, vertical_flip=False)

    # Initialize the dataset
    train_dataset = DICOMDataset(train_root_dirs, transform=train_transform)
    test_dataset = DICOMDataset(test_root_dirs, transform=test_transform)

    # Training loop with validation
    num_epochs = 30
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    best_loss = float('inf')
    train_losses = []
    val_losses = []

    print("Start training loop")

    criterion = CombinedLoss(weight_ce=1.0, weight_dice=1.0)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        # Initialize progress bar for training
        with tqdm(total=len(train_loader), desc=f'Epoch [{epoch+1}/{num_epochs}] - Training', unit='batch', leave=True, miniters=1, mininterval=0.1) as train_bar:
            for images, masks in train_loader:
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                
                # Update the progress bar with the current loss
                train_bar.set_postfix(loss=loss.item())
                train_bar.update(1)

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss:.4f}')

        # Validation loop with progress bar
        model.eval()
        val_running_loss = 0.0
        with tqdm(total=len(test_loader), desc=f'Epoch [{epoch+1}/{num_epochs}] - Validation', unit='batch') as val_bar:
            with torch.no_grad():
                for images, masks in test_loader:
                    images = images.to(device)
                    masks = masks.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    val_running_loss += loss.item()
                    
                    val_bar.set_postfix(val_loss=loss.item())
                    val_bar.update(1)

        val_loss = val_running_loss / len(test_loader)
        val_losses.append(val_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}')

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
            }, f'./checkpoints/best_unet_{num_angles_config}_angles_checkpoint_epoch_{epoch+1}.pth')
            print(f'Checkpoint saved at epoch {epoch+1} with validation loss {val_loss:.4f}')

    torch.save(model.state_dict(), f'unet_model_{num_angles_config}_final.pth')

    folder_path = './loss_plots/male'
    os.makedirs(folder_path, exist_ok=True)

    print(train_losses)
    print(val_losses)
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs+1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss with a dataset of {num_angles_config} on male dataset')
    plt.legend()
    plt.savefig(f'./loss_plots/male/loss_plot_{num_angles_config} angles')
