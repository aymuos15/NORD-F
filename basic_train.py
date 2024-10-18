import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import StanfordDogsDataset
from model import Backbone, UpperBranch, ClassificationBranch, ReconstructionBranch
from model import ConvNeXtPerceptualLoss as VGGPerceptualLoss
# from loss import VGGPerceptualLossV2 as VGGPerceptualLoss
import os
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print()


# Hyperparameters
LEARNING_RATE = 0.001
BATCH_SIZE = 2
NUM_EPOCHS = 10
NUM_WORKERS = 2
CLASS_NUM = 5

DATA_DIR = "/home/localssk23/Downloads/ishika/data/DOGS"

trans_train = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(224, scale=(0.7, 1), ratio=(3/4, 4/3)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

trans_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create dataset and dataloaders
train_dataset = StanfordDogsDataset(root_dir=DATA_DIR, dataset_type='train', transform=trans_train)
val_dataset = StanfordDogsDataset(root_dir=DATA_DIR, dataset_type='val', transform=trans_val)
test_dataset = StanfordDogsDataset(root_dir=DATA_DIR, dataset_type='test', transform=trans_val)


train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=BATCH_SIZE)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=BATCH_SIZE)

# Verify data loading
print("Train dataset size:", len(train_dataset))
print("Validation dataset size:", len(val_dataset))
print()

backbone = Backbone().to(device)
encoder_output_channels = 1024
spatial_dim = 49
flattened_encoder_output_size = 1024 * 49

upper_branch = UpperBranch(encoder_output_channels).to(device)
classification_branch = ClassificationBranch(encoder_output_channels, flattened_encoder_output_size, CLASS_NUM).to(device)
reconstruction_branch = ReconstructionBranch(encoder_output_channels, output_channels=3).to(device)

# Define optimizer
optimizer = optim.Adam([
    {'params': backbone.parameters()},
    {'params': upper_branch.parameters()},
    {'params': classification_branch.parameters()},
    {'params': reconstruction_branch.parameters()}
], lr=LEARNING_RATE)

# Define loss functions
criterion_sim = nn.BCELoss()
criterion_cls = nn.CrossEntropyLoss()
criterion_recon = VGGPerceptualLoss()

# List of specific hyperparameter combinations to check
specific_combinations = [
    (1, 1, 1)
]

# Function to create directories if they don't exist
def create_directories():
    if not os.path.exists('loss_curves_dogs'):
        os.makedirs('loss_curves_dogs')

    if not os.path.exists('accuracy_curves_dogs'):
        os.makedirs('accuracy_curves_dogs')

    if not os.path.exists('models_dogs'):
        os.makedirs('models_dogs')

def grl_weight_scheduler(epoch, initial_weight=0.01, final_weight=1.0, step_size=10):
    steps = epoch // step_size
    increment = (final_weight - initial_weight) / ((final_weight / initial_weight) * step_size)
    weight = min(final_weight, initial_weight + steps * increment)
    return weight


# Training loop
def train_model(backbone, upper_branch, classification_branch, reconstruction_branch, train_loader, val_loader, criterion_sim, criterion_cls, criterion_recon, optimizer, lambda_sim, lambda_cls, lambda_recon, num_epochs=NUM_EPOCHS):
    sim_losses = []
    class_losses = []
    recon_losses = []
    total_losses = []
    train_accuracies = []
    val_accuracies = []
    val_sim_losses = []
    val_class_losses = []
    val_recon_losses = []

    for epoch in range(num_epochs):
        backbone.train()
        upper_branch.train()
        classification_branch.train()
        reconstruction_branch.train()

        grl_weight = grl_weight_scheduler(epoch)
        total_sim_loss = 0.0
        total_cls_loss = 0.0
        total_recon_loss = 0.0
        total_correct_1 = 0
        total_correct_2 = 0
        total_samples_1 = 0
        total_samples_2 = 0

        for i, (img1, img2, target1, target2, labels) in enumerate(train_loader):
            img1, img2, target1, target2, labels = img1.to(device), img2.to(device), target1.to(device), target2.to(device), labels.to(device)


            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass for upper branch
            features1 = backbone(img1)
            features2 = backbone(img2)


            similarity_score, upper_features1, upper_features2 = upper_branch(features1, features2, grl_weight)
            
            # If the sample was rejected, skip it
            #todo: print the targets and labels to understand the low batch error.
            if similarity_score is None:
                continue

            print('Shape of labels:', labels.shape)

            sim_loss = criterion_sim(similarity_score, labels.float())
            total_sim_loss += sim_loss.item()

            # Forward pass for classification branch
            lower_features1, cls_output1 = classification_branch(features1)
            lower_features2, cls_output2 = classification_branch(features2)

            cls_loss1 = criterion_cls(cls_output1, target1)
            cls_loss2 = criterion_cls(cls_output2, target2)
            avg_cls_loss = (cls_loss1 + cls_loss2) / 2
            total_cls_loss += avg_cls_loss.item()

            # Forward pass for reconstruction branch
            recon_output1 = reconstruction_branch(upper_features1, lower_features1)
            recon_output2 = reconstruction_branch(upper_features2, lower_features2)

            img1_resized = F.interpolate(img1, size=recon_output1.shape[2:], mode='bilinear', align_corners=False)
            img2_resized = F.interpolate(img2, size=recon_output2.shape[2:], mode='bilinear', align_corners=False)

            recon_loss1 = criterion_recon(recon_output1, img1_resized)
            recon_loss2 = criterion_recon(recon_output2, img2_resized)
            avg_recon_loss = (recon_loss1 + recon_loss2) / 2
            total_recon_loss += avg_recon_loss.item()

            # Combine losses
            total_loss = lambda_sim * sim_loss + lambda_cls * avg_cls_loss + lambda_recon * avg_recon_loss
            total_loss.backward()

            # Update weights
            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.empty_cache()

            # Calculating accuracy
            _, predicted1 = torch.max(cls_output1, 1)
            _, predicted2 = torch.max(cls_output2, 1)

            total_correct_1 += (predicted1 == target1).sum().item()
            total_correct_2 += (predicted2 == target2).sum().item()
            total_samples_1 += target1.size(0)
            total_samples_2 += target2.size(0)

        # Calculate average losses and accuracy
        avg_sim_loss = total_sim_loss / len(train_loader)
        avg_cls_loss = total_cls_loss / len(train_loader)
        avg_recon_loss = total_recon_loss / len(train_loader)
        train_accuracy_1 = total_correct_1 / total_samples_1 * 100
        train_accuracy_2 = total_correct_2 / total_samples_2 * 100

        sim_losses.append(avg_sim_loss)
        class_losses.append(avg_cls_loss)
        recon_losses.append(avg_recon_loss)
        total_losses.append((avg_sim_loss + avg_cls_loss + avg_recon_loss) / 3)
        train_accuracies.append((train_accuracy_1 + train_accuracy_2) / 2)

        # Validation phase
        backbone.eval()
        upper_branch.eval()
        classification_branch.eval()
        reconstruction_branch.eval()

        total_val_correct_1 = 0
        total_val_correct_2 = 0
        total_val_samples_1 = 0
        total_val_samples_2 = 0
        total_val_sim_loss = 0.0
        total_val_cls_loss = 0.0
        total_val_recon_loss = 0.0

        with torch.no_grad():
            for img1, img2, target1, target2, labels in val_loader:
                img1, img2, target1, target2, labels = img1.to(device), img2.to(device), target1.to(device), target2.to(device), labels.to(device)

                features1 = backbone(img1)
                features2 = backbone(img2)

                similarity_score, upper_features1, upper_features2 = upper_branch(features1, features2, grl_weight)
                sim_loss = criterion_sim(similarity_score, labels.float())
                total_val_sim_loss += sim_loss.item()

                lower_features1, cls_output1 = classification_branch(features1)
                lower_features2, cls_output2 = classification_branch(features2)

                cls_loss1 = criterion_cls(cls_output1, target1)
                cls_loss2 = criterion_cls(cls_output2, target2)
                avg_val_cls_loss = (cls_loss1 + cls_loss2) / 2
                total_val_cls_loss += avg_val_cls_loss.item()

                recon_output1 = reconstruction_branch(upper_features1, lower_features1)
                recon_output2 = reconstruction_branch(upper_features2, lower_features2)

                img1_resized = F.interpolate(img1, size=recon_output1.shape[2:], mode='bilinear', align_corners=False)
                img2_resized = F.interpolate(img2, size=recon_output2.shape[2:], mode='bilinear', align_corners=False)

                recon_loss1 = criterion_recon(recon_output1, img1_resized)
                recon_loss2 = criterion_recon(recon_output2, img2_resized)
                avg_val_recon_loss = (recon_loss1 + recon_loss2) / 2
                total_val_recon_loss += avg_val_recon_loss.item()

                _, predicted1 = torch.max(cls_output1, 1)
                _, predicted2 = torch.max(cls_output2, 1)

                total_val_correct_1 += (predicted1 == target1).sum().item()
                total_val_correct_2 += (predicted2 == target2).sum().item()
                total_val_samples_1 += target1.size(0)
                total_val_samples_2 += target2.size(0)

        avg_val_sim_loss = total_val_sim_loss / len(val_loader)
        avg_val_cls_loss = total_val_cls_loss / len(val_loader)
        avg_val_recon_loss = total_val_recon_loss / len(val_loader)
        val_accuracy_1 = total_val_correct_1 / total_val_samples_1 * 100
        val_accuracy_2 = total_val_correct_2 / total_val_samples_2 * 100

        val_sim_losses.append(avg_val_sim_loss)
        val_class_losses.append(avg_val_cls_loss)
        val_recon_losses.append(avg_val_recon_loss)
        val_accuracies.append((val_accuracy_1 + val_accuracy_2) / 2)

        print(f"""
        Epoch [{epoch+1}/{num_epochs}]
        Training:
            Similarity Loss:  {avg_sim_loss:.4f}
            Classification Loss: {avg_cls_loss:.4f}
            Reconstruction Loss: {avg_recon_loss:.4f}
            Accuracy 1: {train_accuracy_1:.2f}%
            Accuracy 2: {train_accuracy_2:.2f}%

        Validation:
            Similarity Loss:  {avg_val_sim_loss:.4f}
            Classification Loss: {avg_val_cls_loss:.4f}
            Reconstruction Loss: {avg_val_recon_loss:.4f}
            Accuracy 1: {val_accuracy_1:.2f}%
            Accuracy 2: {val_accuracy_2:.2f}%
        """)
        print()

        # save the weights of the classification branch and the upper branch
        if (epoch+1) % 2 == 0:
            torch.save(classification_branch.state_dict(), f"/home/localssk23/Downloads/ishika/weights/classification_branch_epoch_{epoch+1}.pt")
            torch.save(upper_branch.state_dict(), f"/home/localssk23/Downloads/ishika/weights/upper_branch_epoch_{epoch+1}.pt")
            torch.save(backbone.state_dict(), f"/home/localssk23/Downloads/ishika/weights/backbone_epoch_{epoch+1}.pt")

    return sim_losses, class_losses, recon_losses, total_losses, train_accuracies, val_accuracies

# Create necessary directories
create_directories()

# Evaluate the specific hyperparameter combinations
results = {}
for (lambda_sim, lambda_cls, lambda_recon) in specific_combinations:
    key = f"lambda_sim_{lambda_sim}_lambda_cls_{lambda_cls}_lambda_recon_{lambda_recon}"
    print(f"Evaluating combination: {key}")
    print()
    
    results[key] = train_model(backbone, upper_branch, classification_branch, reconstruction_branch, train_loader, val_loader, criterion_sim, criterion_cls, criterion_recon, optimizer, lambda_sim, lambda_cls, lambda_recon)