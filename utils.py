from imblearn.over_sampling import RandomOverSampler
import numpy as np
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm.notebook import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import gc
from torch.cuda.amp import autocast, GradScaler
from PIL import Image
from torchvision import transforms
from torch.utils.data import SubsetRandomSampler

def convert_string_to_int_array(df):
    """
    Convert pixel data stored as strings in a pd dataframe to a 2D NumPy array of integers.

    Parameters:
    - df: DataFrame containing a column named "pixels" with pixel data stored as strings.

    Returns:
    - result: 2D NumPy array containing integer values of pixel data.
    """
    # Apply a lambda function to convert each string of space-separated integers to a NumPy array of integers
    result = np.vstack(df["pixels"].apply(lambda x: np.array(list(map(int, x.split())))))
    
    return result


def plot_emotion_count(data, emotions_dict, set_name):
    """
    Plot countplot for emotions in the given dataset.

    Parameters:
    - data: The dataset, either a DataFrame or a DataLoader.
    - set_name (str): Name of the dataset ('Train', 'Validation', or 'Test').
    """
    sns.set(style="darkgrid")
    plt.figure(figsize=(8, 6))

    if isinstance(data, DataLoader):
        # Initialize an empty list to store labels
        labels = []

        # Iterate over the DataLoader to get labels
        for batch in data:
            labels.extend([int(item) for item in batch[1]])

        # Create a DataFrame from the labels
        df = pd.DataFrame({'emotion': labels})

        sns.countplot(x='emotion', data=df, palette='viridis', hue='emotion', dodge=False, legend=False)

    elif isinstance(data, pd.DataFrame):
        df = data
        sns.countplot(x='emotion', data=df, palette='viridis', hue='emotion', dodge=False, legend=False)

    else:
        raise ValueError("Unsupported data type. Use either a DataFrame or a DataLoader.")

    # Replace x-axis ticks with emotion labels
    plt.xticks(ticks=df['emotion'].unique(), labels=[emotions_dict[label] for label in df['emotion'].unique()])

    # Add labels and title
    plt.xlabel('Emotion')
    plt.ylabel('Count')
    plt.title(f'Countplot of Emotions - {set_name}')

    # Show the plot
    plt.show()   
    
    
def show_images(dataset, emotions_dict, num_images=25, num_cols=5):
    """
    Display a grid of images with their corresponding emotion labels.

    Parameters:
    - dataset: The dataset containing pixel data and emotion labels.
    - emotions_dict: A dictionary mapping emotion indices to emotion names.
    - num_images (int): Number of images to display.
    - num_cols (int): Number of columns in the grid.

    Returns:
    None
    """
    # Get the first 'num_images' samples from the dataset
    samples = dataset[:num_images]
    images = convert_string_to_int_array(samples)
    labels = dataset.emotion

    # Map label indices to emotions using the provided dictionary
    emotion_labels = [emotions_dict[label] for label in labels]

    # Reshape the images to 48x48
    images = images.reshape(-1, 48, 48)

    # Create a grid of images
    num_rows = int(np.ceil(num_images / num_cols))
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 12))

    for i in range(num_rows):
        for j in range(num_cols):
            index = i * num_cols + j
            if index < num_images:
                ax = axes[i, j] if num_rows > 1 else axes[j]
                ax.imshow(images[index], cmap='gray')
                ax.set_title(f'Emotion: {emotion_labels[index]}')
                ax.axis('off')

    plt.show()
    
    
    
def split_labels_and_data(df):
    """
    Split labels and data from the given DataFrame.

    Parameters:
    - df (DataFrame): The DataFrame containing the dataset.

    Returns:
    - labels (Series): The emotion labels.
    - data (DataFrame): The pixel data.
    """
    labels = df['emotion']
    data = df.drop(['emotion', 'Usage'], axis=1)  # Assuming 'Usage' is the column specifying the split
    return labels, data
    
    
    
def get_datapoint_weights(dataset):
    count_list = []
    for i in dataset:
        count_list.append(int(i[1].item()))
    
    _, class_counts = np.unique(count_list, return_counts=True)
    
    train_class_weights = [1/class_counts[i] for i in count_list]

    return train_class_weights



class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx].reshape(1, 48, 48).astype(np.uint8)
        label = torch.Tensor([self.labels[idx]])

        # Convert to PIL Image
        image_pil = Image.fromarray(image.squeeze(), mode='L')

        # Apply the transformation
        if self.transform:
            image_transformed = self.transform(image_pil)
        else:
            image_transformed = transforms.ToTensor()(image_pil)

        return image_transformed, label
        
        
        
def train(train_dataloader, valid_dataloader, model, criterion, optimizer, epochs, model_save_path, writer, device, scaler, patience, scheduler=None):
    """
    Train a neural network model and validate it on a separate validation set.

    Parameters:
    - train_dataset (torch.utils.data.Dataset): The training dataset.
    - valid_dataloader (torch.utils.data.DataLoader): DataLoader for the validation set.
    - model (torch.nn.Module): The neural network model to be trained.
    - criterion (torch.nn.Module): The loss function.
    - optimizer (torch.optim.Optimizer): The optimization algorithm.
    - epochs (int): The number of training epochs.
    - model_save_path (str): Path to save the best model.
    - writer (torch.utils.tensorboard.SummaryWriter): TensorBoard writer for logging.
    - device (str): Device on which to perform training (e.g., 'cuda' or 'cpu').
    - scaler (torch.cuda.amp.GradScaler): GradScaler for mixed-precision training.
    - patience (int): Number of epochs with no improvement on the validation loss before stopping.

    Returns:
    None
    """
    # Initialize variables to track the best validation loss
    best_valid_loss = float('inf')
    no_improvement_count = 0  # Counter for consecutive epochs with no improvement

    # Print initial losses and accuracies
    print("Initial Stats: \n")
    initial_train_loss, initial_train_accuracy = evaluate_model(model, train_dataloader, criterion, device)
    initial_valid_loss, initial_valid_accuracy = evaluate_model(model, valid_dataloader, criterion, device)
    print(f'Train Loss: {initial_train_loss:.4f}, Train Accuracy: {initial_train_accuracy:.4f}')
    print(f'Validation Loss: {initial_valid_loss:.4f}, Validation Accuracy: {initial_valid_accuracy:.4f}')
    print("-" * 50)

    # Logging to TensorBoard
    writer.add_scalar('Loss/Train', initial_train_loss, 0)
    writer.add_scalar('Loss/Valid', initial_valid_loss, 0)
    writer.add_scalar('Accuracy/Train', initial_train_accuracy, 0)
    writer.add_scalar('Accuracy/Valid', initial_valid_accuracy, 0)

    print("\nStarting training: \n")
    # Training loop
    for epoch in range(1, epochs + 1):

        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in tqdm(train_dataloader, desc="Training", 
                                   position=0, leave=False):
            # Move inputs to the specified device
            inputs = inputs.float().to(device)
            labels = labels.squeeze().long().to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # get prediction and loss
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            # use scaler to scale loss for mixed-precision training
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Accumulate the training loss
            train_loss += loss.item() * inputs.size(0)

            outputs = outputs.to("cpu")
            labels = labels.to("cpu")

            # Compute the number of correctly predicted samples
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            # Free up memory
            del inputs, labels, outputs, loss
            torch.cuda.empty_cache()
            gc.collect()

        # Average the training loss over the dataset
        train_loss /= len(train_dataloader.dataset)

        # Compute training accuracy
        train_accuracy = train_correct / train_total

        # Use evaluate_model function for validation
        valid_loss, valid_accuracy = evaluate_model(model, valid_dataloader, criterion, device)
        
        # Scheduler step based on the validation loss (if scheduler is provided)
        if scheduler is not None:
            scheduler.step(valid_loss)

        # Save the best model based on validation loss
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), model_save_path)
            no_improvement_count = 0  # Reset the counter on improvement
        else:
            no_improvement_count += 1

        # Logging to TensorBoard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Valid', valid_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_accuracy, epoch)
        writer.add_scalar('Accuracy/Valid', valid_accuracy, epoch)

        # Print stats every 5 epochs and the last epoch
        if epoch % 5 == 0:
            print(f'Epoch {epoch}/{epochs} ==> '
                  f'Train Loss: {train_loss:.4f}, '
                  f'Train Accuracy: {train_accuracy:.4f}',
                  f'Validation Loss: {valid_loss:.4f}, '
                  f'Validation Accuracy: {valid_accuracy:.4f}')

        # Check for early stopping
        if no_improvement_count >= patience:
            print(f'No improvement for {patience} consecutive epochs. Stopping training.')
            break

    # Close TensorBoard writer
    writer.close()

def evaluate_model(model, dataloader, criterion, device):
    """
    Evaluate the model on the given dataloader and return the loss and accuracy.

    Parameters:
    - model (torch.nn.Module): The neural network model to be evaluated.
    - dataloader (torch.utils.data.DataLoader): DataLoader for the dataset.
    - criterion (torch.nn.Module): The loss function.
    - device (str): Device on which to perform evaluation (e.g., 'cuda' or 'cpu').

    Returns:
    - loss (float): Average loss on the dataset.
    - accuracy (float): Accuracy on the dataset.
    """
    model.eval()
    loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluation", position=0, leave=False):
            inputs = inputs.float().to(device)
            labels = labels.squeeze().long().to(device)
            outputs = model(inputs)
            loss += criterion(outputs, labels).item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(dataloader.dataset)
    accuracy = correct / total
    return loss, accuracy
    
    
def test(model, test_dataloader, emotions_dict, device, plot_misclassified=False, num_misclassified=5):
    """
    Evaluate the model on the test set and display performance metrics.

    Parameters:
    - model (torch.nn.Module): The trained neural network model.
    - test_dataloader (torch.utils.data.DataLoader): DataLoader for the test set.
    - device (str): Device on which to perform the evaluation (e.g., 'cuda' or 'cpu').
    - plot_misclassified (bool): Whether to plot misclassified images.
    - num_misclassified (int): Number of misclassified images to plot.

    Returns:
    None
    """
    # Set the model to evaluation mode
    model.eval()
    model.to(device)

    # Lists to store true labels, predicted labels, and misclassified images
    all_labels = []
    all_predictions = []
    misclassified_images = []

    # Loop through the test DataLoader
    with torch.no_grad():
        for inputs, labels in tqdm(test_dataloader, desc="Testing", position=0, leave=False):
            # Move inputs to the specified device
            inputs = inputs.float().to(device)
            labels = labels.squeeze().long().to(device)

            # Forward pass to get predictions
            outputs = model(inputs)
            
            # Convert labels and predictions to numpy arrays
            outputs = outputs.to("cpu")
            labels = labels.to("cpu").numpy()
            
            _, predicted = torch.max(outputs.data, 1)
            predicted = predicted.numpy()

            # Store true labels and predicted labels
            all_labels.extend(labels)
            all_predictions.extend(predicted)

            # If requested, store misclassified images
            if plot_misclassified and len(misclassified_images) < num_misclassified:
                misclassified_indices = np.where(labels != predicted)[0]
                for mis_idx in misclassified_indices:
                    misclassified_images.append((inputs[mis_idx], labels[mis_idx], predicted[mis_idx]))

    # Classification Report
    report = classification_report(all_labels, all_predictions, target_names=emotions_dict.values(), digits=4)
    print("\nClassification Report:\n")
    print(report)

    # Add extra space
    print("\n" + "=" * 100 + "\n")

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_predictions)

    # Plot Confusion Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=emotions_dict.values(), yticklabels=emotions_dict.values())
    plt.title("Confusion Matrix", fontsize=16)  # Increase title font size
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig('confusion_matrix.png')  # Save the plot as a PNG file
    plt.show()

    # Add extra space
    print("\n" + "=" * 100 + "\n")

    print("Misclassified Images: \n")

    # Plot Misclassified Images
    if plot_misclassified:
        num_cols = min(num_misclassified, 5)
        num_rows = int(np.ceil(num_misclassified / num_cols))
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 12))

        for i in range(num_rows):
            for j in range(num_cols):
                index = i * num_cols + j
                if index < num_misclassified:
                    ax = axes[i, j] if num_rows > 1 else axes[j]
                    image, true_label, misclassified_label = misclassified_images[index]

                    if image.shape[0] == 3:
                        image = image[0]

                    
                    ax.imshow(image.squeeze().cpu(), cmap='gray')
                    ax.set_title(f'True: {emotions_dict[true_label]}, Predicted: {emotions_dict[misclassified_label]}', fontsize=7)
                    ax.axis('off')

        plt.show()
        
        
        
