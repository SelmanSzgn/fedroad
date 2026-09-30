import torch
import torch.nn as nn


def get_test_accuracy(testloader, device, global_model):
    correct = 0
    total = 0
    global_model.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = global_model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def get_test_loss(testloader, device, global_model):
    global_model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    total_samples = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = global_model(images)
            loss = criterion(outputs, labels)
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
    avg_loss = total_loss / total_samples
    return avg_loss

def get_client_accuracy(active_clients, global_model, device, batch_size=32):
    client_accuracy = []
    for client in active_clients:
        local_data_loader = torch.utils.data.DataLoader(
            client.local_data, batch_size=batch_size, shuffle=True
        )
        accuracy = get_test_accuracy(local_data_loader, device, global_model)
        client_accuracy.append(accuracy)
    return client_accuracy

def get_client_loss(active_clients, global_model, device, batch_size=32):
    client_loss = []
    for client in active_clients:
        local_data_loader = torch.utils.data.DataLoader(
            client.local_data, batch_size=batch_size, shuffle=True
        )
        loss = get_test_loss(local_data_loader, device, global_model)
        client_loss.append(loss)
    return client_loss

