import torch
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

class MyConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_p):
        kernel_size = 3
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.MaxPool2d(2, stride=2)
        )

    def forward(self, x):
        return self.model(x)

def get_batch_accuracy(output, y, N):
    pred = output.argmax(dim=1, keepdim=True)
    correct = pred.eq(y.view_as(pred)).sum().item()
    return correct / N


def train(model, train_loader, train_N, random_trans, optimizer, loss_function, device):
    total_loss = 0
    accuracy = 0
    
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        output = model(random_trans(x))
        optimizer.zero_grad()
        batch_loss = loss_function(output, y)
        batch_loss.backward()
        optimizer.step()

        total_loss += batch_loss.item()
        accuracy += get_batch_accuracy(output, y, train_N)

    # print('Train - Loss: {:.4f} Accuracy: {:.4f}'.format(total_loss, accuracy))
    return total_loss, accuracy

def validate(model, valid_loader, valid_N, loss_function, device):
    total_loss = 0
    accuracy = 0

    # cm
    y_true, y_pred = [], []
    
    model.eval()
    with torch.no_grad():
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)

            total_loss += loss_function(output, y).item()
            accuracy += get_batch_accuracy(output, y, valid_N)
            
            # cm
            preds = torch.argmax(output, dim=1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
     # Confusion matrix 생성
    cm = confusion_matrix(y_true, y_pred)

    # print('Valid - Loss: {:.4f} Accuracy: {:.4f}'.format(total_loss, accuracy))
    return total_loss, accuracy, cm, y_true, y_pred

