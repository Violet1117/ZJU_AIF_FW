import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import inspect
import textwrap
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from model.dataset_FER2013 import FER2013
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from streamlit.logger import get_logger
LOGGER = get_logger(__name__)

'''
show_code: Show the code of the demo.
'''
def show_code(demo):
    """Showing the code of the demo."""
    is_show_code = st.sidebar.checkbox("Show code", True)
    if is_show_code:
        # Showing the code of the demo.
        st.markdown("## Code")
        sourcelines, _ = inspect.getsourcelines(demo)
        st.code(textwrap.dedent("".join(sourcelines[1:])))
    return is_show_code

'''
show_model_code: Show the model code.
'''
def show_model_code(demo):
    is_show_model_code = st.sidebar.checkbox("Show model code", True)
    if is_show_model_code:
        st.markdown("## Model Code")
        sourcelines, _ = inspect.getsourcelines(demo)
        st.code(textwrap.dedent("".join(sourcelines[1:])))
    return is_show_model_code

'''
show_model_details: Show the model details.
'''
def show_model_details(model):
    is_show_model_details = st.sidebar.checkbox(
        label="Show model details", 
        value=True
    )
    if is_show_model_details:
        st.markdown("## Model Details")
        st.write(model)
    return is_show_model_details

'''
show_train_details: Show the train details.
'''
def show_train_details(device, loss_function, optimizer):
    is_show_train_details = st.sidebar.checkbox(
        label="Show train details", 
        value = True
    )
    if is_show_train_details:
        st.markdown("## Train Details")
        st.write(f"Device: {device}")
        st.write(f"Loss function: {loss_function}")
        st.write(f"Optimizer: {optimizer}")
    return is_show_train_details


'''
load_data: Load the data.
for streamlit, we need to use st.cache_data to cache the data.
'''
@st.cache_data
def load_data():
    LOGGER.info('Loading data...')
    Fer2013_data = pd.read_csv('dataset/fer2013.csv')
    train_data = Fer2013_data[Fer2013_data['Usage'] == 'Training']
    val_data = Fer2013_data[Fer2013_data['Usage'] == 'PrivateTest']
    test_data = Fer2013_data[Fer2013_data['Usage'] == 'PublicTest']
    train_data = train_data.reset_index(drop=True)
    val_data = val_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)
    LOGGER.info('Data loaded.')
    return train_data, val_data, test_data

'''
data_pre: Data preprocessing.
'''
def data_pre():
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    BATCH_SIZE = 64
    train_data, val_data, test_data = load_data()
    train_data = FER2013(train_data, transform)
    val_data = FER2013(val_data, transform)
    test_data = FER2013(test_data, transform)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    test_loader= DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader, test_loader

def data_pre_AlexNet():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    BATCH_SIZE = 64
    train_data, val_data, test_data = load_data()
    train_data = FER2013(train_data, transform)
    val_data = FER2013(val_data, transform)
    test_data = FER2013(test_data, transform)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    test_loader= DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader, test_loader

def save_result(result, name):
    pd.DataFrame(result).to_csv(name+'.csv', index=False)

def plot_roc_curve(labels, probs, num_classes):
    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(labels, probs[:, i], pos_label=i)
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure()
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (area = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    st.pyplot(plt)

def train_init(epoch, model, train_loader, val_loader, loss_function, optimizer, device, resultpth):
    
    best_acc = 0
    result = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()
    st.markdown("### Training loss")
    train_loss_chart = st.line_chart()
    st.markdown('### Training accuracy')
    train_acc_chart = st.line_chart()

    for e in range(1, epoch+1):
        model.train()
        running_loss = 0
        correct = 0
        total = 0

        for img, label in train_loader:
            img, label = img.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(img)
            loss = loss_function(output, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * img.size(0)
            correct += (torch.argmax(output, dim=1) == label).sum().item()
            total += label.size(0)
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        result['train_loss'].append(epoch_loss)
        result['train_acc'].append(epoch_acc)

        model.eval()
        running_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for img, label in val_loader:
                img, label = img.to(device), label.to(device)
                output = model(img)
                loss = loss_function(output, label)
                running_loss += loss.item() * img.size(0)
                correct += (torch.argmax(output, dim=1) == label).sum().item()
                total += label.size(0)
            epoch_loss = running_loss / total
            epoch_acc = correct / total
            result['val_loss'].append(epoch_loss)
            result['val_acc'].append(epoch_acc)
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), resultpth)
        print(f'Epoch {e}/{epoch} loss: {result["train_loss"][-1]:.4f} acc: {result["train_acc"][-1]:.4f} val_loss: {result["val_loss"][-1]:.4f} val_acc: {result["val_acc"][-1]:.4f}')
        per = e*100/epoch
        status_text.text(f"{int(per)}% Complete")
        progress_bar.progress(int(per))
        train_loss_chart.add_rows(pd.DataFrame({
            'train_loss':[result['train_loss'][-1]],
            'val_loss':[result['val_loss'][-1]]
        }))
        train_acc_chart.add_rows(pd.DataFrame({
            'train_acc':[result['train_acc'][-1]],
            'val_acc':[result['val_acc'][-1]]
        }))
    progress_bar.empty()
    return result

def evaluate(model, test_loader, device):
    model.eval()
    all_preds = np.array([])
    all_labels = np.array([])
    all_probs = np.empty((0, 7))

    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            all_preds = np.concatenate((all_preds, preds.cpu().numpy()))
            all_labels = np.concatenate((all_labels, labels.cpu().numpy()))
            all_probs = np.vstack((all_probs, probs.cpu().numpy()))

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='macro')
    rec = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')

    return acc, prec, rec, f1, all_labels, all_probs