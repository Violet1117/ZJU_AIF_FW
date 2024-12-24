from model.AlexNet import AlexNet
import streamlit as st
from utils import *
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np


def AlexNet_page():
    st.markdown("# AlexNet")
    epoch = st.slider("Epoch", 1, 100)
    learning_rate = st.number_input(
        label = "Enter the learning rate",
        min_value = 0.0001,
        max_value = 0.1,
        value = 0.001,
        step = 0.0001,
        format = "%f",
        help = "Enter the learning"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlexNet().to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    detals_train = show_train_details(device, loss_function, optimizer)
    detals_model = show_model_details(model)
    if st.button("Train"):
        train_loader, val_loader , test_loader= data_pre_AlexNet()
        result = train_init(epoch, model, train_loader, val_loader, loss_function, optimizer, device, resultpth='results/AlexNet.pth')
        save_result(result, 'results/AlexNet')
        st.success("Training completed.")
        acc, prec, rec, f1, labels, probs = evaluate(model, test_loader, device)
        st.write(f"Test Accuracy: {acc:.4f}")
        st.write(f"Test Precision: {prec:.4f}")
        st.write(f"Test Recall: {rec:.4f}")
        st.write(f"Test F1 Score: {f1:.4f}")
        # 绘制 ROC 曲线
        plot_roc_curve(labels, probs, num_classes=7)

AlexNet_page()