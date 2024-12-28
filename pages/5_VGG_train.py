from model.VGG import Vgg
import streamlit as st
from utils import *
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

def VGG_page():
    st.markdown("# VGG")
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
    model = Vgg().to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    detals_train = show_train_details(device, loss_function, optimizer)
    detals_model = show_model_details(model)
    if st.button("Train"):
        train_loader, val_loader , test_loader= data_pre()
        result = train_init(epoch, model, train_loader, val_loader, loss_function, optimizer, device, resultpth='results/VGG.pth')
        save_result(result, 'results/VGG')
        st.success("Training completed.")
        acc, prec, rec, f1, labels, probs = evaluate(model, test_loader, device)

        save_result_eval(acc, prec, rec, f1, labels, probs, 'results/VGG_eval.csv')

VGG_page()