from model.MLP import MLP
from utils import *
import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np



def MLP_page():
    st.markdown("# MPL")
    epoch = st.number_input(
        label="Enter the epochs you want to trian.",
        min_value = 1,
        max_value = 200,
        value = 50,
        step = 1,
        help = "Enter the epochs you want to trian."
    )
    layer_size1 = st.selectbox(
        label = "Select the size for the first layer of MLP",
        options = [2048, 1024, 512],
        index = 1,
        help = "The default option is 1024",
    )
    layer_size2 = st.selectbox(
        label = "Select the size for the second layer of MLP",
        options = [1024, 512, 256],
        index = 1,
        help = "The default option is 512"
    )
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
    model = MLP(layer_size1=layer_size1, layer_size2=layer_size2).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    detals_train=show_train_details(device, loss_function, optimizer)
    detals_model=show_model_details(model)
    if st.button("Train"):
        train_loader, val_loader , test_loader =  data_pre()
        result = train_init(epoch, model, train_loader, val_loader, loss_function, optimizer, device, resultpth='results/MLP.pth')
        save_result(result, 'results/MLP')
        st.success("Training completed.")
        acc, prec, rec, f1, labels, probs = evaluate(model, test_loader, device)

        save_result_eval(acc, prec, rec, f1, labels, probs, 'results/MLP_eval.csv')

MLP_page()