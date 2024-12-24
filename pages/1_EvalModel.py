from model.TYCNN import TYCNN
from model.AlexNet import AlexNet
from model.MLP import MLP
from model.VGG import Vgg
import streamlit as st
from utils import *

def EvalModel_page():
    _,_,test_loader = data_pre()
    st.markdown("# Eval Model")
    model_name = st.selectbox(
        label = "Select the model",
        options = ["TYCNN", "AlexNet", "MLP", "VGG"]
    )
    if model_name == "TYCNN":
        model = TYCNN()
        model.load_state_dict(torch.load('results/TYCNN.pth'))
    elif model_name == "AlexNet":
        model = AlexNet()
        model.load_state_dict(torch.load('results/AlexNet.pth'))
    elif model_name == "MLP":
        model = MLP()
        model.load_state_dict(torch.load('results/MLP.pth'))
    elif model_name == "VGG":
        model = Vgg()
    detals_model = show_model_details(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if st.button("Eval"):
        acc, prec, rec, f1, labels, probs = evaluate(model, test_loader, device)
        st.write(f"Test Accuracy: {acc:.4f}")
        st.write(f"Test Precision: {prec:.4f}")
        st.write(f"Test Recall: {rec:.4f}")
        st.write(f"Test F1 Score: {f1:.4f}")
        # 绘制 ROC 曲线
        plot_roc_curve(labels, probs, num_classes=7)

EvalModel_page()