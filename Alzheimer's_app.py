import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import mne
import numpy as np
from sklearn.preprocessing import StandardScaler
from ssqueezepy import cwt
import timm


# Define ChronoNet Model
class ChronoNet(nn.Module):
    def __init__(self):
        super(ChronoNet, self).__init__()
        self.inception1 = Inception(19)
        self.inception2 = Inception(96)
        self.inception3 = Inception(96)
        self.gru1 = nn.GRU(96, 32, batch_first=True)
        self.gru2 = nn.GRU(32, 32, batch_first=True)
        self.gru3 = nn.GRU(64, 32, batch_first=True)
        self.gru4 = nn.GRU(96, 2, batch_first=True)
        self.affine1 = None

    def forward(self, x):
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.inception3(x)
        x = x.view(x.size(0), -1, 96)
        x, _ = self.gru1(x)
        x_res = x
        x, _ = self.gru2(x)
        x_res2 = x
        x_cat1 = torch.cat([x_res, x], dim=2)
        x, _ = self.gru3(x_cat1)
        x = torch.cat([x_res, x_res2, x], dim=2)
        x = x.view(x.size(0), -1)
        if self.affine1 is None:
            self.affine1 = nn.Linear(x.size(1), 96)
        x = F.elu(self.affine1(x))
        x = x.view(x.size(0), -1, 96)
        x, _ = self.gru4(x)
        x = torch.squeeze(x, dim=1)
        return F.softmax(x, dim=-1)


# Define Inception Module
class Inception(nn.Module):
    def __init__(self, in_channels):
        super(Inception, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv1d(in_channels, 32, kernel_size=2, stride=2)
        self.conv3 = nn.Conv1d(in_channels, 32, kernel_size=8, stride=2, padding=3)

    def forward(self, x):
        x1 = F.elu(self.conv1(x))
        x2 = F.elu(self.conv2(x))
        x3 = F.elu(self.conv3(x))
        return torch.cat([x1, x2, x3], dim=1)


# Define OurModel Class
class OurModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.reduce_channels = nn.Conv2d(1280, 14, kernel_size=1)
        self.model = timm.create_model('resnest26d', pretrained=True)
        self.model.conv1[0] = nn.Conv2d(14, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.fc1 = nn.Linear(1000, 500)
        self.fc2 = nn.Linear(500, 1)
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # Reorder dimensions for the model
        x = self.reduce_channels(x)
        x = self.model(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x


# Load Models
@st.cache_resource
def load_models():
    model_1d = torch.load('/Users/anody/Downloads/senior project/best_model_ChronoNet.pth', map_location="cpu")
    model_2d = torch.load('/Users/anody/Downloads/senior project/best_model_scaleogram.pth', map_location="cpu")
    model_1d.eval()
    model_2d.eval()
    return model_1d, model_2d


model_1d, model_2d = load_models()


# Preprocessing for 1D model
def preprocess_1d(file_path, duration=60, overlap=30):
    raw = mne.io.read_raw_eeglab(file_path, preload=True)
    sampling_rate = raw.info['sfreq']
    epochs = mne.make_fixed_length_epochs(raw, duration=duration, overlap=overlap)
    epoch_data = epochs.get_data()
    if epoch_data.size == 0:
        raise ValueError(f"No valid epochs in {file_path}.")
    scaler = StandardScaler()
    standardized_data = np.array([scaler.fit_transform(epoch.T).T for epoch in epoch_data])
    reshaped_data = np.moveaxis(standardized_data, 1, 2)
    input_tensor = torch.tensor(reshaped_data, dtype=torch.float32).transpose(1, 2)
    return input_tensor


# Preprocessing for 2D model
def preprocess_2d(file_path, epoch_duration=10, overlap=5, sampling_rate=128, target_length=None):
    raw = mne.io.read_raw_eeglab(file_path, preload=True)
    raw.filter(l_freq=1.0, h_freq=50.0)
    raw_data = raw.get_data()
    if target_length:
        n_channels, n_samples = raw_data.shape
        if n_samples > target_length:
            raw_data = raw_data[:, :target_length]
        elif n_samples < target_length:
            padding = np.zeros((n_channels, target_length - n_samples))
            raw_data = np.hstack([raw_data, padding])
    raw_data = (raw_data - np.mean(raw_data, axis=-1, keepdims=True)) / np.std(raw_data, axis=-1, keepdims=True)
    n_channels, n_samples = raw_data.shape
    epoch_length = epoch_duration * sampling_rate
    step_size = (epoch_duration - overlap) * sampling_rate
    epochs = [
        raw_data[:, start:start + epoch_length]
        for start in range(0, n_samples - epoch_length + 1, step_size)
    ]
    epochs_data = np.array(epochs)
    scaleograms = []
    for epoch in epochs_data:
        epoch_scaleograms = []
        for ch_data in epoch:
            Wx, _ = cwt(ch_data, 'morlet')
            epoch_scaleograms.append(np.abs(Wx))
        scaleograms.append(np.array(epoch_scaleograms))
    scaleograms = np.array(scaleograms)[..., np.newaxis]
    input_tensor = torch.tensor(scaleograms, dtype=torch.float32).squeeze(-1)
    return input_tensor


# Prediction function
def predict(file_path):
    st.write("Preprocessing data...")
    data_1d = preprocess_1d(file_path, duration=60, overlap=30)
    data_2d = preprocess_2d(file_path)
    st.write("Running models...")
    with torch.no_grad():
        output_1d = model_1d(data_1d)
        output_2d = model_2d(data_2d)
    class_probs_1d = torch.softmax(output_1d, dim=1).mean(dim=0).detach().cpu().numpy()
    class_1_prob_1d = class_probs_1d[1]
    prob_2d = output_2d.mean().item()
    weight_1d = 0.3
    weight_2d = 0.7
    final_score = weight_1d * class_1_prob_1d + weight_2d * prob_2d
    threshold = 0.45
    final_decision = final_score > threshold
    return final_decision, final_score, class_1_prob_1d, prob_2d


# Sidebar
def sidebar():
    st.sidebar.title("Navigation")
    st.sidebar.subheader("Add New Patient")
    patient_name = st.sidebar.text_input("Patient Name")
    patient_age = st.sidebar.number_input("Age", min_value=0, step=1)
    patient_notes = st.sidebar.text_area("Notes")
    if st.sidebar.button("Save New Patient"):
        st.sidebar.success(f"Patient {patient_name} added successfully!")
    st.sidebar.write("---")
    st.sidebar.subheader("Search Patient Records")
    search_query = st.sidebar.text_input("Search by Name or ID")
    if st.sidebar.button("Search"):
        st.sidebar.info(f"Results for: {search_query}")


# Main page
def main():
    sidebar()
    st.title("Alzheimer's Detection from EEG")
    st.write("Upload an EEG file (.set) to analyze Alzheimer's likelihood.")
    uploaded_file = st.file_uploader("Choose an EEG file", type="set")
    if uploaded_file is not None:
        temp_file_path = f"/tmp/{uploaded_file.name}"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        try:
            final_decision, final_score, prob_1d, prob_2d = predict(temp_file_path)
            st.write(f"1D Model Probability (Alzheimer's): {prob_1d:.4f}")
            st.write(f"2D Model Probability (Alzheimer's): {prob_2d:.4f}")
            st.write(f"Final Score: {final_score:.4f}")
            decision = "Alzheimer's Detected" if final_decision else "No Alzheimer's Detected"
            st.write(f"Final Decision: {decision}")
        except Exception as e:
            st.error(f"Error: {e}")

# Login Page
def login():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == "admin" and password == "admin":
            st.session_state["authenticated"] = True
        else:
            st.error("Invalid username or password")

# Session state for login
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    login()
else:
    main()
