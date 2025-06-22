import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Define Generator architecture
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(100 + 10, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 28 * 28),
            nn.Tanh()
        )

    def forward(self, z, labels):
        label_input = torch.nn.functional.one_hot(labels, 10).float()
        x = torch.cat((z, label_input), dim=1)
        out = self.model(x)
        return out.view(-1, 1, 28, 28)

# Load model
device = torch.device("cpu")
G = Generator().to(device)
G.load_state_dict(torch.load("generator.pth", map_location=device))
G.eval()

# Streamlit UI
st.title("🧠 Handwritten Digit Generator")
st.markdown("This app generates **5 images** of handwritten digits (0-9) using a custom-trained GAN model.")

digit = st.selectbox("Select a digit to generate:", list(range(10)))
generate = st.button("Generate Images")

if generate:
    z = torch.randn(5, 100)
    labels = torch.tensor([digit] * 5)
    with torch.no_grad():
        images = G(z, labels)
    images = images.squeeze().numpy()

    fig, axs = plt.subplots(1, 5, figsize=(15, 3))
    for i in range(5):
        axs[i].imshow(images[i], cmap="gray")
        axs[i].axis("off")
    st.pyplot(fig)
