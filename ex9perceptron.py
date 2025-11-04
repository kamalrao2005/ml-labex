# Installed required packages
!pip install streamlit scikit-learn numpy -q

# Downloaded cloudflared for tunneling
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared
!chmod +x cloudflared

print("🚀 Installation complete!")
print("✅ Streamlit, scikit-learn, numpy installed")
print("✅ Cloudflared tunnel ready")

%%writefile enhanced_app.py
import streamlit as st
import numpy as np
import pickle
import os

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def train_model():
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    outputs = np.array([[0], [0], [0], [1]])

    np.random.seed(42)
    weights = np.random.rand(2, 1)
    bias = np.random.rand(1)
    learning_rate = 0.1
    epochs = 10000

    for epoch in range(epochs):
        weighted_sum = np.dot(inputs, weights) + bias
        predictions = sigmoid(weighted_sum)
        error = predictions - outputs
        adjustments = error * sigmoid_derivative(predictions)
        weights -= np.dot(inputs.T, adjustments) * learning_rate
        bias -= np.sum(adjustments) * learning_rate

    return weights, bias

def save_model(weights, bias, filename='and_gate_model.pkl'):
    model_data = {'weights': weights, 'bias': bias}
    with open(filename, 'wb') as f:
        pickle.dump(model_data, f)

def load_model(filename='and_gate_model.pkl'):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        return model_data['weights'], model_data['bias']
    return None, None

# Streamlit App UI
st.set_page_config(page_title="AND Gate Neural Network", layout="centered")
st.title("🔮 Smart Neural Network - AND Logic Gate")
st.caption("⚡ Powered by NumPy and visualized with Streamlit")
st.markdown("*Running on Google Colab with Cloudflare Tunnel*")

# Load or train model
weights, bias = load_model()

if weights is None:
    st.info("🎯 No saved model found. Training new model...")
    with st.spinner('🧠 Training neural network...'):
        weights, bias = train_model()
        save_model(weights, bias)
    st.success("✨ Model trained and saved successfully!")
else:
    st.success("💾 Loaded pre-trained model!")

# Model information
col1, col2 = st.columns(2)
with col1:
    st.metric("🎯 Model Status", "Ready")
with col2:
    st.metric("🔧 Architecture", "2-1-1")

st.divider()

# User input section
st.subheader("🎛️ Test Your Inputs")
col1, col2 = st.columns(2)
with col1:
    a = st.selectbox("🔵 Input A", [0, 1], index=0)
with col2:
    b = st.selectbox("🟢 Input B", [0, 1], index=0)

input_data = np.array([a, b])

# Prediction
result = sigmoid(np.dot(input_data, weights) + bias)[0]
binary_result = round(result)

st.subheader("🎪 Prediction Results")
col1, col2, col3 = st.columns(3)
with col1:
    st.write(f"📥 **Input: [{a}, {b}]**")
with col2:
    st.write(f"🔬 **Raw Output: {result:.4f}**")
with col3:
    if binary_result == 1:
        st.success(f"✅ **Binary Output: {binary_result}**")
    else:
        st.error(f"❌ **Binary Output: {binary_result}**")

# Truth table
st.subheader("📋 AND Gate Truth Table")
truth_table_data = {
    'Input A': [0, 0, 1, 1],
    'Input B': [0, 1, 0, 1],
    'Expected Output': [0, 0, 0, 1]
}

# Test all combinations
predictions = []
for i in range(4):
    test_input = np.array([truth_table_data['Input A'][i], truth_table_data['Input B'][i]])
    pred = round(sigmoid(np.dot(test_input, weights) + bias)[0])
    predictions.append(pred)

truth_table_data['Model Output'] = predictions
st.dataframe(truth_table_data, use_container_width=True)

# Accuracy calculation
accuracy = sum(1 for i in range(4) if truth_table_data['Expected Output'][i] == predictions[i]) / 4 * 100
st.metric("🎯 Model Accuracy", f"{accuracy:.1f}%")

# Debug info
with st.expander("🔍 Model Parameters & Debug Info"):
    col1, col2 = st.columns(2)
    with col1:
        st.write("⚖️ **Weights:**")
        st.code(str(weights))
    with col2:
        st.write("🔧 **Bias:**")
        st.code(str(bias))

    if st.button("🔄 Retrain Model"):
        with st.spinner('🧠 Retraining neural network...'):
            weights, bias = train_model()
            save_model(weights, bias)
        st.success("✨ Model retrained successfully!")
        st.rerun()

# Footer
st.markdown("---")
st.markdown("**🔬 Built with:** Python • NumPy • Streamlit • Google Colab")

print("📝 Enhanced app.py created successfully!")
import subprocess
import time
import re
import threading
import sys

def run_streamlit():
    """Run Streamlit in background"""
    cmd = [sys.executable, "-m", "streamlit", "run", "enhanced_app.py",
           "--server.port", "8501", "--server.headless", "true",
           "--server.enableCORS", "false", "--server.enableXsrfProtection", "false"]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def run_cloudflared():
    """Run cloudflared tunnel"""
    cmd = ["./cloudflared", "tunnel", "--url", "http://localhost:8501"]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

# Kill any existing processes
!pkill -f streamlit &>/dev/null || true
!pkill -f cloudflared &>/dev/null || true

print("🚀 Starting Enhanced Neural Network AND Gate App")
print("=" * 60)

# Start Streamlit
print("🎬 Launching Streamlit server...")
streamlit_process = run_streamlit()
time.sleep(5)

# Start Cloudflared tunnel
print("🌐 Creating secure public tunnel...")
tunnel_process = run_cloudflared()

# Extract public URL
print("🔍 Getting your public URL...")
public_url = None
timeout = 30
start_time = time.time()

while time.time() - start_time < timeout:
    line = tunnel_process.stdout.readline()
    if line:
        # Look for the URL pattern
        if "trycloudflare.com" in line:
            # Extract URL using regex
            url_match = re.search(r'https://[^\s]+\.trycloudflare\.com', line)
            if url_match:
                public_url = url_match.group(0)
                break
    time.sleep(0.5)

print("=" * 60)
if public_url:
    print("🎉 SUCCESS! Your Neural Network app is now live!")
    print(f"🔗 Public URL: {public_url}")
    print("")
    print("🌟 App Features:")
    print("   🔮 Smart model persistence (auto save/load)")
    print("   🎛️ Interactive AND gate testing")
    print("   📋 Complete truth table display")
    print("   🎯 Real-time accuracy metrics")
    print("   🔄 One-click model retraini")
