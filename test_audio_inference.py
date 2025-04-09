import numpy as np
import librosa
import tflite_runtime.interpreter as tflite

# Load class labels
labels = np.array(["Normal", "Murmur", "Extrasystolic"])

# Load TFLite model
interpreter = tflite.Interpreter(model_path="heartbeat_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Parameters
sr = 22050
max_pad_len = 128
file_path = "murmur__162_1307101835989_B.wav"  # Update this if file name changes

def extract_mfcc(audio, sr=22050, max_pad_len=128):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    if mfcc.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_pad_len]
    return mfcc

# Load and process audio file
audio, _ = librosa.load(file_path, sr=sr, mono=True)
mfcc = extract_mfcc(audio)
input_data = np.expand_dims(mfcc, axis=(0, -1)).astype(np.float32)

# Run inference
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])

# Get prediction
predicted_label = labels[np.argmax(output_data)]
confidence = np.max(output_data)

print(f"✅ Prediction: {predicted_label} ({confidence*100:.2f}%)")
print("🎯 Expected: Murmur")