# 🎙️ Speaker Recognition with LSTM, BiLSTM, and LSTM-CNN

## 📌 Project Overview
This project develops a **speaker recognition system** to identify individuals based on audio data using three deep learning approaches: **LSTM**, **BiLSTM**, and **LSTM-CNN**. The system processes audio files, extracts MFCC features, and trains models to classify speakers. The goal is to compare the performance of these architectures for speaker identification.

---

## 📂 Dataset
- **Speaker Recognition Audio Dataset**: Sourced from Kaggle, containing audio recordings from 50 speakers.  
  🔗 [Speaker Recognition Audio Dataset](https://www.kaggle.com/datasets/vjcalling/speaker-recognition-audio-dataset)

---

## 🔍 Project Workflow

### **1. Audio Preprocessing**
Split audio files into 2-second segments for consistent input.

```python
from pydub import AudioSegment
import os

def split_audio(file_path, segment_duration=2000):
    audio = AudioSegment.from_file(file_path)
    segments = []
    for i in range(0, len(audio), segment_duration):
        segment = audio[i:i + segment_duration]
        segments.append(segment)
    return segments

def process_speakers(input_directory, output_directory, segment_duration=2000):
    speaker_dirs = [d for d in os.listdir(input_directory) if os.path.isdir(os.path.join(input_directory, d))]
    for idx, speaker in enumerate(speaker_dirs, start=1):
        speaker_path = os.path.join(input_directory, speaker)
        output_speaker_path = os.path.join(output_directory, f"S{idx}")
        os.makedirs(output_speaker_path, exist_ok=True)
        for filename in os.listdir(speaker_path):
            if filename.endswith(".wav"):
                file_path = os.path.join(speaker_path, filename)
                segments = split_audio(file_path, segment_duration)
                for i, segment in enumerate(segments):
                    segment_filename = f"{os.path.splitext(filename)[0]}_part{i}.wav"
                    segment_path = os.path.join(output_speaker_path, segment_filename)
                    segment.export(segment_path, format="wav")
```

### **2. Feature Extraction**
Extract MFCC features from audio segments.

```python
import librosa
import numpy as np

def extract_mfcc(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    n_fft = min(2048, len(audio))
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, n_fft=n_fft)
    return mfccs
```

### **3. Model Architectures**
Three models are implemented: **LSTM**, **BiLSTM**, and **LSTM-CNN**, each with early stopping and learning rate reduction.

#### **LSTM Model**
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

model = Sequential([
    LSTM(256, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
    Dropout(0.3),
    LSTM(128, return_sequences=False),
    Dropout(0.3),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dense(64, activation='relu'),
    Dense(len(speaker_folders), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
lr_reduction = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32, callbacks=[early_stopping, lr_reduction])
```

#### **BiLSTM Model**
```python
from tensorflow.keras.layers import Bidirectional

model = Sequential([
    Bidirectional(LSTM(256, return_sequences=True), input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.3),
    Bidirectional(LSTM(128, return_sequences=False)),
    Dropout(0.3),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dense(64, activation='relu'),
    Dense(len(speaker_folders), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32, callbacks=[early_stopping, lr_reduction])
```

#### **LSTM-CNN Model**
```python
from tensorflow.keras.layers import Conv1D, MaxPooling1D

model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),
    Conv1D(filters=128, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),
    LSTM(128, return_sequences=False),
    Dropout(0.3),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dense(64, activation='relu'),
    Dense(len(speaker_folders), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32, callbacks=[early_stopping, lr_reduction])
```

### **4. Model Evaluation**
Evaluate each model on the test set.

```python
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Evaluation Accuracy: {accuracy}")
```

---

## 📊 Model Results
| Model        | Test Accuracy |
|--------------|---------------|
| LSTM         | 0.9179746386450297 (~91.8%) |
| BiLSTM       | 0.9260441606810322 (~92.6%) |
| LSTM-CNN     | 0.9092843841447193 (~90.9%) |


---

## ▶️ How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/ali27kh/Voice_recognition_LSTM_BILSTM_CNN.git
   cd Voice_recognition_LSTM_BILSTM_CNN
   ```
2. Download the dataset from Kaggle and place it in `50_speakers_audio_data/`.
3. Install dependencies.
4. Train the models and fine-tune them with your dataset or other techniques.

---

## 📌 Key Insights
- **BiLSTM** outperforms others with a test accuracy of ~92.6%, likely due to its ability to capture bidirectional temporal dependencies.
- **LSTM-CNN** integrates spatial and temporal features but achieves slightly lower accuracy (~90.9%).
- Audio segmentation into 2-second clips ensures consistent input for model training.
- Early stopping and learning rate reduction prevent overfitting and optimize training.
- MFCC features effectively capture speaker-specific audio characteristics.

---

## 📜 License
MIT License
