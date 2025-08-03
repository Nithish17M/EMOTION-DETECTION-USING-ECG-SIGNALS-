import os
import scipy.io
import numpy as np

# Path to your DEAP dataset directory
data_dir = 'C:/Users/Nithish/Downloads/data_preprocessed_matlab'

# DEAP has 32 subjects: s01.mat to s32.mat
num_subjects = 32
all_data = []
all_labels = []

for i in range(1, num_subjects + 1):
    filename = os.path.join(data_dir, f's{i:02d}.mat')  # Format s01.mat, s02.mat, ..., s32.mat
    
    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        continue
    
    # Load the .mat file
    mat = scipy.io.loadmat(filename)
    
    # Each file contains 'data' and 'labels'
    data = mat['data']        # Shape: (40 trials, 40 channels, 8064 samples)
    labels = mat['labels']    # Shape: (40 trials, 4 labels)
    
    all_data.append(data)
    all_labels.append(labels)

# Convert to numpy arrays if needed
all_data = np.array(all_data)      # Shape: (32, 40, 40, 8064)
all_labels = np.array(all_labels)  # Shape: (32, 40, 4)

print("Loaded DEAP dataset")
print("All data shape:", all_data.shape)
print("All labels shape:", all_labels.shape)


import pandas as pd

output_csv = "deap_ecg_data.csv"

all_data = []

for subject_id in range(1, 33):  # Subjects: 1 to 32
    filename = os.path.join(data_dir, f"s{subject_id:02d}.mat")

    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        continue

    mat = scipy.io.loadmat(filename)
    data = mat['data']     # shape: (40, 40, 8064)
    labels = mat['labels'] # shape: (40, 4)

    for trial_index in range(40):
        ecg_signal = data[trial_index, 38, :]  # ECG is channel 39 (index 38)

        for sample_index, value in enumerate(ecg_signal):
            row = {
                "Subject": f"Subject {subject_id}",
                "Trial": trial_index + 1,
                "Sample": sample_index,
                "ECG_Value": value,
                "Valence": labels[trial_index, 0],
                "Arousal": labels[trial_index, 1],
                "Dominance": labels[trial_index, 2],
                "Liking": labels[trial_index, 3]
            }
            all_data.append(row)

    print(f"Processed Subject {subject_id}")

# Save to CSV
df = pd.DataFrame(all_data)
df.to_csv(output_csv, index=False)
print(f"Saved ECG signal data to {output_csv}")


import pandas as pd
csv_file = "deap_ecg_data.csv"

# Load the CSV file
df = pd.read_csv(csv_file)

# Display the first 5 rows
print("Head (first 5 rows):")
print(df.head())

# Display 10 random rows
print("\nSample 10 random rows:")
print(df.sample(10))

# Display the last 5 rows
print("\nTail (last 5 rows):")
print(df.tail())
rows, columns = df.shape
print(f"The CSV file contains:\n➡️ {rows} rows\n➡️ {columns} columns")


import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.signal import butter, filtfilt
import numpy as np

# Load CSV
df = pd.read_csv("deap_ecg_data.csv")

# Drop missing values
df.dropna(inplace=True)

# Convert to float
df['ECG_Value'] = df['ECG_Value'].astype(float)

# Normalize ECG signal
scaler = StandardScaler()
df['ECG_Signal_Normalized'] = scaler.fit_transform(df[['ECG_Value']])

# ----- Step 5: Filter ECG signal using Butterworth bandpass filter -----
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype='band')

def apply_bandpass_filter(data, lowcut=0.5, highcut=45.0, fs=128.0):
    b, a = butter_bandpass(lowcut, highcut, fs)
    return filtfilt(b, a, data)

df['ECG_Signal_Filtered'] = apply_bandpass_filter(df['ECG_Value'])

# ----- Step 6: Encode Labels -----
# Convert valence/arousal/dominance/liking to binary classes (low/high)
df['Valence_Label'] = ['High' if v >= 5 else 'Low' for v in df['Valence']]
df['Arousal_Label'] = ['High' if v >= 5 else 'Low' for v in df['Arousal']]
df['Dominance_Label'] = ['High' if v >= 5 else 'Low' for v in df['Dominance']]
df['Liking_Label'] = ['High' if v >= 5 else 'Low' for v in df['Liking']]

# Optionally encode using LabelEncoder
le = LabelEncoder()
df['Valence_Label_Encoded'] = le.fit_transform(df['Valence_Label'])
df['Arousal_Label_Encoded'] = le.fit_transform(df['Arousal_Label'])

# Save final processed CSV
df.to_csv("deap_ecg_preprocessed.csv", index=False)
print("✅ Preprocessing complete. Saved to 'deap_ecg_preprocessed.csv'.")

import scipy.io
import numpy as np
import scipy.stats as stats
from scipy.signal import welch
import os

# -------------------- ECG Feature Function --------------------
def extract_ecg_features(ecg_signal, fs=128):
    mean_val = np.mean(ecg_signal)
    std_val = np.std(ecg_signal)
    skewness = stats.skew(ecg_signal)
    kurt = stats.kurtosis(ecg_signal)
    rms = np.sqrt(np.mean(ecg_signal ** 2))
    ptp = np.ptp(ecg_signal)
    energy = np.sum(ecg_signal ** 2)

    freqs, psd = welch(ecg_signal, fs)

    def bandpower(fmin, fmax):
        idx = np.logical_and(freqs >= fmin, freqs <= fmax)
        return np.trapz(psd[idx], freqs[idx])

    vlf = bandpower(0.003, 0.04)
    lf = bandpower(0.04, 0.15)
    hf = bandpower(0.15, 0.4)
    lf_hf_ratio = lf / hf if hf > 0 else 0

    return [
        mean_val, std_val, skewness, kurt, rms, ptp, energy,
        vlf, lf, hf, lf_hf_ratio
    ]

# -------------------- Load & Extract Features --------------------
X = []
y_valence = []
y_arousal = []

for i in range(1, 33):  # Subjects s01 to s32
    data_dir = "C:/Users/Udith/Downloads/data_preprocessed_matlab"  # or the full path: "C:/Users/YourName/Documents/deap_data"
    filename = os.path.join(data_dir, f"s{i:02d}.mat")

    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        continue

    mat = scipy.io.loadmat(filename)
    data = mat['data']  # shape: (40, 40, 8064)
    labels = mat['labels']  # shape: (40, 4)

    for trial in range(40):
        ecg_signal = data[trial, 38, :]  # ECG is channel 39 (index 38)
        features = extract_ecg_features(ecg_signal)
        X.append(features)
        y_valence.append(labels[trial, 0])
        y_arousal.append(labels[trial, 1])

X = np.array(X)
y_valence = np.array(y_valence)
y_arousal = np.array(y_arousal)

print("✅ Extracted features for all subjects")
print("Feature matrix shape:", X.shape)  # Should be (1280, 11)
print("Sample features:", X[0])


import pandas as pd

df = pd.DataFrame(X, columns=[
    'mean', 'std', 'skewness', 'kurtosis', 'rms', 'ptp', 'energy',
    'vlf_power', 'lf_power', 'hf_power', 'lf_hf_ratio'
])
df['valence'] = y_valence
df['arousal'] = y_arousal

df.to_csv("ecg_features_deap.csv", index=False)
print("✅ Saved features to ecg_features_deap.csv")
print("Sample labels - Valence:", y_valence[0], "Arousal:", y_arousal[0])


import pandas as pd

# Load the CSV
df = pd.read_csv("ecg_features_deap.csv")

# Show first 5 rows
print("🔹 Head (first 5 rows):")
print(df.head())

# Show last 5 rows
print("\n🔹 Tail (last 5 rows):")
print(df.tail())


import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import os

data_dir = "C:/Users/Udith/Downloads/data_preprocessed_matlab"  # Change this to your directory containing s01.mat ... s32.mat
trial_index = 0         # Which trial to plot (0 to 39)

plt.figure(figsize=(20, 15))
for i in range(1, 33):
    filename = os.path.join(data_dir, f"s{i:02d}.mat")
    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        continue

    mat = scipy.io.loadmat(filename)
    data = mat['data']  # (40 trials, 40 channels, 8064 samples)

    # Get ECG signal from trial 0 (or whichever you choose)
    ecg_signal = data[trial_index, 38, :]  # channel 39 is ECG

    plt.subplot(8, 4, i)  # 8 rows x 4 columns = 32 subplots
    plt.plot(ecg_signal[:1000])  # Plot only first 1000 samples for clarity
    plt.title(f"Subject {i:02d}")
    plt.axis('off')  # Clean look

plt.suptitle(f"ECG Signals from {trial_index + 1}st Trial of All Subjects", fontsize=18)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import os

data_dir = "C:/Users/Udith/Downloads/data_preprocessed_matlab"  # Change this to your directory containing s01.mat ... s32.mat
trial_index = 25         # Which trial to plot (0 to 39)

plt.figure(figsize=(20, 15))
for i in range(1, 33):
    filename = os.path.join(data_dir, f"s{i:02d}.mat")
    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        continue

    mat = scipy.io.loadmat(filename)
    data = mat['data']  # (40 trials, 40 channels, 8064 samples)

    # Get ECG signal from trial 0 (or whichever you choose)
    ecg_signal = data[trial_index, 38, :]  # channel 39 is ECG

    plt.subplot(8, 4, i)  # 8 rows x 4 columns = 32 subplots
    plt.plot(ecg_signal[:1000])  # Plot only first 1000 samples for clarity
    plt.title(f"Subject {i:02d}")
    plt.axis('off')  # Clean look

plt.suptitle(f"ECG Signals from {trial_index + 1}th Trial of All Subjects", fontsize=18)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


import pandas as pd

# Load your feature + label CSV
df = pd.read_csv("ecg_features_deap.csv")

# Define emotion based on valence-arousal space
def map_emotion(row):
    v, a = row['valence'], row['arousal']
    if v >= 5 and a >= 5:
        return "Happy"
    elif v >= 5 and a < 5:
        return "Relaxed"
    elif v < 5 and a >= 5:
        return "Angry"
    else:
        return "Sad"

# Add emotion label
df['Emotion'] = df.apply(map_emotion, axis=1)

# Save the labeled data
df.to_csv("ecg_features_with_emotions.csv", index=False)
print(df[['valence', 'arousal', 'Emotion']].head())


import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the features CSV
df = pd.read_csv("ecg_features_deap.csv")

# Optional: Check for 0 power values and print how many would be dropped
zero_power_rows = df[['vlf_power', 'lf_power', 'hf_power']].sum(axis=1) == 0
print(f"Rows with zero power values: {zero_power_rows.sum()}")

# Keep rows where total power is not zero
df = df[~zero_power_rows]

# Drop missing values
df.dropna(inplace=True)

# Feature columns
feature_cols = ['mean', 'std', 'skewness', 'kurtosis', 'rms', 'ptp', 
                'energy', 'vlf_power', 'lf_power', 'hf_power', 'lf_hf_ratio']

# Check if any data is left
if df.shape[0] == 0:
    print("No rows left after filtering. Check your data or filtering condition.")
else:
    # Normalize features
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    # Emotion label mapping (optional)
    def map_emotion(row):
        if row['valence'] >= 5 and row['arousal'] >= 5:
            return 'Happy'
        elif row['valence'] >= 5 and row['arousal'] < 5:
            return 'Relaxed'
        elif row['valence'] < 5 and row['arousal'] >= 5:
            return 'Angry'
        else:
            return 'Sad'

    df['emotion'] = df.apply(map_emotion, axis=1)

    # Save preprocessed data
    df.to_csv("ecg_preprocessed_features.csv", index=False)
    print("Preprocessing complete. Sample:")
    print(df.head())


import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load your CSV
df = pd.read_csv("ecg_features_deap.csv")  # replace with your actual file name

# Drop rows with NaN or infinite values (optional but useful)
df.replace([float('inf'), float('-inf')], pd.NA, inplace=True)
df.dropna(inplace=True)

# Option 1: Drop power features completely if they're all zeros
feature_cols = ['mean', 'std', 'skewness', 'kurtosis', 'rms', 'ptp', 'energy']

# Check if DataFrame is now empty
if df.shape[0] == 0:
    print("⚠️ No data available after cleaning. Please check your file or skip power features.")
else:
    # Normalize features
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    # Emotion labeling based on valence-arousal model
    def map_emotion(row):
        if row['valence'] >= 5 and row['arousal'] >= 5:
            return 'happy'
        elif row['valence'] < 5 and row['arousal'] >= 5:
            return 'angry'
        elif row['valence'] < 5 and row['arousal'] < 5:
            return 'sad'
        else:
            return 'relaxed'

    df['emotion'] = df.apply(map_emotion, axis=1)

    # ✅ Now you can visualize or train a model
    print(df[['emotion'] + feature_cols].head())
    df.to_csv("cleaned_ecg_features_deap.csv", index=False)


    import seaborn as sns
import matplotlib.pyplot as plt

# Pairplot by emotion
sns.pairplot(df, hue='emotion', vars=feature_cols)
plt.suptitle("Feature Distributions by Emotion", y=1.02)
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df, 
    x='valence', 
    y='arousal', 
    hue='emotion', 
    palette='Set2', 
    alpha=0.7,
    s=80
)

plt.axhline(0.5, color='gray', linestyle='--')  # Horizontal midpoint
plt.axvline(0.5, color='gray', linestyle='--')  # Vertical midpoint
plt.title('Emotion Distribution in Valence-Arousal Space')
plt.xlabel('Valence (Negative → Positive)')
plt.ylabel('Arousal (Calm → Excited)')
plt.grid(True)
plt.legend(title='Emotion', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

import pandas as pd
import matplotlib.pyplot as plt

# Load the cleaned data
df_raw = pd.read_csv('cleaned_ecg_features_deap.csv')

# Get the unique emotions
emotions = df_raw['emotion'].unique()

# Plot settings
plt.figure(figsize=(12, 8))

# Loop over each emotion and plot corresponding valence and arousal
for idx, emotion in enumerate(emotions):
    # Filter the data for the current emotion
    emotion_signals = df_raw[df_raw['emotion'] == emotion]

    # Extract valence and arousal values for plotting
    valence_data = emotion_signals['valence'].values
    arousal_data = emotion_signals['arousal'].values

    # Create a subplot for each emotion
    plt.subplot(len(emotions), 1, idx + 1)

    # Plot both valence and arousal on the same plot (two signals)
    plt.plot(valence_data, label='Valence', color='blue')
    plt.plot(arousal_data, label='Arousal', color='red')

    # Add titles and labels
    plt.title(f'{emotion} Emotion (Valence and Arousal)')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)

# Adjust layout and show the plot
plt.tight_layout()
plt.show()

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Load the data
df_raw = pd.read_csv('cleaned_ecg_features_deap.csv')

# Convert emotions to labels
label_encoder = LabelEncoder()
df_raw['emotion_label'] = label_encoder.fit_transform(df_raw['emotion'])

# Features: valence and arousal
X = df_raw[['valence', 'arousal']].values

# Target: emotion labels
y = df_raw['emotion_label'].values

# Reshape X to be 3D (samples, time_steps, features) for LSTM
X = np.expand_dims(X, axis=1)  # This adds the time_step dimension (1 time step per sample)

# Normalize the features (valence, arousal)
scaler = StandardScaler()
X = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

# One-hot encode the target labels
y = to_categorical(y)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the LSTM model using Input() to avoid the warning
model = Sequential()
model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))  # This explicitly defines the input shape
model.add(LSTM(units=64, return_sequences=False))
model.add(Dropout(0.2))  # Add dropout to reduce overfitting
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=y_train.shape[1], activation='softmax'))  # Output layer with softmax activation for multi-class classification

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Generate classification report and confusion matrix
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)  # Convert predictions to class labels
y_true = np.argmax(y_test, axis=1)  # Convert true labels to class labels

# Classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes, target_names=label_encoder.classes_))

# Confusion matrix
print("\nConfusion Matrix:")
conf_matrix = confusion_matrix(y_true, y_pred_classes)
print(conf_matrix)

# Plot training and validation accuracy/loss
# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Load the data
df_raw = pd.read_csv('cleaned_ecg_features_deap.csv')

# Convert emotions to labels
label_encoder = LabelEncoder()
df_raw['emotion_label'] = label_encoder.fit_transform(df_raw['emotion'])

# Features: all the necessary columns including 'valence' and 'arousal'
feature_cols = ['mean', 'std', 'skewness', 'kurtosis', 'rms', 'ptp', 'energy', 'valence', 'arousal']
X = df_raw[feature_cols].values

# Target: emotion labels
y = df_raw['emotion_label'].values

# Reshape X to be 3D (samples, time_steps, features) for LSTM
X = np.expand_dims(X, axis=1)  # Adding the time_step dimension (1 time step per sample)

# Normalize the features
scaler = StandardScaler()
X = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

# One-hot encode the target labels
y = to_categorical(y)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the LSTM model using Input() to avoid the warning
model = Sequential()
model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))  # Define the input shape explicitly
model.add(LSTM(units=64, return_sequences=False))
model.add(Dropout(0.2))  # Add dropout to reduce overfitting
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=y_train.shape[1], activation='softmax'))  # Output layer with softmax activation for multi-class classification

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Generate classification report
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)  # Convert predictions to class labels
y_true = np.argmax(y_test, axis=1)  # Convert true labels to class labels

# Classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes, target_names=label_encoder.classes_))


# Plot training and validation accuracy/loss
# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()



