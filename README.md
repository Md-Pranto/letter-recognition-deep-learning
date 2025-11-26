# EMNIST Handwritten Letter Classification (A–Z)

**CNN Model using TensorFlow/Keras** • ~92% Accuracy

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project implements a Convolutional Neural Network (CNN) to classify handwritten English letters (A–Z) using the **EMNIST Letters** dataset.  
After 100 epochs, the model achieves **~92% training accuracy** and **~87% validation accuracy**.
### I used 100k+ .mat format images and converted it to .npy format

---

## Project Structure

```text
project/
├── Data/                              # Supporting sample files
├── emnist-dataset/                    # Store 'emnist-letters.mat' here
├── 100epoch_90%accuracy_model.keras   # Saved trained model
├── letterclassify_version(1).ipynb    # Full training notebook
└── README.md
```
## Model Architecture
```
- LayerDetailsConv2D32 filters
-  3×3, ReLUMaxPooling2D2×2Conv2D64 filters
   -x3, ReLUMaxPooling2D2×2FlattenDense128 units
-  ReLUOutput Dense26 units, softmax
```
### Training Settings
```
Optimizer: adam
Loss: sparse_categorical_crossentropy
Input: 28×28 grayscale (normalized)
Batch size: 128
Epochs: 100
```

### Overfitting Reduction Methods

```
20% validation split
EarlyStopping (patience=10)
ReduceLROnPlateau (factor=0.5, patience=5)
Pixel normalization (/255.0)
```

## Model Performance

- MetricValueTraining Accuracy~95%
- Test Accuracy (approx)~92%

## How to Run

`Download the EMNIST Letters dataset:`
```
→ https://www.nist.gov/itl/products-and-services/emnist-dataset
```
or you can download it from my github.

Place emnist-letters.mat in the emnist-dataset/ folder.

#### Open and run the notebook:

```
jupyter notebook "letterclassify_version(1).ipynb" or
"letterclassify_version(2).ipynb"
```

Execute all cells **sequentially**.

## Loading the Saved Model
```
from tensorflow.keras.models import load_model
model = load_model("100epoch_90%accuracy_model.keras")
```

# image_array shape: (28, 28, 1)
prediction = model.predict(image_array[np.newaxis, ...])
letter = chr(65 + np.argmax(prediction))  # 0=A, 1=B, ..., 25=Z
print("Predicted letter:", letter)

# Requirements
```
BashPython >= 3.8
TensorFlow >= 2.10
NumPy
SciPy
Matplotlib
Jupyter Notebook
Install with:
Bashpip install tensorflow numpy scipy matplotlib jupyter
```
# Future Improvements

 - Test on custom handwritten images
 - Build a web app (Flask)
 - Convert to TensorFlow Lite for mobile
 - increase more accuracy with more data and also reduce overfit


**Star** this repo if you find it helpful!
Feel free to **fork** and **improve** the model .
