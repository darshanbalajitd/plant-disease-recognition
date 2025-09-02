# Plant Disease Detection App
## Description
This application uses deep learning (MobileNetV2) to detect plant diseases from leaf images. It is designed for educational and experimental purposes and may not be highly accurate due to limited training data.

![Demo]([./assets/demo.png](https://github.com/darshanbalajitd/plant-disease-recognition/blob/main/demo.png))

## Dataset
Get the original dataset here: [https://www.kaggle.com/datasets/itselif/comprehensive-plant-disease-dataset]

## Requirements
- Python 3.x
- PyTorch, torchvision, streamlit, PIL
- A GPU is recommended for faster training and inference, but CPU will also work.

## Note
- Before running the program, make sure to adjust the number of workers based on the number of CPU cores available on your machine. This ensures optimal performance and resource utilization.
## Setup Instructions
1. Download the GitHub Repository
   
   - Clone or download this repository to your local machine.
   - Run:
     ```
     git clone https://github.com/darshanbalajitd/plant-disease-recognition.git
     ```
2. Preprocess the Data
   
   - Edit the paths in pre-processing.py to match your dataset location.
   - Run:
     ```
     python pre-processing.py
     ```
3. Train the Model
   
   - Edit the paths in train_MobileNetV2.py as needed.
   - Run:
     ```
     python train_MobileNetV2.py
     ```
4. Start the App
   
   - Edit the paths in app.py if necessary.
   - Run the app using Streamlit:
     ```
     streamlit run app.py
     ```
## Usage Tips
- For best results, upload a clear image of a single leaf.
- The prediction accuracy may be limited due to the small training dataset.
- Results are for demonstration and may not be suitable for real-world diagnosis.
