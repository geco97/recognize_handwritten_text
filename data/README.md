# Data Folder

This folder contains all datasets and image files used in the project.

## Folder Structure
- `raw`: Contains the original datasets downloaded from [EMNIST Letters](https://www.nist.gov/itl/products-and-services/emnist-dataset).
- `processed`: Contains preprocessed training and testing data in NumPy format.
- `new_images`: Contains user-uploaded images for prediction.

## Steps to Download and Preprocess Data
1. Download the dataset from the official EMNIST Letters website.
2. Place the downloaded file in the `raw` folder.
3. Run the preprocessing script:
   ```bash
   python src/data_processing.py
   ```
4. Preprocessed files will be saved in the processed folder.