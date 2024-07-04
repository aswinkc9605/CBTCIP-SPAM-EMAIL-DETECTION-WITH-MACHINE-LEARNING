# Spam Email Classifier

This project implements a spam email classifier using machine learning techniques in Python. It utilizes the scikit-learn library for text preprocessing, feature extraction (TF-IDF vectorization), and model training (Random Forest Classifier). The classifier distinguishes between spam and non-spam emails based on the dataset provided.

## Features

- **Text Preprocessing**: Converts text to lowercase and removes punctuation.
- **TF-IDF Vectorization**: Transforms text data into numerical features using TF-IDF.
- **Random Forest Classifier**: Trains a classifier to predict whether an email is spam or not.

## Setup

To run the project locally, follow these steps:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/aswinkc9605/SPAM-EMAIL-DETECTION-WITH-MACHINE-LEARNING.git
   cd SPAM-EMAIL-DETECTION-WITH-MACHINE-LEARNING
   ```

2. **Install dependencies:**

   Ensure you have Python installed (preferably Python 3.6+). Install the required libraries using pip:

   ```bash
   pip install pandas scikit-learn
   ```

3. **Dataset**

   Place your dataset in CSV format with columns 'v1' (label: spam/ham) and 'v2' (email text) in the root directory of the project. Update the `file_path` variable in the script to point to your dataset.

4. **Run the script:**

   Execute the Python script to preprocess data, train the classifier, evaluate its performance, and classify new emails:

   ```bash
   python sample.py
   ```

5. **Output**

   The script will output evaluation metrics (accuracy, precision, recall, F1-score) and save classified emails into `spam_emails.txt` and `non_spam_emails.txt`.

## Usage

To classify new emails:

1. Modify the `Spam.csv` list in the script with new email examples.
2. Run the script again (`python sample.py`).

## Contributing

Contributions are welcome! If you have any suggestions, improvements, or bug fixes, feel free to open an issue or create a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

