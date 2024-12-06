# **Sentiment Analysis Using Fine-Tuned BERT**  
### **By Ansh Sharma**  
---
# **Deployed Link**--
- ## Google Colab -- https://colab.research.google.com/drive/1xtIr_MBp--2J8Qhasv7I1Bv6KYLlSHJv?usp=sharing
- ## Wandb AI -- https://wandb.ai/anshsharma21050421/sentiment-analysis-api/runs/5neuwzts/overview

Welcome to my sentiment analysis project! This repository showcases the fine-tuning of a **BERT** model using **Hugging Face**, deployment on **WandB AI** and **Heroku** via **FastAPI**, and the complete workflow of testing, evaluating, and productizing the model for real-world applications.  

---
![logo_cover_image](https://github.com/user-attachments/assets/adbf19c2-6768-4158-b028-8aef76c0185e)



## **Project Overview**  

The project aims to classify sentiment in student reviews about campus events and amenities into **Positive**, **Negative**, or **Neutral** categories. This end-to-end implementation covers:  
- **Data Preparation**: Creating a custom dataset with `ReviewText` and `SentimentLabels`.  
- **Fine-Tuning**: Leveraging the pre-trained **BERT model** to adapt to the dataset.  
- **Evaluation**: Measuring performance with accuracy, F1-score, and confusion matrix.  
- **Visualization**: Showcasing sentiment distributions and insights via Matplotlib.  
- **Productization**: Deploying the model for public use via **FastAPI** on both local and cloud servers.  

---

## **Key Features**  

1. **Fine-Tuned BERT Model**  
   - Adapted pre-trained BERT using Hugging Face for sentiment classification.  
   - Optimized with **AdamW optimizer** and evaluated on metrics like **accuracy** and **F1-score**.  

2. **Deployment**  
   - **FastAPI**: Designed an intuitive REST API for real-time sentiment predictions.  
   - Hosted on **Heroku** and tested locally for seamless integration.  

3. **Monitoring & Logging**  
   - Tracked model performance and fine-tuning experiments on **WandB AI**.  

4. **Evaluation Metrics**  
   - Performance visualized with confusion matrices and sentiment distribution graphs.  

---

## **Technologies Used**  

- **Programming**: Python  
- **Frameworks**: Hugging Face Transformers, FastAPI  
- **Deployment**: Heroku, Local Server  
- **Monitoring**: WandB AI  
- **Visualization**: Matplotlib, Seaborn  

---

## **Project Workflow**  

### 1. **Dataset Creation**  
- Custom dataset with `ReviewText` and `SentimentLabels` (Positive, Negative, Neutral).  
- Preprocessing: Text cleaning, tokenization, and encoding of labels.  

### 2. **Model Fine-Tuning**  
- Base Model: **BERT (bert-base-uncased)**.  
- Added a classification head for sentiment analysis.  
- Optimizer: AdamW with learning rate scheduling.  
- Metrics: Accuracy, F1-score, and confusion matrix.

  ![Screenshot (4)](https://github.com/user-attachments/assets/3c63fb57-7ffc-41e3-9f57-fdb5f70b8563)
 

### 3. **Evaluation**  
- Tested on unseen data with detailed analysis of predictions.  
- Plotted sentiment distributions for better insights.
  

### 4. **Productization**  
- Built a REST API with **FastAPI**:  
  - Endpoint: `/predict`  
  - Input: `ReviewText`  
  - Output: Predicted sentiment and confidence score.  
- Deployed on **Heroku** and tested via a local server.  

---

## **Getting Started**  

### **Structure of the Repository** 
├── app.py              # FastAPI application code  
├── requirements.txt    # Python dependencies  
├── data/               # Dataset folder  
├── models/             # Saved fine-tuned BERT model  
├── visualizations/     # Evaluation metrics and graphs  
└── README.md           # Project documentation  
![Screenshot 2024-12-06 204819](https://github.com/user-attachments/assets/34a43424-48f8-46f7-bc28-aaf149482025)




