# Sentiment Analysis using Pre-Trained Roberta Model

This project demonstrates sentiment analysis using the Roberta model for natural language processing. The dataset used for training and testing is a collection of tweets labeled as positive or negative sentiment.

## Environment Setup

1. **Install Required Libraries**

   Open a new cell in your Colab notebook and run the following commands:

   ```python
   !pip install transformers torch tqdm pandas numpy matplotlib nltk wordcloud seaborn

## Mounting the Drive

If you are using Google Drive to store your dataset, make sure to mount your Google Drive in the Colab notebook. Add the following code in a new cell:

  ```
  from google.colab import drive
  drive.mount('/content/drive')
  ```

## Data Preprocessing
The notebook begins by loading the necessary libraries, installing required packages, and mounting Google Drive.

## Tokenization and Preprocessing
The dataset is preprocessed to remove URLs, user references, punctuations, and stopwords. Additionally, stemming and lemmatization are applied for further text normalization.

## Exploratory Data Analysis (EDA)
### Word Clouds
The notebook generates word clouds for both stemmed and lemmatized words to visualize the most frequent words in the dataset.

![image](https://github.com/Dev-Goyal02/NLP_Resources/assets/106668945/c363587d-04c3-479c-92a2-bdbfb92c7377)

![image](https://github.com/Dev-Goyal02/NLP_Resources/assets/106668945/f024bbb0-5ef0-40ee-a5a2-18dbcbd34228)


## Model Training and Zero-Shot Classification
The Roberta model is used for zero-shot classification. The notebook includes code for training the model on a labeled dataset and then performing zero-shot classification on a test dataset.

### Accuracy Calculation
The accuracy of the zero-shot classification is calculated and printed.

## Evaluation and Visualization
### Confusion Matrix

![image](https://github.com/Dev-Goyal02/NLP_Resources/assets/106668945/59463daf-8544-4854-acc1-013e64a2a848)

The notebook generates a confusion matrix to visualize the model's performance.


### Classification Report
The classification report, including precision, recall, and F1-score, is printed for detailed performance metrics.

Classification Report:
              precision    recall  f1-score   support

           0       0.48      0.56      0.52       100
           1       0.48      0.40      0.43       100

    accuracy                           0.48       200
   macro avg       0.48      0.48      0.48       200
weighted avg       0.48      0.48      0.48       

# Few Shot Learning

## Train-Test Split
```
from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(full_data, test_size=0.3)
train_data = train_data.reset_index(drop=True)
test_data = test_data.reset_index(drop=True)
positiv_samples["label"]=[1]*NUM_SAMPLES

full_data = pd.concat([negative_samples,  positiv_samples])

```

## Tokenization and Data Preprocessing
```
from transformers import RobertaTokenizer

tokenizer = RobertaTokenizer.from_pretrained('roberta-base',
                                             truncation=True,
                                             do_lower_case=True)
MAX_LEN = 130

train_tokenized_data = [tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=MAX_LEN,
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True
        )
        for text in train_data['text']]

test_tokenized_data = [tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=MAX_LEN,
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True
        )
        for text in test_data['text']]

```

## Model Training and Evaluation
```
from torch.utils.data import Dataset, DataLoader

# ... (rest of the code for defining SentimentData, DataLoader, model, optimizer, etc.)

# Set the number of epochs
num_epochs = 5  # You can adjust this value based on your preference

# Run the training loop
train_loop(num_epochs)

# ... (rest of the code for plotting and saving the model)

```
## Results and Visualizations

Include visualizations such as loss curves and accuracy trends.

## Model Save
```
save_path="./"
torch.save(model, save_path+'trained_roberta.pt')
print('All files saved')

```

# Hyperparameter Tuning for Custom Roberta Model

This project explores hyperparameter tuning for sentiment analysis using a custom Roberta model. The code includes training loops, model evaluation, and hyperparameter search using different configurations.

## Model Architecture

The custom Roberta model is defined with configurable hyperparameters such as the number of hidden layers, attention heads, and hidden size.

```
class CustomRobertaClass(torch.nn.Module):
    # ... (model architecture code)
```

# Training Loop and Evaluation
The training loop is encapsulated in the custom_train_loop function, which includes early stopping based on test loss. The train_and_evaluate function is then used to train and evaluate the model for a specific configuration.

```
def custom_train_loop(model, train_loader, test_loader, optimizer, scheduler, num_epochs, loss):
    # ... (training loop code)

def train_and_evaluate(config, model, train_loader, test_loader, num_epochs, loss):
    # ... (train and evaluate code)
```
# Hyperparameter Search
Hyperparameter tuning is performed using an exhaustive search over a defined space of hyperparameters. The search space includes variations in the number of hidden layers, attention heads, learning rates, optimizers, freezing pretrained layers, batch sizes, and loss functions.

```
hyperparameter_space = {
    'num_hidden_layers': [6, 9, 12],
    'num_attention_heads': [6, 9, 12],
    'hidden_size': [768],
    'learning_rate': [1e-5, 5e-5, 1e-4],
    'optimizer': [AdamW],
    'num_epochs': [4, 6],
    'freeze_pretrained': [True, False],
    'loss': [torch.nn.CrossEntropyLoss(), torch.nn.NLLLoss()],
    'batch_size': [1, 8, 16]
    ... (hyperparameter search code)
}
```
## Results Visualization
Results are visualized using bar charts to compare the impact of different hyperparameters on both training and test accuracy.

Feel free to explore the hyperparameter space further or customize the code according to your specific needs.

Enjoy experimenting with hyperparameter tuning for sentiment analysis using the custom Roberta model!

## Usage
Open the provided Colab notebook in a Colab environment.

Run each cell sequentially to execute the code.

Make sure to adjust any file paths or specific configurations based on your setup.

Ensure GPU acceleration is enabled in the Colab notebook.

## Note
This README assumes that you have access to a GPU accelerator type in your Colab environment.

If you encounter any issues related to missing NLTK resources, follow the provided instructions to download the necessary resources.

Feel free to experiment with different hyperparameters, models, or preprocessing techniques to improve performance.

Enjoy exploring sentiment analysis with the Roberta model!
