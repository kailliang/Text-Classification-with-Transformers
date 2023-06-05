# Text-Classification-with-Transformers
## Performing text classification with MobileBERT, DistilBERT, and BERT-Base

Code breakdown:

1. Imports necessary libraries and modules. Sets configurations for ignoring warnings and sets random states, epochs, batch sizes and learning rate for the model.
2. Reads and processes a job data CSV file from a directory. This includes converting certain columns to datetime format, handling missing data, encoding categorical variables, and generating new features based on date and time.
3. Generates a new feature 'on_time', which is a binary indicator of whether the job was finished on time or not.
4. Creates new features based on the time of job reporting and the target finish time.
5. Combines various features to generate a text representation of each record.
6. Tokenizes this text representation using a pre-trained BERT tokenizer. The text is also padded and attention masks are created.
7. Splits the tokenized data into a training and validation set.
8. Transforms the labels using LabelEncoder.
9. Computes class weights to handle potential class imbalance.
10. Creates PyTorch DataLoaders for training and validation sets.
11. Initializes a pre-trained BERT model for sequence classification. Several alternative models are commented out.
12. Sets up the loss function to be Cross Entropy Loss, with class weights for handling class imbalance. Also, sets up the AdamW optimizer and a learning rate scheduler.
13. Trains the model on the training set for a given number of epochs. After each epoch, it evaluates the model on the validation set.
14. For every 10 epochs, it prints the training loss, accuracy on the validation set, and a confusion matrix.
15. After training, it computes and prints the precision of the model on the validation set.
16. It computes and prints the Area Under the Receiver Operating Characteristic Curve (AUC-ROC) for the model on the validation set.
17. Finally, it plots the ROC curve.
