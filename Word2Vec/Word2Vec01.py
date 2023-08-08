import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Assuming df is your DataFrame and it has columns 'advert', 'price', 'rooms', 'footage'
df = pd.read_csv('your_data.csv')

# Preprocess and tokenize the text
stop_words = set(stopwords.words('english'))
df['tokenized_advert'] = df['advert'].apply(lambda x: [word for word in word_tokenize(
    x.lower()) if word.isalpha() and word not in stop_words])

# Train Word2Vec model
word2vec_model = Word2Vec(
    df['tokenized_advert'].to_list(), vector_size=100, min_count=2, window=5)

# Create document vectors by averaging word vectors for each advert
df['doc_vector'] = df['tokenized_advert'].apply(lambda x: np.mean(
    [word2vec_model.wv[word] for word in x if word in word2vec_model.wv.index_to_key], axis=0))

# Handle missing values (if any word is not in the Word2Vec vocabulary)
df = df.dropna()

# Split data into train and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Create train and test data for the model
X_train = np.array(train_df[['rooms', 'footage']].join(
    train_df['doc_vector'].apply(pd.Series)).values.tolist())
y_train = train_df['price'].values

X_test = np.array(test_df[['rooms', 'footage']].join(
    test_df['doc_vector'].apply(pd.Series)).values.tolist())
y_test = test_df['price'].values

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"Mean Absolute Error: {mae}, Mean Squared Error: {mse}")
