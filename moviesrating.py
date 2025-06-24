import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
df = pd.read_csv('IMDb movies.csv')
df.head()
df.info()
df.describe()
df.isnull().sum()

# Drop irrelevant columns
df.drop(['imdb_title_id', 'title', 'original_title', 'description'], axis=1, inplace=True)
df = df.dropna(subset=['avg_vote', 'genre', 'director'])
df['year'] = df['year'].fillna(df['year'].mode()[0])
df['duration'] = df['duration'].fillna(df['duration'].median())
# Convert multi-label genre into binary columns
df['genre'] = df['genre'].apply(lambda x: x.split(',')[0])  # simplify to main genre
genre_dummies = pd.get_dummies(df['genre'], prefix='genre')
df['director'] = df['director'].fillna('Unknown')
le = LabelEncoder()
df['director_encoded'] = le.fit_transform(df['director'])
X = pd.concat([genre_dummies, df[['year', 'duration', 'director_encoded']]], axis=1)
y = df['avg_vote']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score:", r2_score(y_test, y_pred))

sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Rating")
plt.ylabel("Predicted Rating")
plt.title("Actual vs Predicted Movie Ratings")
plt.show()
