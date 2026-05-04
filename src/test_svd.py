import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

ratings = pd.read_csv('dataset/ratings.csv')
movies = pd.read_csv('dataset/movies.csv')
train_ratings, test_ratings = train_test_split(ratings, test_size=0.2, random_state=42)

user_movie_matrix = train_ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
user_ids = list(user_movie_matrix.index)
movie_ids = list(user_movie_matrix.columns)
user_id_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
movie_id_to_idx = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}

R = user_movie_matrix.values
user_ratings_mean = np.mean(R, axis=1)
R_demeaned = R - user_ratings_mean.reshape(-1, 1)

U, sigma, Vt = svds(R_demeaned, k=50)
sigma = np.diag(sigma)
all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
preds_df = pd.DataFrame(all_user_predicted_ratings, columns=movie_ids, index=user_ids)

predicted_train = []
actual_train = []
for _, row in train_ratings.iterrows():
    u = row['userId']
    m = row['movieId']
    if u in user_id_to_idx and m in movie_id_to_idx:
        predicted_train.append(preds_df.loc[u, m])
        actual_train.append(row['rating'])

rmse_svd = np.sqrt(mean_squared_error(actual_train, predicted_train))
mae_svd = mean_absolute_error(actual_train, predicted_train)
print(f"Train RMSE: {rmse_svd}, Train MAE: {mae_svd}")
