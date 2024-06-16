import pandas as pd

# Load datasets
movies_data = pd.read_csv('ml-latest-small/movies.csv')
ratings_data = pd.read_csv('ml-latest-small/ratings.csv')

# Display the dimensions of the datasets
print('Movies dataframe shape:', movies_data.shape)
print('Ratings dataframe shape:', ratings_data.shape)

# Display the first few rows of each dataframe
print('First few rows of movies dataframe:\n', movies_data.head())
print('First few rows of ratings dataframe:\n', ratings_data.head())

# Create a mapping from movie IDs to movie titles
movie_titles = movies_data.set_index('movieId')['title'].to_dict()

# Calculate the number of unique users and movies
unique_users = ratings_data.userId.nunique()
unique_movies = ratings_data.movieId.nunique()

print("Unique users count:", unique_users)
print("Unique movies count:", unique_movies)

# Calculate the sparsity of the rating matrix
total_elements = unique_users * unique_movies
print("Total elements in the rating matrix:", total_elements)

# Display the number of ratings and the matrix density
num_ratings = len(ratings_data)
matrix_density = (num_ratings / total_elements) * 100
print("Number of ratings:", num_ratings)
print(f"Matrix density: {matrix_density:.4f}%")
print("The matrix is very sparse.")

import torch
import numpy as np
from torch.autograd import Variable
from tqdm import tqdm

class MFModel(torch.nn.Module):
    def __init__(self, num_users, num_movies, num_factors=20):
        super(MFModel, self).__init__()
        self.user_factors = torch.nn.Embedding(num_users, num_factors)
        self.movie_factors = torch.nn.Embedding(num_movies, num_factors)
        self.user_factors.weight.data.uniform_(0, 0.05)
        self.movie_factors.weight.data.uniform_(0, 0.05)
        
    def forward(self, data):
        users, movies = data[:, 0], data[:, 1]
        return (self.user_factors(users) * self.movie_factors(movies)).sum(1)
    
    def predict(self, user, movie):
        return self.forward(user, movie)

from torch.utils.data import Dataset, DataLoader

class RatingsDataset(Dataset):
    def __init__(self):
        self.ratings = ratings_data.copy()
        users = ratings_data.userId.unique()
        movies = ratings_data.movieId.unique()
        
        self.user_id_map = {u: i for i, u in enumerate(users)}
        self.movie_id_map = {m: i for i, m in enumerate(movies)}
        
        self.idx_to_user_id = {i: u for u, i in self.user_id_map.items()}
        self.idx_to_movie_id = {i: m for m, i in self.movie_id_map.items()}
        
        self.ratings.movieId = self.ratings.movieId.apply(lambda x: self.movie_id_map[x])
        self.ratings.userId = self.ratings.userId.apply(lambda x: self.user_id_map[x])
        
        self.inputs = torch.tensor(self.ratings[['userId', 'movieId']].values)
        self.targets = torch.tensor(self.ratings['rating'].values, dtype=torch.float32)

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]

    def __len__(self):
        return len(self.ratings)

epochs = 128
use_cuda = torch.cuda.is_available()
print("Using GPU:", use_cuda)

model = MFModel(unique_users, unique_movies, num_factors=8)
if use_cuda:
    model = model.cuda()

loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_data = RatingsDataset()
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)

for epoch in tqdm(range(epochs)):
    losses = []
    for inputs, targets in train_loader:
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        predictions = model(inputs)
        loss = loss_function(predictions.squeeze(), targets)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {np.mean(losses):.4f}")

user_factors = model.user_factors.weight.data.cpu().numpy()
movie_factors = model.movie_factors.weight.data.cpu().numpy()

from sklearn.cluster import KMeans

num_clusters = 10
kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(movie_factors)

for cluster_num in range(num_clusters):
    print(f"Cluster {cluster_num}")
    cluster_movies = []
    for idx in np.where(kmeans.labels_ == cluster_num)[0]:
        movie_id = train_data.idx_to_movie_id[idx]
        rating_count = ratings_data[ratings_data['movieId'] == movie_id].shape[0]
        cluster_movies.append((movie_titles[movie_id], rating_count))
    for movie in sorted(cluster_movies, key=lambda x: x[1], reverse=True)[:10]:
        print(f"\t{movie[0]}")

