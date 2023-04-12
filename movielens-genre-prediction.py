# -*- coding: utf-8 -*-
"""PyData Workshop Notebook.ipynb
"""
# TODO: Run linter on this file
# TODO: Run through the notebook and make sure it works
# TODO: Run through Grammarly
# TODO: Uncomment install commands
# TODO: Check headlines
# TODO: Improve hyperparameters
# TODO: discuss use-case with Matthias
# TODO: Uncomment asserts

# This notebook runs faster on a GPU.
# You can enable GPU acceleration by going to Edit -> Notebook Settings -> Hardware Accelerator -> GPU

import torch
from torch import Tensor

print(torch.__version__)

# Install required packages.
import os

os.environ['TORCH'] = torch.__version__

#!pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}.html
#!pip install torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}.html
#!pip install pyg-lib -f https://data.pyg.org/whl/nightly/torch-${TORCH}.html
#!pip install git+https://github.com/pyg-team/pytorch_geometric.git
"""# Node Prediction on MovieLens

This colab notebook shows how to load a set of `*.csv` files as input and construct a heterogeneous graph from it.
We will then use this dataset as input into a [heterogeneous graph model](https://pytorch-geometric.readthedocs.io/en/latest/notes/heterogeneous.html#hgtutorial), and use it for the task of node prediction.

We are going to use the [MovieLens dataset](https://grouplens.org/datasets/movielens/) collected by the GroupLens research group.
This toy dataset describes movies, tagging activity and ratings from MovieLens.
The dataset contains approximately 100k ratings across more than 9k movies from more than 600 users.
We are going to use this dataset to generate two node types holding data for movies and users, respectively, and one edge type connecting users and movies, representing the relation of whether a user has rated a specific movie.

The node prediction task then predicts missing movie genres.

## Heterogeneous Graph Creation

First, we download the dataset to an arbitrary folder (in this case, the current directory):
"""

from torch_geometric.data import download_url, extract_zip

dataset_name = 'ml-latest'

url = f'https://files.grouplens.org/datasets/movielens/{dataset_name}.zip'
extract_zip(download_url(url, '.'), '.')

movies_path = f'./{dataset_name}/movies.csv'
ratings_path = f'./{dataset_name}/ratings.csv'
tags_path = f'./{dataset_name}/tags.csv'
"""Before we create the heterogeneous graph, let’s take a look at the data."""

import pandas as pd

# Load the entire ratings data frame into memory:
ratings_df = pd.read_csv(ratings_path)

# Load the entire movie data frame into memory:
movies_df = pd.read_csv(movies_path)

# Load the entire tags data frame into memory:
tags_df = pd.read_csv(tags_path)

print('movies.csv:')
print('===========')
print(movies_df[["movieId", "genres", "title"]].head())
print(f"Number of movies: {len(movies_df)}")
print()
print('ratings.csv:')
print('============')
print(ratings_df[["userId", "movieId", "rating"]].head())
print(f"Number of ratings: {len(ratings_df)}")
print()
print('tags.csv:')
print('============')
print(tags_df[['userId', 'movieId', 'tag']].head())
print(f"Number of tags: {len(tags_df)}")
"""We are going to use the `genres` column in the `movie.csv` as the target of our node prediction task. 
Every movie is assigned to one or more genres. For simplicity, we are going to use only the most popular genres in 
our node prediction task. Moreover, we filter out movies that are assigned to more than one popular genre. This is to
avoid having to predict multiple genres for each movie. 
"""

# Filter out movies that do not belong to the most popular genres:
num_genres = 3
genres = movies_df['genres'].str.get_dummies('|')
popular_genres = genres.sum().sort_values(ascending=False).index[:num_genres]
movies_df = movies_df[movies_df['genres'].str.contains(
    '|'.join(popular_genres))]

# Filter out movies that are assigned to more than one popular genre:
genres = movies_df['genres'].str.get_dummies('|')
movies_df = movies_df[genres[popular_genres].sum(axis=1) == 1].reset_index(
    drop=True)

# Split genres and convert into indicator variables:
genres = movies_df['genres'].str.get_dummies('|')
genres = genres[popular_genres]
print(genres[popular_genres].head())

# Use genres as movie targets:
movie_target = torch.from_numpy(genres.values).to(torch.float)
"""Let's split the movies into train, validation, and test sets. We are going to use 80% of the movies for training, 10% for validation, and 10% for testing.
"""
from sklearn.model_selection import train_test_split

num_movies = len(movies_df)

# Next, split the indices of the nodes into train, validation, and test sets using `train_test_split`
train_idx, valtest_idx = train_test_split(range(num_movies),
                                          train_size=0.8,
                                          test_size=0.2,
                                          random_state=42)
val_idx, test_idx = train_test_split(valtest_idx,
                                     train_size=0.5,
                                     test_size=0.5,
                                     random_state=42)
"""After preparing the target column and the split we can look at the data again and prepare the features for every movie.

We see that the `movies.csv` file provides the additional columns: `movieId` and `title`.
The `movieId` assigns a unique identifier to each movie. The `title` column contains both the title and the release year of the movie.
We are going to split the year from the title and use both as features to predict the movie genre.
The title needs to be converted into a representation that can help the model to learn the relationship between movies. We are using a
bag-of-words representation of the title.
"""

# Extract the year from the title field and create a new 'year' column
movies_df['year'] = movies_df['title'].str.extract('(\(\d{4}\))', expand=False)
movies_df['year'] = movies_df['year'].str.extract('(\d{4})', expand=False)

# Remove the year and any trailing/leading whitespace from the title
movies_df['title'] = movies_df['title'].str.replace('(\(\d{4}\))', '')
movies_df['title'] = movies_df['title'].str.strip()

# Use a CountVectorizer to create a bag-of-words representation of the titles
from sklearn.feature_extraction.text import CountVectorizer
title_vectorizer = CountVectorizer(stop_words='english', analyzer='char_wb', ngram_range=(2, 2)) #, max_features=10000)
title_vectorizer.fit(movies_df['title'].iloc[train_idx]) 
title_features = title_vectorizer.transform(movies_df['title'])

# Create binary indicator variables for the year
year_indicator = pd.get_dummies(movies_df['year']).values

import numpy as np

movie_feat = np.concatenate((title_features.toarray(), year_indicator), axis=1)
"""Now, as we prepared the features based on the title and the year, we can add the tag features to the movie features as well. 
Similar to the title, we are using a bag-of-words representation of the tags. Finally, we combine the tag features with the title and year features into a single feature matrix for every movie."""

# Filter out tags that do not belong to the movies in the training set
tags_df = tags_df[tags_df['movieId'].isin(movies_df.index)]
tags_df = tags_df[tags_df['tag'].notna()]
tags_df = tags_df.groupby('movieId')['tag'].apply(
    lambda x: ' '.join(x)).reset_index('movieId')

# Merge the tags into the movies data frame
movies_df = pd.merge(movies_df, tags_df, on='movieId', how='left')
movies_df['tag'].fillna('', inplace=True)
tag_vectorizer = CountVectorizer(stop_words='english', analyzer='char_wb', ngram_range=(2, 2)) #, max_features=10000)
tag_vectorizer.fit(movies_df['tag'].iloc[train_idx])
tag_features = tag_vectorizer.transform(movies_df['tag'])

# Combine all movie features into a single feature matrix
movie_feat = np.concatenate((movie_feat, tag_features.toarray()), axis=1)
movie_feat = torch.tensor(movie_feat, dtype=torch.float)
"""The `ratings.csv` data connects users (as given by `userId`) and movies (as given by `movieId`).
Due to simplicity, we do not make use of the additional `timestamp` and `rating` information.
Here, we create a mapping that maps entry IDs to a consecutive value in the range `{ 0, ..., num_rows - 1 }`.
This is needed as we want our final data representation to be as compact as possible, *e.g.*, the representation of a movie in the first row should be accessible via `x[0]`.

Afterwards, we obtain the final `edge_index` representation of shape `[2, num_ratings]` from `ratings.csv` by merging mapped user and movie indices with the raw indices given by the original data frame.
"""

# Filter ratings to only include movies in movies_df
ratings_df = ratings_df[ratings_df['movieId'].isin(movies_df.index)]

# Create a mapping from unique user indices to range [0, num_user_nodes):
unique_user_id = ratings_df['userId'].unique()
unique_user_id = pd.DataFrame(data={
    'userId': unique_user_id,
    'mappedID': pd.RangeIndex(len(unique_user_id)),
})
print("Mapping of user IDs to consecutive values:")
print("==========================================")
print(unique_user_id.head())
print()
# Create a mapping from unique movie indices to range [0, num_movie_nodes):
unique_movie_id = ratings_df['movieId'].unique()
unique_movie_id = pd.DataFrame(data={
    'movieId': unique_movie_id,
    'mappedID': pd.RangeIndex(len(unique_movie_id)),
})
print("Mapping of movie IDs to consecutive values:")
print("===========================================")
print(unique_movie_id.head())

# Perform merge to obtain the edges from users and movies:
ratings_user_id = pd.merge(ratings_df['userId'],
                           unique_user_id,
                           left_on='userId',
                           right_on='userId',
                           how='left')
ratings_user_id = torch.from_numpy(ratings_user_id['mappedID'].values)
ratings_movie_id = pd.merge(ratings_df['movieId'],
                            unique_movie_id,
                            left_on='movieId',
                            right_on='movieId',
                            how='left')
ratings_movie_id = torch.from_numpy(ratings_movie_id['mappedID'].values)

# With this, we are ready to construct our `edge_index` in COO format
# following PyG semantics:
edge_index_user_to_movie = torch.stack([ratings_user_id, ratings_movie_id],
                                       dim=0)
# assert edge_index_user_to_movie.size() == (2, 100836)

print()
print("Final edge indices pointing from users to movies:")
print("=================================================")
print(edge_index_user_to_movie)
"""With this, we are ready to initialize our `HeteroData` object and pass the necessary information to it.
Note that we also pass in a `node_id` vector to each node type in order to reconstruct the original node indices from sampled subgraphs.
We also take care of adding reverse edges to the `HeteroData` object.
This allows our GNN model to use both directions of the edge for message passing:
"""

import torch_geometric.transforms as T
from torch_geometric.data import HeteroData

data = HeteroData()

# Save node indices:
data["user"].node_id = torch.arange(len(unique_user_id))
data["movie"].node_id = torch.arange(len(movies_df))

# Add the node features and edge indices:
data["movie"].x = movie_feat
data["user", "rates", "movie"].edge_index = edge_index_user_to_movie
data["user", "rates",
     "movie"].edge_attr = torch.from_numpy(ratings_df["rating"].to_numpy())

# We also need to make sure to add the reverse edges from movies to users
# in order to let a GNN be able to pass messages in both directions.
# We can leverage the `T.ToUndirected()` transform for this from PyG:
data = T.ToUndirected()(data)

# Add the node labels:
data['movie'].y = movie_target

print(data)

assert data.node_types == ["user", "movie"]
assert data.edge_types == [("user", "rates", "movie"),
                           ("movie", "rev_rates", "user")]
# assert data["user"].num_nodes == 610
# assert data["user"].num_features == 0
# assert data["movie"].num_nodes == 9742
# assert data["movie"].num_features == 9180  # TODO: try out with other number of features
#assert data["user", "rates", "movie"].num_edges == 100836
#assert data["movie", "rev_rates", "user"].num_edges == 100836
#assert data["movie", "rev_rates", "user"].num_edge_features == 1
"""We can now split our data into train, validation, and test sets based on the indices of the movies."""

# First, create a numpy array with the same number of rows as your dataset, and fill it with False values
data['movie'].train_mask = np.zeros(data['movie'].num_nodes, dtype=bool)
data['movie'].test_mask = np.zeros(data['movie'].num_nodes, dtype=bool)
data['movie'].val_mask = np.zeros(data['movie'].num_nodes, dtype=bool)

# Update the corresponding indices in the mask array to True for the train, validation, and test sets
data['movie'].train_mask[train_idx] = True
data['movie'].val_mask[val_idx] = True
data['movie'].test_mask[test_idx] = True

assert data['movie'].train_mask.sum() == int(0.8 * data['movie'].num_nodes)
# assert data['movie'].val_mask.sum() == int(0.1 * data['movie'].num_nodes)
# assert data['movie'].test_mask.sum() == int(0.1 * data['movie'].num_nodes) + 1
"""## Defining Mini-batch Loaders

We are now ready to create a mini-batch loader that will generate subgraphs that can be used as input into our GNN.
While this step is not strictly necessary for small-scale graphs, it is absolutely necessary to apply GNNs on larger graphs that do not fit onto GPU memory otherwise.
Here, we make use of the [`loader.LinkNeighborLoader`](https://pytorch-geometric.readthedocs.io/en/latest/modules/loader.html#torch_geometric.loader.LinkNeighborLoader) which samples multiple hops from a node and creates a subgraph from it.
"""

from torch_geometric.loader import NeighborLoader

train_loader = NeighborLoader(
    data,
    # Sample 30 neighbors for each node and edge type for 2 iterations
    num_neighbors={key: [30] * 2
                   for key in data.edge_types},
    # Use a batch size of 128 for sampling training nodes of type paper
    batch_size=256,
    input_nodes=('movie', data['movie'].train_mask),
    shuffle=True,
)
val_loader = NeighborLoader(
    data,
    # Sample 30 neighbors for each node and edge type for 2 iterations
    num_neighbors={key: [30] * 2
                   for key in data.edge_types},
    # Use a batch size of 128 for sampling training nodes of type paper
    batch_size=128,
    input_nodes=('movie', data['movie'].val_mask),
)

sampled_hetero_data = next(iter(train_loader))
print(sampled_hetero_data['movie'].batch_size)

# Inspect a sample:
sampled_data = next(iter(train_loader))

print("Sampled mini-batch:")
print("===================")
print(sampled_data)

assert sampled_data["movie"].batch_size == 128
"""## Creating a Heterogeneous GNN

We are now ready to create our heterogeneous GNN.
The GNN is responsible for learning enriched node representations from the surrounding subgraphs, which can be then used to derive node-level predictions.
For defining our heterogenous GNN, we make use of [`nn.SAGEConv`](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.SAGEConv) and the [`nn.to_hetero()`](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.to_hetero_transformer.to_hetero) function, which transforms a GNN defined on homogeneous graphs to be applied on heterogeneous ones.
"""

import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, to_hetero


class GNN(torch.nn.Module):

    def __init__(self, hidden_channels, num_genres):
        super().__init__()

        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, num_genres)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class Model(torch.nn.Module):

    def __init__(self, hidden_channels, num_genres):
        super().__init__()
        # Since the dataset does not come with rich features for the user, we also learn an
        # embedding matrix for users. For the movies, we use a linear layer to transform
        # the features into the same dimensionality as the user embeddings.
        self.user_emb = torch.nn.Embedding(data["user"].num_nodes,
                                           hidden_channels)
        self.movie_lin = torch.nn.Linear(data["movie"].x.shape[1],
                                         hidden_channels)

        # Instantiate homogeneous GNN:
        self.gnn = GNN(hidden_channels, num_genres)

        # Convert GNN model into a heterogeneous variant:
        self.gnn = to_hetero(self.gnn, metadata=data.metadata())

    def forward(self, data: HeteroData) -> Tensor:
        x_dict = {
          "user": self.user_emb(data["user"].node_id),
          "movie": self.movie_lin(torch.ones_like(data["movie"].x)) # self.movie_lin(data["movie"].x),
        } 

        # `x_dict` holds feature matrices of all node types
        # `edge_index_dict` holds all edge indices of all edge types
        x = self.gnn(x_dict, data.edge_index_dict)

        # Return the node embeddings for the movie nodes:
        return x['movie']

model = Model(hidden_channels=8, num_genres=num_genres)

print(model)
"""## Training a Heterogeneous GNN

Training our GNN is then similar to training any PyTorch model.
We move the model to the desired device, and initialize an optimizer that takes care of adjusting model parameters via stochastic gradient descent.

The training loop then iterates over our mini-batches, applies the forward computation of the model, computes the loss from ground-truth labels and obtained predictions, and adjusts model parameters via back-propagation and stochastic gradient descent.
"""

import torch.nn.functional as F
import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: '{device}'")

model = model.to(device)
for lr in [0.01, 0.005, 0.001]:
    print(f"Learning rate: {lr}")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

for epoch in range(1, 10):
    pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch:02d}")
    for batch in pbar:
        model.train()
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch)
        train_logits = logits[:batch['movie'].batch_size]
        train_y = batch["movie"].y[:batch["movie"].batch_size].argmax(dim=1)
        loss = F.cross_entropy(train_logits, train_y)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        model.eval()
        logits = model(data.cuda())
        train_logits = logits[data["movie"].train_mask].argmax(-1)
        val_logits = logits[data["movie"].val_mask].argmax(-1)
        test_logits = logits[data["movie"].test_mask].argmax(-1)
        train_y = data["movie"].y[data["movie"].train_mask].argmax(dim=1)
        val_y = data["movie"].y[data["movie"].val_mask].argmax(dim=1)
        test_y = data["movie"].y[data["movie"].test_mask].argmax(dim=1)

        print('Train Acc:',
              (train_logits == train_y).sum().item() / train_y.size(0))
        print('Val Acc:', (val_logits == val_y).sum().item() / val_y.size(0))
        print('Test Acc:',
              (test_logits == test_y).sum().item() / test_y.size(0))

        # val_loss = F.cross_entropy(val_logits, val_y)
        # val_acc = torch.sum(val_logits.argmax(
        #     dim=1) == val_y).item() / batch["movie"].val_mask.sum().item()

    # print(
    # f"Epoch: {epoch:03d}, Train Loss: {loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}"
    # )
"""## Evaluating a Heterogeneous GNN

After training, we evaluate our model on useen data coming from the test set.

"""

from sklearn.metrics import classification_report

with torch.no_grad():
    data.to(device)
    test_logits = model(data)[data['movie'].test_mask]
    test_y = data['movie'].y[data['movie'].test_mask].argmax(dim=1)
    test_acc = torch.sum(test_logits.argmax(dim=1) == test_y).item() / data['movie'].test_mask.sum().item()
    print(f"Test Accuracy: {test_acc:.4f}")

    print(
        classification_report(test_y.cpu().numpy(),
                              test_logits.argmax(dim=1).cpu().numpy(),
                              target_names=genres.columns))
