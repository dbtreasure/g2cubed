import numpy as np
from sklearn.decomposition import PCA
import plotly.express as px
import pandas as pd
import json

with open("g2g-response-embeddings.json", "r", encoding="utf-8") as file:
    response = json.load(file)
    embeddings = [term["snippet_embedding"] for term in response]
    # Assuming 'topic' and 'full_text' are keys in your JSON objects
    topics = [term["term"] for term in response]  # Replace 'topic' with the correct key
    full_texts = [term["snippet"] for term in response]  # Replace 'full_text' with the correct key

embeddings_array = np.array(embeddings)
print(embeddings_array.shape)

# Creating PCA model and transforming embeddings
pca_model = PCA(n_components=2)
pca_embeddings_values = pca_model.fit_transform(embeddings_array)
print(pca_embeddings_values.shape)

# Creating a DataFrame for Plotly
df = pd.DataFrame({
    'x': pca_embeddings_values[:, 0],
    'y': pca_embeddings_values[:, 1],
    'topic': topics,
    'full_text': full_texts
})

# Using an existing qualitative color scale from Plotly
fig = px.scatter(
    df,
    x='x',
    y='y',
    color='topic',
    hover_name='full_text',
    title='PCA embeddings',
    width=800,
    height=600,
    color_discrete_sequence=px.colors.qualitative.Plotly
)

fig.update_layout(
    xaxis_title='First Component',
    yaxis_title='Second Component'
)
fig.show()
