#!/usr/bin/env python
# coding: utf-8

# # Wordcloud by using the PubMed API

# In[ ]:


get_ipython().system('pip install biopython')


# In[ ]:


from Bio import Entrez

# Enter your email address (required for using the Entrez API)
Entrez.email = ""

# Search for papers related to ChEMBL
search_query = "(ChEMBL[Title/Abstract]) AND 2010:2024[Date - Publication]"

# Use the Entrez API to search PubMed
handle = Entrez.esearch(db="pubmed", term=search_query, retmax=1000)  # retmax specifies the max number of records
record = Entrez.read(handle)

# Fetch the PubMed IDs of the articles
pmids = record["IdList"]

# Calculate and display the number of papers
num_papers = len(pmids)
print(f"Number of papers retrieved: {num_papers}")

# Fetch detailed information (including abstracts) using the PubMed IDs
handle = Entrez.efetch(db="pubmed", id=",".join(pmids), retmode="xml", rettype="abstract")
records = Entrez.read(handle)

# Extract and print the abstracts
abstracts = []
for r in records["PubmedArticle"]:
    try:
        # Extract the title and abstract
        title = r["MedlineCitation"]["Article"]["ArticleTitle"]
        abstract = r["MedlineCitation"]["Article"]["Abstract"]["AbstractText"][0]
        abstracts.append(f"Title: {title}\nAbstract: {abstract}\n")
    except KeyError:
        # Handle cases where no abstract is available
        abstracts.append(f"Title: {title}\nAbstract: Not available\n")

# Print the abstracts
for abstract in abstracts:
    print(abstract)


# In[ ]:


import nltk
from nltk.corpus import stopwords
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Function to preprocess and filter out stopwords (including custom ones)
def process_text(text_list):
    # Join all the abstracts into one large string
    full_text = ' '.join(text_list).lower()
    
    # Tokenize the text (split into words)
    tokens = nltk.word_tokenize(full_text)
    
    # Custom stopwords to remove (e.g., 'title', 'abstract')
    custom_stopwords = set(['however', 'developed','identified','development','approach', 'results','small','one','two','sub','based','compounds','discovery','search','also', 'available','databases','database','chemical', 'model', 'compound','used','chembl','title', 'abstract', 'data', 'information', 'study', 'analysis', 'different', 'using', 'sets', 'set','method', 'new'])
    
    # Combine NLTK's stopwords with your custom stopwords
    stop_words = set(stopwords.words('english')).union(custom_stopwords)
    
    # Filter out stopwords and non-alphabetical tokens
    filtered_tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    
    return filtered_tokens

# Function to count most common keywords
def get_most_common_keywords(tokens, n=30):
    freq = Counter(tokens)
    return freq.most_common(n)

# Function to visualize the most common keywords as a word cloud
# Enter path to file and filename
def visualize_keywords(keywords, filename=''):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(dict(keywords))
    
    # Save the word cloud image to a file
    wordcloud.to_file(filename)
    print(f"Word cloud saved successfully at: {filename}")
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()


# Process text to remove common stopwords (including 'title' and 'abstract')
tokens = process_text(abstracts)

# Get the most common keywords
common_keywords = get_most_common_keywords(tokens, 30)


# Visualize the keywords
visualize_keywords(common_keywords)


# Output most common keywords
print(common_keywords)


# # Topic modelling

# In[ ]:


get_ipython().system('pip install matplotlib==3.7.3')


# In[ ]:


get_ipython().system('pip install sentence-transformers')


# In[ ]:


from Bio import Entrez
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Configure Entrez API
Entrez.email = ""  

# Predefined labels for possible topics
labels = [
    "Machine Learning", "Artificial Intelligence", "Drug Inhibition", "Prediction", "Structure-Activity",
     "Similarity",  "SARSCOV2", "Screening",
    "Deep Learning", "QSAR", "Toxicity", "Repurposing", "Infectious Disease", "Neurodegenerative Disease", "Antibiotic Resistance", "Antimicrobial",
    "Molecular Modeling", "Precision Medicine", "Personalized Medicine",
     "Target-Based Drug Discovery"
]

# Initialize the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Adjust time period
def search_pubmed(term, max_results=1000):
    """Search PubMed for a given term and return a list of PubMed IDs, filtering by publication date (2010-2019)."""
    query = f"{term} AND ({'2010/01/01'[0:4]}[PDAT] : {'2019/12/31'[0:4]}[PDAT])"
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
    record = Entrez.read(handle)
    handle.close()
    return record["IdList"]


def fetch_pubmed_details(pubmed_ids):
    """Fetch details for a list of PubMed IDs."""
    ids = ",".join(pubmed_ids)
    handle = Entrez.efetch(db="pubmed", id=ids, rettype="medline", retmode="text")
    records = handle.read()
    handle.close()
    return records


def extract_titles_and_abstracts(records):
    """Extract titles and abstracts from PubMed records."""
    papers = []
    for record in records.split('\n\n'):
        title_match = re.search(r"TI  - (.+)", record)
        abstract_match = re.search(r"AB  - (.+)", record)
        title = title_match.group(1) if title_match else ""
        abstract = abstract_match.group(1) if abstract_match else ""
        if title or abstract:
            papers.append(title + " " + abstract)
    return papers


def preprocess_text(texts, exclude_words=None):
    """Preprocess text for topic modeling and exclude certain words."""
    if exclude_words is None:
        exclude_words = []  # Default to empty list if no words are specified for exclusion
    
    # Basic cleanup
    texts = [re.sub(r"[^a-zA-Z ]", "", text.lower()) for text in texts]
    
    # Remove the excluded words
    texts = [
        " ".join([word for word in text.split() if word not in exclude_words])
        for text in texts
    ]
    
    return texts

def perform_topic_modeling(texts, num_topics=5, num_words=10):
    """Perform topic modeling on a list of texts."""
    vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(texts)

    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(tfidf_matrix)

    topics = {}
    for idx, topic in enumerate(lda.components_):
        words = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-num_words:]]
        topics[f"Topic {idx + 1}"] = words

    return lda, tfidf_matrix, vectorizer, topics


def get_topic_label(top_words, model, labels):
    """Generate a representative label for a topic using predefined labels and word embeddings."""
    word_embeddings = model.encode(top_words)
    topic_embedding = np.mean(word_embeddings, axis=0).reshape(1, -1)
    label_embeddings = model.encode(labels)
    similarities = cosine_similarity(topic_embedding, label_embeddings)
    most_similar_idx = np.argmax(similarities)
    return labels[most_similar_idx]


def assign_labels_to_topics(topics, model, labels):
    """Assign labels to topics based on their top words using SentenceTransformer embeddings."""
    topic_labels = {}
    for topic_name, top_words in topics.items():
        label = get_topic_label(top_words, model, labels)
        topic_labels[topic_name] = label
    return topic_labels

# Enter filename

def visualize_topics_using_dictionary(topics, lda, vectorizer, topic_labels, filename=""):
    """Visualize topics using the given dictionary of topics and save the heatmap as a file."""
    topic_word_distributions = lda.components_ / lda.components_.sum(axis=1)[:, None]
    feature_names = vectorizer.get_feature_names_out()

    # Extract the top words and their weights for each topic
    data = []
    for topic_idx, top_words in enumerate(topics.values()):
        row = {word: topic_word_distributions[topic_idx][feature_names.tolist().index(word)]
               for word in top_words if word in feature_names}
        data.append(row)

    # Convert to DataFrame
    heatmap_data = pd.DataFrame(data).fillna(0)
    # Use topic labels as y-axis labels
    heatmap_data.index = [topic_labels.get(f"Topic {i+1}", f"Topic {i+1}") for i in range(len(topics))]

    # Plot heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, cmap="YlGnBu", cbar_kws={'label': 'Word Weight'})
    plt.title("Topic-Word Heatmap for papers 2010-2019", fontsize = 16)
    plt.xlabel("Words", fontsize = 14)
    plt.ylabel("Topics", fontsize = 14)
    plt.tight_layout()

    # Save the heatmap to a file
    plt.savefig(filename, dpi = 600)
    print(f"Heatmap saved as {filename}")
    plt.show()

def main():
    search_term = "ChEMBL"
    max_results = 1000

    # List of words to exclude
    exclude_words = ["using", "novel", "background", "chembl", "analysis", "study"]

    print("Searching PubMed...")
    pubmed_ids = search_pubmed(search_term, max_results)

    print(f"Found {len(pubmed_ids)} papers. Fetching details...")
    records = fetch_pubmed_details(pubmed_ids)

    print("Extracting titles and abstracts...")
    papers = extract_titles_and_abstracts(records)

    print(f"Preprocessing {len(papers)} papers...")
    # Pass the list of exclude words to the preprocess_text function
    preprocessed_papers = preprocess_text(papers, exclude_words=exclude_words)

    print("Performing topic modeling...")
    lda, tfidf_matrix, vectorizer, topics = perform_topic_modeling(preprocessed_papers)

    print("Identified Topics:")
    for topic, words in topics.items():
        print(f"{topic}: {', '.join(words)}")

    print("Generating single-word topic labels using SentenceTransformer...")
    topic_labels = assign_labels_to_topics(topics, model, labels)
    for topic, label in topic_labels.items():
        print(f"{topic}: {label}")

    print("Visualizing topics...")
    # Pass topic_labels to the visualize function
    visualize_topics_using_dictionary(topics, lda, vectorizer, topic_labels, filename="")

if __name__ == "__main__":
    main()
    

