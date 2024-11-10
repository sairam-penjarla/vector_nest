# Vector Nest 🪺

[![Website](https://img.shields.io/badge/Website-Visit-blue)](https://psairam9301.wixsite.com/website)
[![YouTube Channel](https://img.shields.io/badge/YouTube-Visit-red)](https://www.youtube.com/@sairampenjarla)
[![GitHub Page](https://img.shields.io/badge/GitHub-Visit-gray)](https://github.com/sairam-penjarla)
[![LinkedIn Page](https://img.shields.io/badge/LinkedIn-Visit-blue)](https://www.linkedin.com/in/sairam-penjarla-b5041b121/)
[![Instagram Page](https://img.shields.io/badge/Instagram-Visit-purple)](https://www.instagram.com/sairam.ipynb/)


### Installation

```bash
pip install vector_db
```

### ⚡ Project Details

The project is a **database management system for handling vector embeddings and metadata**. The main functionalities include creating a database, adding data (in the form of text, metadata, and embeddings), and performing queries based on cosine similarity. This project is ideal for use in AI applications where you need to search, filter, and organize large amounts of vector data.

#### Key Features:
- **Create a database**: You can create a new database with either `overwrite` or `append` mode.
- **Create a collection**: Define a collection to store documents (such as research papers) and their embeddings.
- **Add data**: Add synthetic or real data to the collection, including associated metadata and vector embeddings.
- **Search and retrieve**: Use cosine similarity to retrieve the most relevant documents to a query. Filters such as author or category can be applied.
- **Advanced queries**: Support for setting a similarity threshold to filter out low-relevance results.

## Example Usage

### **⚡ Creating and adding data to a collection:**

```python
import random

# Initialize the VectorNest
manager = VectorNest()

# Step 1: Create a database named 'research_database' with mode='overwrite' or 'append'
db_name = 'research_database'
manager.create_database(db_name, mode='overwrite')  # Use 'overwrite' to start fresh or 'append' to keep existing data

db_name = 'research_database'
manager.use_database(db_name)

# Step 2: Create a collection for storing research papers, with mode='overwrite' or 'append'
collection_name = 'research_papers'
manager.create_collection(collection_name, mode='overwrite')  # 'overwrite' replaces existing collection, 'append' keeps it if it exists

# Step 3: Generate synthetic data for research papers
authors = ["Alice Johnson", "Bob Smith", "Carol Lee", "David Wu", "Eve Brown"]
categories = ["AI", "Data Science", "Quantum Computing", "Cybersecurity", "Blockchain"]
publication_years = [2019, 2020, 2021, 2022, 2023]

def generate_fake_abstract(category):
    return f"This paper discusses advancements in {category}. It covers recent trends, methodologies, and potential future applications."

# Step 4: Add synthetic research papers to the collection
for i in range(50):  # Adding 50 synthetic papers
    title = f"Research Paper {i+1}"
    category = random.choice(categories)
    author = random.choice(authors)
    year = random.choice(publication_years)
    abstract = generate_fake_abstract(category)
    
    metadata = {
        "title": title,
        "author": author,
        "year": str(year),
        "category": category
    }
    manager.add_to_collection(collection_name, text=abstract, metadata=metadata)

```

### **⚡ Retrieving from collection:**

Example 1: Retrieve top 5 research papers similar to a specific topic, filtering by category

```python
query_text = "advancements in AI"
filters = {"category": "AI"}
top_n = 5
retrieved_texts = manager.retrieve_from_collection(collection_name, query_text, filters=filters, top_n=top_n)

print("\nTop 5 research papers similar to the query in the 'AI' category:")
for result in retrieved_texts:
    print(f"Text: {result['text']}\nMetadata: {result['metadata']}\nSimilarity: {result['similarity']}\n")

```


### **⚡ Close the database connection:**

```python
manager.close_connection()
```

### **⚡ Close the database connection:**

Example 2. Identify the most similar research papers in the entire collection, regardless of category, with a high similarity threshold

```python
db_name = 'research_database'
collection_name = 'research_papers'

manager.use_database(db_name)


query_text = "applications of blockchain in security"
filters = {'author': 'Carol Lee'}
top_n = 5
threshold = 0.01
retrieved_texts = manager.retrieve_from_collection(collection_name, query_text, filters=filters, top_n=top_n, threshold=threshold)

print("\nTop 5 research papers related to 'blockchain' with high similarity:")
for result in retrieved_texts:
    print(f"Text: {result['text']}\nMetadata: {result['metadata']}\nSimilarity: {result['similarity']}\n")
```

### More Information

For a detailed explanation and walkthrough of this project, check out the blog post on my website:

[Link to Blog Post](https://psairam9301.wixsite.com/website/post/vector-nest-a-database-for-vector-embeddings)

You can also watch the YouTube video on this project for further understanding:

[YouTube Video Link](https://www.youtube.com/@sairampenjarla)