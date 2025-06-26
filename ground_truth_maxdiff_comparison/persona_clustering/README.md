# Persona Clustering with Model2Vec

Python script for semantic clustering of persona descriptions using the fast and efficient Model2Vec embedding model.

## Overview

This tool analyzes persona descriptions by clustering them based on semantic similarity. It helps researchers identify patterns, themes, and groupings in their persona collections, which can provide valuable insights into persona design diversity and potential redundancies.

## Features

- **Fast Semantic Clustering**: Uses Model2Vec for fast, high-quality text embeddings
- **Zero-Shot Classification**: Optional advanced classification across multiple dimensions (seniority, industry, expertise, etc.)
- **Hybrid Clustering Approaches**: Combine semantic embeddings with classification features for enhanced analysis
- **Automatic Optimization**: Finds optimal number of clusters using silhouette analysis
- **Multiple Clustering Methods**: Supports K-means and DBSCAN algorithms with configurable parameters
- **Rich Interactive Visualizations**: 2D scatter plots with PCA or t-SNE, featuring hover tooltips and transparency controls
- **Comprehensive Analysis**: Detailed cluster statistics and common theme extraction
- **File Output**: Generates cluster assignments, classification results, and dissimilarity matrix files
- **Similarity Search**: Find semantically similar personas
- **Flexible Input**: Command-line and programmatic interfaces
- **Quality Metrics**: Silhouette scores and cluster quality assessments with intelligent recommendations

## Installation

### Prerequisites

- Python 3.8 or higher
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Quick Setup with uv (Recommended)

1. **Install uv** (if not already installed):
   ```bash
   # On macOS and Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # On Windows
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   
   # Or with pip
   pip install uv
   ```

2. **Create and activate virtual environment**:
   ```bash
   # Create virtual environment
   uv venv persona-clustering
   
   # Activate virtual environment
   # On macOS/Linux:
   source persona-clustering/bin/activate
   # On Windows:
   persona-clustering\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   uv pip install -r requirements.txt
   ```

### Alternative Setup with pip

```bash
# Create virtual environment
python -m venv persona-clustering

# Activate virtual environment
# On macOS/Linux:
source persona-clustering/bin/activate
# On Windows:
persona-clustering\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Manual Installation

If you prefer to install packages individually:

```bash
# Core dependencies only
uv pip install model2vec pandas scikit-learn matplotlib seaborn numpy plotly

# Include zero-shot classification features
uv pip install model2vec pandas scikit-learn matplotlib seaborn numpy plotly torch transformers tokenizers
```

## Zero-Shot Classification Features

The advanced version includes zero-shot classification capabilities that can automatically categorize personas across multiple dimensions without training data.

### Available Classification Dimensions

- **seniority**: entry level, junior, associate, senior, principal, staff, director
- **industry**: healthcare, financial services, gaming, enterprise tools, consumer tech, e-commerce, education, government, automotive, media, social media, developer tools, logistics, travel, hardware/IoT
- **expertise**: quantitative research, qualitative research, research operations, mixed methods, strategic research, prototyping, analytics, experimental design
- **priorities**: learning, strategic influence, team building, process optimization, stakeholder management, compliance, innovation, quality assurance
- **challenges**: technical/data, organizational/political, resource constraints, regulatory, stakeholder alignment, scale/complexity, user access, time pressure

### Zero-Shot Classification Usage

```bash
# Basic classification with default dimensions (seniority, industry, expertise)
python enhanced_persona_clustering.py UXR_personas.txt --use-classification

# Custom dimensions
python enhanced_persona_clustering.py UXR_personas.txt --use-classification --dimensions seniority industry

# Color visualization by classification dimension
python enhanced_persona_clustering.py UXR_personas.txt --use-classification --color-by seniority

# Hybrid approach combining semantic and classification features
python enhanced_persona_clustering.py UXR_personas.txt --use-classification --classification-method hybrid
```

### Classification Methods

1. **Semantic (Default)**: Uses Model2Vec embeddings for clustering, with classification as additional analysis
2. **Classification**: Clusters based purely on zero-shot classification features
3. **Hybrid**: Combines both semantic embeddings and classification features

### Model Information

- **Classification Model**: `typeform/distilbert-base-uncased-mnli`
- **Performance**: Typically 60-80% confidence on persona classifications
- **Processing Speed**: ~1-2 seconds per persona

## Quick Start

### Basic Usage

```python
from persona_clustering import run_persona_clustering

# Analyze your personas
results = run_persona_clustering('UXR_personas.txt')
```

### Command Line Usage

```bash
# Basic analysis with K-means
python persona_clustering.py UXR_personas.txt

# DBSCAN clustering with custom parameters
python persona_clustering.py UXR_personas.txt --method dbscan --eps 0.6 --min-samples 3

# Full customization
python persona_clustering.py UXR_personas.txt --clusters 6 --method dbscan --eps 0.7 --min-samples 2 --viz tsne --output my_results
```

## Data Requirements

Your personas file should contain persona descriptions separated by headers in the format:

```
=== PERSONA 1 ===
You are a Junior UX Researcher specializing in quantitative analysis...
Core expertise: Data analysis + quantitative methods...
Key priorities: Learning advanced statistical techniques...
Primary challenges: Limited autonomy in study design...

=== PERSONA 2 ===
You are a Senior UX Researcher specializing in mixed methods...
Core expertise: Analytics + quantitative methods...
...
```

The script automatically parses this format and extracts each persona description. **Note**: The script does NOT deduplicate personas - all personas in the file will be analyzed.

## Usage Examples

### 1. Simple Analysis

```python
# Run with default settings
results = run_persona_clustering('UXR_personas.txt')

# Access results
print(f"Found {results['n_clusters']} clusters")
print(f"Analyzed {len(results['personas'])} personas")
```

### 2. Custom Configuration

```python
# Use a larger model and DBSCAN clustering
results = run_persona_clustering(
    personas_file_path='UXR_personas.txt',
    model_name='minishlab/potion-base-32M',  # Larger, more accurate model
    clustering_method='dbscan',              # Use density-based clustering
    eps=0.6,                                 # DBSCAN distance parameter
    min_samples=3,                           # Minimum points per cluster
    visualization_method='tsne',             # Use t-SNE for visualization
    output_dir='my_results'                  # Custom output directory
)
```

### 3. Advanced Analysis

```python
# Access detailed results
personas = results['personas']
embeddings = results['embeddings']
dissimilarity_matrix = results['dissimilarity_matrix']
cluster_labels = results['cluster_labels']
cluster_analysis = results['cluster_analysis']

# Find similar personas
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

query_persona = "You are a data scientist working on machine learning projects"
query_embedding = results['model'].encode([query_persona])
similarities = cosine_similarity(query_embedding, embeddings)[0]

# Get top 5 most similar personas
top_indices = np.argsort(similarities)[-5:][::-1]
for idx in top_indices:
    print(f"Persona {idx} - Similarity: {similarities[idx]:.3f}")
    print(f"  {personas[idx][:100]}...")
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `personas_file` | Path to personas text file | Required |
| `--model` | Model2Vec model to use | `minishlab/potion-base-8M` |
| `--clusters` | Number of clusters (auto-determined if not specified) | Auto |
| `--method` | Clustering method (`kmeans` or `dbscan`) | `kmeans` |
| `--eps` | DBSCAN eps parameter: max distance between points in same cluster | `0.5` |
| `--min-samples` | DBSCAN min_samples parameter: min points to form a cluster | `3` |
| `--viz` | Visualization method (`pca` or `tsne`) | `pca` |
| `--output` | Output directory for results | `results` |

### DBSCAN Parameter Guidance

**For persona clustering, recommended starting points:**

| Persona Collection Size | Recommended eps | Recommended min_samples | Use Case |
|------------------------|-----------------|------------------------|----------|
| 20-50 personas | 0.6 | 2 | More clusters, smaller groups |
| 50-100 personas | 0.5 | 3 | Balanced clustering |
| 100+ personas | 0.4 | 4 | Fewer, tighter clusters |

**Parameter Effects:**
- **Higher eps** (0.6-0.8): More permissive, larger clusters, fewer noise points
- **Lower eps** (0.3-0.4): Stricter, smaller clusters, more noise points
- **Higher min_samples**: Requires more points to form clusters, reduces small clusters
- **Lower min_samples**: Allows smaller clusters, may create many tiny clusters

The script provides intelligent recommendations if your parameters aren't working well!

### Available Model2Vec Models

| Model | Size | Performance | Use Case |
|-------|------|-------------|----------|
| `minishlab/potion-base-2M` | ~8 MB | Good | Quick analysis, limited resources |
| `minishlab/potion-base-4M` | ~16 MB | Better | Balanced speed/accuracy |
| `minishlab/potion-base-8M` | ~30 MB | Best (default) | Recommended for most use cases |
| `minishlab/potion-base-32M` | ~120 MB | Highest | Maximum accuracy |

## Output Files

The script automatically generates files in the specified output directory:

### 1. Persona Clusters CSV (`persona_clusters_YYYYMMDD_HHMMSS.csv`)

Contains the cluster assignment for each persona:

```csv
persona_index,cluster_id
0,0
1,2
2,0
3,1
...
```

- `persona_index`: Zero-indexed position of persona in input file (0, 1, 2, ...)
- `cluster_id`: Assigned cluster identifier (0, 1, 2, ... or -1 for noise in DBSCAN)

### 2. Dissimilarity Matrix CSV (`dissimilarity_matrix_YYYYMMDD_HHMMSS.csv`)

A square matrix where each cell `(i,j)` contains the dissimilarity score (1 - cosine_similarity) between persona `i` and persona `j`:

```csv
,0,1,2,3,...
0,0.0,0.234,0.567,0.123,...
1,0.234,0.0,0.789,0.456,...
2,0.567,0.789,0.0,0.345,...
3,0.123,0.456,0.345,0.0,...
...
```

- Values range from 0 (identical) to 2 (completely opposite)
- Matrix is symmetric: `dissimilarity(i,j) == dissimilarity(j,i)`
- Diagonal values are always 0 (persona compared to itself)

### 3. Interactive Visualization HTML (`persona_clusters_visualization_YYYYMMDD_HHMMSS.html`)

Standalone HTML file containing the interactive Plotly visualization that can be:
- Opened in any web browser
- Shared with team members
- Embedded in presentations or reports
- Used for detailed cluster exploration

## Console Output

The script provides detailed console output including:

- Data loading statistics and validation
- Model download progress (if needed)
- Embedding generation with progress indicators
- Optimal cluster analysis with silhouette scores
- DBSCAN parameter recommendations and warnings
- Detailed cluster composition and common themes
- Quality metrics and summary statistics
- File output locations and access instructions

## Clustering Method Comparison

### K-means
- **Best for**: Well-separated, spherical clusters
- **Pros**: Fast, consistent results, good for exploration
- **Cons**: Requires pre-specifying number of clusters
- **Recommended when**: You have an idea of how many persona types you expect

### DBSCAN
- **Best for**: Natural groupings, handling outliers
- **Pros**: Finds clusters of varying shapes, identifies noise/outliers automatically
- **Cons**: Sensitive to parameter tuning, may struggle with varying densities
- **Recommended when**: You want to discover natural persona groupings without pre-assumptions

## Returned Results Dictionary

```python
results = {
    'personas': list,                     # List of persona descriptions
    'embeddings': np.ndarray,             # Model2Vec embeddings (n_personas, embedding_dim)
    'dissimilarity_matrix': np.ndarray,   # Pairwise dissimilarity matrix (n_personas, n_personas)
    'cluster_labels': np.ndarray,         # Cluster assignments for each persona
    'cluster_analysis': dict,             # Detailed cluster information
    'model': StaticModel,                 # Model2Vec model for further use
    'embeddings_2d': np.ndarray,          # 2D coordinates for visualization
    'n_clusters': int,                    # Number of clusters found
    'silhouette_scores': list,            # Clustering quality scores (if auto-determined)
    'output_files': {                     # Paths to generated files
        'cluster_file': str,
        'dissimilarity_file': str
    }
}
```

## Use Cases for Persona Research

### 1. Persona Quality Assessment
- Identify redundant or overly similar personas
- Ensure diverse representation across persona types
- Detect potential biases in persona design

### 2. Persona Collection Analysis
- Understand the conceptual space covered by your personas
- Optimize persona selection for balanced research studies
- Group personas by similarity for targeted analysis

### 3. Research Design
- Verify that personas represent distinct user segments
- Analyze the semantic relationships between different persona types
- Create persona taxonomies based on empirical clustering

### 4. Methodology Improvement
- Refine persona development processes
- Create templates for future persona creation
- Establish quality criteria for persona diversity

### 5. Similarity Analysis
- Find the most similar personas to a given description
- Identify outlier personas that don't fit established patterns
- Understand the semantic distance between different persona types

## Troubleshooting

### Common Issues

1. **DBSCAN returns only 1 cluster or too many noise points**
   - **Solution**: Increase `eps` (try 0.6-0.8) or decrease `min_samples` (try 2)
   - **The script will suggest specific parameter adjustments**

2. **DBSCAN finds too many small clusters**
   - **Solution**: Decrease `eps` (try 0.3-0.4) or increase `min_samples` (try 4-5)

3. **File format errors**
   - Ensure your personas file uses the exact format: `=== PERSONA X ===`
   - Check for consistent encoding (UTF-8 recommended)
   - Verify that each persona section contains content

4. **Memory errors with large datasets**
   - Use a smaller model (e.g., `potion-base-2M`)
   - Consider breaking very large persona collections into smaller batches

5. **Poor clustering results**
   - Try different clustering methods (kmeans vs dbscan)
   - Adjust DBSCAN parameters based on script recommendations
   - Use a larger Model2Vec model for better embeddings
   - Check if your personas are truly diverse enough to cluster

6. **Visualization display issues**
   - Switch between PCA and t-SNE for different perspectives
   - The HTML file can be opened in any modern web browser
   - Check that Plotly is properly installed

### Performance Tips

- Use `potion-base-8M` for best balance of speed and accuracy
- For very large persona collections (>500), consider sampling for initial exploration
- DBSCAN may be slower but often finds more meaningful natural clusters
- PCA is faster than t-SNE for visualization with large datasets
- The script shows progress bars for large collections

## Advanced Usage

### Fine-tuning DBSCAN Parameters

```bash
# Start with default parameters
python persona_clustering.py UXR_personas.txt --method dbscan

# If you get warnings about too few clusters, try:
python persona_clustering.py UXR_personas.txt --method dbscan --eps 0.7 --min-samples 2

# If you get too many noise points, try:
python persona_clustering.py UXR_personas.txt --method dbscan --eps 0.6 --min-samples 3

# For very tight, high-quality clusters:
python persona_clustering.py UXR_personas.txt --method dbscan --eps 0.4 --min-samples 4
```

### Custom Similarity Queries

```python
# Load results
results = run_persona_clustering('UXR_personas.txt')

# Find personas most similar to a custom description
custom_description = "Senior researcher focusing on healthcare AI applications"
custom_embedding = results['model'].encode([custom_description])

# Calculate similarities
from sklearn.metrics.pairwise import cosine_similarity
similarities = cosine_similarity(custom_embedding, results['embeddings'])[0]

# Get top matches
top_5_indices = np.argsort(similarities)[-5:][::-1]
print("Most similar personas to your description:")
for i, idx in enumerate(top_5_indices):
    print(f"{i+1}. Persona {idx} (similarity: {similarities[idx]:.3f})")
    print(f"   {results['personas'][idx][:100]}...")
```

### Cluster Analysis

```python
# Analyze specific clusters
cluster_analysis = results['cluster_analysis']

for cluster_id, info in cluster_analysis.items():
    print(f"\n--- Cluster {cluster_id} Analysis ---")
    print(f"Size: {info['size']} personas")
    print(f"Indices: {info['indices']}")
    print(f"Top themes: {[word for word, count in info['common_words'][:5]]}")
    
    # Calculate average dissimilarity within cluster
    cluster_indices = info['indices']
    dissim_matrix = results['dissimilarity_matrix']
    cluster_dissim = dissim_matrix[np.ix_(cluster_indices, cluster_indices)]
    avg_internal_dissim = np.mean(cluster_dissim[np.triu_indices_from(cluster_dissim, k=1)])
    print(f"Average internal dissimilarity: {avg_internal_dissim:.3f}")
```

### Export Analysis

```python
# Export detailed analysis to files
import json

# Save cluster analysis as JSON
with open('cluster_analysis.json', 'w') as f:
    # Convert numpy arrays to lists for JSON serialization
    exportable_analysis = {}
    for cluster_id, info in results['cluster_analysis'].items():
        exportable_analysis[cluster_id] = {
            'size': info['size'],
            'indices': info['indices'],
            'common_words': info['common_words'],
            'sample_personas': info['personas'][:3]  # Just first 3 for brevity
        }
    json.dump(exportable_analysis, f, indent=2)

# Export embeddings for external analysis
np.save('persona_embeddings.npy', results['embeddings'])
```

## Example Workflows

### Workflow 1: Initial Exploration
```bash
# Start with K-means to understand basic structure
python persona_clustering.py UXR_personas.txt --viz pca

# Then try DBSCAN to find natural groupings
python persona_clustering.py UXR_personas.txt --method dbscan --viz tsne
```

### Workflow 2: Parameter Optimization
```bash
# Start conservative
python persona_clustering.py UXR_personas.txt --method dbscan --eps 0.4 --min-samples 4

# If too restrictive, relax parameters
python persona_clustering.py UXR_personas.txt --method dbscan --eps 0.6 --min-samples 3

# If still too restrictive, try more permissive
python persona_clustering.py UXR_personas.txt --method dbscan --eps 0.7 --min-samples 2
```

### Workflow 3: Quality Assessment
```bash
# Use larger model for highest accuracy
python persona_clustering.py UXR_personas.txt --model minishlab/potion-base-32M --method dbscan --eps 0.5 --min-samples 3
```

The script's intelligent feedback will guide you toward optimal parameters for your specific persona collection!