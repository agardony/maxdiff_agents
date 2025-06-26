import pandas as pd
import numpy as np
from model2vec import StaticModel
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import Counter
import re
import os
from datetime import datetime
import sys
import warnings

# Import transformers for zero-shot classification
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    warnings.warn("transformers not installed. Zero-shot classification features will be unavailable.")

# Fix tokenizers parallelism warning and multiprocessing issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"  # Limit OpenMP threads
os.environ["MKL_NUM_THREADS"] = "1"  # Limit MKL threads

# Add multiprocessing fix for macOS
import multiprocessing
if hasattr(multiprocessing, 'set_start_method'):
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set

# Classification dimensions and their labels
CLASSIFICATION_DIMENSIONS = {
    "seniority": [
        "junior level researcher",
        "associate level researcher",
        "senior level researcher",
        "principal level researcher",
        "staff level researcher",
        "director level researcher",
        "ux researcher"
    ],
    "industry": [
        "healthcare and medical technology",
        "financial services and banking",
        "gaming and entertainment",
        "business software and enterprise tools",
        "consumer technology products",
        "e-commerce and retail",
        "education technology",
        "government and civic technology",
        "automotive technology",
        "media and content platforms",
        "social media platforms",
        "developer tools and platforms",
        "logistics and supply chain",
        "travel and hospitality",
        "hardware and IoT devices",
        "non-profit and social impact",
        "cybersecurity",
        "consulting",
        "climate technology",
        "two-sided marketplaces"
    ],
    "expertise": [
        "quantitative research and data analysis",
        "research operations and process management",
        "mixed methods research",
        "prototyping and design research",
        "analytics and metrics",
        "strategic research and planning",
        "experimental design and A/B testing"
    ],
    "priorities": [
        "learning and skill development",
        "strategic influence and business impact",
        "team building and mentoring",
        "process optimization and efficiency",
        "compliance and regulatory adherence",
        "quality assurance and rigor",
        "innovation and methodology development",
        "stakeholder management and communication"
    ],
    "challenges": [
        "technical and data challenges",
        "organizational and political challenges",
        "resource and budget constraints",
        "regulatory and compliance requirements",
        "stakeholder alignment and buy-in",
        "scale and complexity management",
        "user access and recruitment",
        "time pressure and quick turnaround"
    ]
}

def load_and_explore_personas(file_path):
    """
    Load and explore the personas file
    """
    print(f"{'='*60}")
    print(f"LOADING PERSONAS")
    print(f"{'='*60}")
    print(f"Reading file: {file_path}")
    
    # Read the file content
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        print("✓ File loaded successfully")
    except FileNotFoundError:
        print(f"✗ Error: File '{file_path}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Error reading file: {e}")
        sys.exit(1)
    
    print("Parsing persona sections...")
    
    # Parse personas based on the format === PERSONA X ===
    personas = []
    persona_blocks = re.split(r'=== PERSONA \d+ ===', content)
    
    # Remove empty first element and process each persona block
    for i, block in enumerate(persona_blocks[1:]):
        if block.strip():
            # Clean up the text but preserve all content
            persona_text = block.strip()
            personas.append(persona_text)
    
    print(f"✓ Successfully parsed {len(personas)} personas")
    
    if len(personas) == 0:
        print("✗ Error: No personas found in file. Check format.")
        print("Expected format: === PERSONA X === followed by persona content")
        sys.exit(1)
    
    # Basic statistics
    persona_lengths = [len(persona) for persona in personas]
    print(f"\nPersona statistics:")
    print(f"  Count: {len(personas)}")
    print(f"  Average length: {np.mean(persona_lengths):.0f} characters")
    print(f"  Length range: {min(persona_lengths)} - {max(persona_lengths)} characters")
    
    # Show sample personas
    print(f"\nFirst 3 personas (preview):")
    for i, persona in enumerate(personas[:3]):
        preview = persona.replace('\n', ' ')[:120]
        print(f"  {i}: {preview}...")
    
    return personas

def preprocess_personas(personas):
    """
    Clean and preprocess persona text for better clustering
    Note: We don't deduplicate as requested
    """
    print(f"\n{'='*60}")
    print(f"PREPROCESSING PERSONAS")
    print(f"{'='*60}")
    print(f"Cleaning and preparing {len(personas)} personas...")
    print("Note: Preserving all personas (no deduplication)")
    
    cleaned_personas = []
    
    for i, persona in enumerate(personas):
        # Convert to string and clean
        persona_str = str(persona)
        
        # Remove extra whitespace
        persona_str = re.sub(r'\s+', ' ', persona_str.strip())
        
        # Remove any special characters that might cause issues
        persona_str = re.sub(r'[^\w\s.,!?;:-]', '', persona_str)
        
        cleaned_personas.append(persona_str)
        
        # Show progress for large collections
        if len(personas) > 50 and (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{len(personas)} personas...")
    
    print(f"✓ Preprocessing complete")
    print(f"✓ {len(cleaned_personas)} personas ready for analysis")
    
    return cleaned_personas

def setup_zero_shot_classifier():
    """
    Initialize the zero-shot classification model
    """
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("transformers package required for zero-shot classification. Install with: pip install transformers torch")
    
    print(f"\n{'='*60}")
    print(f"INITIALIZING ZERO-SHOT CLASSIFIER")
    print(f"{'='*60}")
    print("Loading BART-large-MNLI model for classification...")
    print("This may take a moment on first run...")
    
    try:
        # Add these settings to prevent multiprocessing issues on macOS
        import torch
        torch.set_num_threads(1)  # Limit threads to prevent conflicts
        
        # Check if GPU is available and use it (prefer MPS on Mac, then CUDA)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
            device_name = "GPU (MPS)"
        elif torch.cuda.is_available():
            device = 0
            device_name = "GPU (CUDA)"
        else:
            device = -1  # CPU
            device_name = "CPU"
        print(f"Device set to use {device_name.lower()}")
        
        classifier = pipeline(
            "zero-shot-classification", 
            model="facebook/bart-large-mnli",
            device=device,  # Use GPU if available, CPU otherwise
            torch_dtype=torch.float32,  # Explicit dtype for compatibility
            return_all_scores=True  # Get all scores for better analysis
        )
        print("✓ Zero-shot classifier loaded successfully")
        return classifier
    except Exception as e:
        print(f"✗ Error loading classifier: {e}")
        print("Trying fallback approach with CPU...")
        
        # Fallback: try with CPU if GPU fails
        try:
            import torch
            torch.set_num_threads(1)
            
            classifier = pipeline(
                "zero-shot-classification", 
                model="facebook/bart-large-mnli",
                device=-1,  # Force CPU for fallback
                framework="pt"  # Explicitly use PyTorch
            )
            print("✓ Zero-shot classifier loaded successfully (CPU fallback mode)")
            return classifier
        except Exception as e2:
            print(f"✗ Fallback also failed: {e2}")
            print("Make sure you have torch and transformers installed:")
            print("  pip install torch transformers")
            raise

def classify_personas_by_dimensions(personas, dimensions, classifier):
    """
    Classify personas across specified dimensions using zero-shot classification
    """
    print(f"\n{'='*60}")
    print(f"ZERO-SHOT CLASSIFICATION")
    print(f"{'='*60}")
    print(f"Classifying {len(personas)} personas across {len(dimensions)} dimensions:")
    for dim in dimensions:
        print(f"  • {dim} ({len(CLASSIFICATION_DIMENSIONS[dim])} categories)")
    
    results = {}
    
    for dimension in dimensions:
        print(f"\nClassifying dimension: {dimension}")
        labels = CLASSIFICATION_DIMENSIONS[dimension]
        dimension_results = []
        
        # Process in smaller batches to avoid memory issues
        batch_size = 5  # Smaller batch size for stability
        total_batches = (len(personas) + batch_size - 1) // batch_size
        
        print(f"Processing in {total_batches} batches of {batch_size}...")
        
        for batch_idx in range(0, len(personas), batch_size):
            batch_end = min(batch_idx + batch_size, len(personas))
            batch_personas = personas[batch_idx:batch_end]
            
            print(f"  Batch {batch_idx//batch_size + 1}/{total_batches}: personas {batch_idx}-{batch_end-1}")
            
            for i, persona in enumerate(batch_personas):
                actual_idx = batch_idx + i
                try:
                    # Add error handling and retry logic
                    max_retries = 3
                    for retry in range(max_retries):
                        try:
                            result = classifier(persona, labels)
                            dimension_results.append({
                                'label': result['labels'][0],
                                'confidence': result['scores'][0],
                                'all_scores': dict(zip(result['labels'], result['scores']))
                            })
                            break  # Success, exit retry loop
                        except Exception as retry_error:
                            if retry < max_retries - 1:
                                print(f"    Retry {retry + 1} for persona {actual_idx}...")
                                import time
                                time.sleep(1)  # Brief pause before retry
                                continue
                            else:
                                raise retry_error
                                
                except Exception as e:
                    print(f"    Warning: Classification failed for persona {actual_idx}: {e}")
                    # Fallback to unknown classification
                    dimension_results.append({
                        'label': 'unknown',
                        'confidence': 0.0,
                        'all_scores': {'unknown': 1.0}
                    })
            
            # Brief pause between batches to prevent memory buildup
            if batch_idx + batch_size < len(personas):
                import time
                time.sleep(0.5)
        
        results[dimension] = dimension_results
        
        # Show classification summary
        labels_count = Counter([r['label'] for r in dimension_results])
        avg_confidence = np.mean([r['confidence'] for r in dimension_results])
        print(f"  ✓ Average confidence: {avg_confidence:.3f}")
        print(f"  ✓ Top categories: {dict(labels_count.most_common(3))}")
    
    print(f"\n✓ Zero-shot classification complete across all dimensions")
    return results

def create_classification_features(classification_results, method='categorical'):
    """
    Convert classification results into feature vectors for clustering
    
    Parameters:
    -----------
    classification_results : dict
        Results from classify_personas_by_dimensions
    method : str
        'categorical' - use label encodings
        'confidence' - use confidence score vectors
        'hybrid' - combine both approaches
    """
    print(f"\n{'='*60}")
    print(f"CREATING CLASSIFICATION FEATURE VECTORS")
    print(f"{'='*60}")
    print(f"Method: {method}")
    
    n_personas = len(next(iter(classification_results.values())))
    dimensions = list(classification_results.keys())
    
    if method == 'categorical':
        # One-hot encode the predicted labels
        feature_vectors = []
        feature_names = []
        
        for persona_idx in range(n_personas):
            persona_features = []
            
            for dimension in dimensions:
                # Get all unique labels for this dimension
                all_labels = CLASSIFICATION_DIMENSIONS[dimension]
                
                # Create one-hot encoding for this persona's prediction
                dimension_features = np.zeros(len(all_labels))
                predicted_label = classification_results[dimension][persona_idx]['label']
                if predicted_label in all_labels:
                    label_idx = all_labels.index(predicted_label)
                    dimension_features[label_idx] = 1.0
                
                persona_features.extend(dimension_features)
                
                # Add feature names (only for first persona)
                if persona_idx == 0:
                    feature_names.extend([f"{dimension}_{label}" for label in all_labels])
            
            feature_vectors.append(persona_features)
        
        feature_matrix = np.array(feature_vectors)
        
    elif method == 'confidence':
        # Use confidence scores for all possible labels
        feature_matrix = []
        feature_names = []
        
        for persona_idx in range(n_personas):
            persona_features = []
            
            for dimension in dimensions:
                all_labels = CLASSIFICATION_DIMENSIONS[dimension]
                result = classification_results[dimension][persona_idx]
                
                # Create confidence vector for all possible labels
                confidence_vector = []
                for label in all_labels:
                    confidence = result['all_scores'].get(label, 0.0)
                    confidence_vector.append(confidence)
                
                persona_features.extend(confidence_vector)
                
                # Add feature names (only for first persona)
                if persona_idx == 0:
                    feature_names.extend([f"{dimension}_conf_{label}" for label in all_labels])
            
            feature_matrix.append(persona_features)
        
        feature_matrix = np.array(feature_matrix)
        
    elif method == 'hybrid':
        # Combine categorical and confidence approaches
        print("  Creating categorical features...")
        cat_features, _ = create_classification_features(classification_results, 'categorical')
        print("  Creating confidence features...")
        conf_features, _ = create_classification_features(classification_results, 'confidence')
        
        # Combine features
        feature_matrix = np.hstack([cat_features, conf_features])
        feature_names = None  # Would be too long to track
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Normalize features
    scaler = StandardScaler()
    feature_matrix = scaler.fit_transform(feature_matrix)
    
    print(f"✓ Created feature matrix shape: {feature_matrix.shape}")
    print(f"✓ Features per persona: {feature_matrix.shape[1]}")
    
    return feature_matrix, feature_names

def cluster_personas_with_model2vec(personas, model_name="minishlab/potion-base-8M"):
    """
    Cluster personas using Model2Vec embeddings
    """
    print(f"\n{'='*60}")
    print(f"LOADING MODEL2VEC MODEL")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    
    # Check if model needs to be downloaded
    try:
        print("Checking if model is cached locally...")
        # Try to load without downloading first
        model = StaticModel.from_pretrained(model_name)
        print("✓ Model loaded from local cache")
    except Exception as e:
        print("Model not found locally, downloading...")
        print("This may take a few moments depending on your internet connection...")
        
        # Show download progress
        print(f"Downloading {model_name}...")
        model = StaticModel.from_pretrained(model_name)
        print("✓ Model downloaded and loaded successfully")
    
    print(f"\n{'='*60}")
    print(f"GENERATING EMBEDDINGS")
    print(f"{'='*60}")
    print(f"Processing {len(personas)} personas...")
    print("This may take a moment for large persona collections...")
    
    # Show progress for large collections
    if len(personas) > 50:
        print("Progress: [", end="", flush=True)
        batch_size = max(1, len(personas) // 20)  # Show 20 progress markers
        
        embeddings_list = []
        for i in range(0, len(personas), batch_size):
            batch = personas[i:i+batch_size]
            batch_embeddings = model.encode(batch)
            embeddings_list.append(batch_embeddings)
            print("▓", end="", flush=True)
        
        print("] Complete!")
        embeddings = np.vstack(embeddings_list)
    else:
        # For smaller collections, process all at once
        embeddings = model.encode(personas)
        print("✓ Embeddings generated")
    
    print(f"✓ Generated embeddings shape: {embeddings.shape}")
    print(f"✓ Embedding dimensions: {embeddings.shape[1]}")
    
    return embeddings, model

def find_optimal_clusters(embeddings, max_clusters=15):
    """
    Find optimal number of clusters using silhouette analysis
    """
    print(f"\n{'='*60}")
    print(f"FINDING OPTIMAL NUMBER OF CLUSTERS")
    print(f"{'='*60}")
    print(f"Testing cluster counts from 2 to {min(max_clusters, len(embeddings)-1)}...")
    print("Using silhouette analysis to determine optimal clustering...")
    
    silhouette_scores = []
    k_range = range(2, min(max_clusters + 1, len(embeddings)))
    
    print("\nCluster analysis progress:")
    for i, k in enumerate(k_range):
        if k >= len(embeddings):
            break
        
        print(f"  Testing K={k}...", end="", flush=True)
        
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Calculate silhouette score
        score = silhouette_score(embeddings, cluster_labels, metric='cosine')
        silhouette_scores.append(score)
        
        print(f" Silhouette Score = {score:.3f}")
    
    # Find optimal k
    optimal_k = k_range[np.argmax(silhouette_scores)]
    best_score = max(silhouette_scores)
    
    print(f"\n✓ Analysis complete!")
    print(f"✓ Optimal number of clusters: {optimal_k}")
    print(f"✓ Best silhouette score: {best_score:.3f}")
    
    return optimal_k, silhouette_scores

def perform_clustering(embeddings, personas, n_clusters=None, method='kmeans', eps=0.3, min_samples=2):
    """
    Perform clustering on persona embeddings
    """
    if n_clusters is None and method == 'kmeans':
        n_clusters, _ = find_optimal_clusters(embeddings)
    
    print(f"\n{'='*60}")
    print(f"PERFORMING {method.upper()} CLUSTERING")
    print(f"{'='*60}")
    
    if method == 'kmeans':
        print(f"Target clusters: {n_clusters}")
        print(f"Method: {method}")
        print("Running clustering algorithm...")
        print("  Initializing K-means with multiple random starts...")
        clustering_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        print("  Fitting model to embeddings...")
        cluster_labels = clustering_model.fit_predict(embeddings)
        print("  ✓ K-means clustering complete")
        
    elif method == 'dbscan':
        print(f"Method: {method}")
        print(f"Parameters: eps={eps}, min_samples={min_samples}")
        print("Running clustering algorithm...")
        print("  Initializing DBSCAN with cosine metric...")
        clustering_model = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        print("  Fitting model to embeddings...")
        cluster_labels = clustering_model.fit_predict(embeddings)
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        print(f"  ✓ DBSCAN found {n_clusters} clusters")
        
        # Report noise points if any
        noise_count = sum(1 for label in cluster_labels if label == -1)
        if noise_count > 0:
            print(f"  ✓ Identified {noise_count} noise/outlier points")
        
        # Give recommendations if clustering didn't work well
        if n_clusters <= 1:
            print(f"  ⚠️  Warning: Only {n_clusters} cluster(s) found!")
            print(f"  💡 Try increasing eps (currently {eps}) to be more permissive")
            print(f"  💡 Or decreasing min_samples (currently {min_samples}) to allow smaller clusters")
        elif noise_count > len(embeddings) * 0.5:
            print(f"  ⚠️  Warning: High noise ratio ({noise_count}/{len(embeddings)} points)")
            print(f"  💡 Consider increasing eps or decreasing min_samples")
    
    # Calculate clustering quality metrics
    if len(set(cluster_labels)) > 1:
        print("  Calculating quality metrics...")
        silhouette_avg = silhouette_score(embeddings, cluster_labels, metric='cosine')
        print(f"  ✓ Silhouette Score: {silhouette_avg:.3f}")
    
    print(f"✓ Clustering complete with {n_clusters} clusters")
    
    return cluster_labels, n_clusters

def analyze_persona_clusters(personas, cluster_labels, n_clusters, classification_results=None):
    """
    Analyze and interpret the persona clusters
    """
    print(f"\n{'='*60}")
    print(f"PERSONA CLUSTER ANALYSIS ({n_clusters} clusters)")
    print(f"{'='*60}")
    
    cluster_analysis = {}
    
    for cluster_id in range(n_clusters):
        cluster_personas = [personas[i] for i, label in enumerate(cluster_labels) if label == cluster_id]
        cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
        
        print(f"\n--- Cluster {cluster_id} ({len(cluster_personas)} personas) ---")
        print(f"Persona indices: {cluster_indices[:10]}" + ("..." if len(cluster_indices) > 10 else ""))
        
        # Show classification breakdown if available
        if classification_results:
            print("Classification breakdown:")
            for dimension, results in classification_results.items():
                cluster_classifications = [results[i]['label'] for i in cluster_indices]
                classification_counts = Counter(cluster_classifications)
                top_classifications = classification_counts.most_common(3)
                print(f"  {dimension}: {dict(top_classifications)}")
        
        # Show sample personas from this cluster
        for i, persona in enumerate(cluster_personas[:3]):  # Show first 3
            print(f"{i+1}. {persona[:200]}...")
        
        if len(cluster_personas) > 3:
            print(f"   ... and {len(cluster_personas) - 3} more personas")
        
        # Extract key characteristics
        cluster_text = " ".join(cluster_personas)
        
        # Find common words/phrases (simple approach)
        words = re.findall(r'\b\w{4,}\b', cluster_text.lower())
        word_counts = Counter(words)
        common_words = word_counts.most_common(10)
        
        print(f"Common terms: {', '.join([word for word, count in common_words[:5]])}")
        
        cluster_analysis[cluster_id] = {
            'size': len(cluster_personas),
            'personas': cluster_personas,
            'indices': cluster_indices,
            'common_words': common_words
        }
        
        # Add classification info if available
        if classification_results:
            cluster_analysis[cluster_id]['classifications'] = {}
            for dimension, results in classification_results.items():
                cluster_classifications = [results[i] for i in cluster_indices]
                cluster_analysis[cluster_id]['classifications'][dimension] = cluster_classifications
    
    # Handle noise points for DBSCAN
    noise_personas = [personas[i] for i, label in enumerate(cluster_labels) if label == -1]
    noise_indices = [i for i, label in enumerate(cluster_labels) if label == -1]
    if noise_personas:
        print(f"\n--- Noise/Outliers ({len(noise_personas)} personas) ---")
        print(f"Persona indices: {noise_indices}")
        for i, persona in enumerate(noise_personas[:3]):
            print(f"{i+1}. {persona[:200]}...")
    
    return cluster_analysis

def visualize_persona_clusters(embeddings, cluster_labels, personas, method='pca', 
                             classification_results=None, color_by='cluster'):
    """
    Visualize persona clusters in 2D using Plotly with interactive hover
    """
    print(f"\n{'='*60}")
    print(f"CREATING INTERACTIVE VISUALIZATION")
    print(f"{'='*60}")
    print(f"Generating 2D visualization using {method.upper()}...")
    print(f"Coloring by: {color_by}")
    
    # Reduce dimensionality for visualization
    if method == 'pca':
        print("  Applying Principal Component Analysis...")
        reducer = PCA(n_components=2, random_state=42)
        embeddings_2d = reducer.fit_transform(embeddings)
        x_label = f'PC1 ({reducer.explained_variance_ratio_[0]:.1%} variance)'
        y_label = f'PC2 ({reducer.explained_variance_ratio_[1]:.1%} variance)'
        total_variance = sum(reducer.explained_variance_ratio_)
        print(f"  ✓ Total variance explained: {total_variance:.1%}")
    elif method == 'tsne':
        print("  Applying t-SNE dimensionality reduction...")
        print("  This may take longer for large datasets...")
        perplexity = min(30, len(embeddings)-1)
        reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        embeddings_2d = reducer.fit_transform(embeddings)
        x_label = 't-SNE 1'
        y_label = 't-SNE 2'
        print(f"  ✓ t-SNE complete (perplexity={perplexity})")
    
    print("  Creating interactive scatter plot with hover functionality...")
    
    # Prepare data for Plotly
    x_coords = embeddings_2d[:, 0]
    y_coords = embeddings_2d[:, 1]
    
    # Determine coloring scheme
    if color_by == 'cluster':
        color_values = cluster_labels
        color_labels = [f"Cluster {label}" if label != -1 else "Noise" for label in cluster_labels]
        color_title = "Cluster"
    elif color_by in CLASSIFICATION_DIMENSIONS and classification_results:
        color_values = [classification_results[color_by][i]['label'] for i in range(len(personas))]
        color_labels = color_values
        color_title = color_by.title()
    else:
        # Fallback to cluster coloring
        color_values = cluster_labels
        color_labels = [f"Cluster {label}" if label != -1 else "Noise" for label in cluster_labels]
        color_title = "Cluster"
    
    # Create shorter hover text that's more readable
    hover_texts = []
    persona_previews = []
    for i, persona in enumerate(personas):
        # Create a longer preview (first 300 chars instead of 150)
        preview = persona
        # Clean up text for better display
        preview = preview.replace('\n', ' ').replace('\r', ' ')
        preview = re.sub(r'\s+', ' ', preview)  # Normalize whitespace
        
        # Add word wrapping by inserting line breaks every ~50 characters at word boundaries
        words = preview.split(' ')
        wrapped_lines = []
        current_line = ""
        
        for word in words:
            if len(current_line + " " + word) <= 50:
                current_line += (" " + word) if current_line else word
            else:
                if current_line:
                    wrapped_lines.append(current_line)
                current_line = word
        
        if current_line:
            wrapped_lines.append(current_line)
        
        wrapped_preview = "<br>".join(wrapped_lines)
        persona_previews.append(wrapped_preview)
        
        # Create detailed hover text with classification info if available
        hover_components = [f"Persona {i}"]
        
        if classification_results:
            for dimension, results in classification_results.items():
                label = results[i]['label']
                confidence = results[i]['confidence']
                hover_components.append(f"{dimension.title()}: {label} ({confidence:.2f})")
        
        # Join with line breaks for vertical stacking instead of horizontal
        hover_text = "<br>".join(hover_components)
        hover_texts.append(hover_text)
    
    # Get unique color values and create color mapping
    unique_values = sorted(set(color_values))
    
    # Create the plotly figure
    fig = go.Figure()
    
    # Define colors for clusters (using a nice color palette)
    colors = px.colors.qualitative.Set3
    if len(unique_values) > len(colors):
        # Repeat colors if we have more clusters than available colors
        colors = colors * (len(unique_values) // len(colors) + 1)
    
    # Add scatter plot for each unique value
    for i, value in enumerate(unique_values):
        if value == -1:
            # Noise points (for DBSCAN)
            group_name = 'Noise/Outliers'
            color = 'black'
            symbol = 'x'
        else:
            group_name = str(value)
            color = colors[i % len(colors)]
            symbol = 'circle'
        
        # Get indices for this group
        mask = np.array(color_values) == value
        group_indices = np.where(mask)[0]
        
        fig.add_trace(go.Scatter(
            x=x_coords[mask],
            y=y_coords[mask],
            mode='markers',
            name=group_name,
            marker=dict(
                color=color,
                size=12,
                symbol=symbol,
                line=dict(width=1, color='black'),
                opacity=0.5  # Set to 50% transparency
            ),
            hovertemplate="<b>%{customdata[0]}</b><br>" +
                         f"<b>{x_label}:</b> %{{x:.3f}}<br>" +
                         f"<b>{y_label}:</b> %{{y:.3f}}<br>" +
                         "<b>Description:</b><br>%{customdata[1]}" +
                         "<extra></extra>",  # This removes the trace name from hover
            customdata=[[hover_texts[idx], persona_previews[idx]] for idx in group_indices],
            showlegend=True
        ))
    
    # Update layout for better appearance and hover readability
    title_text = f'Semantic Clustering of Personas (Colored by {color_title})'
    if color_by != 'cluster':
        title_text += f'<br><sub>Zero-shot classification dimension: {color_by}</sub>'
    
    fig.update_layout(
        title={
            'text': title_text + '<br>' + 
                   '<sub>Hover over points to see persona details • Click legend to toggle groups</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        xaxis_title=dict(text=x_label, font=dict(size=14)),
        yaxis_title=dict(text=y_label, font=dict(size=14)),
        width=1200,
        height=800,
        hovermode='closest',
        template='plotly_white',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.01,
            font=dict(size=12),
            title=None  # Remove legend title for cleaner look
        ),
        margin=dict(r=200, l=80, t=100, b=80),  # Make room for legend and title
        hoverlabel=dict(
            bgcolor='rgba(255,255,255,0.75)',  # 75% transparent white background
            bordercolor='rgba(0,0,0,0.3)',     # 30% transparent black border
            font=dict(
                size=11,
                family="Arial, sans-serif",
                color='black'
            ),
            align="left",
            namelength=-1
        )
    )
    
    # Add grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    
    # No additional layout updates needed - CSS injection handles transparency
    
    print("  ✓ Interactive visualization ready")
    print("  Opening in browser...")
    print("  💡 Hover over points to see persona details!")
    print("  💡 Click legend items to show/hide clusters!")
    print("  💡 Use zoom and pan tools for detailed exploration!")
    
    # Show the plot with custom config for transparency
    config = {
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToRemove': ['pan2d', 'lasso2d']
    }
    fig.show(config=config)
    
    # Also save as HTML file with CSS injection for transparent hover boxes
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    color_suffix = f"_{color_by}" if color_by != 'cluster' else ""
    html_file = f"persona_clusters_visualization{color_suffix}_{timestamp}.html"
    
    # Write HTML with custom CSS for transparent hover boxes
    html_string = fig.to_html(include_plotlyjs=True)
    
    # Add CSS to make hover boxes transparent using the suggested method
    css_injection = """
    <style>
    .hovertext { 
        fill-opacity: 0.4 !important; 
        stroke-opacity: 1 !important;
        background-color: rgba(255, 255, 255, 0.4) !important;
        border-color: rgba(0, 0, 0, 0.3) !important;
    }
    .hoverlayer .hovertext {
        background-color: rgba(255, 255, 255, 0.4) !important;
        border-color: rgba(0, 0, 0, 0.3) !important;
    }
    </style>
    """
    
    # Insert CSS before closing head tag
    html_string = html_string.replace('</head>', css_injection + '</head>')
    
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_string)
    
    print(f"  ✓ Visualization saved as: {html_file}")
    
    return embeddings_2d

def calculate_dissimilarity_matrix(embeddings):
    """
    Calculate pairwise dissimilarity matrix from embeddings
    """
    print(f"\n{'='*60}")
    print(f"CALCULATING DISSIMILARITY MATRIX")
    print(f"{'='*60}")
    print(f"Computing pairwise distances for {len(embeddings)} personas...")
    print("This creates a symmetric matrix of semantic distances...")
    
    # Calculate cosine similarity matrix
    print("  Computing cosine similarities...")
    similarity_matrix = cosine_similarity(embeddings)
    
    # Convert to dissimilarity (1 - similarity)
    print("  Converting to dissimilarity scores...")
    dissimilarity_matrix = 1 - similarity_matrix
    
    print(f"✓ Dissimilarity matrix complete")
    print(f"✓ Matrix shape: {dissimilarity_matrix.shape}")
    print(f"✓ Distance range: {dissimilarity_matrix.min():.3f} to {dissimilarity_matrix.max():.3f}")
    
    return dissimilarity_matrix

def save_results(cluster_labels, dissimilarity_matrix, classification_results=None, output_dir="results"):
    """
    Save clustering results to files
    """
    print(f"\n{'='*60}")
    print(f"SAVING RESULTS")
    print(f"{'='*60}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {os.path.abspath(output_dir)}")
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save persona-cluster mapping
    print("Saving persona-cluster assignments...")
    cluster_df = pd.DataFrame({
        'persona_index': range(len(cluster_labels)),
        'cluster_id': cluster_labels
    })
    
    # Add classification results if available
    if classification_results:
        for dimension, results in classification_results.items():
            cluster_df[f'{dimension}_label'] = [r['label'] for r in results]
            cluster_df[f'{dimension}_confidence'] = [r['confidence'] for r in results]
    
    cluster_file = os.path.join(output_dir, f"persona_clusters_{timestamp}.csv")
    cluster_df.to_csv(cluster_file, index=False)
    print(f"✓ Saved: {os.path.basename(cluster_file)}")
    
    # Save dissimilarity matrix
    print("Saving dissimilarity matrix...")
    dissimilarity_file = os.path.join(output_dir, f"dissimilarity_matrix_{timestamp}.csv")
    pd.DataFrame(dissimilarity_matrix).to_csv(dissimilarity_file, index=False)
    print(f"✓ Saved: {os.path.basename(dissimilarity_file)}")
    
    # Save detailed classification results if available
    classification_file = None
    if classification_results:
        print("Saving detailed classification results...")
        classification_data = []
        for i in range(len(cluster_labels)):
            row = {'persona_index': i, 'cluster_id': cluster_labels[i]}
            for dimension, results in classification_results.items():
                result = results[i]
                row[f'{dimension}_label'] = result['label']
                row[f'{dimension}_confidence'] = result['confidence']
                # Add all scores for transparency
                for label, score in result['all_scores'].items():
                    row[f'{dimension}_{label}_score'] = score
            classification_data.append(row)
        
        classification_df = pd.DataFrame(classification_data)
        classification_file = os.path.join(output_dir, f"classification_details_{timestamp}.csv")
        classification_df.to_csv(classification_file, index=False)
        print(f"✓ Saved: {os.path.basename(classification_file)}")
    
    print(f"✓ All results saved to: {os.path.abspath(output_dir)}")
    
    return cluster_file, dissimilarity_file, classification_file

def generate_cluster_summary_report(cluster_analysis, total_personas, classification_results=None):
    """
    Generate a summary report of the clustering results
    """
    print(f"\n{'='*60}")
    print("CLUSTER SUMMARY REPORT")
    print(f"{'='*60}")
    
    print(f"Total personas analyzed: {total_personas}")
    print(f"Number of clusters found: {len(cluster_analysis)}")
    
    # Cluster size distribution
    sizes = [analysis['size'] for analysis in cluster_analysis.values()]
    print(f"\nCluster size statistics:")
    
    if len(sizes) > 0:
        print(f"Average cluster size: {np.mean(sizes):.1f}")
        print(f"Largest cluster: {max(sizes)} personas")
        print(f"Smallest cluster: {min(sizes)} personas")
    else:
        print("No clusters found - all points classified as noise/outliers")
        print("💡 Consider adjusting clustering parameters:")
        print("   • For DBSCAN: increase eps or decrease min_samples")
        print("   • For K-means: check if data is suitable for clustering")
    
    # Classification distribution if available
    if classification_results:
        print(f"\nClassification summary across all personas:")
        for dimension, results in classification_results.items():
            all_labels = [r['label'] for r in results]
            label_counts = Counter(all_labels)
            avg_confidence = np.mean([r['confidence'] for r in results])
            print(f"{dimension.title()}:")
            print(f"  Average confidence: {avg_confidence:.3f}")
            print(f"  Distribution: {dict(label_counts.most_common(5))}")
    
    # Most common words across all clusters
    all_words = []
    for analysis in cluster_analysis.values():
        all_words.extend([word for word, count in analysis['common_words']])
    
    overall_common = Counter(all_words).most_common(15)
    print(f"\nMost common terms across all personas:")
    print(", ".join([word for word, count in overall_common[:10]]))
    
    return cluster_analysis

def main_clustering_pipeline(file_path, model_name="minishlab/potion-base-8M", 
                           n_clusters=None, clustering_method='kmeans', 
                           visualization_method='pca', output_dir="results",
                           eps=0.3, min_samples=2, use_classification=False,
                           classification_dimensions=None, classification_method='semantic',
                           color_by='cluster'):
    """
    Main pipeline for clustering persona descriptions
    
    Parameters:
    -----------
    file_path : str
        Path to the personas text file
    model_name : str, default="minishlab/potion-base-8M"
        Model2Vec model to use for embeddings
    n_clusters : int, optional
        Number of clusters (if None, will be determined automatically)
    clustering_method : str, default='kmeans'
        Clustering method ('kmeans' or 'dbscan')
    visualization_method : str, default='pca'
        Dimensionality reduction method for visualization ('pca' or 'tsne')
    output_dir : str, default='results'
        Directory to save output files
    eps : float, default=0.3
        DBSCAN parameter: maximum distance between points in the same cluster
    min_samples : int, default=2
        DBSCAN parameter: minimum number of points to form a cluster
    use_classification : bool, default=False
        Whether to use zero-shot classification features
    classification_dimensions : list, optional
        List of dimensions to classify on (from CLASSIFICATION_DIMENSIONS keys)
    classification_method : str, default='semantic'
        How to use classification: 'semantic' (Model2Vec only), 'classification' (zero-shot only), 
        'hybrid' (combine both)
    color_by : str, default='cluster'
        What to color visualization by ('cluster' or any classification dimension)
    
    Returns:
    --------
    dict : Dictionary containing all analysis results
    """
    try:
        # Step 1: Load personas
        personas = load_and_explore_personas(file_path)
        
        # Step 2: Preprocess personas (no deduplication)
        cleaned_personas = preprocess_personas(personas)
        
        # Step 3: Initialize classification if requested
        classification_results = None
        classifier = None
        
        if use_classification and classification_dimensions:
            if not TRANSFORMERS_AVAILABLE:
                print("⚠️  Warning: transformers not available, falling back to semantic clustering only")
                use_classification = False
            else:
                classifier = setup_zero_shot_classifier()
                classification_results = classify_personas_by_dimensions(
                    cleaned_personas, classification_dimensions, classifier
                )
        
        # Step 4: Choose clustering approach based on method
        if classification_method == 'semantic' or not use_classification:
            # Use Model2Vec embeddings only
            print(f"\nUsing semantic embeddings for clustering...")
            embeddings, model = cluster_personas_with_model2vec(cleaned_personas, model_name)
            
        elif classification_method == 'classification' and use_classification:
            # Use zero-shot classification features only
            print(f"\nUsing classification features for clustering...")
            embeddings, feature_names = create_classification_features(
                classification_results, method='hybrid'
            )
            model = classifier  # Store classifier as model for consistency
            
        elif classification_method == 'hybrid' and use_classification:
            # Combine both approaches
            print(f"\nUsing hybrid approach (semantic + classification)...")
            semantic_embeddings, model = cluster_personas_with_model2vec(cleaned_personas, model_name)
            classification_embeddings, _ = create_classification_features(
                classification_results, method='confidence'
            )
            
            # Combine embeddings (weighted)
            semantic_weight = 0.7  # Give more weight to semantic features
            classification_weight = 0.3
            
            # Normalize both embedding types
            semantic_norm = semantic_embeddings / np.linalg.norm(semantic_embeddings, axis=1, keepdims=True)
            classification_norm = classification_embeddings / np.linalg.norm(classification_embeddings, axis=1, keepdims=True)
            
            embeddings = np.hstack([
                semantic_norm * semantic_weight,
                classification_norm * classification_weight
            ])
            print(f"✓ Combined embeddings shape: {embeddings.shape}")
            
        else:
            # Default fallback
            embeddings, model = cluster_personas_with_model2vec(cleaned_personas, model_name)
        
        # Step 5: Calculate dissimilarity matrix
        dissimilarity_matrix = calculate_dissimilarity_matrix(embeddings)
        
        # Step 6: Find optimal number of clusters (if not specified and using kmeans)
        if n_clusters is None and clustering_method == 'kmeans':
            optimal_k, silhouette_scores = find_optimal_clusters(embeddings)
        else:
            optimal_k = n_clusters
            silhouette_scores = None
        
        # Step 7: Perform clustering
        cluster_labels, n_clusters_found = perform_clustering(embeddings, cleaned_personas, 
                                                             n_clusters=optimal_k, 
                                                             method=clustering_method,
                                                             eps=eps,
                                                             min_samples=min_samples)
        
        # Step 8: Save results to files
        cluster_file, dissimilarity_file, classification_file = save_results(
            cluster_labels, dissimilarity_matrix, classification_results, output_dir
        )
        
        # Step 9: Analyze clusters
        cluster_analysis = analyze_persona_clusters(
            cleaned_personas, cluster_labels, n_clusters_found, classification_results
        )
        
        # Step 10: Generate summary report
        generate_cluster_summary_report(cluster_analysis, len(cleaned_personas), classification_results)
        
        # Step 11: Visualize results
        embeddings_2d = visualize_persona_clusters(
            embeddings, cluster_labels, cleaned_personas, 
            method=visualization_method, classification_results=classification_results,
            color_by=color_by
        )
        
        # Return results for further analysis
        results = {
            'personas': cleaned_personas,
            'embeddings': embeddings,
            'dissimilarity_matrix': dissimilarity_matrix,
            'cluster_labels': cluster_labels,
            'cluster_analysis': cluster_analysis,
            'model': model,
            'embeddings_2d': embeddings_2d,
            'n_clusters': n_clusters_found,
            'silhouette_scores': silhouette_scores,
            'classification_results': classification_results,
            'classifier': classifier,
            'output_files': {
                'cluster_file': cluster_file,
                'dissimilarity_file': dissimilarity_file,
                'classification_file': classification_file
            }
        }
        
        return results
        
    except Exception as e:
        print(f"Error in clustering pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def run_persona_clustering(personas_file_path, **kwargs):
    """
    Convenience function to run persona clustering analysis
    
    Parameters:
    -----------
    personas_file_path : str
        Path to your personas text file
    **kwargs : optional parameters
        model_name : str, Model2Vec model to use
        n_clusters : int, number of clusters (auto-determined if None)
        clustering_method : str, 'kmeans' or 'dbscan'
        visualization_method : str, 'pca' or 'tsne'
        output_dir : str, directory to save results
        eps : float, DBSCAN eps parameter
        min_samples : int, DBSCAN min_samples parameter
        use_classification : bool, whether to use zero-shot classification
        classification_dimensions : list, dimensions to classify on
        classification_method : str, 'semantic', 'classification', or 'hybrid'
        color_by : str, what to color visualization by
    
    Returns:
    --------
    dict : Complete analysis results
    """
    
    print("Starting Enhanced Persona Clustering Analysis...")
    print("=" * 60)
    print(f"Input file: {personas_file_path}")
    print(f"Model: {kwargs.get('model_name', 'minishlab/potion-base-8M')}")
    print(f"Method: {kwargs.get('clustering_method', 'kmeans')}")
    
    # Show classification info if enabled
    if kwargs.get('use_classification', False):
        dims = kwargs.get('classification_dimensions', [])
        method = kwargs.get('classification_method', 'semantic')
        print(f"Zero-shot classification: ENABLED")
        print(f"Classification dimensions: {dims}")
        print(f"Classification method: {method}")
        print(f"Visualization coloring: {kwargs.get('color_by', 'cluster')}")
    else:
        print(f"Zero-shot classification: DISABLED")
    
    if kwargs.get('clustering_method') == 'dbscan':
        print(f"DBSCAN eps: {kwargs.get('eps', 0.5)}")
        print(f"DBSCAN min_samples: {kwargs.get('min_samples', 3)}")
    if kwargs.get('n_clusters'):
        print(f"Clusters: {kwargs.get('n_clusters')} (fixed)")
    else:
        print("Clusters: Auto-determined")
    print(f"Output: {kwargs.get('output_dir', 'results')}")
    
    # Run the clustering pipeline
    results = main_clustering_pipeline(personas_file_path, **kwargs)
    
    if results:
        print("\n" + "=" * 60)
        print("CLUSTERING ANALYSIS COMPLETE!")
        print("=" * 60)
        
        # Provide access information
        print(f"\n📊 RESULTS SUMMARY:")
        print(f"  • {len(results['personas'])} personas analyzed")
        print(f"  • {results['n_clusters']} clusters identified") 
        print(f"  • Embedding dimensions: {results['embeddings'].shape}")
        
        if results['classification_results']:
            print(f"  • Classification dimensions: {list(results['classification_results'].keys())}")
        
        print(f"\n📁 OUTPUT FILES:")
        print(f"  • Clusters: {os.path.basename(results['output_files']['cluster_file'])}")
        print(f"  • Dissimilarity matrix: {os.path.basename(results['output_files']['dissimilarity_file'])}")
        if results['output_files']['classification_file']:
            print(f"  • Classification details: {os.path.basename(results['output_files']['classification_file'])}")
        print(f"  • Location: {os.path.dirname(results['output_files']['cluster_file'])}")
        
        # Example: Find similar personas to a specific one
        if len(results['personas']) > 0:
            print(f"\n🔍 SIMILARITY EXAMPLE:")
            sample_persona = results['personas'][0]
            
            if hasattr(results['model'], 'encode'):
                # Model2Vec model
                sample_embedding = results['model'].encode([sample_persona])
                similarities = cosine_similarity(sample_embedding, results['embeddings'])[0]
                top_indices = np.argsort(similarities)[-6:-1][::-1]  # Top 5 similar (excluding itself)
                
                print(f"Most similar personas to Persona 0:")
                print(f"  '{sample_persona[:80]}...'")
                for i, idx in enumerate(top_indices):
                    print(f"  {i+1}. Persona {idx} (similarity: {similarities[idx]:.3f})")
                    print(f"     {results['personas'][idx][:80]}...")
        
        print(f"\n💡 ACCESS YOUR RESULTS:")
        print(f"  results['personas']              # List of personas")
        print(f"  results['embeddings']            # Embeddings used for clustering") 
        print(f"  results['dissimilarity_matrix']  # Pairwise distances")
        print(f"  results['cluster_labels']        # Cluster assignments")
        print(f"  results['cluster_analysis']      # Detailed cluster info")
        print(f"  results['classification_results'] # Zero-shot classification data")
        print(f"  results['model']                 # Model used for embeddings")
        
        return results
    
    else:
        print("\n❌ CLUSTERING ANALYSIS FAILED")
        print("Please check your data file and try again.")
        return None

# Example usage and main execution
if __name__ == "__main__":
    import sys
    import argparse
    
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Cluster persona descriptions using Model2Vec and/or zero-shot classification')
    parser.add_argument('personas_file', help='Path to personas text file')
    parser.add_argument('--model', default='minishlab/potion-base-8M', 
                       help='Model2Vec model to use (default: minishlab/potion-base-8M)')
    parser.add_argument('--clusters', type=int, default=None,
                       help='Number of clusters (auto-determined if not specified)')
    parser.add_argument('--method', choices=['kmeans', 'dbscan'], default='kmeans',
                       help='Clustering method (default: kmeans)')
    parser.add_argument('--viz', choices=['pca', 'tsne'], default='pca',
                       help='Visualization method (default: pca)')
    parser.add_argument('--output', default='results',
                       help='Output directory for results (default: results)')
    parser.add_argument('--eps', type=float, default=0.5,
                       help='DBSCAN eps parameter: max distance between points in same cluster (default: 0.5)')
    parser.add_argument('--min-samples', type=int, default=3,
                       help='DBSCAN min_samples parameter: min points to form a cluster (default: 3)')
    
    # New classification arguments
    parser.add_argument('--use-classification', action='store_true',
                       help='Enable zero-shot classification features')
    parser.add_argument('--dimensions', nargs='+', 
                       choices=list(CLASSIFICATION_DIMENSIONS.keys()),
                       default=['seniority', 'industry', 'expertise', 'priorities', 'challenges'],
                       help='Classification dimensions to use (default: seniority industry expertise priorities challenges)')
    parser.add_argument('--classification-method', 
                       choices=['semantic', 'classification', 'hybrid'], 
                       default='semantic',
                       help='Clustering approach: semantic (Model2Vec only), classification (zero-shot only), hybrid (both) (default: semantic)')
    parser.add_argument('--color-by', default='cluster',
                       help='What to color visualization by: cluster or any classification dimension (default: cluster)')
    
    # If running as script with command line arguments
    if len(sys.argv) > 1:
        args = parser.parse_args()
        
        # Validate color_by argument
        valid_color_options = ['cluster'] + list(CLASSIFICATION_DIMENSIONS.keys())
        if args.color_by not in valid_color_options:
            print(f"Error: --color-by must be one of: {valid_color_options}")
            sys.exit(1)
        
        # If color_by is a classification dimension, ensure use_classification is enabled
        if args.color_by != 'cluster' and not args.use_classification:
            print(f"Warning: --color-by {args.color_by} requires --use-classification. Enabling classification...")
            args.use_classification = True
            if args.color_by not in args.dimensions:
                args.dimensions.append(args.color_by)
        
        results = run_persona_clustering(
            personas_file_path=args.personas_file,
            model_name=args.model,
            n_clusters=args.clusters,
            clustering_method=args.method,
            visualization_method=args.viz,
            output_dir=args.output,
            eps=args.eps,
            min_samples=args.min_samples,
            use_classification=args.use_classification,
            classification_dimensions=args.dimensions,
            classification_method=args.classification_method,
            color_by=args.color_by
        )
    
    # If running interactively or no arguments provided
    else:
        print("Enhanced Persona Clustering Tool")
        print("=" * 40)
        print("Interactive mode: Please provide your personas file path")
        print("\nBasic usage:")
        print("  results = run_persona_clustering('UXR_personas.txt')")
        print("\nWith zero-shot classification:")
        print("  results = run_persona_clustering('UXR_personas.txt',")
        print("                                   use_classification=True,")
        print("                                   classification_dimensions=['seniority', 'industry'],")
        print("                                   classification_method='hybrid',")
        print("                                   color_by='seniority')")
        print("\nCommand line examples:")
        print("  # Basic semantic clustering")
        print("  python persona_clustering.py UXR_personas.txt")
        print("\n  # With zero-shot classification")
        print("  python persona_clustering.py UXR_personas.txt --use-classification --dimensions seniority industry")
        print("\n  # Hybrid approach with custom visualization")
        print("  python persona_clustering.py UXR_personas.txt --use-classification --classification-method hybrid --color-by seniority")
        print("\n  # DBSCAN with classification")
        print("  python persona_clustering.py UXR_personas.txt --method dbscan --use-classification --dimensions seniority industry expertise")
        
        print(f"\nAvailable classification dimensions: {list(CLASSIFICATION_DIMENSIONS.keys())}")
        
        # For your specific file, you can run:
        # results = run_persona_clustering('UXR_personas.txt', use_classification=True)