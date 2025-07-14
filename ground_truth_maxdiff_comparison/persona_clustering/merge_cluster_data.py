#!/usr/bin/env python3
"""
Script to merge cluster data from persona clustering directories into maxdiff_responses.csv

This script:
1. Creates a copy of maxdiff_responses.csv
2. Adds columns for each cluster type (seniority, industry, expertise, priorities, challenges)
3. For each cluster, finds the most common label for each cluster_id
4. Joins cluster information to maxdiff_responses on persona_index
"""

import pandas as pd
import os
from pathlib import Path
import re
from collections import Counter

def find_most_recent_cluster_file(cluster_dir):
    """Find the most recent persona_clusters CSV file in a directory"""
    cluster_files = list(cluster_dir.glob("persona_clusters_*.csv"))
    if not cluster_files:
        raise FileNotFoundError(f"No persona_clusters CSV files found in {cluster_dir}")
    
    # Sort by modification time and return the most recent
    cluster_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return cluster_files[0]

def get_cluster_label_mapping(df, label_col):
    """
    Create a mapping from cluster_id to the most common label in that cluster.
    Returns a dict where keys are cluster_ids and values are "cluster_id_most_common_label"
    """
    cluster_label_map = {}
    
    # Group by cluster_id and find the most common label for each cluster
    for cluster_id in df['cluster_id'].unique():
        if cluster_id == -1:  # Handle outliers/noise points
            cluster_label_map[cluster_id] = f"{cluster_id}_outlier"
            continue
            
        cluster_data = df[df['cluster_id'] == cluster_id]
        label_counts = Counter(cluster_data[label_col])
        most_common_label = label_counts.most_common(1)[0][0]
        
        # Create the combined label: cluster_id + most_common_label
        cluster_label_map[cluster_id] = f"{cluster_id}_{most_common_label}"
    
    return cluster_label_map

def main():
    # Set up paths
    base_dir = Path("/Users/agardony/Projects/maxdiff_agents/ground_truth_maxdiff_comparison/persona_clustering")
    maxdiff_file = base_dir / "maxdiff_responses.csv"
    
    # Define cluster directories and their corresponding label columns
    cluster_dirs = {
        'seniority': ('seniority clusters', 'seniority_label'),
        'industry': ('industry clusters', 'industry_label'), 
        'expertise': ('expertise clusters', 'expertise_label'),
        'priorities': ('priorities_clusters', 'priorities_label'),
        'challenges': ('challenges_clusters', 'challenges_label')
    }
    
    print("Loading maxdiff_responses.csv...")
    # Load the original maxdiff responses file
    df_maxdiff = pd.read_csv(maxdiff_file)
    print(f"Loaded {len(df_maxdiff)} rows from maxdiff_responses.csv")
    
    # Process each cluster type
    cluster_mappings = {}
    
    for cluster_type, (dir_name, label_col) in cluster_dirs.items():
        print(f"\nProcessing {cluster_type} clusters...")
        
        cluster_dir = base_dir / dir_name
        if not cluster_dir.exists():
            print(f"Warning: Directory {cluster_dir} does not exist, skipping...")
            continue
        
        # Find the most recent cluster file
        try:
            cluster_file = find_most_recent_cluster_file(cluster_dir)
            print(f"Found cluster file: {cluster_file.name}")
        except FileNotFoundError as e:
            print(f"Warning: {e}, skipping...")
            continue
        
        # Load cluster data
        df_cluster = pd.read_csv(cluster_file)
        print(f"Loaded {len(df_cluster)} rows from {cluster_file.name}")
        
        # Get cluster label mapping
        cluster_label_map = get_cluster_label_mapping(df_cluster, label_col)
        print(f"Found {len(cluster_label_map)} unique clusters:")
        for cluster_id, label in sorted(cluster_label_map.items()):
            count = len(df_cluster[df_cluster['cluster_id'] == cluster_id])
            print(f"  Cluster {cluster_id}: {label} ({count} personas)")
        
        # Create the mapping from persona_index to cluster label
        persona_cluster_map = {}
        for _, row in df_cluster.iterrows():
            persona_idx = row['persona_index']
            cluster_id = row['cluster_id']
            persona_cluster_map[persona_idx] = cluster_label_map[cluster_id]
        
        cluster_mappings[cluster_type] = persona_cluster_map
    
    # Add cluster columns to the maxdiff dataframe
    print(f"\nAdding cluster columns to maxdiff data...")
    
    for cluster_type, persona_cluster_map in cluster_mappings.items():
        column_name = f"{cluster_type}_cluster"
        df_maxdiff[column_name] = df_maxdiff['persona_index'].map(persona_cluster_map)
        
        # Count how many mappings were successful
        mapped_count = df_maxdiff[column_name].notna().sum()
        print(f"  Added {column_name}: {mapped_count}/{len(df_maxdiff)} mappings successful")
    
    # Create output filename
    output_file = base_dir / "maxdiff_responses_with_clusters.csv"
    
    # Save the enhanced dataframe
    print(f"\nSaving enhanced data to {output_file}...")
    df_maxdiff.to_csv(output_file, index=False)
    print(f"Saved {len(df_maxdiff)} rows with {len(df_maxdiff.columns)} columns")
    
    # Display summary of new columns
    print(f"\nSummary of new cluster columns:")
    for cluster_type in cluster_mappings.keys():
        column_name = f"{cluster_type}_cluster"
        if column_name in df_maxdiff.columns:
            unique_values = df_maxdiff[column_name].value_counts()
            print(f"\n{column_name}:")
            for value, count in unique_values.head(10).items():
                print(f"  {value}: {count}")
            if len(unique_values) > 10:
                print(f"  ... and {len(unique_values) - 10} more")
    
    print(f"\nScript completed successfully!")
    print(f"Original file: {maxdiff_file}")
    print(f"Enhanced file: {output_file}")

if __name__ == "__main__":
    main()
