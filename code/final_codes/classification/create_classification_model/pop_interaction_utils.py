#!/usr/bin/env python3
"""
Population Interaction Utilities

This module provides functions to manipulate sample populations created by the interactive_sample_selection tool.
It allows merging populations from multiple files, filtering by category, and removing specific samples.

Usage:
  python pop_interaction_utils.py merge-samples --input file1.json file2.json --output merged.json
  python pop_interaction_utils.py merge-samples-from-dir --input-dir samples_dir/ --output merged.json
  python pop_interaction_utils.py remove-category --input samples.json --category "Forest" --output filtered.json
  python pop_interaction_utils.py remove-samples --input samples.json --indices 0 5 10 --output filtered.json
"""

import os
import sys
import json
import argparse
import glob
from collections import Counter

def load_json_file(json_path):
    """
    Load sample data from a JSON file.
    
    Args:
        json_path: Path to the JSON file
        
    Returns:
        JSON data as a dictionary
    """
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON file {json_path}: {e}")
        return None

def save_json_file(data, output_path):
    """
    Save data to a JSON file.
    
    Args:
        data: Data to save
        output_path: Path to save the JSON file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving JSON file {output_path}: {e}")
        return False

def count_samples_by_category(samples_data):
    """
    Count samples by category in a samples dataset.
    
    Args:
        samples_data: Dictionary containing samples data
        
    Returns:
        Counter object with categories and their counts
    """
    categories = []
    for sample in samples_data.get("samples", []):
        if "category" in sample:
            categories.append(sample["category"])
    
    return Counter(categories)

def print_population_stats(population_name, samples_data):
    """
    Print statistics about a sample population.
    
    Args:
        population_name: Name to display for the population
        samples_data: Dictionary containing samples data
    """
    counts = count_samples_by_category(samples_data)
    total = sum(counts.values())
    
    print(f"\n{population_name} statistics:")
    print(f"Total samples: {total}")
    
    if counts:
        print("Samples per category:")
        for category, count in sorted(counts.items()):
            print(f"  - {category}: {count} samples")
    else:
        print("No categorized samples found.")

def merge_samples(input_files, output_file):
    """
    Merge samples from multiple JSON files into a single JSON file.
    
    Args:
        input_files: List of paths to JSON files containing samples
        output_file: Path to save the merged samples
        
    Returns:
        Merged samples data
    """
    print(f"Merging samples from {len(input_files)} files...")
    
    # Create an empty structure for the merged result
    merged_data = {
        "n_samples_x": None,
        "n_samples_y": None,
        "samples": []
    }
    
    # Process each input file
    for file_path in input_files:
        print(f"Processing {file_path}...")
        
        # Load samples from the JSON file
        file_data = load_json_file(file_path)
        if file_data is None:
            print(f"Skipping {file_path} due to load error")
            continue
        
        # Print statistics for this file
        print_population_stats(f"File {os.path.basename(file_path)}", file_data)
        
        # Add all samples to the merged data
        merged_data["samples"].extend(file_data.get("samples", []))
    
    # Print statistics for the merged set
    print_population_stats("Merged population", merged_data)
    
    # Save the merged data to the output file
    success = save_json_file(merged_data, output_file)
    if success:
        print(f"Saved {len(merged_data['samples'])} merged samples to {output_file}")
    
    return merged_data

def merge_samples_from_dir(input_dir, output_file, exclude_files=None):
    """
    Merge samples from all JSON files in a directory.
    
    Args:
        input_dir: Directory containing JSON files with samples
        output_file: Path to save the merged samples
        exclude_files: List of file paths to exclude from the merge
        
    Returns:
        Merged samples data
    """
    # Find all JSON files in the directory
    json_files = glob.glob(os.path.join(input_dir, "*.json"))
    
    # Filter out excluded files
    if exclude_files:
        # Convert exclude_files to absolute paths for accurate comparison
        exclude_files_abs = [os.path.abspath(f) for f in exclude_files]
        json_files = [f for f in json_files if os.path.abspath(f) not in exclude_files_abs]
    
    print(f"Found {len(json_files)} JSON files in {input_dir}")
    if exclude_files:
        print(f"Excluding {len(exclude_files)} files: {', '.join(exclude_files)}")
    
    # Use the merge_samples function to merge all files
    return merge_samples(json_files, output_file)

def remove_category(input_file, category, output_file):
    """
    Remove samples of a specific category from a population.
    
    Args:
        input_file: Path to JSON file containing samples
        category: Category of samples to remove
        output_file: Path to save the filtered samples
        
    Returns:
        Filtered samples data
    """
    print(f"Removing samples of category '{category}' from {input_file}...")
    
    # Load samples from the JSON file
    input_data = load_json_file(input_file)
    if input_data is None:
        return None
    
    # Print statistics for the input set
    print_population_stats("Input population", input_data)
    
    # Create a new structure for the filtered result
    filtered_data = {
        "n_samples_x": input_data.get("n_samples_x"),
        "n_samples_y": input_data.get("n_samples_y"),
        "samples": []
    }
    
    # Add only samples of other categories
    for sample in input_data.get("samples", []):
        if "category" not in sample or sample["category"] != category:
            filtered_data["samples"].append(sample)
    
    # Print statistics for the filtered set
    print_population_stats("Filtered population", filtered_data)
    
    # Save the filtered data to the output file
    save_json_file(filtered_data, output_file)
    print(f"Saved {len(filtered_data['samples'])} filtered samples to {output_file}")
    
    return filtered_data

def remove_samples(input_file, indices, output_file):
    """
    Remove samples at specific indices from a population.
    
    Args:
        input_file: Path to JSON file containing samples
        indices: List of indices of samples to remove
        output_file: Path to save the filtered samples
        
    Returns:
        Filtered samples data
    """
    print(f"Removing {len(indices)} samples from {input_file}...")
    
    # Load the JSON data
    json_data = load_json_file(input_file)
    if json_data is None:
        return None
    
    # Check if indices are valid
    sample_count = len(json_data.get("samples", []))
    valid_indices = [i for i in indices if 0 <= i < sample_count]
    invalid_indices = [i for i in indices if i not in valid_indices]
    
    if invalid_indices:
        print(f"Warning: Ignoring invalid indices: {invalid_indices}")
    
    # Create a set of indices to remove for efficient lookup
    indices_to_remove = set(valid_indices)
    
    # Filter the samples
    filtered_samples = []
    for i, sample in enumerate(json_data.get("samples", [])):
        if i not in indices_to_remove:
            filtered_samples.append(sample)
    
    # Create a new JSON structure
    filtered_data = {
        "n_samples_x": json_data.get("n_samples_x"),
        "n_samples_y": json_data.get("n_samples_y"),
        "samples": filtered_samples
    }
    
    # Save the filtered data
    save_json_file(filtered_data, output_file)
    print(f"Removed {len(valid_indices)} samples. Saved {len(filtered_samples)} samples to {output_file}")
    
    # Print statistics for the filtered set
    print_population_stats("Filtered population", filtered_data)
    
    return filtered_data

def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Utilities for manipulating sample populations")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # merge-samples command
    merge_parser = subparsers.add_parser("merge-samples", help="Merge samples from multiple JSON files")
    merge_parser.add_argument("--input", required=True, nargs="+", help="Input JSON files")
    merge_parser.add_argument("--output", required=True, help="Output JSON file")
    
    # merge-samples-from-dir command
    merge_dir_parser = subparsers.add_parser("merge-samples-from-dir", help="Merge samples from all JSON files in a directory")
    merge_dir_parser.add_argument("--input-dir", required=True, help="Input directory containing JSON files")
    merge_dir_parser.add_argument("--output", required=True, help="Output JSON file")
    merge_dir_parser.add_argument("--exclude", nargs="+", help="Files to exclude from the merge")
    
    # remove-category command
    remove_cat_parser = subparsers.add_parser("remove-category", help="Remove samples of a specific category")
    remove_cat_parser.add_argument("--input", required=True, help="Input JSON file")
    remove_cat_parser.add_argument("--category", required=True, help="Category to remove")
    remove_cat_parser.add_argument("--output", required=True, help="Output JSON file")
    
    # remove-samples command
    remove_samples_parser = subparsers.add_parser("remove-samples", help="Remove samples at specific indices")
    remove_samples_parser.add_argument("--input", required=True, help="Input JSON file")
    remove_samples_parser.add_argument("--indices", required=True, type=int, nargs="+", help="Indices of samples to remove")
    remove_samples_parser.add_argument("--output", required=True, help="Output JSON file")
    
    return parser.parse_args()

def main():
    """Main entry point for the script."""
    args = parse_args()
    
    if args.command == "merge-samples":
        merge_samples(args.input, args.output)
    elif args.command == "merge-samples-from-dir":
        merge_samples_from_dir(args.input_dir, args.output, args.exclude)
    elif args.command == "remove-category":
        remove_category(args.input, args.category, args.output)
    elif args.command == "remove-samples":
        remove_samples(args.input, args.indices, args.output)
    else:
        print("No command specified. Use --help for usage information.")

if __name__ == "__main__":
    main()


"""
python code/final_codes/classification/create_classification_model/pop_interaction_utils.py merge-samples\
    --input data/selection_test/Through_1.json data/selection_test/Through_2.json data/selection_test/Green_1.json \
    --output data/selection_test/merged.json

python code/final_codes/classification/create_classification_model/pop_interaction_utils.py merge-samples-from-dir \
    --input-dir data/selection_test/ --output data/selection_test/merged/merged2.json

python code/final_codes/classification/create_classification_model/pop_interaction_utils.py remove-category \
    --input data/selection_test/merged/merged2.json \
    --category "Green" \
    --output data/selection_test/merged/filtered.json

python code/final_codes/classification/create_classification_model/pop_interaction_utils.py remove-samples \
    --input data/selection_test/Green_1.json --indices 1 2 3 4 5 --output data/selection_test/merged/Green_filtered.json

"""