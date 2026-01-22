import pickle
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple


def load_results(filepath: str) -> Dict:
    """Load the results from pickle file."""
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    return data


def calculate_ece(confidences: np.ndarray, accuracies: np.ndarray, n_bins: int = 10) -> float:
    """
    Calculate Expected Calibration Error (ECE).
    
    Args:
        confidences: Array of confidence scores (max probabilities)
        accuracies: Array of binary correctness (1 for correct, 0 for incorrect)
        n_bins: Number of bins for calibration
    
    Returns:
        ECE score
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece


def create_calibration_data(results: List[Dict], n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create calibration data for plotting.
    
    Returns:
        bin_centers, bin_accuracies, bin_confidences, bin_counts
    """
    # Extract confidences (max probability for predicted letter) and correctness
    confidences = []
    accuracies = []
    
    for result in results:
        pred_letter = result["pred_letter"]
        option_probs = result["option_probs"]
        
        # Get the confidence for the predicted letter
        confidence = option_probs[pred_letter]
        confidences.append(confidence)
        
        # Get correctness
        accuracies.append(1.0 if result["is_correct"] else 0.0)
    
    confidences = np.array(confidences)
    accuracies = np.array(accuracies)
    
    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    bin_centers = (bin_lowers + bin_uppers) / 2
    
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        count = in_bin.sum()
        
        if count > 0:
            bin_accuracies.append(accuracies[in_bin].mean())
            bin_confidences.append(confidences[in_bin].mean())
        else:
            bin_accuracies.append(0.0)
            bin_confidences.append((bin_lower + bin_upper) / 2)
        
        bin_counts.append(count)
    
    return bin_centers, np.array(bin_accuracies), np.array(bin_confidences), np.array(bin_counts)


def plot_calibration(results: List[Dict], n_bins: int = 10, output_file: str = "calibration_plot.png"):
    """
    Create a calibration plot (reliability diagram) showing accuracy vs. confidence.
    """
    # Get calibration data
    bin_centers, bin_accuracies, bin_confidences, bin_counts = create_calibration_data(results, n_bins)
    
    # Calculate ECE
    confidences = np.array([r["option_probs"][r["pred_letter"]] for r in results])
    accuracies = np.array([1.0 if r["is_correct"] else 0.0 for r in results])
    ece = calculate_ece(confidences, accuracies, n_bins)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Bar chart showing accuracy vs confidence
    bar_width = 1.0 / n_bins * 0.8
    colors = plt.cm.RdYlGn(bin_accuracies)
    
    bars = ax1.bar(bin_centers, bin_accuracies, width=bar_width, 
                   color=colors, edgecolor='black', linewidth=1.5, alpha=0.8,
                   label='Accuracy')
    
    # Add confidence line
    ax1.plot(bin_centers, bin_confidences, 'o-', color='blue', 
             linewidth=2, markersize=8, label='Average Confidence')
    
    # Add perfect calibration line
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration')
    
    ax1.set_xlabel('Confidence Bin', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy / Confidence', fontsize=12, fontweight='bold')
    ax1.set_title(f'Calibration Plot (ECE: {ece:.4f})', fontsize=14, fontweight='bold')
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Add sample counts as text on bars
    for i, (center, count) in enumerate(zip(bin_centers, bin_counts)):
        if count > 0:
            ax1.text(center, bin_accuracies[i] + 0.02, f'n={count}', 
                    ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Distribution of predictions across confidence bins
    ax2.bar(bin_centers, bin_counts, width=bar_width, 
            color='steelblue', edgecolor='black', linewidth=1.5, alpha=0.8)
    ax2.set_xlabel('Confidence Bin', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of Predictions', fontsize=12, fontweight='bold')
    ax2.set_title('Distribution of Predictions by Confidence', fontsize=14, fontweight='bold')
    ax2.set_xlim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Calibration plot saved to {output_file}")
    plt.close()


def print_statistics(data: Dict):
    """Print detailed statistics about the results."""
    results = data["results"]
    
    print("=" * 70)
    print("CALIBRATION ANALYSIS RESULTS")
    print("=" * 70)
    print(f"\nTotal samples: {data['total_samples']}")
    print(f"Correct predictions: {data['correct_predictions']}")
    print(f"Overall accuracy: {data['accuracy']:.4f}")
    
    # Calculate confidence statistics
    confidences = np.array([r["option_probs"][r["pred_letter"]] for r in results])
    accuracies = np.array([1.0 if r["is_correct"] else 0.0 for r in results])
    
    print(f"\nConfidence Statistics:")
    print(f"  Mean confidence: {confidences.mean():.4f}")
    print(f"  Median confidence: {np.median(confidences):.4f}")
    print(f"  Std confidence: {confidences.std():.4f}")
    print(f"  Min confidence: {confidences.min():.4f}")
    print(f"  Max confidence: {confidences.max():.4f}")
    
    # Calculate ECE for different bin sizes
    print(f"\nExpected Calibration Error (ECE):")
    for n_bins in [5, 10, 15, 20]:
        ece = calculate_ece(confidences, accuracies, n_bins)
        print(f"  {n_bins:2d} bins: {ece:.6f}")
    
    # Analyze by confidence ranges
    print(f"\nAccuracy by Confidence Range:")
    ranges = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]
    for low, high in ranges:
        mask = (confidences >= low) & (confidences < high)
        if mask.sum() > 0:
            acc = accuracies[mask].mean()
            count = mask.sum()
            avg_conf = confidences[mask].mean()
            print(f"  [{low:.1f}, {high:.1f}): {count:4d} samples, "
                  f"accuracy: {acc:.4f}, avg confidence: {avg_conf:.4f}")
    
    print("=" * 70)


def main():
    # Load results
    print("Loading results from imagenet_r_results.pkl...")
    data = load_results("imagenet_r_results.pkl")
    
    # Print statistics
    print_statistics(data)
    
    # Create calibration plot
    print("\nGenerating calibration plot...")
    plot_calibration(data["results"], n_bins=10, output_file="calibration_plot.png")
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
