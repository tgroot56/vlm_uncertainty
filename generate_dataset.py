def get_supervision_samples(mc_dataset, num_samples: int = 10):
    """
    Get the first num_samples from the pre-constructed MC dataset for supervision.
    
    Args:
        mc_dataset: Pre-constructed multiple choice dataset
        num_samples: Number of samples to retrieve (default: 10)
        
    Returns:
        List of the first num_samples from mc_dataset
    """
    num_samples = min(num_samples, len(mc_dataset))
    return mc_dataset[:num_samples]

