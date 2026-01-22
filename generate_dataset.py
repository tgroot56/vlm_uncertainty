# get only first 10 datapoints
def get_data_split(dataset, split: str = "test", num_points: int = 10):
    """
    Get a subset of the dataset split with only the first num_points datapoints.
    
    Args:
        dataset: HuggingFace dataset object
        split: Dataset split to use ("train", "test", etc.)
        num_points: Number of datapoints to retrieve
    """
    return dataset[split].select(range(num_points))

