# """Example script showing how to train probes with different feature combinations."""

# import os
# from probe.train import train_probe

# # Path to the supervision dataset
# DATA_PATH = "supervision_dataset_with_features.pkl"

# # Create output directory
# os.makedirs("probe_experiments", exist_ok=True)

# # Example 1: Train on vision features only (final layer)
# print("\n" + "="*80)
# print("Experiment 1: Vision Features (Final Layer)")
# print("="*80)
# train_probe(
#     data_path=DATA_PATH,
#     feature_names=['vision_final_layer_features'],
#     output_dir='probe_experiments/vision_final',
#     num_epochs=100,
#     learning_rate=0.01,
# )

# # Example 2: Train on LM answer features (final layer)
# print("\n" + "="*80)
# print("Experiment 2: LM Answer Features (Final Layer)")
# print("="*80)
# train_probe(
#     data_path=DATA_PATH,
#     feature_names=['lm_final_answer_features'],
#     output_dir='probe_experiments/lm_answer_final',
#     num_epochs=100,
#     learning_rate=0.01,
# )

# # Example 3: Train on combined vision + LM features
# print("\n" + "="*80)
# print("Experiment 3: Combined Vision + LM Features")
# print("="*80)
# train_probe(
#     data_path=DATA_PATH,
#     feature_names=[
#         'vision_final_layer_features',
#         'lm_final_visual_features',
#         'lm_final_prompt_features',
#         'lm_final_answer_features',
#     ],
#     output_dir='probe_experiments/combined_final',
#     num_epochs=100,
#     learning_rate=0.01,
# )

# # Example 4: Train on all available features
# print("\n" + "="*80)
# print("Experiment 4: All Features")
# print("="*80)
# train_probe(
#     data_path=DATA_PATH,
#     feature_names=[
#         'vision_middle_layer_features',
#         'vision_final_layer_features',
#         'lm_middle_visual_features',
#         'lm_final_visual_features',
#         'lm_middle_prompt_features',
#         'lm_final_prompt_features',
#         'lm_middle_answer_features',
#         'lm_final_answer_features',
#     ],
#     output_dir='probe_experiments/all_features',
#     num_epochs=100,
#     learning_rate=0.01,
# )

# print("\n" + "="*80)
# print("All experiments completed!")
# print("="*80)
