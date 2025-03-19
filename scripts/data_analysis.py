# import pandas as pd
# import os

# def analyze_fluctuations(files, output_file):
#     results = []
    
#     for file_path in files:
#         df = pd.read_csv(file_path)
        
#         # Calculate statistical information
#         avg_time_gap = round(df["time_gap_minutes"].mean(),2)
#         count_below_15 = (df["time_gap_minutes"] < 15).sum()
#         avg_time_gap_15min = round(df[df["time_gap_minutes"] < 15]["time_gap_minutes"].mean(),2)
#         direction_distribution = df["direction"].value_counts().to_dict()
        
#         # Get file name
#         file_name = os.path.basename(file_path)
        
#         # Organize results
#         results.append({
#             "file": file_name,
#             "avg_time_gap": avg_time_gap,
#             "count_below_15": count_below_15,
#             "avg_time_gap_15min": avg_time_gap_15min,
#             "before_count": direction_distribution.get("before", 0),
#             "after_count": direction_distribution.get("after", 0)
#         })
    
#     # Convert to DataFrame and write to CSV
#     result_df = pd.DataFrame(results)
#     result_df.to_csv(output_file, index=False)
#     print(f"Results saved to {output_file}")

# # Define file paths
# # input_dir = "data/processed/2018/training"
# # files = [
# #     os.path.join(input_dir, f) for f in [
# #         "559_fluctuations.csv",
# #         "563_fluctuations.csv",
# #         "570_fluctuations.csv",
# #         "575_fluctuations.csv",
# #         "588_fluctuations.csv",
# #         "591_fluctuations.csv"
# #     ]
# # ]

# input_dir = "data/processed/2020/training"
# files = [
#     os.path.join(input_dir, f) for f in [
#         "540_fluctuations.csv",
#         "544_fluctuations.csv",
#         "552_fluctuations.csv",
#         "567_fluctuations.csv",
#         "584_fluctuations.csv",
#         "596_fluctuations.csv"
#     ]
# ]

# output_file = os.path.join(input_dir, "result.csv")

# # Run analysis
# analyze_fluctuations(files, output_file)

import pandas as pd
import os

def analyze_fluctuations(files, output_file):
    results = []
    
    for file_path in files:
        df = pd.read_csv(file_path)
        
        # Calculate statistical information
        avg_time_gap = round(df["time_gap_minutes"].mean(), 2)
        count_below_15 = (df["time_gap_minutes"] < 15).sum()
        avg_time_gap_15min = round(df[df["time_gap_minutes"] < 15]["time_gap_minutes"].mean(), 2)
        
        # Normalize value counts and round to 2 decimal places
        direction_distribution = {k: round(v, 2) for k, v in df["direction"].value_counts(normalize=True).to_dict().items()}
        pre_fluctuation_category_dist = {k: round(v, 2) for k, v in df["pre_fluctuation_category"].value_counts(normalize=True).to_dict().items()}
        post_fluctuation_category_dist = {k: round(v, 2) for k, v in df["post_fluctuation_category"].value_counts(normalize=True).to_dict().items()}
        pre_fluctuation_direction_dist = {k: round(v, 2) for k, v in df["pre_fluctuation_direction"].value_counts(normalize=True).to_dict().items()}
        post_fluctuation_direction_dist = {k: round(v, 2) for k, v in df["post_fluctuation_direction"].value_counts(normalize=True).to_dict().items()}
        
        # Get file name
        file_name = os.path.basename(file_path)
        
        # Organize results
        results.append({
            "file": file_name,
            "avg_time_gap": avg_time_gap,
            "count_below_15": count_below_15,
            "avg_time_gap_15min": avg_time_gap_15min,
            "before_count": direction_distribution.get("before", 0),
            "after_count": direction_distribution.get("after", 0),
            "pre_fluctuation_category": pre_fluctuation_category_dist,
            "post_fluctuation_category": post_fluctuation_category_dist,
            "pre_fluctuation_direction_UP": pre_fluctuation_direction_dist.get("UP", 0),
            "pre_fluctuation_direction_DOWN": pre_fluctuation_direction_dist.get("DOWN", 0),
            "post_fluctuation_direction_UP": post_fluctuation_direction_dist.get("UP", 0),
            "post_fluctuation_direction_DOWN": post_fluctuation_direction_dist.get("DOWN", 0),
        })
    
    # Convert to DataFrame and write to CSV
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

# Define file paths
# input_dir = "data/processed/2020/training"
# files = [
#     os.path.join(input_dir, f) for f in [
#         "540_fluctuations.csv",
#         "544_fluctuations.csv",
#         "552_fluctuations.csv",
#         "567_fluctuations.csv",
#         "584_fluctuations.csv",
#         "596_fluctuations.csv"
#     ]
# ]

# Define file paths
input_dir = "data/processed/2018/training"
files = [
    os.path.join(input_dir, f) for f in [
        "559_fluctuations.csv",
        "563_fluctuations.csv",
        "570_fluctuations.csv",
        "575_fluctuations.csv",
        "588_fluctuations.csv",
        "591_fluctuations.csv"
    ]
]

output_file = os.path.join(input_dir, "result.csv")

# Run analysis
analyze_fluctuations(files, output_file)

