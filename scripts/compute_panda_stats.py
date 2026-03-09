import os
import glob
import json
import numpy as np

def compute_panda_stats(dataset_root, output_path):
    # Find all episodes_stats.jsonl files
    pattern = os.path.join(dataset_root, "no_noise_demo_1_round/config_*_train/lerobot_data/meta/episodes_stats.jsonl")
    stat_files = glob.glob(pattern)
    
    if not stat_files:
        print(f"No stats files found matching {pattern}")
        return
        
    print(f"Found {len(stat_files)} stats files.")
    
    # We will accumulate means and vars using Welford's online algorithm or basic weighted average
    # For min/max it's just min/max over all files
    # observation.state is 9 dims, action is 8 dims
    
    agg_stats = {
        "observation.state": {
            "min": np.inf * np.ones(9),
            "max": -np.inf * np.ones(9),
            "sum": np.zeros(9),
            "sum_sq": np.zeros(9),
            "count": 0
        },
        "action": {
            "min": np.inf * np.ones(8),
            "max": -np.inf * np.ones(8),
            "sum": np.zeros(8),
            "sum_sq": np.zeros(8),
            "count": 0
        }
    }
    
    for file_path in stat_files:
        with open(file_path, 'r') as f:
            for line in f:
                ep_stat = json.loads(line)
                stats = ep_stat["stats"]
                
                for key in ["observation.state", "action"]:
                    if key in stats:
                        s = stats[key]
                        c = s["count"][0]
                        min_v = np.array(s["min"])
                        max_v = np.array(s["max"])
                        mean_v = np.array(s["mean"])
                        std_v = np.array(s["std"])
                        
                        agg_stats[key]["min"] = np.minimum(agg_stats[key]["min"], min_v)
                        agg_stats[key]["max"] = np.maximum(agg_stats[key]["max"], max_v)
                        agg_stats[key]["sum"] += mean_v * c
                        # E[x^2] = Var(x) + E[x]^2
                        var_v = std_v ** 2
                        agg_stats[key]["sum_sq"] += (var_v + mean_v ** 2) * c
                        agg_stats[key]["count"] += c

    final_stats = {
        "observation.state": {},
        "action": {}
    }
    
    for key in ["observation.state", "action"]:
        c = agg_stats[key]["count"]
        if c > 0:
            mean = agg_stats[key]["sum"] / c
            # Var(x) = E[x^2] - E[x]^2
            var = (agg_stats[key]["sum_sq"] / c) - (mean ** 2)
            # Handle float precision issues
            var = np.maximum(var, 0)
            std = np.sqrt(var)
            
            final_stats[key]["min"] = agg_stats[key]["min"].tolist()
            final_stats[key]["max"] = agg_stats[key]["max"].tolist()
            final_stats[key]["mean"] = mean.tolist()
            final_stats[key]["std"] = std.tolist()
            
            # Optional: Add q01 and q99 as approximations or fallbacks if needed
            # For simplicity, we can use min/max or set them to min/max
            final_stats[key]["q01"] = final_stats[key]["min"]
            final_stats[key]["q99"] = final_stats[key]["max"]
            
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(final_stats, f, indent=4)
        
    print(f"Stats successfully written to {output_path}")

if __name__ == "__main__":
    compute_panda_stats("/root/wise_dataset_0.3.2", "/root/DreamDojo/shared_meta/panda_stats.json")
