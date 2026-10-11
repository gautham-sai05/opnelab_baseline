import scipy.io as sio
import numpy as np
import json
import matplotlib.pyplot as plt

def inspect_people_mat(filepath):
    print("Loading People.mat...")
    mat = sio.loadmat(filepath, simplify_cells=True)
    
    if 'people' not in mat:
        print("Error: 'people' key not found.")
        return
        
    people = mat['people'] # Should be an array of length 10
    
    num_people = len(people)
    print(f"Number of people: {num_people}")
    
    stats = {
        'num_people': num_people,
        'trace_lengths': [],
        'mins': [],
        'maxs': [],
        'means': [],
        'stds': [],
        'nan_count': 0,
        'inf_count': 0,
        'sensors_per_person': [],
        'speeds_per_sensor': [],
        'traces_per_speed': [],
        'amplifications_per_trace': []
    }
    
    examples = []
    
    for p_idx, person in enumerate(people):
        # person is a dict-like if simplify_cells=True worked, or struct array
        # Let's see what keys are in person
        if not isinstance(person, dict):
            # It might still be a numpy record array
            pass
            
        sensors = person['Sen'] if isinstance(person, dict) and 'Sen' in person else person
        stats['sensors_per_person'].append(len(sensors))
        
        for s_idx, sensor in enumerate(sensors):
            speeds = sensor['S'] if isinstance(sensor, dict) and 'S' in sensor else sensor
            stats['speeds_per_sensor'].append(len(speeds))
            
            for sp_idx, speed in enumerate(speeds):
                traces = speed['T'] if isinstance(speed, dict) and 'T' in speed else speed
                stats['traces_per_speed'].append(len(traces))
                
                for t_idx, trace in enumerate(traces):
                    amps = trace['A'] if isinstance(trace, dict) and 'A' in trace else trace
                    stats['amplifications_per_trace'].append(len(amps))
                    
                    for a_idx, amp_data in enumerate(amps):
                        if isinstance(amp_data, np.ndarray):
                            data = amp_data
                        else:
                            data = np.array(amp_data)
                            
                        data = data.flatten()
                        stats['trace_lengths'].append(len(data))
                        stats['mins'].append(np.min(data))
                        stats['maxs'].append(np.max(data))
                        stats['means'].append(np.mean(data))
                        stats['stds'].append(np.std(data))
                        stats['nan_count'] += np.isnan(data).sum()
                        stats['inf_count'] += np.isinf(data).sum()
                        
                        if len(examples) < 3:
                            examples.append(data)
                            
    # Aggregate
    result = {
        'num_people': num_people,
        'avg_sensors': np.mean(stats['sensors_per_person']),
        'avg_speeds': np.mean(stats['speeds_per_sensor']),
        'avg_traces': np.mean(stats['traces_per_speed']),
        'avg_amps': np.mean(stats['amplifications_per_trace']),
        'total_traces': len(stats['trace_lengths']),
        'min_trace_length': int(np.min(stats['trace_lengths'])),
        'max_trace_length': int(np.max(stats['trace_lengths'])),
        'median_trace_length': int(np.median(stats['trace_lengths'])),
        'global_min': float(np.min(stats['mins'])),
        'global_max': float(np.max(stats['maxs'])),
        'global_mean': float(np.mean(stats['means'])),
        'global_std': float(np.mean(stats['stds'])),
        'total_nans': int(stats['nan_count']),
        'total_infs': int(stats['inf_count'])
    }
    
    with open('../results/dataset_audit.json', 'w') as f:
        json.dump(result, f, indent=4)
        
    print("Saved dataset_audit.json")
    
    # Plot examples
    if len(examples) > 0:
        plt.figure(figsize=(12, 8))
        for i, ex in enumerate(examples):
            plt.subplot(3, 1, i+1)
            plt.plot(ex)
            plt.title(f"Example Human Trace {i+1}")
        plt.tight_layout()
        plt.savefig('../results/figures/dataset/human_raw_examples.png')
        print("Saved human_raw_examples.png")

if __name__ == "__main__":
    inspect_people_mat('../../People.mat')
