import pandas as pd
import pickle
from itertools import chain
import torch

class TraceEncoder:
    def __init__(self, log_name):
        self.log_name = log_name
        self.log = pd.read_csv(f'event_log/{log_name}.csv')
        self.distinct_traces = {}
        self.label2id = {}
        self.id2label = {}
        self.process_log()

    def process_log(self):
        print("Processing log to extract distinct traces and their frequency percentages...")

        # Clean activity names
        self.log['activity'] = self.log['activity'].str.replace(' ', '').str.replace('+', '').str.replace('-', '').str.replace('_', '')

        # Extract unique traces per case
        grouped = self.log.groupby('case', sort=False)['activity'].apply(tuple)  # Convert lists to immutable tuples
        
        # Count distinct traces
        total_traces = len(grouped)  # Total number of traces in the event log
        trace_counts = grouped.value_counts()  # Get count of each unique trace

        # Calculate frequency as percentage
        self.distinct_traces = {trace: (count / total_traces) * 100 for trace, count in trace_counts.items()}

        # Generate label mappings
        all_activities = set(chain(*self.distinct_traces.keys()))
        self.label2id = {label: idx for idx, label in enumerate(all_activities)}
        self.id2label = {idx: label for label, idx in self.label2id.items()}

        # Encode traces using label2id mapping
        encoded_traces = {tuple(self.label2id[a] for a in trace): freq for trace, freq in self.distinct_traces.items()}

        # Convert to tensors
        self.tensor_traces = {torch.tensor(trace): freq for trace, freq in encoded_traces.items()}

        # Save encoded traces and mappings
        self.save_data("distinct_traces.pkl", self.tensor_traces)
        self.save_data("label2id.pkl", self.label2id)
        self.save_data("id2label.pkl", self.id2label)
        print("Distinct traces and their frequencies (%) encoded and stored successfully.")

    def save_data(self, filename, data):
        with open(f'semantic_data/{self.log_name}/{filename}', 'wb') as f:
            pickle.dump(data, f)

# Usage example
log_encoder = TraceEncoder('your_log_file_name')
