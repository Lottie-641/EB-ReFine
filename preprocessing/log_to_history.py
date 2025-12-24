from utility import log_config as lg
from jinja2 import Template
import pandas as pd
import numpy as np
import pickle
import torch
from itertools import chain, repeat, islice
import os


class Log:
    def __init__(self, log_name, setting):
        """
        Initialize the Log object:
          - Read and preprocess the log file.
          - Build label mappings.
          - Compute trace statistics and extract timestamp-based features.
          - Split into training and testing sets.
          - Generate prefix histories, suffixes, and corresponding prefixes.
          - Convert labels to tensors and serialize objects.
        """
        self.__log_name = log_name
        self.__setting = setting
        self.__log = pd.read_csv(f'event_log/{log_name}.csv')
        
        # These will be defined during processing.
        self.__train = None
        self.__test = None
        self.__history_train = None
        self.__history_test = None
        self.__prefixes_train = None  # Holds activity prefixes for training cases.
        self.__prefixes_test = None   # Holds activity prefixes for testing cases.
        self.__len_prefix_train = []
        self.__len_prefix_test = []
        self.__dict_label_train = {}
        self.__dict_label_test = {}
        self.__id2label = {}
        self.__label2id = {}
        self.__max_length = 0
        self.__cont_trace = 0
        self.__max_trace = 0
        self.__mean_trace = 0

        # Process the log step by step.
        self._preprocess_log()
        self._assign_label_mappings()
        self._compute_trace_statistics()
        self._apply_timestamp_features()
        self._split_train_test()
        self._generate_prefix_histories()
        self._convert_labels_to_tensors()
        self._serialize_all_objects()
        self._serialize_label_mappings()

    @staticmethod
    def pad_infinite(iterable, padding=None):
        """Returns an iterator that yields the items from iterable followed by infinite padding."""
        return chain(iterable, repeat(padding))

    @staticmethod
    def pad(iterable, size, padding=None):
        """Pads the iterable to the specified size using the given padding token."""
        return islice(Log.pad_infinite(iterable, padding), size)

    def _preprocess_log(self):
        """
        Clean the log:
          - Remove spaces and special characters from 'activity' (and 'resource', if applicable).
          - Fill missing values.
          - Convert the 'timestamp' column to datetime.
        """
        self.__log['activity'] = self.__log['activity'].str.replace(' ', '')
        self.__log['activity'] = self.__log['activity'].str.replace('+', '')
        self.__log['activity'] = self.__log['activity'].str.replace('-', '')
        self.__log['activity'] = self.__log['activity'].str.replace('_', '')
        
        if self.__log_name not in ['sepsis', 'wastewater']:
            self.__log['resource'] = self.__log['resource'].astype(str)
            self.__log['resource'] = self.__log['resource'].str.replace(' ', '')
            self.__log['resource'] = self.__log['resource'].str.replace('+', '')
            self.__log['resource'] = self.__log['resource'].str.replace('-', '')
            self.__log['resource'] = self.__log['resource'].str.replace('_', '')
        
        self.__log.fillna('UNK', inplace=True)
        self.__log['timestamp'] = pd.to_datetime(self.__log['timestamp'])

    def _assign_label_mappings(self):
        """
        Build label mapping dictionaries for each event attribute (except 'timesincecasestart').
        For each attribute, all unique values are extracted and a special terminal token (e.g., 'ENDactivity')
        is added.
        """
        for attribute in lg.log[self.__log_name]['event_attribute']:
            if attribute != 'timesincecasestart':
                all_labels = list(self.__log[attribute].unique())
                all_labels.append('END' + attribute)
                self.__id2label[attribute] = {k: label for k, label in enumerate(all_labels)}
                self.__label2id[attribute] = {label: k for k, label in enumerate(all_labels)}

    def _compute_trace_statistics(self):
        """
        Compute trace statistics:
          - The number of events per case.
          - Maximum trace length and mean trace length.
        """
        self.__cont_trace = self.__log['case'].value_counts(dropna=False)
        self.__max_trace = int(self.__cont_trace.max())
        self.__mean_trace = int(round(np.mean(self.__cont_trace)))
        self.__max_length = int(self.__cont_trace.max())

    def _apply_timestamp_features(self):
        """
        Compute timestamp-based features:
          - 'timesincelastevent': Time (in seconds) from the previous event.
          - 'timesincecasestart': Elapsed time (in seconds) since the case began.
        """
        self.__log = self.__log.groupby('case', group_keys=False).apply(self._extract_timestamp_features)
        self.__log = self.__log.reset_index(drop=True)
        self.__log['timesincecasestart'] = self.__log['timesincecasestart'].astype(int)

    def _extract_timestamp_features(self, group):
        """
        For a given case (group), sort by timestamp and compute:
          - The time difference with the previous event.
          - The elapsed time since the case start.
        """
        group = group.sort_values('timestamp', ascending=True)
        start_date = group['timestamp'].iloc[0]
        group["timesincelastevent"] = group['timestamp'].diff() \
            .fillna(pd.Timedelta(seconds=0)) \
            .apply(lambda x: float(x / np.timedelta64(1, 's')))
        group["timesincecasestart"] = (group['timestamp'] - start_date) \
            .fillna(pd.Timedelta(seconds=0)) \
            .apply(lambda x: float(x / np.timedelta64(1, 's')))
        return group

    def _split_train_test(self):
        """
        Split the log into training and testing sets based on the case start times.
          - Approximately the first 66% of cases are used for training.
        """
        grouped = self.__log.groupby("case")
        start_timestamps = grouped["timestamp"].min().reset_index()
        start_timestamps = start_timestamps.sort_values("timestamp", ascending=True, kind="mergesort")
        train_ids = list(start_timestamps["case"])[:int(0.66 * len(start_timestamps))]
        self.__train = self.__log[self.__log["case"].isin(train_ids)] \
            .sort_values("timestamp", ascending=True, kind='mergesort')
        self.__test = self.__log[~self.__log["case"].isin(train_ids)] \
            .sort_values("timestamp", ascending=True, kind='mergesort')

    def _generate_prefix_history(self, df):
        """
        Generate prefix histories, suffixes (padded to fixed length), and the corresponding label dictionaries.
        
        Additionally, generate a list of padded activity prefixes corresponding to the suffixes.
        
        Returns:
            tuple:
              - list_seq: List of prefix history strings (concatenation of event and trace texts).
              - prefixes: List of padded activity prefix sequences corresponding to the suffixes.
              - dict_event_label: Dictionary mapping each event attribute to the list of "next event" labels.
              - list_len_prefix: List containing the length (i.e. number of events) for each prefix.
              - dict_len_label: Dictionary (keyed by position) with label IDs for each suffix position.
              - dict_entire_label: Dictionary (keyed by position) with label IDs for the full padded activity sequences.
        """
        list_seq = []
        list_len_prefix = []
        dict_entire_label = {i: [] for i in range(self.__max_length)}

        event_template = Template(lg.log[self.__log_name]['event_template'])
        trace_template = Template(lg.log[self.__log_name]['trace_template'])

        dict_event_label = {attr: [] for attr in lg.log[self.__log_name]['event_attribute']}
        dict_trace_label = {attr: [] for attr in lg.log[self.__log_name]['trace_attribute']}
        dict_len_label = {i: [] for i in range(self.__max_length)}
        dict_prefix_label = {i: [] for i in range(self.__max_length)}

        prefixes = []  # Holds the activity prefixes corresponding to each suffix.

        # Process each trace (grouped by case).
        all_traces = []
        for group_name, group_data in df.groupby('case', sort=False):
            event_dict_hist = {}
            trace_dict_hist = {}
            event_text = ''
            len_prefix = 1
            activities_per_trace = []

            # Build the event history text and record the activities.
            for index, row in group_data.iterrows():
                activities_per_trace.append(row['activity'])
                for v in lg.log[self.__log_name]['event_attribute']:
                    value = row[v]
                    event_dict_hist[v] = value.replace(' ', '') if isinstance(value, str) else value
                event_text += event_template.render(event_dict_hist) + ' '
                for w in lg.log[self.__log_name]['trace_attribute']:
                    value = row[w]
                    trace_dict_hist[w] = value.replace(' ', '') if isinstance(value, str) else value
                trace_text = trace_template.render(trace_dict_hist)

                prefix_hist = event_text + trace_text
                list_seq.append(prefix_hist)
                list_len_prefix.append(len_prefix)
                len_prefix += 1
                
            for i in range(len(activities_per_trace)):
                all_traces.append(list(self.pad(activities_per_trace, self.__max_length, 'ENDactivity')))

            # Save the original activity list for prefix generation.
            activity_list = activities_per_trace.copy()
            original_activities = activities_per_trace.copy()
            
            # Modify activity_list for suffix generation.
            if activity_list:
                activity_list.pop(0)
            activity_list.append('ENDactivity')

            # Generate suffixes and the full padded activity sequences.
            suffixes = []
            activities = []
            for i in range(len(activity_list)):
                suffix = list(self.pad(activity_list[i:], self.__max_length, 'ENDactivity'))
                suffixes.append(suffix)
                #activities.append(list(self.pad(activity_list, self.__max_length, 'ENDactivity')))
                
            # Generate the corresponding activity prefixes using the original activity list.
            for i in range(len(original_activities)):
                prefix = list(self.pad(original_activities[:i+1], self.__max_length, 'ENDactivity'))
                #prefix = [activity for activity in prefix if activity != 'ENDactivity']
                activities.append(list(self.pad(original_activities, self.__max_length, 'ENDactivity')))
                prefixes.append(prefix)
                
            # Build dict_len_label and dict_entire_label using the suffixes.
            for s in suffixes:
                for i in range(len(s)):
                    dict_len_label[i].append(self.__label2id['activity'][s[i]])
            for a in activities:
                for i in range(len(a)):
                    dict_entire_label[i].append(self.__label2id['activity'][a[i]])
            for s in prefixes:
                for i in range(len(s)):
                    dict_prefix_label[i].append(self.__label2id['activity'][s[i]])

            # Build next-event labels for each event attribute.
            for v in lg.log[self.__log_name]['event_attribute']:
                if v != 'timesincecasestart':
                    shifted = group_data[v].shift(-1).fillna('END' + v).tolist()
                    dict_event_label[v].extend(shifted)
                else:
                    shifted = group_data[v].shift(-1).fillna(0).tolist()
                    dict_event_label[v].extend(shifted)

        return list_seq, dict_prefix_label, dict_event_label, list_len_prefix, dict_len_label, dict_entire_label, all_traces
        
    def _extract_distinct_trace_frequencies(self, all_traces):
        """
        Given a list of traces (all_traces) where each trace is a sequence of activities,
        count how many times each unique trace appears, encode each trace using self.__label2id['activity'],
        calculate the frequency percentages (relative to the total number of traces), and save the resulting
        dictionary to a pickle file.
    
        Args:
            all_traces (list): A list of traces, where each trace is a list (or sequence) of activity strings.
        """
        trace_counts = {}
        total_cases = len(all_traces)
        
        # Count how many times each distinct trace appears.
        for trace in all_traces:
            # Convert the trace to a tuple so it can be used as a dictionary key.
            trace_tuple = tuple(trace)
            trace_counts[trace_tuple] = trace_counts.get(trace_tuple, 0) + 1
        encoded_trace_freq = {}
        # Encode each trace and compute its frequency percentage.
        for trace, count in trace_counts.items():
            # Encode the trace using the activity label mapping.
            #print("************trace*********", trace)
            encoded_trace = tuple(self.__label2id['activity'][act] for act in trace)
            encoded_trace_freq[encoded_trace] = (count / total_cases) * 100
        #print("encoded_trace", encoded_trace)
        # Order the encoded traces based on frequency (highest frequency first).
        sorted_encoded_trace_freq = dict(
            sorted(encoded_trace_freq.items(), key=lambda x: x[1], reverse=True)
        )
        #print("sorted_encoded_trace_freq", sorted_encoded_trace_freq)

        # Save the resulting dictionary.
        self._serialize_object(sorted_encoded_trace_freq, 'encoded_trace_frequencies')
        
    def _generate_prefix_histories(self):
        """
        Generate prefix histories for both the training and testing sets using _generate_prefix_history.
        This updated version also captures and serializes the activity prefixes.
        """
        (self.__history_train, self.__prefixes_train, self.__dict_label_train,
         self.__len_prefix_train, dict_suffix_train, entire_activities_train, all_traces_train) = self._generate_prefix_history(self.__train)

        (self.__history_test, self.__prefixes_test, self.__dict_label_test,
         self.__len_prefix_test, dict_suffix_test, entire_activities_test, all_traces_test) = self._generate_prefix_history(self.__test)

        All_traces = all_traces_train + all_traces_test
        print("**************", All_traces[:10])

        # Convert event attribute labels to PyTorch tensors.
        for label_dict in (self.__dict_label_train, self.__dict_label_test):
            for key, values in label_dict.items():
                if key != 'timesincecasestart':
                    tensor_values = [self.__label2id[key].get(val) for val in values]
                    label_dict[key] = torch.tensor(tensor_values)
                else:
                    label_dict[key] = torch.tensor(values).view(-1, 1)

        # Serialize the generated objects.
        self._serialize_object(self.__history_train, 'train')
        self._serialize_object(self.__history_test, 'test')
        self._serialize_object(self.__len_prefix_train, 'len_train')
        self._serialize_object(self.__len_prefix_test, 'len_test')
        self._serialize_object(dict_suffix_train, 'suffix_train')
        self._serialize_object(dict_suffix_test, 'suffix_test')
        self._serialize_object(entire_activities_train, 'activities_train')
        self._serialize_object(entire_activities_test, 'activities_test')
        self._serialize_object(self.__dict_label_train[lg.log[self.__log_name]['target']], 'label_train')
        self._serialize_object(self.__dict_label_test[lg.log[self.__log_name]['target']], 'label_test')
        # Serialize the activity prefixes.
        self._serialize_object(self.__prefixes_train, 'prefixes_train')
        self._serialize_object(self.__prefixes_test, 'prefixes_test')
        self._extract_distinct_trace_frequencies(All_traces)

    def _convert_labels_to_tensors(self):
        """
        (Optional) Additional conversion if needed.
        Note: Label conversion is already performed in _generate_prefix_histories.
        """
        pass

    def _serialize_object(self, obj, obj_type):
        """
        Serialize an object to a pickle file.
        
        Args:
            obj: The object to be serialized.
            obj_type (str): A label used in the filename.
        """
        os.makedirs(f"semantic_data/{self.__log_name}", exist_ok=True)
        filename = f'semantic_data/{self.__log_name}/{self.__log_name}_{obj_type}_{self.__setting}.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(obj, f)

    def _serialize_all_objects(self):
        """
        (Optional) Serialize additional objects if necessary.
        Note: Serialization of key objects is already performed in _generate_prefix_histories.
        """
        pass

    def _serialize_label_mappings(self):
        """
        Serialize the label mapping dictionaries.
        """
        os.makedirs(f"semantic_data/{self.__log_name}", exist_ok=True)
        with open(f'semantic_data/{self.__log_name}/{self.__log_name}_id2label_{self.__setting}.pkl', 'wb') as f:
            pickle.dump(self.__id2label, f)
        with open(f'semantic_data/{self.__log_name}/{self.__log_name}_label2id_{self.__setting}.pkl', 'wb') as f:
            pickle.dump(self.__label2id, f)

    def get_id2label(self):
        """Return the dictionary mapping IDs to labels."""
        return self.__id2label

    def get_label2id(self):
        """Return the dictionary mapping labels to IDs."""
        return self.__label2id
