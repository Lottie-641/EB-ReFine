from collections import Counter 
from utility import log_config as lg
from jinja2 import Template
import pandas as pd
import numpy as np
import pickle
import torch
from itertools import chain, repeat, islice

class Log():
    def __init__(self, log, setting):
        self.__log_name = log
        self.__log = pd.read_csv('event_log/'+log+'.csv')
        self.__train = []
        self.__test = []
        self.__len_prefix_train = []
        self.__len_prefix_test = []
        self.__history_train = []
        self.__history_test = []
        self.__dict_label_train = []
        self.__dict_label_test = []
        self.__id2label = {}
        self.__label2id = {}
        self.__setting = setting
        self.__max_length = 0
        self.__cont_trace = 0
        self.__max_trace = 0
        self.__mean_trace = 0
        self.__split_log()

    def pad_infinite(self, iterable, padding=None):
        return chain(iterable, repeat(padding))

    def pad(self, iterable, size, padding=None):
        return islice(self.pad_infinite(iterable, padding), size)

    def __gen_prefix_history(self, df):
        list_seq = []
        list_len_prefix = []
        dict_entire_label = {i: [] for i in range(self.__max_length)}
        sequence = df.groupby('case', sort=False)
        event_template = Template(lg.log[self.__log_name]['event_template'])
        trace_template = Template(lg.log[self.__log_name]['trace_template'])

        dict_event_label = {}
        for v in lg.log[self.__log_name]['event_attribute']:
            dict_event_label[v] = []
        dict_trace_label = {}
        for v in lg.log[self.__log_name]['trace_attribute']:
            dict_trace_label[v] = []

        dict_len_label = {}
        for i in range(self.__max_length):
            dict_len_label[i] = []

        for group_name, group_data in sequence:
            event_dict_hist = {}
            trace_dict_hist = {}
            event_text = ''
            len_prefix = 1
            activity_list = []
            for index, row in group_data.iterrows():
                activity_list.append(row['activity'])
                for v in lg.log[self.__log_name]['event_attribute']:
                    value = row[v]
                    if isinstance(value, str):
                        event_dict_hist[v] = value.replace(' ', '')
                    else:
                        event_dict_hist[v] = value
                event_text = event_text + event_template.render(event_dict_hist) + ' '
                for w in lg.log[self.__log_name]['trace_attribute']:
                    value = row[w]
                    if isinstance(value, str):
                        trace_dict_hist[w] = value.replace(' ', '')
                    else:
                        trace_dict_hist[w] = value
                trace_text = trace_template.render(trace_dict_hist)

                prefix_hist = event_text + trace_text
                list_seq.append(prefix_hist)
                list_len_prefix.append(len_prefix)
                len_prefix = len_prefix + 1

            suffixes = []
            activities = []
            activity_list.pop(0)
            activity_list.append('ENDactivity')

            for i in range(len(activity_list)):
                suffixes.append(list(self.pad(activity_list[i:], self.__max_length, 'ENDactivity')))
                activities.append(list(self.pad(activity_list, self.__max_length, 'ENDactivity')))
            for s in suffixes:
                for i in range(len(s)):
                    dict_len_label[i].append(self.__label2id['activity'][s[i]])
            for a in activities:
                for i in range(len(a)):
                    dict_entire_label[i].append(self.__label2id['activity'][a[i]])

            for v in lg.log[self.__log_name]['event_attribute']:
                if v != 'timesincecasestart':
                    dict_event_label[v].extend(group_data[v].shift(-1).fillna('END' + v).tolist())
                else:
                    dict_event_label[v].extend(group_data[v].shift(-1).fillna(0).tolist())
        return list_seq, dict_event_label, list_len_prefix, dict_len_label, dict_entire_label

    def __extract_timestamp_features(self, group):
        timestamp_col = 'timestamp'
        group = group.sort_values(timestamp_col, ascending=True)
        start_date = group[timestamp_col].iloc[0]

        timesincelastevent = group[timestamp_col].diff()
        timesincelastevent = timesincelastevent.fillna(pd.Timedelta(seconds=0))
        group["timesincelastevent"] = timesincelastevent.apply(
            lambda x: float(x / np.timedelta64(1, 's')))

        elapsed = group[timestamp_col] - start_date
        elapsed = elapsed.fillna(pd.Timedelta(seconds=0))
        group["timesincecasestart"] = elapsed.apply(lambda x: float(x / np.timedelta64(1, 's')))
        return group

    def __split_log(self):
        self.__log['activity'] = self.__log['activity'].str.replace(' ', '')
        self.__log['activity'] = self.__log['activity'].str.replace('+', '')
        self.__log['activity'] = self.__log['activity'].str.replace('-', '')
        self.__log['activity'] = self.__log['activity'].str.replace('_', '')

        if self.__log_name != 'sepsis':
            self.__log['resource'] = self.__log['resource'].astype(str)
            self.__log['resource'] = self.__log['resource'].str.replace(' ', '')
            self.__log['resource'] = self.__log['resource'].str.replace('+', '')
            self.__log['resource'] = self.__log['resource'].str.replace('-', '')
            self.__log['resource'] = self.__log['resource'].str.replace('_', '')

        self.__log.fillna('UNK', inplace=True)

        self.__cont_trace = self.__log['case'].value_counts(dropna=False)
        self.__max_trace = max(self.__cont_trace)

        self.__mean_trace = int(round(np.mean(self.__cont_trace)))

        self.__log['timestamp'] = pd.to_datetime(self.__log['timestamp'])
        for c in lg.log[self.__log_name]['event_attribute']:
            if c != 'timesincecasestart':
                ALL_LABEL = list(self.__log[c].unique())
                ALL_LABEL.append('END' + c)
                self.__id2label[c] = {k: l for k, l in enumerate(ALL_LABEL)}
                self.__label2id[c] = {l: k for k, l in enumerate(ALL_LABEL)}

        cont_trace = self.__log['case'].value_counts(dropna=False)
        self.__max_length = max(cont_trace)

        self.__log = self.__log.groupby('case', group_keys=False).apply(self.__extract_timestamp_features)
        self.__log = self.__log.reset_index(drop=True)
        self.__log['timesincecasestart'] = self.__log['timesincecasestart']
        self.__log['timesincecasestart'] = self.__log['timesincecasestart'].astype(int)

        grouped = self.__log.groupby("case")
        start_timestamps = grouped["timestamp"].min().reset_index()
        start_timestamps = start_timestamps.sort_values("timestamp", ascending=True, kind="mergesort")
        train_ids = list(start_timestamps["case"])[:int(0.66 * len(start_timestamps))]
        self.__train = self.__log[self.__log["case"].isin(train_ids)].sort_values("timestamp", ascending=True, kind='mergesort')
        self.__test = self.__log[~self.__log["case"].isin(train_ids)].sort_values("timestamp", ascending=True, kind='mergesort')

        self.__history_train, self.__dict_label_train, self.__len_prefix_train, dict_suffix_train, entire_activities_train = self.__gen_prefix_history(self.__train)
        self.__history_test, self.__dict_label_test, self.__len_prefix_test, dict_suffix_test, entire_activities_test = self.__gen_prefix_history(self.__test)

        for v in self.__dict_label_train:
            if v != 'timesincecasestart':
                temp_list = []
                for key in self.__dict_label_train[v]:
                    temp_list.append(self.__label2id[v].get(key))
                self.__dict_label_train[v] = torch.tensor(temp_list)
            else:
                self.__dict_label_train[v] = torch.tensor(self.__dict_label_train[v]).view(-1, 1)

        for v in self.__dict_label_test:
            if v != 'timesincecasestart':
                temp_list = []
                for key in self.__dict_label_test[v]:
                    temp_list.append(self.__label2id[v].get(key))
                self.__dict_label_test[v] = torch.tensor(temp_list)
            else:
                self.__dict_label_test[v] = torch.tensor(self.__dict_label_test[v]).view(-1, 1)

        #extract_encoded_distinct_traces_with_frequency
        grouped = self.__log.groupby('case')
        encoded_traces = []
        for case, group in grouped:
            trace = tuple(self.__label2id['activity'][activity] for activity in group['activity'].tolist())
            encoded_traces.append(trace)
        
        trace_counter = Counter(encoded_traces)
        total_traces = sum(trace_counter.values())
        percentage_counter = {trace: round((freq / total_traces) * 100, 2)
                              for trace, freq in trace_counter.items()}

        self.__serialize_object(percentage_counter, 'encoded_trace_frequency')
        self.__serialize_object(self.__history_train, 'train')
        self.__serialize_object(self.__history_test, 'test')
        self.__serialize_object(self.__len_prefix_train, 'len_train')
        self.__serialize_object(self.__len_prefix_test, 'len_test')
        self.__serialize_object(dict_suffix_train, 'suffix_train')
        self.__serialize_object(dict_suffix_test, 'suffix_test')
        self.__serialize_object(entire_activities_train, 'activities_train')
        self.__serialize_object(entire_activities_test, 'activities_test')

        self.__serialize_object(self.__dict_label_train[lg.log[self.__log_name]['target']], 'label_train')
        self.__serialize_object(self.__dict_label_test[lg.log[self.__log_name]['target']], 'label_test')

        with open('semantic_data/' + self.__log_name + '/' + self.__log_name + '_id2label_' + self.__setting + '.pkl', 'wb') as f:
            pickle.dump(self.__id2label, f)

        with open('semantic_data/' + self.__log_name + '/' + self.__log_name + '_label2id_' + self.__setting + '.pkl', 'wb') as f:
            pickle.dump(self.__label2id, f)

    def __utility_function(self, list_seq, dict_event_label):
        for l, a, r, t in zip(list_seq, dict_event_label['activity'], dict_event_label['resource'],
                                dict_event_label['timesincecasestart']):
            print(l, 'label-->', a, r, t)
            print('-------------------------')

    def __serialize_object(self, lista, type):
        with open('semantic_data/' + self.__log_name + '/' + self.__log_name + '_' + type + '_' + self.__setting + '.pkl', 'wb') as f:
            pickle.dump(lista, f)

    def get_id2label(self):
        return self.__id2label

    def get_label2id(self):
        return self.__label2id

