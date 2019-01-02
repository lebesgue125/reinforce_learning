from collections import namedtuple
import numpy as np


class HistoryStored(object):

    def __init__(self, object_name, field_names, max_size):
        self.memory = []
        self.mem_size = 0
        self.max_size = max_size
        self.total_record = 0
        self.object_name = object_name
        self.field_names = field_names
        self.store_obj = namedtuple(object_name, field_names)

    def _create_obj(self, data_dict):
        for f_n in self.field_names:
            if f_n not in data_dict:
                data_dict[f_n] = None
        return self.store_obj(**data_dict)

    def add(self, data_dict):
        if self.total_record < self.max_size:
            self.memory.append(self._create_obj(data_dict))
        else:
            self.memory[self.mem_size] = self._create_obj(data_dict)
        self.mem_size = self.mem_size + 1 if self.mem_size + 1 < self.max_size else 0
        self.total_record += 1

    def add_head(self, data_dict):
        self.memory.insert(0, self._create_obj(data_dict))
        self.total_record += 1

    def take_sample(self, batch_size):
        sample_data = dict()
        b_point = min(self.max_size, self.total_record) - batch_size - 1
        if b_point < 0:
            return None
        start_point = np.random.choice(b_point)
        for i in range(batch_size):
            obj = self.memory[start_point + i]
            for f_n in self.field_names:
                if f_n in sample_data:
                    sample_data[f_n].append(getattr(obj, f_n))
                else:
                    sample_data[f_n] = [getattr(obj, f_n)]
        for k in sample_data.keys():
            sample_data[k] = np.array(sample_data[k])
        return sample_data

    def clear(self):
        self.memory = []
        self.mem_size = 0
        self.total_record = 0

    def copy(self, other_stored):
        if isinstance(other_stored, HistoryStored):
            self.memory = other_stored.memory.copy()
            self.mem_size = other_stored.mem_size
            self.total_record = other_stored.total_record
            self.max_size = other_stored.max_size
