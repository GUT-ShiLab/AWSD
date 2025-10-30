from abc import ABC, abstractmethod
import collections
import numpy as np
from copy import deepcopy
import pandas as pd
import pickle
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import OneHotEncoder
import pdb

# samples in sequence (this is used to check to make sure that the subjects are in sequence)
SAMPLES_SEQUENCE = [['biopsy', 0], ['stool', 0], ['stool', 4], ['stool', 12], ['biopsy', 52], ['stool', 52]]
missing_num_samples = {'biopsy_0': 0, 'stool_0': 0, 'stool_4': 0, 'stool_12': 0, 'biopsy_52': 0, 'stool_52': 0}
total_num_samples = {'biopsy_0': 0, 'stool_0': 0, 'stool_4': 0, 'stool_12': 0, 'biopsy_52': 0, 'stool_52': 0}
imputer_dict = {'mean': SimpleImputer(missing_values=np.nan, strategy='mean'),
                'mice': IterativeImputer(random_state=0, missing_values=np.nan)}
index_to_order = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']


class ProcessDataset(ABC):

    @property
    def data_dict(self):
        raise NotImplementedError

    @property
    def dataset_otu_columns(self):
        raise NotImplementedError

    def process_data(self, pad_in_sequence=True, imputer=None, normalize=True, taxonomy_order=None):
        # since GAIN will use pad_in_sequence
        if imputer == 'GAIN':
            assert pad_in_sequence is True

    @abstractmethod
    def process_columns(self, order='phylum'):
        """

        An example of a column would be k__bacteria|p__etc|c__etc|o__etc|f__etc|g__etc|s__etc
        The order separate in this example would be |
        The feature separate in this example is __

        :param order_separate:
        :param feature_separate:
        :return: a dictionary of all the indexes of the category order:
                {'Proteobacteria: [0, 3, 4, 6, 8, ...], 'Tenericutes': [2, 5, 7, ...]}
                The numbers here refers to the column index of the OTU
        """
        ...

    def get_selected_features(self, intersecting_dict):
        """

        When we are doing transfer learning we need to ensure that we are getting the appropriate features columns in
        order. This is to get the selected features in order. The "categorize" function needs to be run before this
        function can work


        :param intersecting_dict: { 'Bacteria': [0, 1, 2, 3], 'Archea': [4, 5, 6, 7] }
        :return:
        """
        intersecting_features = sorted(list(intersecting_dict.keys()))

        # re-create the new otu features in a sorted column order
        new_index_ordering = []
        for intersecting_feature in intersecting_features:
            selected_index = self.categorized_otu_columns.index(intersecting_feature)
            new_index_ordering.append(selected_index)

        self._data_dict['sorted_data'] = self._data_dict['sorted_data'][:, :, new_index_ordering]
        return self._data_dict

    def _categorize(self, output_dict, order='phylum'):
        """
        Categorize the data into a specific order group
        if phylum is specified, then only get the features of phylum (by aggregating all the values together)

        :param pad_in_sequence:
        :param imputer:
        :param order:
        :return:
        """
        sorted_data = output_dict['sorted_data']
        missing_data = output_dict.get('missing_data', None)
        order_dict = self.process_columns(order)

        #  ##### aggregate the orders of the columns value and put into a single column #####  #
        # category is the name of the category of the specific order, (phylum) e.g.
        # current_indexes are the indexes to get the values for that category (Firmicutes) e.g.
        new_columns = []
        new_values = []
        new_missing_values = []
        for current_category, current_indexes in order_dict.items():
            aggregated_value = np.sum(sorted_data[:, :, current_indexes], 2)
            new_columns.append(current_category)
            new_values.append(aggregated_value)

            if missing_data is not None:
                new_missing_values.append(np.sum(missing_data[:, :, current_indexes], 2))

        new_values = np.array(new_values).transpose(1, 2, 0)
        new_missing_values = np.array(new_missing_values).transpose(1, 2, 0)
        new_missing_values[new_missing_values > 1] = 1

        self.categorized_otu_columns = new_columns
        output_dict['sorted_data'] = new_values
        output_dict['missing_data'] = new_missing_values
        return output_dict


class ProcessDatasetMMC7(ProcessDataset):
    def __init__(self, dataset_filename):
        self.dataset_filename = dataset_filename
        self.dataset = pd.read_csv(f"data/{self.dataset_filename}.csv")
        self.gain_data_filename = 'data/imputed_data_mmc7.npy'
        self._data_dict = None  # will not be None after running process_data

    @property
    def data_dict(self):
        return self._data_dict

    @property
    def dataset_otu_columns(self):
        return list(self.dataset.columns)[24:]

    def process_columns(self, order='phylum'):
        mmc_columns = [item.split('..') for item in self.dataset_otu_columns]
        for mmc_columns_index in range(len(mmc_columns)):
            mmc_columns[mmc_columns_index] = \
                [item.split('__')[1] for item in mmc_columns[mmc_columns_index][:-1]]

        order_dict = {}
        order_index = index_to_order.index(order)

        for sample_index, sample_columns in enumerate(mmc_columns):
            if order_dict.get(sample_columns[order_index], None) is None:
                order_dict[sample_columns[order_index]] = [sample_index]
            else:
                order_dict[sample_columns[order_index]].append(sample_index)

        return order_dict

    # def categorize(self, pad_in_sequence=True, imputer=None, order='phylum'):
    #     raise NotImplementedError

    def process_data(self, pad_in_sequence=True, imputer=None, normalize=True, taxonomy_order=None):
        super(ProcessDatasetMMC7, self).process_data()
        

        # pdb.set_trace()
        self.dataset = self.dataset.sort_values(by=['SubjectID', 'collectionWeek', 'sampleType'])

        MAX_TIME_POINTS = 6
        missing_values = None  # this is only used when pad_in_sequence is True

        # remove biopsy for now because they only occur from week 0 and week52
        # df.drop(df[df['sampleType'] == 'biopsy'].index, inplace=True)
        # MAX_TIME_POINTS = 4 # if we remove biopsy

        values = self.dataset.values
        # remove nan values
        nan_indexes = np.argwhere(values[:, 16] != values[:, 16])
        original_values = np.delete(values, [811, 812], axis=0) # remove the whole subject hardcoded first #TODO remove hardcoded value here mmc7.csv

        ### pad the samples with 0
        if pad_in_sequence:
            missing_values = []  # binary values to tell you if this is a missing value or not
            processed_values = []
            index_sample_sequence = 0
            index = 0
            current_subject_id = original_values[0, 1]

            while index < original_values.shape[0]:
                value = original_values[index]

                if value[1] != current_subject_id:
                    while index_sample_sequence != len(SAMPLES_SEQUENCE):
                        # adding missing samples and keep track of the number of them for different weeks
                        missing_key = '_'.join([str(item) for item in SAMPLES_SEQUENCE[index_sample_sequence]])
                        missing_num_samples[missing_key] += 1

                        empty_value = np.zeros(original_values.shape[1])
                        empty_value[:] = np.nan
                        empty_value[1] = current_subject_id
                        processed_values.append(empty_value)
                        missing_values.append(
                            np.ones(
                                original_values.shape[1]))  # add a binary value to show that this is a missing sample
                        index_sample_sequence += 1

                    current_subject_id = value[1]
                    index_sample_sequence = 0

                if value[2] != SAMPLES_SEQUENCE[index_sample_sequence][0] or value[3] != \
                        SAMPLES_SEQUENCE[index_sample_sequence][1]:
                    # adding missing samples and keep track of the number of them for different weeks
                    missing_key = '_'.join([str(item) for item in SAMPLES_SEQUENCE[index_sample_sequence]])
                    missing_num_samples[missing_key] += 1

                    empty_value = np.zeros(original_values.shape[1])
                    empty_value[:] = np.nan
                    empty_value[1] = original_values[index, 1]
                    processed_values.append(empty_value)
                    missing_values.append(
                        np.ones(original_values.shape[1]))  # add a binary value to show that this is a missing sample
                else:
                    # add total key to see how many samples in each week
                    current_key = '_'.join([str(item) for item in SAMPLES_SEQUENCE[index_sample_sequence]])
                    total_num_samples[current_key] += 1

                    processed_values.append(value)
                    missing_values.append(np.zeros(
                        original_values.shape[1]))  # add a binary value to show that this is not a missing sample
                    index += 1

                index_sample_sequence += 1

            original_values = np.array(processed_values)
            missing_values = np.array(missing_values)

        # get the relevant information
        values = original_values[:, 24:].astype(np.float32)
        if missing_values is not None:
            missing_values = missing_values[:, 24:]

        # impute the data if not using GAIN
        if imputer != 'GAIN' and imputer is not None:
            current_imputer = imputer_dict[imputer]
            values = current_imputer.fit_transform(values)
        # if imputation is none just set them to 0
        elif imputer is None:
            values[np.isnan(values)] = 0

        # ### do normalization ### #
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(values)

        lastSubjectId = original_values[0, 1]
        sorted_data = []  # store all the sorted data
        target_data = []
        sorted_length = []
        current_data = [normalized_features[0]]  # store the temp list for the current subject Id
        zero_padding = np.zeros_like(current_data[0])
        allsubjectid = []

        ### make into shape [samples, timepoints_length, features]
        for current_index in range(1, original_values.shape[0]):
            subjectId = original_values[current_index, 1]
            allsubjectid.append(subjectId)

            # if it's a new subjectId then we append a new row
            if subjectId != lastSubjectId:
                lastSubjectId = subjectId

                # get the actual length
                sorted_length.append(len(current_data))

                # pad data to appropriate time steps
                while len(current_data) < MAX_TIME_POINTS:
                    current_data.append(zero_padding)

                # add the current sample to the whole data and the target value
                sorted_data.append(current_data)
                target_data.append(str(
                    original_values[current_index - 1, 16]))  # previous id because this current index is the next id

                current_data = [normalized_features[current_index]]
            else:
                current_data.append(normalized_features[current_index])

        # ## add the last item to the whole data ## #
        sorted_length.append(len(current_data))
        while len(current_data) < MAX_TIME_POINTS:
            current_data.append(zero_padding)
        sorted_data.append(current_data)
        target_data.append(str(original_values[-1, 16]))  # previous id because this current index is the next id

        target_data = np.array(target_data)
        if missing_values is not None:
            missing_data = missing_values.reshape(target_data.shape[0], 6, -1)
        sorted_length = np.array(sorted_length)
        sorted_data = np.stack(sorted_data, 0).astype(np.float32)

        if imputer == 'GAIN':
            sorted_data = np.load(self.gain_data_filename).astype(np.float32)

        # ### delete nan data ### #
        nan_indexes = np.where(target_data == 'nan')[0]
        target_data = np.delete(target_data, nan_indexes, axis=0)
        sorted_data = np.delete(sorted_data, nan_indexes, axis=0)

        # convert target data to one hot encoding
        target_data = np.expand_dims(target_data, 1)
        self.one_hot_encoder = OneHotEncoder()
        target_data = self.one_hot_encoder.fit_transform(target_data).toarray().astype(np.float32)
        # pdb.set_trace()
        self._data_dict = {'sorted_data': sorted_data, 'sorted_length': sorted_length,
                       'target_data': target_data}
        if missing_values is not None:
            self._data_dict['missing_data'] = missing_data

        if taxonomy_order is not None:
            self._data_dict = self._categorize(self._data_dict, taxonomy_order)

        return self._data_dict


class ProcessDatasetAllergy(ProcessDataset):
    def __init__(self, dataset_filename):
        self.dataset_filename = dataset_filename
        self.dataset_otu = pd.read_csv(f'./data/DIABIMMUNE_data_16s.csv')
        self.dataset_metadata = pd.read_excel('./data/DIABIMMUNE_metadata.xlsx', engine="openpyxl")
        self.gain_data_filename = './data/imputed_data_allergy.npy'
        self.dataset = pd.merge(self.dataset_metadata, self.dataset_otu, on=['SampleID'])
        self._data_dict = None  # will not be None after running process_data

    @property
    def dataset_otu_columns(self):
        return list(self.dataset.columns)[18:]

    def process_columns(self, order='phylum'):
        """
            An example of a column would be k__bacteria|p__etc|c__etc|o__etc|f__etc|g__etc|s__etc
            The order separate in this example would be |
            The feature separate in this example is __

            :param order_separate:
            :param feature_separate:
            :return:
        """
        # get all the names of that particular order
        samples_columns = [item.split('|') for item in self.dataset_otu_columns]
        for samples_columns_index in range(len(samples_columns)):
            samples_columns[samples_columns_index] = \
                [item.split('__')[1] for item in samples_columns[samples_columns_index]]

        idx = index_to_order.index(order)
        # this total_length is to ensure that we are only getting the correct category
        total_length = idx + 1

        order_dict = {}  # the order index of the column of the dataset

        # get all the orders indexes into a dictionary,
        # so later we can select the specific column index and aggregate those orders
        for sample_index, sample_columns in enumerate(samples_columns):
            # if not the correct category we go to next one
            if len(sample_columns) != total_length:
                continue

            if order_dict.get(sample_columns[idx], None) is None:
                order_dict[sample_columns[idx]] = [sample_index]
            else:
                order_dict[sample_columns[idx]].append(sample_index)

        return order_dict

    def process_data(self, pad_in_sequence=True, imputer=None, normalize=True, taxonomy_order=None):
        super(ProcessDatasetAllergy, self).process_data()
        

        # import pdb
        # pdb.set_trace()
        self.dataset = self.dataset.sort_values(by=['subjectID', 'collection_month'])

        all_unique_timepoints = np.unique(self.dataset['collection_month'].values)
        max_timepoints = len(all_unique_timepoints)
        # test = [pd.DataFrame(y).values for x, y in self.dataset.groupby('subjectID', as_index=False)]
        # test.index.__name__
        # pdb.set_trace()
        all_samples = [pd.DataFrame(y).values for x, y in self.dataset.groupby('subjectID', as_index=False)]
        labels = np.array([sample[:, 8:11][-1] for sample in all_samples]).astype(np.float32)
        # pdb.set_trace()
        nan_samples = np.where(np.isnan(labels)[:, 0] == True)
        for index in sorted(nan_samples[0], reverse=True):
            del all_samples[index]

        # all_samples = np.delete(all_samples, nan_samples, axis=0).tolist()
        labels = np.delete(labels, nan_samples, axis=0)
        lengths = []

        # remove duplicated timepoints
        for sample_index in range(len(all_samples)):
            current_timepoints = all_samples[sample_index][:, 3]
            duplicated_timepoints = [item for item, count in collections.Counter(current_timepoints).items() if count > 1]
            for duplicated_timepoint in duplicated_timepoints:
                indexes = np.where(all_samples[sample_index][:, 3] == duplicated_timepoint)[0][1:]
                all_samples[sample_index] = np.delete(all_samples[sample_index], indexes, axis=0)

        if pad_in_sequence is True:
            for sample_index, current_sample in enumerate(all_samples):
                list_to_add = []
                for idx, current_time in enumerate(all_unique_timepoints):
                    if current_time not in current_sample[:, 3]:
                        list_to_add.append(idx)
                for item in list_to_add:
                    # pdb.set_trace()
                    all_samples[sample_index] = np.insert(all_samples[sample_index], item, np.nan, axis=0)
            lengths = np.repeat(max_timepoints, len(all_samples))
        else:
            for sample_index in range(len(all_samples)):
                length = all_samples[sample_index].shape[0]
                all_samples[sample_index] = np.pad(all_samples[sample_index],
                                               [(0, max_timepoints - all_samples[sample_index].shape[0]), (0, 0)],
                                                    constant_values=np.nan)
                lengths.append(length)
            lengths = np.array(lengths)

        all_samples = np.array(all_samples)[:, :, 18:].astype(np.float32)  # TODO make envionment variables too
        missing_data = np.isnan(all_samples).astype(np.float32)

        # impute the data if not using GAIN or there is no imputation setup
        if imputer != 'GAIN' and imputer is not None:
            current_imputer = imputer_dict[imputer]
            original_shape = all_samples.shape
            all_samples = all_samples.reshape(-1, original_shape[-1])
            all_samples = current_imputer.fit_transform(all_samples)
            all_samples = all_samples.reshape(*original_shape)
        # if imputation is none just set them to 0
        elif imputer is None:
            all_samples[np.isnan(all_samples)] = 0

        # ### do normalization ### #
        # if GAIN we don't have to normalize because GAIN already normalizes
        if imputer == 'GAIN':
            all_samples = np.load(self.gain_data_filename).astype(np.float32)
        else:
            # normalize
            if normalize:
                scaler = StandardScaler()
                original_shape = all_samples.shape
                all_samples = all_samples.reshape(-1, original_shape[-1])
                all_samples = scaler.fit_transform(all_samples)
                all_samples = all_samples.reshape(*original_shape)

        self._data_dict = {'sorted_data': all_samples,
                       'sorted_length': lengths, 'target_data': labels, 'missing_data': missing_data}

        if taxonomy_order is not None:
            self._data_dict = self._categorize(self._data_dict, taxonomy_order)

        return self._data_dict
    

class ProcessDatasetMeta(ProcessDataset):

    def __init__(self, dataset_filename):
        with open(f'data/{dataset_filename}.data', 'rb') as f:
            dataset = pickle.load(f)
        self.timepoints = dataset['T']
        self.all_unique_timepoints = np.unique([v for timepoint in self.timepoints for v in timepoint])
        self.max_timepoints = len(self.all_unique_timepoints)
        self.original_x = dataset['X']
        self.labels = dataset['y']

    @property
    def dataset_otu_columns(self):
        return None
    
    def process_columns(self):
        pass

    def process_data(self, pad_in_sequence=True, imputer=None, normalize=True, taxonomy_order=None):
        lengths = []
        self.samples = deepcopy(self.original_x)

        if pad_in_sequence is True:
            for sample_index, timepoint in enumerate(self.timepoints):
                list_to_add = []
                # pdb.set_trace()
                for idx, current_time in enumerate(self.all_unique_timepoints):
                    if current_time not in timepoint:
                        list_to_add.append(idx)
                for item in list_to_add:
                    self.samples[sample_index] = np.insert(self.samples[sample_index], item, np.nan, axis=1)
            lengths = np.repeat(self.max_timepoints, len(self.samples))
        else:
            for sample_index in range(len(self.samples)):
                length = self.samples[sample_index].shape[1]
                self.samples[sample_index] = np.pad(self.samples[sample_index],
                                               [(0, 0), (0,self.max_timepoints - self.samples[sample_index].shape[1])],
                                                    constant_values=np.nan)
                lengths.append(length)
            lengths = np.array(lengths)
        samples = np.array(self.samples).transpose(0, 2, 1).astype(np.float32)
        labels = np.array(self.labels).astype(np.float32)
        samples, labels, lengths = shuffle(samples, labels, lengths, random_state=0)

        # ### normalize ### #
        scaler = StandardScaler()
        original_shape = samples.shape
        samples = samples.reshape(-1, original_shape[-1])
        samples = scaler.fit_transform(samples)
        samples.reshape(*original_shape)

        missing_data = np.isnan(samples).astype(np.float32)

        # impute the data if not using GAIN
        if imputer != 'GAIN' and imputer is not None:
            current_imputer = imputer_dict[imputer]
            original_shape = samples.shape
            samples = samples.reshape(-1, original_shape[-1])
            samples = current_imputer.fit_transform(samples)
            samples = samples.reshape(*original_shape)
        # if imputation is none just set them to 0
        if imputer is None:
            samples[np.isnan(samples)] = 0

        return_dict = {'sorted_data': samples,
                       'sorted_length': lengths, 'target_data': np.expand_dims(labels, 1), 'missing_data': missing_data}
        return return_dict


import re

class ProcessDatasetInfant(ProcessDataset):
    def __init__(self, dataset_filename):
        self.dataset_filename = dataset_filename
        self.dataset_otu = pd.read_csv(f'./data/infant/infant_abundance.csv')
        self.dataset_metadata = pd.read_csv(f'./data/infant/infant_metadata.csv')
        self.init_data()

    @property
    def dataset_otu_columns(self):
        return self.dataset_otu.iloc[0, :]

    def init_data(self):
        # 清理掉不需要的列，比如母亲样本
        target_sample_ids = self.dataset_metadata[
            self.dataset_metadata['Env'].str.contains('-M\(C\)|-M\(V\)')
        ]['SampleID'].tolist()
        columns_to_remove = [col for col in self.dataset_otu.columns if col in target_sample_ids]
        self.dataset_otu = self.dataset_otu.drop(columns=columns_to_remove)
        self.dataset_metadata = self.dataset_metadata[
            ~self.dataset_metadata["SampleID"].isin(columns_to_remove)
        ]
        self.dataset_otu = self.dataset_otu.T

    def parse_env(self, env_str):
        """
        把 '1011-5Y(C)' 或 '1014-NB(C)' 解析成 (individual_id, month_age, label)
        """
        # 先匹配正常样本：1011-5Y(C)
        match = re.match(r'(\d+)-(\d+)([MY])\((C|V)\)', env_str)
        if match:
            indiv_id = match.group(1)
            number = int(match.group(2))
            unit = match.group(3)
            label = match.group(4)

            if unit == 'Y':
                month_age = number * 12
            elif unit == 'M':
                month_age = number
            else:
                raise ValueError(f"Unknown unit: {unit}")

            return indiv_id, month_age, label

        # 再匹配新生儿样本：1014-NB(C)
        match = re.match(r'(\d+)-NB\((C|V)\)', env_str)
        if match:
            indiv_id = match.group(1)
            label = match.group(2)
            month_age = 0  # 新生儿默认 0个月
            return indiv_id, month_age, label

        # 都没匹配上就报错
        raise ValueError(f"Env format not recognized: {env_str}")

    def process_columns(self, order='phylum'):
        samples_columns = [item.split(';') for item in self.dataset_otu_columns]
        taxonomy_levels = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']

        if order not in taxonomy_levels:
            raise ValueError(f"Order {order} not recognized. Must be one of {taxonomy_levels}")

        order_idx = taxonomy_levels.index(order)

        processed_columns = []
        for taxa in samples_columns:
            if len(taxa) > order_idx:
                processed_columns.append(taxa[order_idx])
            else:
                processed_columns.append('Unknown')

        self.processed_columns = processed_columns
        return processed_columns

    def process_data(self, pad_in_sequence=True, imputer=None, normalize=True, taxonomy_order=None):
        if taxonomy_order is not None:
            processed_columns = self.process_columns(order=taxonomy_order)
            self.dataset_otu.columns = processed_columns
            self.dataset_otu = self.dataset_otu.groupby(self.dataset_otu.columns, axis=1).sum()

        self.original_x = []
        self.labels = []
        self.timepoints = []
        
        # 解析 Env 字段，准备 mapping
        sample_id_to_env = {}
        for idx, row in self.dataset_metadata.iterrows():
            indiv_id, month_age, label = self.parse_env(row['Env'])
            sample_id_to_env[row['SampleID']] = (indiv_id, month_age, label)
        
        # group by individual
        indiv_to_samples = {}
        for sample_id, (indiv_id, month_age, label) in sample_id_to_env.items():
            if indiv_id not in indiv_to_samples:
                indiv_to_samples[indiv_id] = []
            indiv_to_samples[indiv_id].append((sample_id, month_age, label))
        
        all_unique_timepoints = sorted(
            list({month_age for samples in indiv_to_samples.values() for (_, month_age, _) in samples})
        )
        max_timepoints = len(all_unique_timepoints)

        sample_id_to_index = {sid: idx for idx, sid in enumerate(self.dataset_otu.index)}

        all_samples = []
        labels = []
        for indiv_id, samples in indiv_to_samples.items():
            samples = sorted(samples, key=lambda x: x[1])  # 按月份排序
            otu_data = []
            sample_timepoints = []
            for sample_id, month_age, label in samples:
                if sample_id not in sample_id_to_index:
                    continue
                sample_idx = sample_id_to_index[sample_id]
                otu_vector = self.dataset_otu.iloc[sample_idx].values
                otu_data.append((month_age, otu_vector))
            
            if len(otu_data) == 0:
                continue
            otu_data = sorted(otu_data, key=lambda x: x[0])  # 按月龄排序一次（保险）
            months, vectors = zip(*otu_data)
            vectors = np.stack(vectors, axis=0)  # (time, feature)
            all_samples.append(vectors)
            
            # 默认把 (V) 标为 1，(C) 标为 0
            first_label = samples[0][2]
            labels.append(1 if first_label == 'V' else 0)

        labels = np.array(labels).astype(np.float32)
        lengths = []
        # pdb.set_trace()
        # 填补时间点
        # all_most_samples = []
        # all_labels = []
        if pad_in_sequence:
            padded_samples = []
            for sample_index, sample in enumerate(all_samples):
                current_timepoints = [month_age[1] for month_age in indiv_to_samples[list(indiv_to_samples.keys())[sample_index]]]
                list_to_add = []
                print(current_timepoints)
                # pdb.set_trace()
                for idx, current_time in enumerate(all_unique_timepoints):
                    if current_time not in current_timepoints:
                        list_to_add.append(idx)
                for item in list_to_add:
                    # pdb.set_trace()
                    all_samples[sample_index] = np.insert(all_samples[sample_index], item, np.nan, axis=0)
            #     if len(list_to_add) == 0:
            #         all_most_samples.append(all_samples[sample_index])
            #         all_labels.append(labels[sample_index])
            # all_samples = all_most_samples
            # labels = np.array(all_labels).astype(np.float32)
            # pdb.set_trace()
            lengths = np.repeat(max_timepoints, len(all_samples))
        else:
            padded_samples = []
            # pdb.set_trace()
            for sample in all_samples:
                length = sample.shape[0]
                sample = np.pad(sample, [(0, max_timepoints - sample.shape[0]), (0, 0)], constant_values=np.nan)
                padded_samples.append(sample)
                lengths.append(length)
            all_samples = padded_samples
            lengths = np.array(lengths)

        all_samples = np.array(all_samples).astype(np.float32)  # shape: (n_samples, max_timepoints, n_features)
        missing_data = np.isnan(all_samples).astype(np.float32)

        # impute
        if imputer != 'GAIN' and imputer is not None:
            current_imputer = imputer_dict[imputer]
            original_shape = all_samples.shape
            all_samples = all_samples.reshape(-1, original_shape[-1])
            all_samples = current_imputer.fit_transform(all_samples)
            all_samples = all_samples.reshape(*original_shape)
        elif imputer is None:
            all_samples[np.isnan(all_samples)] = 0

        # normalize
        if imputer == 'GAIN':
            all_samples = np.load(self.gain_data_filename).astype(np.float32)
        else:
            if normalize:
                scaler = StandardScaler()
                original_shape = all_samples.shape
                all_samples = all_samples.reshape(-1, original_shape[-1])
                all_samples = scaler.fit_transform(all_samples)
                all_samples = all_samples.reshape(*original_shape)

        # 保存成 self._data_dict
        self._data_dict = {
            'sorted_data': all_samples,
            'sorted_length': lengths,
            'target_data': np.expand_dims(labels, 1),
            'missing_data': missing_data
        }

        # pdb.set_trace()

        if taxonomy_order is not None:
            self._data_dict = self._categorize(self._data_dict, taxonomy_order)

        return self._data_dict

# ProcessDatasetAllergy('allergy').process_data()
# ProcessDatasetInfant('infant').process_data()

def find_intersecting_bacteria(dataset1: ProcessDataset, dataset2: ProcessDataset, order='phylum'):
    """

    :param dataset1:    first dataset
    :param dataset2:    second dataset
    :param order:       taxonomy order to categorize
    :return:            return the keys and indexes of the column that is intersecting for dataset1 and dataset2
    """

    dataset1_columns = dataset1.process_columns(order)
    dataset2_columns = dataset2.process_columns(order)

    # find the non intersecting category so we can remove them from the dictionary
    non_intersecting_columns = set(dataset1_columns.keys()) ^ set(dataset2_columns.keys())

    for non_intersecting_column in non_intersecting_columns:
        if dataset1_columns.get(non_intersecting_column, None) is not None: del dataset1_columns[non_intersecting_column]
        if dataset2_columns.get(non_intersecting_column, None) is not None: del dataset2_columns[non_intersecting_column]

    return dataset1_columns, dataset2_columns


dataset_list = {'mmc7': ProcessDatasetMMC7, 'david': ProcessDatasetMeta, 'digiulio': ProcessDatasetMeta,
                't1d': ProcessDatasetMeta, 'allergy': ProcessDatasetAllergy,'infant': ProcessDatasetInfant}
#'allergy'  ##DIABIMMUNE
#'mmc7'  ##PROTECT

# dataset1 = ProcessDatasetMMC7('mmc7')
# dataset2 = ProcessDatasetAllergy('allergy')
# dataset2.process_data()
# pdb.set_trace()
# print(dataset2)
# dataset1_intersecting_columns, dataset2_intersecting_columns = find_intersecting_bacteria(dataset1, dataset2)
# output_dict = dataset1.process_data(True, None, True, 'phylum')
# dataset1.get_selected_features(dataset1_intersecting_columns)

