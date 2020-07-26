import config

from bella import parsers
from bella import write_data
import bella
import target_extraction
from target_extraction.dataset_parsers import semeval_2014
from sklearn.model_selection import train_test_split
import numpy as np

config.small_training_dataset_dir.mkdir(parents=True, exist_ok=True)
config.neural_small_dataset_dir.mkdir(parents=True, exist_ok=True)
dataset_names = ['Dong', 'Election', 'Laptop', 'Restaurant', 'Mitchell']
dataset_fp_mapper = {'Dong train': config.DONG_TRAIN,
                     'Laptop train': config.laptop_train,
                     'Restaurant train': config.restaurant_train,
                     'Mitchell train': config.mitchell_train}

def get_targets_from_spans(dataset: bella.data_types.TargetCollection) -> bella.data_types.TargetCollection:
    '''
    This is required as some of the datasets targets are correct but lower cased 
    from the original. This is no problem for methods that lower case their 
    words but is for methods that require capitalisation.
    '''
    new_dataset = []
    for value in dataset.data_dict():
        target_spans = value['spans']
        new_target = None
        text = value['text']
        assert len(target_spans) == 1
        for span in target_spans:
            new_target = text[span[0] : span[1]]
        assert new_target is not None
        value['target'] = new_target
        new_dataset.append(bella.data_types.Target(**value))
    return bella.data_types.TargetCollection(new_dataset)

size_of_small = len(parsers.semeval_14(config.youtubean_train))
for dataset_name in dataset_names:
    dataset_fp = config.small_training_dataset_dir / f'{dataset_name} train.xml'
    neural_train_dataset_fp = config.neural_small_dataset_dir / f'{dataset_name} train.json'
    neural_val_dataset_fp = config.neural_small_dataset_dir / f'{dataset_name} validation.json'
    if dataset_fp.exists():
        continue
    
    dataset = None
    if dataset_name == 'Election':
        dataset = parsers.election_train(config.ELECTION, name='Election Train')
        dataset = get_targets_from_spans(dataset)
    elif dataset_name == 'Dong':
        dataset = parsers.dong(dataset_fp_mapper[f'{dataset_name} train'])
        new_dong_dataset = []
        for value in dataset.data_dict():
            target_spans = value['spans']
            if len(target_spans) > 1:
                target_spans = [target_spans[0]]
            value['spans'] = target_spans
            value['target'] = value['text'][target_spans[0][0]: target_spans[0][1]] 
            new_dong_dataset.append(bella.data_types.Target(**value))
        dataset = bella.data_types.TargetCollection(new_dong_dataset)
    else:
        dataset = parsers.semeval_14(dataset_fp_mapper[f'{dataset_name} train'])
    assert dataset is not None
    dataset_size = len(dataset)
    test_split_size = size_of_small / dataset_size
    _, small_dataset = bella.data_types.TargetCollection.split_dataset(dataset, test_split_size, 
                                                                       random=False)
    assert len(small_dataset) == size_of_small
    write_data.semeval_14(dataset_fp, small_dataset)
    print(f'Number of targets in {dataset_name} non-neural training dataset {len(small_dataset)}')


    neural_small_dataset = semeval_2014(dataset_fp, conflict=False)
    # Just making sure each sentence contains at least one target.
    assert neural_small_dataset.one_sample_per_span(remove_empty=True).number_targets() == neural_small_dataset.number_targets()
    assert len(neural_small_dataset) == len(neural_small_dataset.samples_with_targets())
    print(f'Number of targets with new format {neural_small_dataset.number_targets()}')
    # For reproducibility reasons
    random_state = 42
    # Validation size is 20% of the training set size based on sentences not
    # targets
    train_size = len(neural_small_dataset)
    val_size = int(train_size * 0.2)
    train_dataset = list(neural_small_dataset.values())
    train, val = train_test_split(train_dataset, test_size=val_size, 
                                  random_state=random_state)
    train_dataset = target_extraction.data_types.TargetTextCollection(train)
    val_dataset = target_extraction.data_types.TargetTextCollection(val)
    print(f'size of neural train {len(train_dataset)} Number targets: {train_dataset.number_targets()}')
    print(f'size of neural validation {len(val_dataset)} Number targets: {val_dataset.number_targets()}')
    train_dataset.to_json_file(neural_train_dataset_fp)
    val_dataset.to_json_file(neural_val_dataset_fp)
