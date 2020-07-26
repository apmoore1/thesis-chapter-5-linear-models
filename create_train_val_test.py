import config

from bella import parsers
from bella import write_data
import bella
import target_extraction
from target_extraction.dataset_parsers import semeval_2014
from sklearn.model_selection import train_test_split

config.neural_dataset_dir.mkdir(parents=True, exist_ok=True)
dataset_names = ['Dong', 'Election', 'Laptop', 'Restaurant', 'Mitchell', 'YouTuBean']
split_names = ['train', 'test']
dataset_fp_mapper = {'Dong train': config.DONG_TRAIN, 'Dong test': config.DONG_TEST,
                     'Laptop train': config.laptop_train, 'Laptop test': config.laptop_test,
                     'Restaurant train': config.restaurant_train, 'Restaurant test': config.restaurant_test,
                     'Mitchell train': config.mitchell_train, 'Mitchell test': config.mitchell_test,
                     'YouTuBean train': config.youtubean_train, 'YouTuBean test': config.youtubean_test}

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

for dataset_name in dataset_names:
    for split_name in split_names:
        dataset_fp = config.neural_dataset_dir / f'{dataset_name} {split_name}.json'
        if dataset_fp.exists():
            continue
        dataset = None
        if dataset_name == 'Election':
            if split_name == 'train':
                election_train = parsers.election_train(config.ELECTION, name='Election Train')
                print(f'Number of targets before {len(election_train)}')
                election_train = get_targets_from_spans(election_train)
                temp_fp = config.neural_dataset_dir / 'Temp Election train.xml'
                write_data.semeval_14(temp_fp, election_train)
                dataset = semeval_2014(temp_fp, conflict=False)
            else:
                election_test = parsers.election_test(config.ELECTION, name='Election Train')
                print(f'Number of targets before {len(election_test)}')
                election_test = get_targets_from_spans(election_test)
                temp_fp = config.neural_dataset_dir / 'Temp Election test.xml'
                write_data.semeval_14(temp_fp, election_test)
                dataset = semeval_2014(temp_fp, conflict=False)
        elif dataset_name == 'Dong':
            dong_dataset = parsers.dong(dataset_fp_mapper[f'{dataset_name} {split_name}'])
            print(f'Number of targets before {len(dong_dataset)}')
            new_dong_dataset = []
            for value in dong_dataset.data_dict():
                target_spans = value['spans']
                if len(target_spans) > 1:
                    target_spans = [target_spans[0]]
                value['spans'] = target_spans
                value['target'] = value['text'][target_spans[0][0]: target_spans[0][1]] 
                new_dong_dataset.append(bella.data_types.Target(**value))
            dong_dataset = bella.data_types.TargetCollection(new_dong_dataset)
            temp_fp = config.neural_dataset_dir / f'Temp Dong {split_name}.xml'
            write_data.semeval_14(temp_fp, dong_dataset)
            dataset = semeval_2014(temp_fp, conflict=False)
        else:
            another_dataset = parsers.semeval_14(dataset_fp_mapper[f'{dataset_name} {split_name}'])
            print(f'Number of targets before {len(another_dataset)}')
            temp_fp = config.neural_dataset_dir / f'Temp {dataset_name} {split_name}.xml'
            write_data.semeval_14(temp_fp, another_dataset)
            dataset = semeval_2014(temp_fp, conflict=False)
        assert dataset is not None
        # Just making sure each sentence contains at least one target.
        assert dataset.one_sample_per_span(remove_empty=True).number_targets() == dataset.number_targets()
        assert len(dataset) == len(dataset.samples_with_targets())
        print(f'Number of targets with new format {dataset.number_targets()}')
        if split_name == 'test':
            dataset.to_json_file(dataset_fp)
            continue
        # For reproducibility reasons
        random_state = 42
        # Validation size is 20% of the training set size based on sentences not
        # targets
        train_size = len(dataset)
        val_size = int(train_size * 0.2)
        train_dataset = list(dataset.values())
        train, val = train_test_split(train_dataset, test_size=val_size, 
                                      random_state=random_state)
        train_dataset = target_extraction.data_types.TargetTextCollection(train)
        val_dataset = target_extraction.data_types.TargetTextCollection(val)
        train_dataset.to_json_file(config.neural_dataset_dir / f'{dataset_name} train.json')
        val_dataset.to_json_file(config.neural_dataset_dir / f'{dataset_name} validation.json')