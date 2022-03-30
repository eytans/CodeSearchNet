from docopt import docopt
import hashlib
import pandas as pd
from dpu_utils.utils import RichPath, run_and_debug
from dpu_utils.codeutils.deduplication import DuplicateDetector
import os
from tqdm import tqdm
from dpu_utils.utils import RichPath
from multiprocessing import Pool, cpu_count
from typing import List, Any
import pickle

def save_file_pickle(fname: str, obj: Any) -> None:
    with open(fname, 'wb') as f:
        pickle.dump(obj, f)


def load_file_pickle(fname: str) -> None:
    with open(fname, 'rb') as f:
        obj = pickle.load(f)
        return obj


def chunkify(df: pd.DataFrame, n: int) -> List[pd.DataFrame]:
    "turn pandas.dataframe into equal size n chunks."
    return [df[i::n] for i in range(n)]


def df_to_jsonl(df: pd.DataFrame, RichPath_obj: RichPath, i: int, basefilename='codedata') -> str:
    dest_filename = f'{basefilename}_{str(i).zfill(5)}.jsonl.gz'
    RichPath_obj.join(dest_filename).save_as_compressed_file(df.to_dict(orient='records'))
    return str(RichPath_obj.join(dest_filename))


def chunked_save_df_to_jsonl(df: pd.DataFrame,
                             output_folder: RichPath,
                             num_chunks: int=None,
                             parallel: bool=True) -> None:
    "Chunk DataFrame (n chunks = num cores) and save as jsonl files."

    df.reset_index(drop=True, inplace=True)
    # parallel saving to jsonl files on azure
    n = cpu_count() if num_chunks is None else num_chunks
    dfs = chunkify(df, n)
    args = zip(dfs, [output_folder]*len(dfs), range(len(dfs)))

    if not parallel:
        for arg in args:
            dest_filename = df_to_jsonl(*arg)
            print(f'Wrote chunk to {dest_filename}')
    else:
        with Pool(cpu_count()) as pool:
            pool.starmap(df_to_jsonl, args)

def label_folds(df: pd.DataFrame, train_ratio: float, valid_ratio: float, test_ratio: float, holdout_ratio: float) -> pd.DataFrame:
    "Adds a partition column to DataFrame with values: {train, valid, test, holdout}."
    assert abs(train_ratio + valid_ratio + test_ratio + holdout_ratio - 1) < 1e-5,  'Ratios must sum up to 1.'
    # code in the same file will always go to the same split
    df['hash_key'] = df.apply(lambda x: f'{x.repo}:{x.path}', axis=1)
    df['hash_val'] = df['hash_key'].apply(lambda x: int(hashlib.md5(x.encode()).hexdigest(), 16) % (2**16))

    train_bound = int(2**16 * train_ratio)
    valid_bound = train_bound + int(2**16 * valid_ratio)
    test_bound = valid_bound + int(2**16 * test_ratio)

    def label_splits(hash_val: int) -> str:
        if hash_val <= train_bound:
            return "train"
        elif hash_val <= valid_bound:
            return "valid"
        elif hash_val <= test_bound:
            return "test"
        else:
            return "holdout"

    # apply partition logic
    df['partition'] = df['hash_val'].apply(lambda x: label_splits(x))
    # display summary statistics
    counts = df.groupby('partition')['repo'].count().rename('count')
    summary_df = pd.concat([counts, (counts / counts.sum()).rename('pct')], axis=1)
    print(summary_df)

    return df

def jsonl_to_df(input_folder: RichPath) -> pd.DataFrame:
    "Concatenates all jsonl files from path and returns them as a single pandas.DataFrame ."

    assert input_folder.is_dir(), 'Argument supplied must be a directory'
    dfs = []
    files = list(input_folder.iterate_filtered_files_in_dir('*.jsonl.gz'))
    assert files, 'There were no jsonl.gz files in the specified directory.'
    print(f'reading files from {input_folder.path}')
    for f in tqdm(files, total=len(files)):
        dfs.append(pd.DataFrame(list(f.read_as_jsonl(error_handling=lambda m,e: print(f'Error while loading {m} : {e}')))))
    return pd.concat(dfs)

def remove_duplicate_code_df(df: pd.DataFrame) -> pd.DataFrame:
    "Resolve near duplicates based upon code_tokens field in data."
    assert 'code_tokens' in df.columns.values, 'Data must contain field code_tokens'
    assert 'language' in df.columns.values, 'Data must contain field language'
    df.reset_index(inplace=True, drop=True)
    df['doc_id'] = df.index.values
    dd = DuplicateDetector(min_num_tokens_per_document=10)
    filter_mask = df.apply(lambda x: dd.add_file(id=x.doc_id,
                                                 tokens=x.code_tokens,
                                                 language=x.language),
                           axis=1)
    # compute fuzzy duplicates
    exclusion_set = dd.compute_ids_to_exclude()
    # compute pandas.series of type boolean which flags whether or not code should be discarded
    # in order to resolve duplicates (discards all but one in each set of duplicate functions)
    exclusion_mask = df['doc_id'].apply(lambda x: x not in exclusion_set)

    # filter the data
    print(f'Removed {sum(~(filter_mask & exclusion_mask)):,} fuzzy duplicates out of {df.shape[0]:,} rows.')
    return df[filter_mask & exclusion_mask]

def run(args, df, appendix=""):

    
    train = float(args['--train-ratio'])
    valid = float(args['--valid-ratio'])
    test = float(args['--test-ratio'])
    holdout = float(args['--holdout-ratio'])
    output_folder = args['--output-dir']

    # get data and process it
    # print('Removing fuzzy duplicates ... this may take some time.')
    # df = remove_duplicate_code_df(df)
    df = df.sample(frac=1, random_state=20181026)  # shuffle order of files
    df = label_folds(df, train_ratio=train, valid_ratio=valid, test_ratio=test, holdout_ratio=holdout)
    splits = ['train', 'valid', 'test', 'holdout']

    for split in splits:
        split_df = df[df.partition == split]

        # save dataframes as chunked jsonl files
        json_file = output_folder + '/' + split + appendix + '.jsonl'
        split_df.to_json(json_file, orient='records', lines=True)
        

        
