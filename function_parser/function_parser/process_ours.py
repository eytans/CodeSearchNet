"""
Usage:
    process_ours.py [options] INPUT_DIR OUTPUT_DIR

Options:
    -h --help
    --language LANGUAGE             Language
    --processes PROCESSES           # of processes to use [default: 16]
    --license-filter FILE           License metadata to filter, every row contains [nwo, license, language, score] (e.g. ['pandas-dev/pandas', 'bsd-3-clause', 'Python', 0.9997])
    --tree-sitter-build FILE        [default: /src/build/py-tree-sitter-languages.so]
"""
import functools
from multiprocessing import Pool
import pickle
from os import PathLike
from posixpath import split
from typing import Optional, Tuple, Type, List, Dict, Any

from docopt import docopt
from dpu_utils.codeutils.deduplication import DuplicateDetector
import pandas as pd
from tree_sitter import Language, Parser

from language_data import LANGUAGE_METADATA
from parsers.language_parser import LanguageParser, tokenize_docstring
from utils import download, get_sha, flatten, remap_nwo, walk
from split_utils import chunkify

class DataProcessor:

    PARSER = Parser()

    def __init__(self, language: str, language_parser: Type[LanguageParser]):
        self.language = language
        self.language_parser = language_parser

    def process_dee(self, nwo, ext) -> List[Dict[str, Any]]:
        # Process dependees (libraries) to get function implementations
        # print("In proccess_dee...")
        indexes = []
        _, nwo = remap_nwo(nwo)
        if nwo is None:
            return indexes

        tmp_dir = download(nwo)
        files = walk(tmp_dir, ext)
        # files = glob.iglob(tmp_dir.name + '/**/*.{}'.format(ext), recursive=True)
        sha = None

        for f in files:
            # print(f"working on file {f}")
            definitions = self.get_function_definitions(f)
            if definitions is None:
                continue
            if sha is None:
                sha = get_sha(tmp_dir, nwo)

            nwo, path, functions = definitions
            indexes.extend((self.extract_function_data(func, nwo, path, sha) for func in functions if len(func['function_tokens']) > 1))
        return indexes

    def process_dent(self, nwo, ext, library_candidates) -> Tuple[List[Dict[str, Any]], List[Tuple[str, str]]]:
        # Process dependents (applications) to get function calls
        dents = []
        edges = []
        _, nwo = remap_nwo(nwo)
        if nwo is None:
            return dents, edges

        tmp_dir = download(nwo)
        files = walk(tmp_dir, ext)
        sha = None

        for f in files:
            context_and_calls = self.get_context_and_function_calls(f)
            if context_and_calls is None:
                continue
            if sha is None:
                sha = get_sha(tmp_dir, nwo)

            nwo, path, context, calls = context_and_calls
            libraries = []
            for cxt in context:
                if type(cxt) == dict:
                    libraries.extend([v.split('.')[0] for v in cxt.values()])
                elif type(cxt) == list:
                    libraries.extend(cxt)

            match_scopes = {}
            for cxt in set(libraries):
                if cxt in library_candidates:
                    match_scopes[cxt] = library_candidates[cxt]

            for call in calls:
                for depended_library_name, dependend_library_functions in match_scopes.items():
                    for depended_library_function in dependend_library_functions:
                        # Other potential filters: len(call['identifier']) > 6 or len(call['identifier'].split('_')) > 1
                        if (call['identifier'] not in self.language_parser.STOPWORDS and
                            ((depended_library_function['identifier'].split('.')[-1] == '__init__' and
                              call['identifier'] == depended_library_function['identifier'].split('.')[0]) or
                             ((len(call['identifier']) > 9 or
                               (not call['identifier'].startswith('_') and len(call['identifier'].split('_')) > 1)) and
                              call['identifier'] == depended_library_function['identifier'])
                            )):
                            dent = {
                                'nwo': nwo,
                                'sha': sha,
                                'path': path,
                                'language': self.language,
                                'identifier': call['identifier'],
                                'argument_list': call['argument_list'],
                                'url': 'https://github.com/{}/blob/{}/{}#L{}-L{}'.format(nwo, sha, path,
                                                                                         call['start_point'][0] + 1,
                                                                                         call['end_point'][0] + 1)
                            }
                            dents.append(dent)
                            edges.append((dent['url'], depended_library_function['url']))
        return dents, edges

    def process_single_file(self, filepath: PathLike) -> List[Dict[str, Any]]:
        definitions = self.get_function_definitions(filepath)
        if definitions is None:
            return []
        _, _, functions = definitions

        return [self.extract_function_data(func, '', '', '') for func in functions if len(func['function_tokens']) > 1]

    def extract_function_data(self, function: Dict[str, Any], nwo, path: str, sha: str):
        return {
            'nwo': nwo,
            'sha': sha,
            'path': path,
            'language': self.language,
            'identifier': function['identifier'],
            'parameters': function.get('parameters', ''),
            'argument_list': function.get('argument_list', ''),
            'return_statement': function.get('return_statement', ''),
            'docstring': function['docstring'].strip(),
            'docstring_summary': function['docstring_summary'].strip(),
            'docstring_tokens': tokenize_docstring(function['docstring_summary']),
            'function': function['function'].strip(),
            'function_tokens': function['function_tokens'],
            'url': 'https://github.com/{}/blob/{}/{}#L{}-L{}'.format(nwo, sha, path, function['start_point'][0] + 1,
                                                                     function['end_point'][0] + 1)
        }

    def get_context_and_function_calls(self, filepath: str) -> Optional[Tuple[str, str, List, List]]:
        nwo = '/'.join(filepath.split('/')[3:5])
        path = '/'.join(filepath.split('/')[5:])
        if any(fp in path.lower() for fp in self.language_parser.FILTER_PATHS):
            return None
        try:
            with open(filepath) as source_code:
                blob = source_code.read()
            tree = DataProcessor.PARSER.parse(blob.encode())
            return (nwo, path, self.language_parser.get_context(tree, blob), self.language_parser.get_calls(tree, blob))
        except (UnicodeDecodeError, FileNotFoundError, IsADirectoryError, ValueError, OSError):
            return None

    def get_function_definitions(self, filepath: str) -> Optional[Tuple[str, str, List]]:
        nwo = '/'.join(filepath.split('/')[3:5])
        path = '/'.join(filepath.split('/')[5:])
        # print(f"in get_function_definitions: filepath={filepath}, nwo={nwo}, path={path}")
        if any(fp in path.lower() for fp in self.language_parser.FILTER_PATHS):
            return None
        try:
            with open(filepath) as source_code:
                blob = source_code.read()
            tree = DataProcessor.PARSER.parse(blob.encode())
            # print(f"* got tree for {filepath}")
            return (nwo, path, self.language_parser.get_definition(tree, blob))
        except (UnicodeDecodeError, FileNotFoundError, IsADirectoryError, ValueError, OSError) as e:
            print(e)
            raise e
            return None


if __name__ == '__main__':
    import time
    # args = docopt(__doc__)
    import json
    args = {"--language": "haskell", "--processes": 1, "--tree-sitter-build": "/src/function-parser/tree-sitter-languages.so", "INPUT_DIR": "/src/function-parser/data/libraries-1.6.0-2020-01-12/", "OUTPUT_DIR": '/src/function-parser/data/'}
    '''
    repository_dependencies = pd.read_csv('/src/repository_dependencies_haskell.csv', index_col=False, dtype="unicode")

    values = ["Cabal", "cabal","haskell","hackage"]
    # repository_dependencies_haskell = repository_dependencies.loc[repository_dependencies['Manifest Platform'].isin(values)]
    # repository_dependencies_haskell.to_csv("repository_dependencies_haskell.csv")
    
    st = time.time()

    projects = pd.read_csv(args['INPUT_DIR'] + 'projects_with_repository_fields-1.6.0-2020-01-12.csv', index_col=False, dtype="unicode")
    print(f"reading took {time.time()-st}")

    st = time.time()

    repository_dependencies['Manifest Platform'] = repository_dependencies['Manifest Platform'].apply(lambda x: x.lower())
    id_to_nwo = {project['ID']: project['Repository Name with Owner'] for project in projects[['ID', 'Repository Name with Owner']].dropna().to_dict(orient='records')}
    nwo_to_name = {project['Repository Name with Owner']: project['Name'] for project in projects[['Repository Name with Owner', 'Name']].dropna().to_dict(orient='records')}

    print(f"reading took {time.time()-st}")
    st = time.time()

    filtered = repository_dependencies[(repository_dependencies['Host Type'] == 'GitHub') & 
    (repository_dependencies['Manifest Platform'].isin(values)  )][['Repository Name with Owner', 'Dependency Project ID']].dropna().to_dict(orient='records')
    print(f"reading took {time.time()-st}")

    st = time.time()

    id_to_nwo_list = list(id_to_nwo.keys())
    dependency_pairs = [(rd['Repository Name with Owner'], id_to_nwo[rd['Dependency Project ID']])
                        for rd in filtered if rd['Dependency Project ID'] in id_to_nwo_list]

    dependency_pairs = list(set(dependency_pairs))
    print(f"reading took {time.time()-st}")


    dents, dees = zip(*dependency_pairs)
    # dents = list(set(dents))
    dees = list(set(dees))
    

    # with open("dees.json", "w") as f:
    #     json.dump(dees, f)

    '''
    import os
    print(f"{os.path.join(__file__, '../..')}")
    # /src/function-parser/dees.json originally
    with open("/src/function-parser/dees.json", "r") as f:
        dees = json.load(f)
    
    print(f"#0 Dees from /src/function-parser/dees.json len: {len(dees)}")


    DataProcessor.PARSER.set_language(Language(args['--tree-sitter-build'], args['--language']))

    # dees = dees[:20]
    # num_dees = len(dees)

    dees_chuncks = chunkify(dees, 100)


    # dees = dees[:num_dees//10] + dees[(num_dees//20)*3:]

    processor = DataProcessor(language=args['--language'],
                              language_parser=LANGUAGE_METADATA[args['--language']]['language_parser'])

    # with Pool(processes=int(args['--processes'])) as pool:
    #     output = pool.imap_unordered(functools.partial(processor.process_dee,
    #                                                    ext=LANGUAGE_METADATA[args['--language']]['ext']),
    #                                  dees)

    for chunck_index, chunck in enumerate(dees_chuncks):
        if chunck_index < 85:
            continue
        print("#" * 40)
        print(f"Starting chunck {chunck_index}")
        print("#" * 40)
        output = []
        st = time.time()
        for dee_num, dee in enumerate(chunck):
            print(f"%% processing dee: {dee}")
            try:
                processed = processor.process_dee(dee, ext=LANGUAGE_METADATA[args['--language']]['ext'])
            except Exception as e:
                print(f"problematic dee {dee} with error {e}")
                continue
            output.extend(processed)
            # if (dee_num+1)%(num_dees//20) == 0:
            #     print(f"processed {(dee_num/num_dees)*100}% of the dees")
            #     print(f"processing last 5% of dees took: {time.time()-st}")
            #     st = time.time()

        definitions = list(flatten(output))
        with open(args['OUTPUT_DIR'] + '{}_definitions.pkl'.format(args['--language']), 'wb') as f:
            pickle.dump(definitions, f)
        
        df = pd.DataFrame(output)

        df.rename(columns = {'nwo':'repo'}, inplace = True)

        from split_utils import run

        output_dir = "/home_local/guest-9530/Documents/program_augmentation/CodeXGLUE/Code-Text/code-to-text/dataset/haskell"
        split_args = {"--train-ratio": 0.7, "--valid-ratio": 0.15, "--test-ratio": 0.15, "--holdout-ratio": 0, "--output-dir":"/src/function-parser/data/"}
        
        print(f"starting translation from DF to splited jsonls files")
        appendix = "_" + str(chunck_index)
        run(split_args, df, appendix)
        print("@" * 40)


        license_filter_file = args.get('--license-filter')
        if license_filter_file is not None:
            with open(license_filter_file, 'rb') as f:
                license_filter = pickle.load(f)
            valid_nwos = dict([(l[0], l[3]) for l in license_filter])

            # Sort function definitions with repository popularity
            definitions = [dict(list(d.items()) + [('score', valid_nwos[d['nwo']])]) for d in definitions if d['nwo'] in valid_nwos]
            definitions = sorted(definitions, key=lambda x: -x['score'])

            # dedupe
            seen = set()
            filtered = []
            for d in definitions:
                if ' '.join(d['function_tokens']) not in seen:
                    filtered.append(d)
                    seen.add(' '.join(d['function_tokens']))

            dd = DuplicateDetector(min_num_tokens_per_document=10)
            filter_mask = [dd.add_file(id=idx,
                                    tokens=d['function_tokens'],
                                    language=d['language']) for idx, d in enumerate(filtered)]
            exclusion_set = dd.compute_ids_to_exclude()
            exclusion_mask = [idx not in exclusion_set for idx, _ in enumerate(filtered)]
            filtered = [d for idx, d in enumerate(filtered) if filter_mask[idx] & exclusion_mask[idx]]

            with open(args['OUTPUT_DIR'] + '{}_dedupe_definitions.pkl'.format(args['--language']), 'wb') as f:
                pickle.dump(filtered, f)
        
