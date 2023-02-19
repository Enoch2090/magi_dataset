import argparse
import requests
import time
import json
import re
import os
import sys
import logging
import hashlib
import pickle
import pandas as pd
import tempfile


from bs4 import BeautifulSoup
from typing import List, Tuple, Union, Dict, Callable, final
from tqdm import tqdm, trange
from datetime import datetime
from pathlib import Path
from github import Github, Repository, UnknownObjectException, RateLimitExceededException
from hn import search_by_date
from markdown import markdown
from requests.exceptions import ConnectionError
from langdetect import detect
from langdetect.detector import LangDetectException
from deep_translator import GoogleTranslator
from deep_translator.exceptions import RequestError as DTRequestError
from dataclasses import dataclass, field, fields, asdict
from collections import defaultdict, deque


logging.basicConfig(format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO, filename='./magi_dataset.log')
magi_dataclasses_logger = logging.getLogger(__name__)

SOURCE_LIST_LINK = 'https://huggingface.co/datasets/Enoch2090/github_semantic_search/resolve/main/list.json'

DEFAULT_FILES = {
    'stopwords': os.path.join(os.path.dirname(__file__), 'data', 'stopwords.txt'),
    'patterns': os.path.join(os.path.dirname(__file__), 'data', 'patterns.txt'),
}

def download_file(
    link: str, 
    local_file_name: str, 
    verbose: bool = False, 
    extra: str = ''
) -> None:
    '''
    Arguments:
        - link (str): http link of file to download
        - local_file_name (str): local file name
        - verbose (bool): whether to show tqdm bar
        - extra (str): extra string displayed on tqdm bar
    '''
    r = requests.get(link, stream=True)
    file_size = int(r.headers.get('content-length'))
    with open(local_file_name, "wb") as f:
        if verbose:
            with tqdm(
                total = file_size / 1024 / 1024,
                desc = f'Downloading {local_file_name}{extra}',
                unit = 'MB',
                dynamic_ncols = True,
                bar_format = '{desc}: {percentage:.2f}%|{bar}| {n:.2f}MB/{total:.2f}MB [{elapsed}<{remaining}]'
            ) as _tqdm:
                chunk_n = 0
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
                    chunk_n += 1
                    _tqdm.update(0.999 / 1024)
            return
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)


class IdleHandler:
    '''
    See https://docs.github.com/en/rest/overview/resources-in-the-rest-api?apiVersion=2022-11-28#rate-limiting for rate limits.
    '''
    def __init__(self):
        pass
    
    def github_rate_limit_exceed_idle(self, idle: float = 120):
        '''
        Used for relieving RateLimitExceededException
        '''
        magi_dataclasses_logger.info(f'Rate limit exceeded, sleep for {idle}s')
        time.sleep(idle)
    
    def github_rate_limit_control_idle(self, idle: float = 0.1):
        '''
        Used for relieving other rate limit exceedings
        '''
        time.sleep(idle)
        
    def translate_rate_exceed_idle(self, idle: float = 60):
        '''
        Used for relieving translation API rate limit
        '''
        magi_dataclasses_logger.info(f'Translate limit exceeded, sleep for {idle}s')
        time.sleep(idle)
    
@dataclass
class GitHubRepo:
    # data
    name: str = field()
    link: str = field(default = '', repr = False)
    tags: Tuple[str] = field(default = tuple(), repr = False)
    stars: int = field(default = 0)
    description: str = field(default = '')
    lang: str = field(default = '', repr = False)
    repo_lang: str = field(default = '', repr = False)
    readme: str = field(default = '', repr = False)
    readme_type: str = field(default = '', repr = False)
    hn_comments: str = field(default = '', repr = False)
    gh_updated_time: str = field(default = '', repr = False)
    gh_accessed_time: str = field(default = '', repr = False)
    hn_accessed_time: str = field(default = '', repr = False)
    
    _fully_initialized: bool = field(default = False)
    
    @property
    def gh_updated_parsed_time(self) -> datetime:
        assert self._fully_initialized, 'Non-fully-initialized GitHubRepo object does not have parsed time'
        return datetime.strptime(self.gh_updated_time, '%Y/%m/%d, %H:%M:%S')
    
    @property
    def gh_accessed_parsed_time(self) -> datetime:
        assert self._fully_initialized, 'Non-fully-initialized GitHubRepo object does not have parsed time'
        return datetime.strptime(self.gh_accessed_time, '%Y/%m/%d, %H:%M:%S')
    
    @property
    def hn_accessed_parsed_time(self) -> datetime:
        assert self._fully_initialized, 'Non-fully-initialized GitHubRepo object does not have parsed time'
        return datetime.strptime(self.hn_accessed_time, '%Y/%m/%d, %H:%M:%S')
    
    @property
    def mentioned_repos(self) -> List[str]:
        assert self._fully_initialized, 'Non-fully-initialized GitHubRepo object does not have mentioned repos'
        pattern = re.compile(r'(https?://)?github.com/(?!{})([^/^\(^\)^\s^<^>^#^\[^\]]*/[^/^\(^\)^\s^<^>^#^\[^\]]*)'.format(self.name))
        return list(set([x[-1] for x in pattern.findall(self.readme)] + [x[-1] for x in pattern.findall(self.hn_comments)]))
    
    def __hash__(self) -> int:
        return (self.name + self.gh_updated_time).__hash__()
    
    @staticmethod
    def _fields(preserved = False) -> set:
        if preserved:
            return set(
                [x.name for x in fields(GitHubRepo)]
            )
        return set(
            [x.name for x in fields(GitHubRepo) if x.name[0] != '_']
        )
    
class GitHubDataset(object):
    MAX_REPO_PER_LANG = 100_000
    MIN_STAR_PER_REPO = 100
    TRANSLATE_MAX_RETRY = 1
    CHECKPOINT_PERIOD = 1000
    LANG_LIST = ['Python', 'C++', 'JavaScript', 'Rust', 'Go']
    ITER_CHUNK_SIZE = 4000
    
    def __init__(
        self, 
        empty: bool = True, 
        lang_list: List[str] = None, 
        file_path: Union[str, os.PathLike, Path] = None, 
        patterns: Union[str, os.PathLike, Path] = None, 
        gh_token: str = None
    ):
        '''
        Arguments:
            - empty (bool): Whether to init the data. If true, the returned GitHubDataset object will be empty. GitHubDataset.init_repos() or GitHubDataset.load() can be called later to initialize the data.
            - lang_list (List[str]): Coding languages included in this GitHubDataset object. Default to ['Python', 'C++', 'JavaScript', 'Rust', 'Go'].
            - file_path (str): If provided and empty=False, will try to load the file at given location. Can be one of str to online source, str to local file, or PathLike objects. Can also be one of the keys in https://huggingface.co/datasets/Enoch2090/github_semantic_search/blob/main/list.json, in this case will download the corresponding files.
            - patterns (Union[List[Union[str, re.Pattern]], str]): Either a str to a file containing regex patterns, or a list of patterns. If str, it must be a text file, with each line a new regex pattern. 
                Example line: ^[\S]*[Mm]achine[-_]*[Ll]earning[\S]*$
            If list, must be either a list of str patterns, or a list of compiled regex object generated with re.compile().
            If left blank, the default list will be used.
        '''
        self.data = []
        self._translate_err_counter = 0
        self._it_idx = 0
        if lang_list:
            self.LANG_LIST = lang_list
        self.lang_stats = defaultdict(int)
        self.reverse_map = {}
        self._init_map = defaultdict(bool) # bool() = False
        self._chunk_map = [[]]
        self._chunk_lang_map = [[]]
        self._chunk_sizes = [0]
        self._init_fingerprint = None
        self._curr_chunk_id = 0
        self.GH_TOKEN = gh_token
        self._init_artifacts()
        self._load_patterns(patterns)
        if empty:
            return
        
        if type(file_path) is str and (not Path(file_path).exists()):
            # non-local source
            try:
                Path('./magi_downloads').mkdir()
            except FileExistsError:
                pass
            if (not file_path.startswith('http')):
                # default source
                # fallback to check online versions of data
                local_source_list = os.path.join(
                    './magi_downloads',
                    SOURCE_LIST_LINK.split('/')[-1]
                )
                download_file(
                    link = SOURCE_LIST_LINK,
                    local_file_name = local_source_list
                )
                
                with open(local_source_list, 'r') as f:
                    source_list = json.load(f)
                
                for index, file in enumerate(source_list[file_path]):
                    local_file_name = os.path.join(
                        './magi_downloads',
                        file.split('/')[-1]
                    )
                    download_file(
                        link = file,
                        local_file_name = local_file_name,
                        verbose = True,
                        extra = f'{index + 1:2d}/{len(source_list[file_path])}'
                    )
                magi_dataclasses_logger.info(
                    f"Default source files {file_path} downloaded to {Path('./magi_downloads').resolve().__str__()}"
                )
                local_file_name = local_file_name.split('-')[0] + '.json'
            else:
                local_file_name = os.path.join('./magi_downloads', 'gh_corpus_tmp.json')
                download_file(
                    link = file_path,
                    local_file_name = local_file_name
                )
                magi_dataclasses_logger.info(
                    f'Online source file {file_path} downloaded to {local_file_name}'
                )
            self.load(local_file_name)
        else:    
            # local source
            self.load(file_path)
        
    def __getitem__(self, idx):
        assert type(idx) is int or type(idx) is str, 'Index must be either int index or repository name str index.'
        if type(idx) is int:
            return self.data[idx]
        return self.data[self.reverse_map[idx]]

    def __setitem__(self, idx, val):
        assert type(val) is GitHubRepo, 'Item must be of GitHubRepo type'
        if type(idx) is int:
            self.data[idx] = val
        else:
            self.data[self.reverse_map[idx]] = val              
    
    def __iter__(self):
        self._it_idx = 0
        return self
        
    def __next__(self) -> GitHubRepo:
        if self._it_idx >= len(self.data):
            raise StopIteration
        next_item = self.data[self._it_idx]
        self._it_idx += 1
        return next_item
    
    def __hash__(self) -> int:
        return sum([repo.__hash__() for repo in self])
    
    def __eq__(self, other) -> bool:
        if (len(self) != len(other)):
            return False
        return self.__hash__() == other.__hash__()
    
    def __len__(self):
        return len(self.data)
    
    def __contains__(self, repo: Union[str, GitHubRepo]):
        assert type(repo) is str or type(repo) is GitHubRepo, 'Item must be either name str or GitHubRepo type.'
        if type(repo) is str:
            return repo in self.reverse_map.keys()
        return repo.name in self.reverse_map.keys()
        
    def __str__(self):
        langs = ', '.join([f'{lang}: List_{self.lang_stats[lang]}' for lang in self.LANG_LIST])
        return f'GitHubDataset({langs})'
    
    @property
    def _chunk_num(self):
        return len(self._chunk_sizes)
    
    def _remove_code_chunk(self):
        pattern = re.compile('```[\s\S]+```')
        for idx in range(self.__len__()):
            self.data[idx].readme = pattern.sub('\n', self.data[idx].readme)
            self.data[idx].hn_comments = pattern.sub('\n', self.data[idx].hn_comments)
            
    def _remove_html_tags(self):
        pattern = re.compile('<[\s\S]+>')
        for idx in range(self.__len__()):
            self.data[idx].readme = pattern.sub(' ', self.data[idx].readme)
            self.data[idx].hn_comments = pattern.sub(' ', self.data[idx].hn_comments)
    
    def _init_artifacts(self):
        '''
        Comments:
            - See https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token
            - If this token is not set, then the rate limit is extremely low, which will be totally unusable 
        '''
        if self.GH_TOKEN is None: 
            self.GH_TOKEN = os.getenv('GH_TOKEN') 
        self._github_artifact = Github(self.GH_TOKEN)
        self._translator_artifact = GoogleTranslator(source='auto', target='en')
        self._idle_handler_artifact = IdleHandler()
                
    def _update_repo(
        self, 
        repo_object: GitHubRepo, 
        repo: Repository.Repository = None
    ):
        if repo is None:
            repo = self._github_artifact.get_repo(repo_object.name)
        repo_object.stars = repo.stargazers_count
        if (not repo_object._fully_initialized) or (repo.updated_at > repo_object.gh_updated_parsed_time):
            root_file_list = repo.get_contents('')
            readme_filename = None
            readme_content = ''
            readme_lang = 'en'
            readme_type = 'text'
            for c in root_file_list:
                if not ('README' in c.name or 'readme' in c.name):
                    continue
                readme_filename = c.name
            if readme_filename is not None:
                if '.md' in readme_filename or '.markdown' in readme_filename:
                    readme_type = 'markdown'
                if '.rst' in readme_filename:
                    readme_type = 'rst'
                repo_readme = repo.get_contents(readme_filename)
                if type(repo_readme) is list:
                    dl_url = repo_readme[0].download_url
                else:
                    dl_url = repo_readme.download_url
                if (dl_url is not None) and (dl_url != ''):
                    readme_content = requests.get(dl_url).text
                    try:
                        readme_lang = detect(readme_content[0:(512 if len(readme_content) <= 512 else -1)])
                    except LangDetectException:
                        readme_lang = 'en'
                if not readme_lang == 'en':
                    readme_content = self._chunk_translate_en(readme_content) 
                if readme_content is None:
                    readme_content = ''
                    magi_dataclasses_logger.error(f'Repo {repo_object.name} translation error') 
            hn_comments = ''
            while True:
                try:
                    hn_comments = self._get_hn_comments(repo.full_name)
                except Exception as e:
                    logging.warning(f"{e}")
                finally: 
                    break
            repo_object.link = repo.html_url
            repo_object.tags = repo.get_topics()
            repo_object.description = repo.description
            repo_object.orig_lang = readme_lang
            repo_object.readme = readme_content
            repo_object.readme_type = readme_type
            repo_object.hn_comments = hn_comments
            repo_object.updated = repo.updated_at.strftime('%Y/%m/%d, %H:%M:%S')
            repo_object.retrieved = datetime.now().strftime('%Y/%m/%d, %H:%M:%S')
        return repo_object
    
    def _rebuild_rmap(self):
        for index, repo in enumerate(self.data):
            self.reverse_map[repo.name] = index
            
    def _append_rmap(
        self, 
        data: GitHubRepo
    ):
        self.reverse_map[data.name] = len(self.reverse_map)
    
    def _translate_wrapper(
        self, 
        x: Union[str, list]
    ) -> Union[str, list, None]:
        result = ''
        self._translate_err_counter = 0
        try:
            if type(x) is str:
                result = self._translator_artifact.translate(x)
            elif type(x) is list:
                x = [_x for _x in x if _x != '']
                result = self._translator_artifact.translate_batch(x)
        except DTRequestError as e:
            self._idle_handler_artifact.translate_rate_exceed_idle()
            if self._translate_err_counter < self.TRANSLATE_MAX_RETRY:
                self._translate_err_counter += 1
                result = self._translate_wrapper(x)
            else:
                return None
        except ConnectionError as e:
            if self._translate_err_counter < self.TRANSLATE_MAX_RETRY:
                self._translate_err_counter += 1
                result = self._translate_wrapper(x)
            else:
                result = ''
        return result
    
    def _divide(
        self,
        text: str, 
        chunk_len: int = 2048
    ) -> List[str]:
        n_chunks = len(text) // chunk_len + 1
        return [
            text[i*chunk_len: i*chunk_len+chunk_len] if i != n_chunks - 1 else text[i*chunk_len::] for i in range(n_chunks)
        ]

    def _chunk_translate_en(
        self, 
        text: str
    ) -> str:
        if text is None:
            return ''
        if len(text) == 0:
            return ''
        try:
            return ''.join(self._translate_wrapper(self._divide(text)))
        except TypeError:
            return ''
        
    def _get_hn_comments(
        self, 
        topic: str
    ) -> str:
        '''
        Arguments: 
            - topic (str) - form of f'{author_name}/{repo_name}' works best.
        Returns:
            - str - concatenated comments
        '''
        text = ''
        for index, r in enumerate(search_by_date(q=topic, stories=True, num_comments__gt=0)):
            if index >= 5:
                break
            hn_comments_raw = requests.get(f'http://hn.algolia.com/api/v1/items/{r["objectID"]}').json()['children']
            hn_comments_text = '<HN_SEP>'.join(
                [
                    BeautifulSoup(x['text'], features="lxml").text for x in hn_comments_raw if x['text'] is not None and len(x['text']) > 0
                ]
            )
            text += f"{hn_comments_text}<HN_SEP>"
        return text
    
    def _load_patterns(self, patterns: Union[List[Union[str, re.Pattern]], str] = None):
        '''
        Load the given filter patterns.
        Argument: 
            - patterns (Union[List[Union[str, re.Pattern]], str]): Either a str to a file containing regex patterns, or a list of patterns. If str, it must be a text file, with each line a new regex pattern. 
                Example line: ^[\S]*[Mm]achine[-_]*[Ll]earning[\S]*$
            If list, must be either a list of str patterns, or a list of compiled regex object generated with re.compile().
            If left blank, the default list will be used.
        '''
        if patterns is None:
            patterns = DEFAULT_FILES['patterns']
        if type(patterns) is str:
            with open(patterns, 'r') as f:
                self.patterns = [x.replace('\n', '') for x in f.readlines()]
        # patterns: List[Union[str, re.Pattern]]
        if type(patterns[0]) is str:
            self.patterns = [re.compile(r'{}'.format(p)) for p in self.patterns]

    def _matches_pattern(self, name:str):
        '''
        Check whether the given name matches any of the pattern loaded by GitHubDataset._load_patterns()
        '''
        for pattern in self.patterns:
            if len(pattern.findall(name.split('/')[-1])) != 0:
                return True
        return False

    def init_repos(
        self, 
        fully_initialize: bool = False, 
    ) -> None:
        '''
        Initialize self.data and update them according to the fully_initialize parameter.
        The recommended way to rebuild the dataset is to run GitHubDataset.init_repos() with fully_initialize = False, and then use GitHubDataset.update_all_repos() to pull the information chunk by chunk, to ensure stability.
        Arguments:
            - fully_initialize (bool): Whether to fully initialize contents in each GitHubRepo, or only initialize names. If fully_initialize=True, will initiailize all contents, this will be time-consuming. If fully_initialize=False, only names are initialized, but the generated GitHubRepo will have attributes fully_initialize=False, and the usage will be very restricted. If call with fully_initialize=False, GitHubDataset.update_repos() must be called before using this dataset.
        Comments:
            - TODO: Add load from checkpoint feature.
        '''
        control_idle = 0.1 if fully_initialize else 0.05
        self.data = []
        for lang in self.LANG_LIST:
            magi_dataclasses_logger.info(f'Initializing language {lang}')
            curr_star_lowerbound = self.MIN_STAR_PER_REPO
            curr_star_upperbound = 1_000_000
            success = 0
            end_curr_lang_search = False
            recent_cache = deque(maxlen=100)
            while not end_curr_lang_search:
                repositories = self._github_artifact.search_repositories(query=f'stars:{curr_star_lowerbound}..{curr_star_upperbound} language:{lang}')
                magi_dataclasses_logger.info(f'Searching scope "stars:{curr_star_lowerbound}..{curr_star_upperbound} language:{lang}"')
                for repo in repositories:
                    if end_curr_lang_search:
                        magi_dataclasses_logger.info(f'Ending search for {lang}')
                        break
                    while True:
                        self._idle_handler_artifact.github_rate_limit_control_idle(control_idle)
                        try:
                            if success >= self.MAX_REPO_PER_LANG:
                                end_curr_lang_search = True
                                break
                            repo_name = repo.full_name
                            star_num = repo.stargazers_count
                            # if repo_name in recent_cache:
                            #     end_curr_lang_search = True
                            #     magi_dataclasses_logger.info(f'Search for {lang} ended, collection number={success} which reached the MAX_REPO_PER_LANG parameter.')
                            #     break
                            if self._matches_pattern(repo_name):
                                magi_dataclasses_logger.info(f'Repo {repo_name} ignored')
                                break
                            repo_object = GitHubRepo(
                                name = repo_name,
                                lang = lang
                            )
                            if fully_initialize:
                                self._update_repo(repo_object, repo)
                                self._init_map[repo_name] = True
                            else:
                                self._init_map[repo_name] = False
                            self.append(repo_object)
                            magi_dataclasses_logger.info(f'Repo {repo_name} added (stars={star_num}), index={success}')
                            success += 1
                            recent_cache.append(repo_name)
                            if self._chunk_sizes[-1] >= self.ITER_CHUNK_SIZE:
                                # create a new virtual chunk
                                self._chunk_sizes.append(0)
                                self._chunk_map.append([])
                                self._chunk_lang_map.append([])
                            self._chunk_sizes[-1] += 1
                            self._chunk_map[-1].append(repo_name)
                            self._chunk_lang_map[-1].append(lang)
                            curr_star_upperbound = star_num - 1
                            
                            if curr_star_upperbound <= self.MIN_STAR_PER_REPO:
                                end_curr_lang_search = True
                                magi_dataclasses_logger.info(f'Search for {lang} ended, curr_star_upperbound={curr_star_upperbound} <= MIN_STAR_PER_REPO={self.MIN_STAR_PER_REPO}')
                            break
                        except RateLimitExceededException as e:
                            self._idle_handler_artifact.github_rate_limit_exceed_idle()
        self._rebuild_rmap()
        for lang in self.LANG_LIST:
            magi_dataclasses_logger.info(f'Coding language {lang} {"retrieved" if fully_initialize else "index built"} with {self.lang_stats[lang]} repositories')
        hashlib.sha1().update(str(time.time()).encode("utf-8"))
        self._init_fingerprint = hashlib.sha1().hexdigest()

    def _update_chunk_repos(self, chunk_id):
        for repo_name in self._chunk_map[chunk_id]:
            while True:
                try:
                    self[repo_name] = self._update_repo(self[repo_name])
                    magi_dataclasses_logger.info(f'Repo {repo_name} info updated (stars={self[repo_name].stars})')
                    self._idle_handler_artifact.github_rate_limit_control_idle(0.05)
                except RateLimitExceededException as e:
                    self._idle_handler_artifact.github_rate_limit_exceed_idle()
                finally:
                    break
            
    def update_repos(self, chunks: Union[List[int], range, int] = -1):
        assert type(chunks) in [list, range, int], 'Argument chunks must be one of list, range or int type.'
        if chunks == -1:
            chunks = list(range(self._chunk_num))
        if type(chunks) is int:
            chunks = [chunks]
        for chunk_id in chunks:
            self._update_chunk_repos(chunk_id)
        magi_dataclasses_logger.info(f'Successfully updated chunk {chunks}, {len(chunks)}/{self._chunk_num} chunks changed.')
    
    def load_fingerprint(
        self, 
        file: Union[str, Path]
    ) -> None:
        '''
        Load fingerprints and chunk indexes from a fingerprint metadata file.
        Arguments:
            - file(Union[str, Path]): .json file containing the fingerprint.
        '''
        with open(file, 'r') as f:
                metadata = json.load(f)
        self._init_fingerprint = metadata['_init_fingerprint']
        self._chunk_map = metadata['_chunk_map']
        self._chunk_lang_map = metadata['_chunk_lang_map']
        self._chunk_sizes = metadata['_chunk_sizes']
        self.data = []
        # Rebuild all repos with name only.
        # No matter how much chunks are loaded, all repos will be rebuilt.
        for chunk_id in range(len(self._chunk_sizes)):
            for repo_id in range(self._chunk_sizes[chunk_id]):
                self.append(
                    GitHubRepo(
                        name = self._chunk_map[chunk_id][repo_id],
                        lang = self._chunk_lang_map[chunk_id][repo_id]
                    )
                )
        self._rebuild_rmap()

    def dump_fingerprint(
        self,
        file: Union[str, Path]
    ) -> None:
        '''
        Dump fingerprints and chunk indexes to a fingerprint metadata file.
        Arguments:
            - file(Union[str, Path]): .json file to dump the fingerprint.
        '''
        with open(file, 'w') as f:
            json.dump(
                {
                    '_chunk_map': self._chunk_map,
                    '_chunk_lang_map': self._chunk_lang_map,
                    '_chunk_sizes': self._chunk_sizes,
                    '_init_fingerprint': self._init_fingerprint
                },
                f
            )
    
    def load(
        self, 
        file: Union[str, Path],
        chunks: Union[List[int], range, int] = -1
    ) -> None:
        '''
        Load from either a .pkl or a .json file dump.
        Arguments:
            - file (Union[str, Path]): either a str or PathLike object. Must be a .json file.
            - chunks: One of list, range or int type to specify chunks to dump. If left empty, defaults to all chunks.
        '''
        if type(file) is str:
            file = Path(file)
        meta_file = file.with_name(file.name.replace(file.suffix, f'-metadata{file.suffix}'))
        file = file.resolve()
        meta_file = meta_file.resolve()
        assert meta_file.exists(), f'{meta_file} does not exist'
        assert file.suffix in ['.json'], f'Unsupported load type {file.suffix}'
        self.load_fingerprint(meta_file)
        
        if chunks == -1:
            chunks = list(range(len(self._chunk_map)))
        if type(chunks) is int:
            chunks = [chunks]
        if file.suffix == '.json':
            for chunk in chunks:
                chunk_file = file.with_name(file.name.replace(file.suffix, f'-{chunk}{file.suffix}'))
                with open(chunk_file ,'r') as f:
                    json_data_object = json.load(f)
                assert json_data_object['_init_fingerprint'] == self._init_fingerprint, f'File {chunk_file} has mismatched fingerprint with metadata.'
                for d in json_data_object['data']:
                    self[d['name']] = GitHubRepo(
                        **{
                            k: d[k] for k in GitHubRepo.__annotations__.keys() if k[0] != '_'
                        },
                        _fully_initialized = True if set(GitHubRepo.__annotations__.keys()) == GitHubRepo._fields(preserved=True) else False
                    )
                
        self.lang_stats = defaultdict(int)
        for d in self.data:
            self.lang_stats[d.lang] += 1
        magi_dataclasses_logger.info(f'Loaded {len(self.data)} repos from {file}')
        
    def dump(
            self, 
            file: Union[str, Path],
            chunks: Union[List[int], range, int] = -1
        ) -> None:
        '''
        Dump to either a .pkl or a .json file dump.
        Arguments:
            - file (Union[str, Path]): Either a str or PathLike object. Must be a .json file.
            - chunks: One of list, range or int type to specify chunks to dump. If left empty, defaults to all chunks.
        '''
        if type(file) is str:
            file = Path(file)
        meta_file = file.with_name(file.name.replace(file.suffix, f'-metadata{file.suffix}'))
        file = file.resolve()
        meta_file = meta_file.resolve()
        assert file.suffix in ['.json'], f'Unsupported dump type {file.suffix}'
        assert self._init_fingerprint, f'Trying to dump a GitHubDataset which is not properly initialized'
        if chunks == -1:
            chunks = list(range(len(self._chunk_map)))
        if type(chunks) is int:
            chunks = [chunks]
        if file.suffix == '.json':
            self.dump_fingerprint(meta_file)
            for chunk in chunks:
                chunk_file = file.with_name(file.name.replace(file.suffix, f'-{chunk}{file.suffix}'))
                with open(chunk_file ,'w') as f:
                    json_data_object = {
                        'data': [
                            {
                                k: v for k, v in asdict(self[name]).items() if k[0] != '_'
                            } for name in self._chunk_map[chunk]
                        ],
                        '_init_fingerprint': self._init_fingerprint
                    }
                    json.dump(json_data_object, f)
        magi_dataclasses_logger.info(f'Dumped to {file}')
        
    def upload_to_es(
        self, 
        es_server: str = 'http://localhost:9200'
    ) -> None:
        '''
        Upload contents of data to a Elasticsearch server.
        Arguments:
            - es_server (str): ES cluster API address.
        '''
        try:
            from elasticsearch import Elasticsearch
        except ImportError:
            magi_dataclasses_logger.error(f'Package elasticsearch not installed. Please use pip install magi_dataclass[elasticsearch] to ensure dependency.')
            return
        
        # TODO: 1.0.5 is a temporary fix for downloading default chunked files.
        # upload to ES feature is therefore still under development.
        raise NotImplemented
        
     
    def append(self, data: GitHubRepo) -> None:
        '''
        Append a new GitHubRepo object to self.data.
        Arguments:
            - data (GitHubRepo): GitHubRepo object to append.
        Comments:
            - For duplicates, this method try to overwrite the old value with the new one.
        '''
        assert type(data) is GitHubRepo
        try:
            self[data.name] = data
        except:
            self.data.append(data)
            self.lang_stats[data.lang] += 1
            self._append_rmap(data)

    @property   
    def statistics(self) -> pd.DataFrame:
        '''
        Get statistics of self.data.
        Returns:
            - pd.DataFrame: Statistics with the following rows:
            [
                'Max Length of README Corpus', 
                'Min Length of README Corpus',
                'Avg Length of README Corpus',
                'Max Length of HN Corpus',
                'Min Length of HN Corpus',
                'Avg Length of HN Corpus',
                '%Data with README file',
                '%Data with HN Comments',
                'Max Stars',
                'Min Stars', 
                'Avg Stars'
            ]
            
        '''
        gh_len = [len(x.readme.split(' ')) for x in self.data]
        hn_len = [len(x.hn_comments.split(' ')) for x in self.data]
        data_df = pd.DataFrame(
            data = [
                max(gh_len),
                min([x for x in gh_len if x > 0]),
                sum(gh_len) / len(gh_len),
                max(hn_len),
                min([x for x in hn_len if x > 0]),
                sum(hn_len) / len(hn_len),
                len([x for x in self.data if len(x.readme) > 0]) / len(self.data),
                len([x for x in self.data if len(x.hn_comments) > 0]) / len(self.data),
                max([x.stars for x in self.data]),
                min([x.stars for x in self.data]),
                sum([x.stars for x in self.data]) / len(self.data),
            ], 
            index = [
                'Max Length of README Corpus', 
                'Min Length of README Corpus',
                'Avg Length of README Corpus',
                'Max Length of HN Corpus',
                'Min Length of HN Corpus',
                'Avg Length of HN Corpus',
                '%Data with README file',
                '%Data with HN Comments',
                'Max Stars',
                'Min Stars', 
                'Avg Stars',
            ],
            columns = ['Value']
        )
        data_df.index.name = 'Statistics'
        return data_df

def entry():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', type=str, default='Python')
    parser.add_argument('--file', type=str, default='gh_data.json')
    parser.add_argument('--max_repo_per_lang', type=int, default=50000)
    parser.add_argument('--min_star_per_repo', type=int, default=500)
    parser.add_argument('--iter_chunk_size', type=int, default=1000)
    parser.add_argument('--gh_token', type=str, default=None)
    parser.add_argument('--meta_only', type=bool, default=False)
    parser.add_argument('--load_meta', type=str, default=None)
    args = parser.parse_args()
    
    github_dataset = GitHubDataset(
        empty = True,
        gh_token = args.gh_token
    )

    github_dataset.MAX_REPO_PER_LANG = args.max_repo_per_lang
    github_dataset.MIN_STAR_PER_REPO = args.min_star_per_repo
    github_dataset.ITER_CHUNK_SIZE = args.iter_chunk_size
    github_dataset.LANG_LIST = [args.lang]
    
    if args.load_meta is not None:
        github_dataset.load_fingerprint(args.load_meta)
    else:
        github_dataset.init_repos(fully_initialize=False)
        if args.meta_only:
            github_dataset.dump_fingerprint(f'{args.file}')
            return

    for chunk in range(github_dataset._chunk_num):
        try:
            github_dataset.update_repos(chunks=chunk)
            github_dataset.dump(args.file, chunks=chunk)
        except Exception as e:
            with open('err.log', 'w+') as f:
                f.write(f'chunk={chunk}, err: {e}\n')
  
if __name__ == '__main__':
    entry()