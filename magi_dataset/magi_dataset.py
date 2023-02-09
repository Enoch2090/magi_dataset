from threading import local
import requests
import time
import json
import re
import os
import sys
import logging
import pickle
import pandas as pd
import tempfile

from bs4 import BeautifulSoup
from typing import List, Tuple, Union, Dict, Callable
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
from dataclasses import dataclass, field, asdict
from collections import defaultdict

logging.basicConfig(format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)
magi_dataclasses_logger = logging.getLogger(__name__)

SOURCE_LIST_LINK = 'https://huggingface.co/datasets/Enoch2090/github_semantic_search/resolve/main/list.json'

DEFAULT_FILES = {
    'stopwords': os.path.join(os.path.dirname(__file__), 'data', 'stopwords.txt'),
    'patterns': os.path.join(os.path.dirname(__file__), 'data', 'patterns.txt'),
}

def download_file(link: str, local_file_name: str) -> None:
    '''
    Arguments:
        - link (str): http link of file to download
        - local_file_name (str): local file name
    '''
    r = requests.get(link, stream=True)
    file_size = int(r.headers.get('content-length'))
    with open(local_file_name, "wb") as f:
        with tqdm(
            total = file_size / 1024 / 1024,
            desc = f'Downloading {local_file_name}',
            unit = 'MB',
            bar_format = '{desc}: {percentage:.2f}%|{bar}| {n:.2f}MB/{total:.2f}MB [{elapsed}<{remaining}]'
        ) as _tqdm:
            chunk_n = 0
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
                chunk_n += 1
                _tqdm.update(1 / 1024)

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
    
class GitHubDataset(object):
    MAX_REPO_PER_LANG = 1000
    MIN_STAR_PER_REPO = 50
    TRANSLATE_MAX_RETRY = 3
    CHECKPOINT_PERIOD = 200
    LANG_LIST = ['Python', 'C++', 'JavaScript', 'Rust', 'Go']
    
    def __init__(
        self, 
        empty: bool = True, 
        lang_list: List[str] = None, 
        file_path: Union[str, os.PathLike, Path] = None, 
    ):
        '''
        Arguments:
            - empty (bool): Whether to init the data. If true, the returned GitHubDataset object will be empty. GitHubDataset.init_repos() or GitHubDataset.load() can be called later to initialize the data.
            - lang_list (List[str]): Coding languages included in this GitHubDataset object. Default to ['Python', 'C++', 'JavaScript', 'Rust', 'Go'].
            - file_path (str): If provided and empty=False, will try to load the file at given location. Can be one of str to online source, str to local file, or PathLike objects.
        '''
        self.data = []
        self._translate_err_counter = 0
        self._it_idx = 0
        if lang_list:
            self.LANG_LIST = lang_list
        self.lang_stats = defaultdict(int)
        self.reverse_map = {}
        self.G = None        
        self._init_artifacts()
        if empty:
            return
        
        if file_path is None:
            # fallback to check online versions of data
            with tempfile.TemporaryDirectory() as tmpdirname:
                local_source_list = os.path.join(
                    tmpdirname,
                    SOURCE_LIST_LINK.split('/')[-1]
                )
                download_file(
                    link = SOURCE_LIST_LINK,
                    local_file_name = local_source_list
                )
                with open(local_source_list, 'r') as f:
                    source_list = json.load(f)
                local_file_name = os.path.join(
                    tmpdirname,
                    source_list['latest_stable'].split('/')[-1]
                )
                if not os.path.exists(local_file_name):
                    download_file(
                        link = source_list['latest_stable'],
                        local_file_name = local_file_name
                    )
                self.load(local_file_name)
        else:    
            if type(file_path) is str and file_path.startswith('http'):
                # online source
                with tempfile.TemporaryDirectory() as tmpdirname:
                    local_file_name = os.path.join(tmpdirname, 'gh_corpus_tmp.json')
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
        self.data[idx] = val
    
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
        self.GH_TOKEN = os.getenv('GH_TOKEN') 
        self._github_artifact = Github(self.GH_TOKEN)
        self._translator_artifact = GoogleTranslator(source='auto', target='en')
        self._idle_handler_artifact = IdleHandler()
                
    def _update_repo(
        self, 
        repo_object: GitHubRepo, 
        repo: Repository.Repository = None
    ):
        repo_object.stars = repo.stargazers_count
        if (not repo_object._fully_initialized) or (repo.updated_at > repo_object.gh_updated_parsed_time):
            root_file_list = repo.get_contents('')
            readme_filename = None
            readme_content = ''
            readme_lang = 'en'
            for c in root_file_list:
                if not ('README' in c.name or 'readme' in c.name):
                    continue
                readme_filename = c.name
            if readme_filename is not None:
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
            repo_object.hn_comments = hn_comments
            repo_object.updated = repo.updated_at.strftime('%Y/%m/%d, %H:%M:%S')
            repo_object.retrieved = datetime.now().strftime('%Y/%m/%d, %H:%M:%S')
    
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
        x: str
    ):
        result = ''
        try:
            result = self._translator_artifact.translate(x) 
        except DTRequestError as e:
            if self._translate_err_counter <= self.TRANSLATE_MAX_RETRY:
                self._idle_handler_artifact.translate_rate_exceed_idle()
                self._translate_err_counter += 1
                result = self._translate_wrapper(x)
            else:
                x_list = x.split(' ')
                x_len = len(x_list) // 2
                magi_dataclasses_logger.warning(f'{e}, split into {x_len} and {len(x_list) - x_len}')
                result = self._translate_wrapper(' '.join(x_list[0:x_len])) + ' ' + self._translate_wrapper(' '.join(x_list[x_len::]))
        except ConnectionError as e:
            if self._translate_err_counter <= self.TRANSLATE_MAX_RETRY:
                self._translate_err_counter += 1
                result = self._translate_wrapper(x)
            else:
                result = ''
        self._translate_err_counter = 0
        return result
    
    def _divide(
        self,
        text: str, 
        chunk_len: int = 2048
    ) -> List[str]:
        n_chunks = len(text) // chunk_len
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
            return ''.join(
                list(
                    map(
                        self._translate_wrapper, self._divide(text)
                    )
                )
            )
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
    
    def init_repos(
        self, 
        fully_initialize: bool = False, 
        checkpoint_path: str = None
    ) -> None:
        '''
        Initialize self.data and update them according to the fully_initialize parameter.
        Arguments:
            - fully_initialize (bool): Whether to fully initialize contents in each GitHubRepo, or only initialize names. If fully_initialize=True, will initiailize all contents, this will be time-consuming. If fully_initialize=False, only names are initialized, but the generated GitHubRepo will have attributes fully_initialize=False, and the usage will be very restricted. If call with fully_initialize=False, GitHubDataset.update_repos() must be called before using this dataset.
            - checkpoint_path (str): str to local checkpoint files. If provided, will periodically dump to the given location.
        Comments:
            - TODO: Add load from checkpoint feature.
        '''
        control_idle = 0.1 if fully_initialize else 0.05
        self.data = []
        lang_report = defaultdict(int)
        for lang in self.LANG_LIST:
            magi_dataclasses_logger.info(f'Initializing language {lang}')
            repositories = self._github_artifact.search_repositories(query=f'stars:>{self.MIN_STAR_PER_REPO} language:{lang}')
            success = 0
            do_break = False
            for index, repo in enumerate(repositories):
                if do_break:
                    break
                while True:
                    self._idle_handler_artifact.github_rate_limit_control_idle(control_idle)
                    try:
                        if success >= self.MAX_REPO_PER_LANG:
                            do_break = True
                            break
                        repo_object = GitHubRepo(
                            name = repo.full_name,
                            lang = lang
                        )
                        if fully_initialize:
                            self._update_repo(repo_object, repo)
                        self.data.append(repo_object)
                        self.lang_stats[lang] += 1
                        success += 1
                        if success % self.CHECKPOINT_PERIOD == 0:
                            magi_dataclasses_logger.info(f'Coding language {lang}, initialization {success:5d} / {self.MAX_REPO_PER_LANG}')
                            if checkpoint_path:
                                self.dump(checkpoint_path)
                        break
                    except RateLimitExceededException as e:
                        self._idle_handler_artifact.github_rate_limit_exceed_idle()
            lang_report[lang] = index
        self._rebuild_rmap()
        if checkpoint_path:
            self.dump(checkpoint_path)
        for lang in self.LANG_LIST:
            magi_dataclasses_logger.info(f'Coding language {lang} retrieved with {index} repositories')

    def update_repos(self, checkpoint_path: str = None):
        '''
        Update all repos in self.data.
        Arguments:
            - checkpoint_path (str): str to local checkpoint files. If provided, will periodically dump to the given location.
        Comments:
            - TODO: Add load from checkpoint feature.
        '''
        for index in trange(len(self.data)):
            while True:
                self._idle_handler_artifact.github_rate_limit_control_idle(0.1)
                try:
                    self._update_repo(repo_object = self.data[index])
                    success += 1
                    if success % self.CHECKPOINT_PERIOD == 0:
                        magi_dataclasses_logger.info(f'Updated {success:5d} / {self.MAX_REPO_PER_LANG}')
                        if checkpoint_path:
                            self.dump(checkpoint_path)
                    break
                except RateLimitExceededException as e:
                    self._idle_handler_artifact.github_rate_limit_exceed_idle()
        if checkpoint_path:
            self.dump(checkpoint_path)
        magi_dataclasses_logger.info(f'Update complete, {success} repos updated.')       
        
    def load(self, file: Union[str, Path]) -> None:
        '''
        Load from either a .pkl or a .json file dump.
        Arguments:
            - file (Union[str, Path]), either a str or PathLike object. Must be a .pkl or .json file.
        Comments:
            - TODO: Add file version check.
        '''
        if type(file) is str:
            file = Path(file)
        file = file.resolve()
        assert file.exists(), f'{file} does not exist'
        assert file.suffix in ['.pkl', '.json'], f'Unsupported load type {file.suffix}'
        if file.suffix == '.pkl':
            magi_dataclasses_logger.warning(f'Loading from a .pkl file from unknown source can be dangerous. See https://www.smartfile.com/blog/python-pickle-security-problems-and-solutions for an example of pickle attack. Therefore, you should consider using .json files for persistence and exchange.')
            with open(file, 'rb') as f:
                pickle_data_object = pickle.load(f)
            for d in pickle_data_object:
                assert type(d) is GitHubRepo, 'Pickled data must be of List[GitHubRepo] type'
            self.data = pickle_data_object
        elif file.suffix == '.json':
            with open(file ,'r') as f:
                json_data_object = json.load(f)
            assert type(json_data_object) is list, 'JSON data must be of List[dict] type'
            for index, d in enumerate(json_data_object):
                assert type(d) is dict, 'JSON data must be of List[dict] type'
                for k in GitHubRepo.__annotations__.keys():
                    if k[0] == '_': # reserved properties
                        continue
                    assert k in d.keys(), f'JSON data of index {index} missing key {k}'
            self.data = []
            for d in json_data_object:
                repo_object = GitHubRepo(
                    **{
                        k: d[k] for k in GitHubRepo.__annotations__.keys() if k[0] != '_'
                    },
                    _fully_initialized = True
                )
                self.data.append(repo_object)
        self.lang_stats = defaultdict(int)
        for d in self.data:
            self.lang_stats[d.lang] += 1
        self._rebuild_rmap()
        magi_dataclasses_logger.info(f'Loaded {len(self.data)} repos from {file}')
        
    def dump(self, file: Union[str, Path]) -> None:
        '''
        Dump to either a .pkl or a .json file dump.
        Arguments:
            - file (Union[str, Path]), either a str or PathLike object. Must be a .pkl or .json file.
        Comments:
            - TODO: Add file version check.
        '''
        if type(file) is str:
            file = Path(file)
        file = file.resolve()
        assert file.suffix in ['.pkl', '.json'], f'Unsupported dump type {file.suffix}'
        if file.suffix == '.pkl':
            magi_dataclasses_logger.warning(f'Loading from a .pkl file from unknown source can be dangerous. See https://www.smartfile.com/blog/python-pickle-security-problems-and-solutions for an example of pickle attack. Therefore, you should consider using .json files for persistence and exchange.')
            with open(file, 'wb') as f:
                pickle.dump(self.data, f)
        elif file.suffix == '.json':
             with open(file ,'w') as f:
                json_data_object = [
                    {
                        k: v for k, v in asdict(d).items() if k[0] != '_'
                    } for d in self.data
                ]
                json.dump(json_data_object, f)
        magi_dataclasses_logger.info(f'Dumped to {file}')
     
    def append(self, data: GitHubRepo) -> None:
        '''
        Append a new GitHubRepo object to self.data.
        Arguments:
            - data (GitHubRepo): GitHubRepo object to append.
        Comments:
            - Note that this method does not check for duplicates. Therefore, user should be aware of appending duplicates to self.data.
        '''
        assert type(data) is GitHubRepo
        self.data.append(data)
        self.lang_stats[data.lang] += 1
        self._append_rmap(data)
        
    def filter_repos(self, patterns: Union[List[Union[str, re.Pattern]], str] = None):
        '''
        Remove GitHubRepo from self.data if its name match any of the pattern provided in patterns argument. By default this method removes the so-called "indexing repos" which are useless to MAGI's usecase.
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
                patterns = [x.replace('\n', '') for x in f.readlines()]
        # patterns: List[Union[str, re.Pattern]]
        if type(patterns[0]) is str:
            patterns = [re.compile(r'{}'.format(p)) for p in patterns]
        new_data = []
        new_stats = defaultdict(int)
        filtered_stats = defaultdict(int)
        old_len = len(self.data)
        remove_list = []
        for repo in self.data:
            found = False
            for pattern in patterns:
                if len(pattern.findall(repo.name.split('/')[-1])) != 0:
                    found = True
                    filtered_stats[repo.lang] += 1
                    break
            if not found:
                new_data.append(repo)
                new_stats[repo.lang] += 1
            else:
                remove_list.append(repo.name)
        self.data = new_data
        self.lang_stats = new_stats
        
        magi_dataclasses_logger.info(f'{old_len - len(self.data)} repos filtered out using {len(patterns)} patterns.')
        magi_dataclasses_logger.info(f'filter_repos removal stats: {dict(filtered_stats)}')
        self._rebuild_rmap()
        return remove_list
        
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
                
if __name__ == '__main__':
    # gd = GitHubDataset()
    # gd.init_repos(fully_initialize=True, checkpoint_path='ghv9.json')
    github_dataset = GitHubDataset(
        empty = False,
        file_path = './ghv9-3.json',
        load_nlp = True,
        load_graph = True
    )