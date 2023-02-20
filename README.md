- # MAGI Dataset

  ## Install

  ```python
  pip install magi_dataset
  ```

  If you plan on using magi_dataset to periodically crawl data, set the following variables in your environment:

  ```shell
  export GH_TOKEN="Your token"
  ```

  Read [Creating a personal access token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token) for more information on creating GitHub personal access token. If using the default data without crawling new data, you may safely ignore this token. You can either provide the GitHub token using `gh_token` argument when initializing the `GitHubDataset` object, or setting it as an environment variable `GH_TOKEN` in your shell. If neither provided, the GitHub API will be initialized with no token, and the rate limit will be not sufficient for subsequent operations.

  ## Usage

  ###  Initiate Using Defaults

  You may initiate a `GitHubDataset` object directly using source we provided. Currently supported sources can be viewed at [list.json](https://huggingface.co/datasets/Enoch2090/github_semantic_search/blob/main/list.json). For example:

  ```python
  >>> from magi_dataset import GitHubDataset
  
  >>> github_dataset = GitHubDataset(
  ...     empty = False,
  ...     file_path = 'rust-latest'
  ... )
  ```

  Which will download `ghv10_rust-metadata.json`, `ghv10_rust-0.json`and `ghv10_rust-1.json` under `./magi_downloads`, and use them to create a dataset. Downloading from curated sources only cost time of downloading files, which is usually <500MB. 

  ### Pull Data by Chunks

  Pulling data from original sources is time-consuming. The recommended way to use `magi_dataset` is to run the collection process in chunked mode. First create an empty dataset and initiate index from GitHub:

  ```python
  >>> from magi_dataset import GitHubDataset
  
  >>> github_dataset = GitHubDataset(
  ...     empty = True
  ... )
  
  >>> github_dataset.init_repos(fully_initialize=False)
  >>> github_dataset.dump('./outputs/gh_data.json')
  ```

  After this process, the fingerprint `./outputs/gh_data-metadata.json` will be generated, which contains both metadata of this dataset and a fixed index of the repos to pull. Based on this metadata file, you can run multiple instances of `GitHubDataset` to pull data from online sources by chunks. For example:

  ```python
  # create a new GitHubDataset object in another terminal
  >>> from magi_dataset import GitHubDataset
  
  >>> github_dataset = GitHubDataset(
  ...     empty = True，
  ...     # use tokens from different accounts to increase throughput
  ...     gh_token = 'ghp_token1'
  ... )
  
  >>> github_dataset.load_fingerprint('./outputs/gh_data-metadata.json')
  
  >>> github_dataset.update_repos(
  ...     chunks = range(0, 50）
  ... )
    
  >>> github_dataset.dump(
  ...     './outputs/gh_data.json',
  ...     chunks = range(0, 50）
  ... )
  ```

  Which dumps `gh_data-0.json`, `gh_data-1.json`, ..., `gh_data-49.json` under the `./outputs` directory. You can also copy the fingerprint metadata file to other machines to pull different chunks, in order to relieve some stress on IP address limits of the translation API. Make sure to use tokens from different GitHub accounts in diffenent terminals/on different machines.

  Alternatively, we provide an entry from shell to do this. You can run for each coding language:

  ```bash
  magi_dataset --lang Python --file ./outputs/gh_data_python-metadata.json --meta_only True --gh_token $GH_TOKEN
  ```

  And after copying `./outputs/gh_data_python-metadata.json` to other machines, run on them separately:

  ```bash
  magi_dataset --lang Python --file ./outputs/gh_data_python.json --meta_only False --load_meta ./outputs/gh_data_python-metadata.json --gh_token $GH_TOKEN
  ```

  ### Pull Data All Together

  If the data is not much (for example, setting `GitHubDataset.MIN_STARS_PER_REPO > 2000`), you can also pull all data together once.To do so, initialize an empty instance and collect data:

  ```python
  >>> from magi_dataset import GitHubDataset
  
  >>> github_dataset = GitHubDataset(
  ...     empty = True
  ... )
  
  >>> github_dataset.init_repos(fully_initialize=True)
  ```

  Or, download the default data (not guranteed to be the newest):

  ```python
  >>> from magi_dataset import GitHubDataset
  
  >>> github_dataset3 = GitHubDataset(
  ...	    empty = False
  ... )
  ```

  The default data may be found at [Enoch2090](https://huggingface.co/Enoch2090)/[github_semantic_search](https://huggingface.co/datasets/Enoch2090/github_semantic_search/blob/main/list.json) on HuggingFace. We will update the data periodically.

  After the dataset is created, access the data with either number index:

  ```python
  >>> github_dataset[5]
  GitHubRepo(name='ytdl-org/youtube-dl', stars=114798, description='Command-line program to download videos from YouTube.com and other video sites', _fully_initialized=True)
  ```

  Or the full name:

  ```python
  >>> github_dataset['ytdl-org/youtube-dl']
  GitHubRepo(name='ytdl-org/youtube-dl', stars=114798, description='Command-line program to download videos from YouTube.com and other video sites', _fully_initialized=True)
  ```

  And you can access the corpus by accessing the `readme` and `hn_comments` attributes of `GitHubRepo` objects.

  ```python
  >>> github_dataset[5].readme[0:100]
  '[![Build Status](https://github.com/ytdl-org/youtube-dl/workflows/CI/badge.svg)](https'
  ```

  ## Future Works

  - The current idle handler design is primordial, will switch to async pipelines to relieve CPU sleep time.
  - Elasticsearch database builder
  - Pinecone database builder (wrapper only)
  - Hash verification of persistence files

  ## Changelogs

  ### v1.0.6
  Temporary fix. Added `redownload` parameter to `GitHubDataset` to avoid redownload of the same file in multiple local runs.

  ### v1.0.5 

  Updated default list of files to ghv10. Users may also retrieve default files with keyname in the latest list. For example if the list states

  ```json
  {
    "python-latest": [
      "https://huggingface.co/datasets/Enoch2090/github_semantic_search/resolve/main/ghv10_python-metadata.json",
      "https://huggingface.co/datasets/Enoch2090/github_semantic_search/resolve/main/ghv10_python-0.json",
      "https://huggingface.co/datasets/Enoch2090/github_semantic_search/resolve/main/ghv10_python-1.json",
    ]
  }
  ```

  User may retrieve these files by simply calling

  ```python
  >>> github_dataset3 = GitHubDataset(
  ...	    empty = False,
  ...     file_path = "python-latest"
  ... )
  ```

  Which is similar to Huggingface model initiation.

  ### v1.0.4

  - Added chunked update/dump/loads. Now when saving to files, only `.json` is allowed. Due to the unsafe nature of `.pkl` files and other reasons, `.pkl` files will not be supported in the future. 
  - If saving a `GitHubDataset` with $N$ chunks to file name `data.json`, will create `data-metadata.json`, and `data-0.json`, `data-1.json` ... `data-$(N-1).json`.