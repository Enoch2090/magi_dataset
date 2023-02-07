# MAGI Dataset

## Install
```python
pip install magi_dataset
```
If you plan on using magi_dataset to periodically crawl data, set the following variables in your environment:

```shell
export GH_TOKEN="Your token"
```

Read [Creating a personal access token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token) for more information on creating GitHub personal access token. If using the default data without crawling new data, you may safely ignore this token.


## Usage
Initialize an empty instance and collect data:

```python
>>> from magi_dataset import GitHubDataset

>>> github_dataset = GitHubDataset(
...     empty = True
... )

github_dataset.init_repos(fully_initialize=True)
```

Download default data (not guranteed to be the newest):

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
