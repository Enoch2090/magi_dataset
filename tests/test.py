from magi_dataset import GitHubDataset

if __name__ == '__main__':
    
    # load local source
    print('testing load local file')
    github_dataset1 = GitHubDataset(
        empty = False,
        file_path = './ghv9_tokenized.json',
    )
    
    # load online source
    print('testing load online source')
    github_dataset2 = GitHubDataset(
        empty = False,
        file_path = 'https://huggingface.co/datasets/Enoch2090/github_semantic_search/resolve/main/ghv9-2.json',
    )
    
    # load online default
    print('testing load online default')
    github_dataset3 = GitHubDataset(
        empty = False
    )