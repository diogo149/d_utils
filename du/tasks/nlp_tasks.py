import os
DATASETS_REPO_BASE = 'https://raw.githubusercontent.com/diogo149/datasets/master/'

def penn_treebank_char(base_dir='~/penn_treebank_char'):
    char_file = 'ptb.char.test.txt'
    ptb_base = DATASETS_REPO_BASE + 'penn_treebank/'
    files = ['ptb.char.%s.txt' % split
             for split in ('train', 'valid', 'test')]
    for filename in files:
        full_file = os.path.join(base_dir, filename)
        _try_download_file(url=ptb_base + filename,
                           path = full_file)
    

    
    
    
