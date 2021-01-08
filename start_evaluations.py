import os


io_files = [('facebook_pages.edgelist',
             'facebook_pages.txt',
             'facebook_pages.csv')]

methods = ['node2vec_snap',
           'node2vec_eliorc',
           'node2vec_custom',
           'deepwalk_phanein',
           'deepwalk_custom']

classifiers = ['logisticalregression',
               'randomforest',
               'gradientboost']

for io_file in io_files:
    for method in methods:
        for classifier in classifiers:
            os.system(f"python .\\main.py --input {io_file[0]} --output {io_file[1]} --results {io_file[2]}" +
                      f" --method {method} --classifier {classifier}")
