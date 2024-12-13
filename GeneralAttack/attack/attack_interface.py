from tqdm import tqdm
import re
import random
from generate_semantic_datasets import target_select_and_extend, template_cluster
from measure_confusion_matrix import process_datasets
from print_fpr_tpr import print_results
import pickle

def perform_attack(orig_template, private_attr_sets):
    private_attrs = re.findall("{(.+?)}", orig_template) # e.g. {name}

    thres = 0.8
    for i in tqdm(range(0, 100), "Collecting datasets:"):
        subsets = []
        skip = False
        for private_attr_set in private_attr_sets:
            subsets.append(random.sample(private_attr_set, 10))
            if None in subsets[-1]:
                skip = True
        if skip:
            continue
        
        file_path = f"../semantic_datasets/data_{i}"
        fw = open(file_path, 'wb')

        False_label, True_label, topk_seed = target_select_and_extend(private_attrs, subsets, orig_template, template_cluster(orig_template, private_attrs), thres)
        if False_label == []:
            continue
        
        # write the label sentences into file
        pickle.dump(False_label, fw)
        pickle.dump(True_label, fw)
        pickle.dump(topk_seed, fw)
        fw.close()
    
    process_datasets()
    print_results()


