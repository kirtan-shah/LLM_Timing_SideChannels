import csv
import os
import time
import re
import random
import argparse
from sentence_transformers import SentenceTransformer, util
from datasets import load_dataset
from gptcache.similarity_evaluation import OnnxModelEvaluation, SbertCrossencoderEvaluation
from names_dataset import NameDataset, NameWrapper
import networkx as nx
from tqdm import tqdm
import pickle
import networkx as nx
from gptcache.embedding import Huggingface
from sklearn import preprocessing
import numpy as np
import copy
import subprocess
import pandas as pd
import json

url = "http://127.0.0.1:8000/query/"
header = "Content-Type: application/json"
flush_command = f'{{"input": "pls flush cache"}}'

# for df creation:
label_type = [] # False or True
label_prompt = [] # from False label or True label
attack_prompt = [] # from attack_sentences
query_response_time = [] #
same_answer = [] # True or False, dependening on if victim and attacker received the same answer for their respective queries
#^ also helps us verify that same answer == cache hit in terms of time, and also true/false positive/negatives
which_dataset = [] # 1 ... 100
attack_sentence_num = [] # 1... 4(?)

onnx = OnnxModelEvaluation()

def greedy_ways(attack, thres):
    sentence_group = [attack[0]]
    for j in range(1, len(attack)):
        new_sentence = attack[j]
        
        flag = 0
        
        for vic in sentence_group:
            score = onnx.evaluation(
                {
                    'question': new_sentence
                },
                {
                    'question': vic
                }
            )
            # HIT at least for half of the victim
            # Then we consider it is HIT
            if score > thres:
                flag = 1
        
        if flag == 1:
            continue
        else:
            sentence_group.append(new_sentence)


    return sentence_group # this is the reduced attack sentences, so we only have one from each sentence space

def main():
    thres = 0.8
    for i in tqdm(range(0, 100), 'processing:'):
        filename = f'./semantic_datasets/data_{i}'
        fr = open(filename, 'rb')

        print("On semantic datasets number", i)

        try:
            False_label = pickle.load(fr) # name and med cond are different
            #print('false label:', False_label)
        except EOFError:
            continue
        
        True_label = pickle.load(fr) # name and med cond are same, rest is semantically similar
        #print('true label:', True_label)
        topk_seed = pickle.load(fr)
        #print('topk_seed:', topk_seed)

        if False_label == []:
            continue

        attack_sentences = greedy_ways(topk_seed, thres)
        #print('attack sentences:', attack_sentences)

        #print(len(True_label))
        #print(len(False_label))
        #print(len(attack_sentences))

        false_label_count = 0
        true_label_count = 0
            
        #hit_calculate(False_label, True_label, attack_sentences, thres, i)
        # victim queries from false label
        for f in False_label:
            #print('false:', f)
            data = f'{{"input": "{f}"}}'
            try:
                result = subprocess.run(
                    ["curl", "-X", "POST", url, "-H", header, "-d", data],
                    text=True,  # Ensures output is returned as a string (not bytes)
                    capture_output=True  # Captures stdout and stderr
                )
                victim_output = (json.loads(result.stdout))["response"]
                #print(victim_output)
            except Exception as e:
                print(f"An error occurred in Dataset {i}, False_label {false_label_count}: {e}")
                continue # go to next t because this one didn't load properly
            

            attack_sentence_count = 0

            # and then we would see if our attack sentences are a cache hit or miss
            # time the query and see if at least one was a cache hit
            for a in attack_sentences:
                print('Dataset #:', i,'False Label #:', false_label_count, ", Attack Sentence #:", attack_sentence_count)
                #print('attack:', a)
                data = f'{{"input": "{a}"}}'
                try:
                    start_time = time.time()
                    result = subprocess.run(
                        ["curl", "-X", "POST", url, "-H", header, "-d", data],
                        text=True,  # Ensures output is returned as a string (not bytes)
                        capture_output=True  # Captures stdout and stderr
                    )
                    end_time = time.time()
                    time_taken = end_time - start_time
                    attacker_output = (json.loads(result.stdout))["response"]
                    #print(attacker_output)
                except Exception as e:
                    print(f"An error occurred in Dataset {i}, False_label {false_label_count}, Attack Sentence {attack_sentence_count}: {e}")
                    continue # go to next attack sentence

                #if (victim_output["response"] != attacker_output["response"]):
                #    print('cache miss')
                print('attack query returned in', time_taken, 'seconds.')

                # here's where we add to our dataframe
                label_type.append(False)
                # label_prompt.append(f) # yes this will get added multiple times for the same prompt
                # attack_prompt.append(a) # but this will be different everytime in this loop
                query_response_time.append(time_taken)
                if (victim_output == attacker_output):
                    same_answer.append(True)
                else:
                    same_answer.append(False)
                which_dataset.append(i)
                attack_sentence_num.append(attack_sentence_count) # 0 indexed


                # FLUSH CACHE
                try:
                    flush = subprocess.run(
                        ["curl", "-X", "POST", url, "-H", header, "-d", flush_command],
                        text=True,  # Ensures output is returned as a string (not bytes)
                        capture_output=True  # Captures stdout and stderr
                    )
                    if ((json.loads(flush.stdout))["response"]) != "cache flush success":
                        print('cache flush error')
                        exit(1)
                    else:
                        print("cache flush success")
                except Exception as e:
                    print(f"Error flushing cache: {e}")
                    exit(1)
                

                # RE-ASK VICTIM PROMPT f
                    # this way, we can see if the next attack sentence is a hit with the same false label prompt (rather than it might be a hit with the previous attack sentence)
                data = f'{{"input": "{f}"}}'
                try:
                    result = subprocess.run(
                        ["curl", "-X", "POST", url, "-H", header, "-d", data],
                        text=True,  # Ensures output is returned as a string (not bytes)
                        capture_output=True  # Captures stdout and stderr
                    )
                    victim_output = (json.loads(result.stdout))["response"]
                except Exception as e:
                    print(f"Error re-asking victim output Dataset {i}, False_label {false_label_count} after Attack Sentence {attack_sentence_count}: {e}")
                    break # go to the next victim prompt

                attack_sentence_count += 1

                #break # for testing

            false_label_count += 1
            #break # for testing

        # the victim will query a sentence from True label
        # then, we will do our attack sentence
        for t in True_label: # victim prompt
            #print('true:', t)
            data = f'{{"input": "{t}"}}'
            try:
                result = subprocess.run(
                    ["curl", "-X", "POST", url, "-H", header, "-d", data],
                    text=True,  # Ensures output is returned as a string (not bytes)
                    capture_output=True  # Captures stdout and stderr
                )
                victim_output = (json.loads(result.stdout))["response"]
                #print(victim_output)
            except Exception as e:
                print(f"An error occurred in Dataset {i}, True_label {true_label_count}: {e}")
                continue # go to next t because this one didn't load properly

            attack_sentence_count = 0

            # and then we would see if our attack sentences are a cache hit or miss
            # time the query and see if at least one was a cache hit
            for a in attack_sentences:
                print('Dataset #:', i,'True Label #:', true_label_count, ", Attack Sentence #:", attack_sentence_count)
                #print('attack:', a)
                data = f'{{"input": "{a}"}}'

                try:
                    start_time = time.time()
                    result = subprocess.run(
                        ["curl", "-X", "POST", url, "-H", header, "-d", data],
                        text=True,  # Ensures output is returned as a string (not bytes)
                        capture_output=True  # Captures stdout and stderr
                    )
                    end_time = time.time()
                    time_taken = end_time - start_time
                    attacker_output = (json.loads(result.stdout))["response"]
                    #print(attacker_output)
                except Exception as e:
                    print(f"An error occurred in Dataset {i}, True_label {true_label_count}, Attack Sentence {attack_sentence_count}: {e}")
                    continue # go to next attack sentence

                #if (victim_output["response"] != attacker_output["response"]):
                #    print('cache miss')
                print('attack query returned in', time_taken, 'seconds.')

                # here's where we add to our dataframe
                label_type.append(True)
                # label_prompt.append(t) # yes this will get added multiple times for the same prompt
                # attack_prompt.append(a) # but this will be different everytime in this loop
                query_response_time.append(time_taken)
                if (victim_output == attacker_output):
                    same_answer.append(True)
                else:
                    same_answer.append(False)
                which_dataset.append(i) # so we can track which dataset it came from
                attack_sentence_num.append(attack_sentence_count) # 0 indexed

                # FLUSH CACHE
                try:
                    flush = subprocess.run(
                        ["curl", "-X", "POST", url, "-H", header, "-d", flush_command],
                        text=True,  # Ensures output is returned as a string (not bytes)
                        capture_output=True  # Captures stdout and stderr
                    )
                    if ((json.loads(flush.stdout))["response"]) != "cache flush success":
                        print('cache flush error')
                        exit(1)
                    else:
                        print("cache flush success")
                except Exception as e:
                    print(f"Error flushing cache: {e}")
                    exit(1)


                # RE-ASK VICTIM PROMPT t
                    # this way, we can see if the next attack sentence is a hit with the same true label prompt (rather than it might be a hit with the previous attack sentence)
                data = f'{{"input": "{t}"}}'
                try:
                    result = subprocess.run(
                        ["curl", "-X", "POST", url, "-H", header, "-d", data],
                        text=True,  # Ensures output is returned as a string (not bytes)
                        capture_output=True  # Captures stdout and stderr
                    )
                    victim_output = (json.loads(result.stdout))["response"]
                except Exception as e:
                    print(f"Error re-asking victim output Dataset {i}, True_label {true_label_count} after Attack Sentence {attack_sentence_count}: {e}")
                    break # go to the next victim prompt

                attack_sentence_count += 1

                # break # for testing

            true_label_count += 1
            # break # for testing


        # break # for testing


    """ print(label_type)
    print(label_prompt)
    print(attack_prompt)
    print(query_response_time)  
    print(same_answer)
    print(attack_sentence_num) """

    #df = pd.DataFrame({"Label Type": label_type, "Label Prompt": label_prompt, "Attack Prompt": attack_prompt, "Query Response Time": query_response_time, "Same Answer": same_answer})
    # if there are any weird discrepancies in the csv, we can run it again and see the label and attack prompts
    # but if all is going well, should be clear that we are using the right prompts everywhere based on query response times and same answer
    df = pd.DataFrame({"Dataset Number": which_dataset, "Label Type": label_type, "Attack Sentence Number": attack_sentence_num, "Query Response Time": query_response_time, "Same Answer": same_answer}) 
    df.to_csv('./CacheHitMiss.csv', index=False)

    return




if __name__ == "__main__":
    main()