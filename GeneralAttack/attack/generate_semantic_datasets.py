## attack based on the ROC graph:


# Compose a meeting agenda for an interdisciplinary team discussing the treatment plan for [Name] with [medical condition]

import csv
import os
import time
import re
import random
import argparse
import itertools

from openai import OpenAI
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
import os

client = OpenAI(
    base_url="https://api.openai.com/v1/",
    # api_key=os.environ.get("OPENAI_API_KEY"),
)

# The library takes time to initialize because the database is massive. A tip is to include its initialization in your app's startup process.
nd = NameDataset()
onnx = OnnxModelEvaluation()

hf = Huggingface(model='distilbert-base-uncased')

def l2_normal(embeding):
    return preprocessing.normalize(embeding)

def calculate_squared_euclidean_distance_sum(embeddings):
    squared_distances_sum = []
    #print(len(embeddings))
    for i in range(len(embeddings)):
        # boardcast in Numpy.
        distances = np.sum((embeddings[i] - embeddings) ** 2, axis=1)
        squared_distances_sum.append(np.sum(distances))
    return squared_distances_sum

def topk_sentences(k, squared_distances_sum, lines):
    distances_with_index = list(zip(squared_distances_sum, range(len(squared_distances_sum))))
    distances_with_index.sort(key=lambda x: x[0])
    if len(distances_with_index) < k:
        k = len(distances_with_index)
    topk_indices = [index for _, index in distances_with_index[:k]]
    topk_sentence = [lines[index] for index in topk_indices]
    return topk_sentence


# False negative means the attack sequence has smaller score with all of the victim
def similarity_evaluation_onnx(attack, victim, thres):
    '''
    For the similarity evaluation
    '''
    score = onnx.evaluation(
        {
            'question': attack
        },
        {
            'question': victim
        }
    )
    # No matter what kind of questions the victim inputs
    # if the score is higher than the thres
    # we return true
    if score > thres:
        return True
    else:
        return False

def similar_eval(attack, victim):
    score = onnx.evaluation(
        {
            'question': attack
        },
        {
            'question': victim
        }
    )
    return score

# [Tested]
# Given the system_prompt and the question, return the string here.
def prompt(system_prompt, question, max_tokens=4096):
    info = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": question,
            }
        ],
        model="gpt-3.5-turbo",
        max_tokens=max_tokens
    )

    return info.choices[0].message.content




# [Tested].
# Given the question from the dataset, summarize the [medical condition]
# Use the word in the sentence without understanding.
def medical_summary(question):
    system_prompt = "Assume you are an experienced doctor. After reading the user's prompt, \
        summarize the question using the name of the relevant disease mentioned in the user's inquiry to determine what specific illness the user is asking about. \
        Always return with Phrase or word extracted from the question sentence, make sure the phrase or the word is the illness itself, not the treatment or other things."
    
    return prompt(system_prompt, question)


def num_string_to_sentenceslist(info):
    pattern = r'^\d+[.]\s.*$'
    seeds = []
    temp = info.splitlines()
    
    for line in temp:
        if re.match(pattern, line):
            line = re.sub("^\d+[.]\s", "", line)
            seeds.append(line)
    return seeds

# [Tested]
# Given the Tempalte of the phenomenon
def template_cluster(orig_template, private_attrs):
    private_attr_list = ", ".join(private_attrs)
    system_prompt = "You are an experienced prompt writer. You will be given a template. The bracket parts {} record the keywords that you cannot change.\
        Your duty is to paraphrase the template into different forms, providing 100 semantically similar templates and returning them back. \
        All of the templates you provide should have the " + private_attr_list + " parts that are unchangeable."

    info = prompt(system_prompt, orig_template)

    return num_string_to_sentenceslist(info)


def matrix(sentences):
    matrix = [[] for _ in range(len(sentences))]
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            score = onnx.evaluation(
                {
                    'question': sentences[i]
                },
                {
                    'question': sentences[j]
                }
            )
            matrix[i].append(score)
    
    return matrix

def orthognal_group(attack, num, thres):
    # if the last_history don't change for a while.
    # then, we simply return the function beforehand.
    for i in range(len(attack)):
        sentence_group = []
        # start from i.
        sentence_group.append(attack[i])
        for j in range(i + 1, len(attack)):
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

        if len(sentence_group) >= num:
            return sentence_group, i
    
    return [], -1

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
        #     if len(sentence_group) == num:
        #         return sentence_group

    return sentence_group

# [Core Function]:
# Select the target and extend the target_seeds.
# Generate sentences group in batch.
def target_select_and_extend(private_attrs, subsets, orig_template, template_seeds, thres):
    format_kwargs = {}
    choices = []
    for attr, subset in zip(private_attrs, subsets):
        choice = random.choice(subset)
        format_kwargs[attr] = choice
        choices.append(choice)
    target = orig_template.format(**format_kwargs)

    # Use target_seeds as Label 1.
    target_seeds = list(map(lambda x: x.format(**format_kwargs), template_seeds))

    # this will be only alternate filled templates  that are semantically similar to the filled in template
    target_coins = list(filter(lambda x: similarity_evaluation_onnx(x, target, thres) == True, target_seeds))
    
    # divide the target_coins into two part, one for True_label, the other for the seed
    True_label = target_coins[:len(target_coins) // 5]
    seed = target_coins[len(target_coins) // 5:]

    # Find the enough number of False_label
    combo = []

    for tup in itertools.product(*subsets):
        can_use = False
        for i, t in enumerate(tup):
            if choices[i] != t:
                can_use = True
        if can_use:
            combo.append(tup)

    # False_label
    False_ds = list(map(lambda x: orig_template.format(**(dict(zip(private_attrs, x)))), combo))
    random.shuffle(False_ds)
    length = min(len(True_label), len(False_ds))
    
    # get the right label samples
    False_label = False_ds[:length]
    True_label = True_label[:length]
    
    # if length is zero, break out beforehand.
    if length == 0:
        return [], [], []
    
    # return False_label, True_label

    embeddings = []
    for line in seed:
        embedding = hf.to_embeddings(line)
        np_data = np.array(embedding).astype("float32").reshape(1, -1)
        np_data = l2_normal(np_data)
        embeddings.append(np_data)
    
    # here we got all of the calculated distance sum, now we want to select top_k sentence and evaluate them altogether.
    squared_distances_sum = calculate_squared_euclidean_distance_sum(embeddings)
    
    topk_seed = topk_sentences(len(seed), squared_distances_sum, seed)

    return False_label, True_label, topk_seed

#     # Use an incremental way to evaluate.
#     # Base value: test of only sentence.

#     # if the TPR case don't grow up, and the  case even 
#     # THIs time for 1.
#     attack_sentences = greedy_ways(topk_seed, thres)
#     attack_len = len(attack_sentences)    
    
#     if attack_sentences == []:
#         return []
    

#     # print(f'attack_sentences: {attack_sentences} attack_len: {attack_len} pos: {pos}')
#     tpr_list = []
#     fpr_list = []
#     # If more than half trial is TRUE, we consider the prediction is TRUE.
#     if attack_len >= 2:
#         tpr_list = list(map(lambda vic: int(sum(list(map(lambda att: int(similarity_evaluation_onnx(att, vic, thres)), attack_sentences))) >= 1), True_label))
#         fpr_list = list(map(lambda vic: int(sum(list(map(lambda att: int(similarity_evaluation_onnx(att, vic, thres)), attack_sentences))) >= 1), False_label))
#     else:
#         tpr_list = list(map(lambda vic: int(sum(list(map(lambda att: int(similarity_evaluation_onnx(att, vic, thres)), attack_sentences))) == 1), True_label))
#         fpr_list = list(map(lambda vic: int(sum(list(map(lambda att: int(similarity_evaluation_onnx(att, vic, thres)), attack_sentences))) == 1), False_label))

#     return [sum(tpr_list), sum(fpr_list), length]
    