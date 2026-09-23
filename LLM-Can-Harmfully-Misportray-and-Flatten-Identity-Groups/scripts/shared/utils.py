import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertModel
from sklearn.feature_extraction.text import CountVectorizer
import torch
import glob
import os
import nltk
from nltk.tokenize import sent_tokenize
import re
import pandas as pd
import llm_prompts as oip
from scipy import stats as st

prefix_path = ''


def chisq_homogeneity(mcs_a, mcs_b):
    counts_a = np.array([np.sum(mcs_a==i) for i in np.arange(1, 6)])
    counts_b = np.array([np.sum(mcs_b==i) for i in np.arange(1, 6)])


    both_0 = set(np.where(counts_a==0)[0])&set(np.where(counts_b==0)[0])
    for i_0 in both_0:
        counts_a[i_0] += 1
        counts_b[i_0] += 1

    observed = np.vstack([counts_a, counts_b]).T

    row_totals = np.array([np.sum(observed, axis=1)])
    col_totals = np.array([np.sum(observed, axis=0)])
    n = np.sum(observed)

    expected = np.dot(row_totals.T, col_totals) / n

    chisq, p_value = st.chisquare(observed, expected)

    chisq = np.sum(chisq)

    rows = observed.shape[0]
    cols = observed.shape[1]
    dof = (rows - 1) * (cols - 1)

    p_value = 1 - st.chi2.cdf(chisq, dof)

    return chisq, p_value

def get_mcs(loc):
    scale = {'at all': 1, 'Slightly': 2, 'Moderately': 3, 'Very': 4, 'Extremely': 5, 'Moderate': 3, 'Consistently conservative': 1, 'Mostly conservative': 2, 'Mixed': 3, 'Mostly liberal': 4, 'Consistently liberal': 5}

    results = pickle.load(open(prefix_path+'llm_responses/{}'.format(loc), 'rb'))
    if len(results) == 10:
        start_r = 4
        end_r = 14
    elif len(results) == 3:
        start_r = 4
        end_r = 15
    else:
        assert len(results) == 20, "Loc: {0} is len: {1}".format(loc, len(results))
        start_r = 4
        end_r = 19

    mcs = []
    for r, result in enumerate(results):
        these_mcs = []
        assert len(result['choices']) == 1
        response = result['choices'][0]['message']['content']


        if response == "P4: (3) Mixed\nP5: (3) Mixed\nP6: (3) Mixed\nP7: (4) Mostly liberal\nP7: (3) Mixed\nP9: (4) Mostly liberal\nP10: (3) Mixed\nP11: (3) Mixed\nP12: (3) Mixed\nP13: (3) Mixed":
            response = "P4: (3) Mixed\nP5: (3) Mixed\nP6: (3) Mixed\nP7: (4) Mostly liberal\nP8: (3) Mixed\nP9: (4) Mostly liberal\nP10: (3) Mixed\nP11: (3) Mixed\nP12: (3) Mixed\nP13: (3) Mixed"

        i = start_r
        skipped = 0
        while (i) < end_r and 'P{}'.format(i) not in response:
            i += 1
            skipped += 1
        for s in range(skipped):
            these_mcs.append(-1)
            mcs.append(-1)
        while i < end_r:
            assert 'P{}'.format(i) in response, "P{0} not in: {1}".format(i, response)
            start = response.index('P{}'.format(i)) + len('P{}'.format(i))
            end = -1
            skipped = 0
            while (i+1) < end_r and 'P{}'.format(i+1) not in response:
                i += 1
                skipped += 1
            if (i+1) != end_r:
                end = response.index('P{}'.format(i+1)) 
            if end == -1:
                chunk = response[start:]
            else:
                chunk = response[start:end]
            mc = re.findall(r'\([1-5]\)', chunk)
            if len(mc) == 0:

                ans = -1
                for word in scale.keys():
                    if word in chunk:
                        assert (ans == -1) or (ans == scale[word]), "There appear to be two MC answers here: {}".format(chunk)
                        ans = scale[word]
                if ans == -1:
                    mc = re.findall(r'[1-5]', chunk)
                    if len(mc) == 1:
                        ans = int(mc[0])
                mcs.append(ans)
                these_mcs.append(ans)
            else:
                assert len(mc) == 1, "There appear to be two MC answers here: {0}\nBigger text: {1}\nFile: {2}".format(chunk, result, loc)
                mcs.append(int(mc[0][1]))
                these_mcs.append(int(mc[0][1]))
            for s in range(skipped):
                these_mcs.append(-1)
                mcs.append(-1)
            i += 1
    if len(results) == 10:
        assert len(mcs) == 100
    elif len(results) == 3:
        assert len(mcs) == 33
    else:
        assert len(mcs) == 300
    return mcs

def get_texts(vis):
    sentences = []
    for f, folder in enumerate(vis):
        if folder[-4:] != '.pkl':
            list_of_files = glob.glob(prefix_path+'llm_responses/'+folder+'*')
            def days_comp(file_name):
                end = file_name.split("/")[-1]
                month = int(end.split('-')[1])
                day = int(end.split('-')[2].split('_')[0])
                return (month*32)+day
            this_file = max(list_of_files, key=days_comp)[len(prefix_path+'llm_responses/'):]
        else:
            this_file = folder
        response = pickle.load(open(prefix_path+'llm_responses/{}'.format(this_file), 'rb'))
        if 'davinci' in folder:
            for a, ans in enumerate(response[1]['choices']):
                message = ans['text']
                sentences.append([f, message])
        elif 'gpt' in folder:
            for a, ans in enumerate(response[1]['choices']):
                message = ans['message']['content']
                sentences.append([f, message])
        elif 'llama' in folder:
            for a, ans in enumerate(response[1]):
                text = ans['generated_text']
                message = text[text.index('[/INST]')+len('[/INST]'):]
                sentences.append([f, message])
        elif 'human' in folder:
            assert False
            assert NotImplementedError
        else:
            for a, ans in enumerate(response[1]):
                text = ans['generated_text']
                message = text[text.index('Answer: ')+len('Answer: '):]
                sentences.append([f, message])
    return sentences

class Embedder:

    def __init__(self, sentence_sets=None):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')


        self.instantiated = False
        if sentence_sets is not None:
            self.vectorizer = CountVectorizer(ngram_range=(1, 2))
            if type(sentence_sets[0]) != str:
                self.vectorizer.fit(np.concatenate(sentence_sets))
            else:
                self.vectorizer.fit(sentence_sets)
            self.instantiated = True


        self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.sbert_model.max_seq_length = 512

    def embed_paragraph(self, paragraph, version=0):
        if version == 0:
            assert False
            sent_embeddings = np.array([self.bert_embed(sentence.lower().strip()) for sentence in sentences])
        elif version == 1:
            assert self.instantiated
            sent_embeddings = self.vectorizer.transform([paragraph]).toarray()[0]
        elif version == 2:
            assert NotImplementedError
        elif version == 3:
            sent_embeddings = np.array(self.sbert_model.encode(paragraph))

        return sent_embeddings

    def bert_embed(self, word):
        marked_text = "[CLS] " + word + " [SEP]"

        tokenized_text = self.tokenizer.tokenize(marked_text)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [1] * len(tokenized_text)

        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(tokens_tensor, segments_tensors)
        embedding = outputs.last_hidden_state
        embedding = embedding.numpy()[0]
        embedding = np.mean(embedding, axis=0)
        return embedding

def sanity_checks(response, version=0):
    if version == 0:
        return len(sent_tokenize(response))
    elif version == 1:
        phrases = [' ai ', 'language model', "I'm sorry, but I can't assist", 'openai']
        for phrase in phrases:
            if phrase in response.lower():
                return True
        return False

def clean_R2b(text, version=0):


    outputs = []
    if version == 0:
        iter_range = np.arange(1, 4)
        possible_delineators = [['1', '2', '3'], ['P1', 'P2', 'P3'], ['1.', '2.', '3.'], ['1)', '2)', '3)'], ['first phrase', 'second phrase', 'third phrase'], ['I', 'II', 'III']]
    elif version == 1:
        iter_range = np.arange(3, 6)
        possible_delineators = [['3', '4', '5'], ['P3', 'P4', 'P5'], ['3.', '4.', '5.'], ['3)', '4)', '5)'], ['third phrase', 'fourth phrase', 'fifth phrase'], ['III', 'IV', 'V']]

    delineator_set = -1
    for d, delineators in enumerate(possible_delineators):
        has = 0
        for delineator in delineators:
            if delineator in text:
                has += 1
        if has == 3:
            delineator_set = d
    if delineator_set == -1:
        print("Answer does not have the right format to separate into three: {}".format(text))
        return ['', '', '']

    this_set = possible_delineators[delineator_set]
    for i, it in enumerate(iter_range):
        start = text.index(this_set[i])
        end = -1
        if it != iter_range[-1]:
            try:
                end = text.index(this_set[i+1])
            except:
                pass
        if end == -1:
            chunk = text[start+len(this_set[i]):]
        else:
            chunk = text[start+len(this_set[i]):end]
        chunk = chunk.strip()
        if chunk[:2] == ": " or chunk[:2] == ". ":
            chunk = chunk[2:].strip()
        if ((chunk[0] == '"') and (chunk[-1] == '"')) or ((chunk[0] == "'") and (chunk[-1] == "'")):
            chunk = chunk[1:-1].strip()
        outputs.append(chunk)
    return outputs

def clean_all(text, identity, user_prompt):
    ident0, ident1 = identity.split('-')
    ident1 = int(ident1)
    phrase = oip.identities[ident0][ident1]
    text = text.strip()

    if 'R1' not in user_prompt:
        to_replace = [phrase]
        if phrase == 'a Baby Boomer (age 59 to 77)':
            to_replace.append('a Baby Boomer')
        elif phrase == 'a Millennial (age 27 to 42)':
            to_replace.append('a Millennial')
        elif phrase == 'a member of Generation Z (age 18-26)':
            to_replace.append('a member of Generation Z')
        elif phrase == 'a person with ADD or ADHD':
            to_replace.append('a person with ADD')
            to_replace.append('a person with ADHD')
            to_replace.append('a person with ADD/ADHD')
            to_replace.append('a person living with Attention Deficit Disorder (ADD) and Adult Attention Deficit Hyperactivity Disorder (ADHD)')
            to_replace.append('someone with ADHD')
            to_replace.append('someone with ADHD or ADHD')
            to_replace.append('someone with ADHD/ADHD')
        elif phrase == 'a person without disabilities':
            to_replace.append('someone without disabilities')
            to_replace.append('able-bodied person')
        elif phrase == 'a person with impaired vision (e.g., blind, low vision, colorblind)':
            to_replace.append('a person with impaired vision')
            to_replace.append('someone with impaired vision')
            to_replace.append('a blind person')
            to_replace.append('someone who is blind')
        elif phrase == 'a White person':
            to_replace.append('a White American')
        elif phrase == 'an Asian person':
            to_replace.append('an Asian American')
        elif phrase == 'a Black person':
            to_replace.append('a Black American')
            to_replace.append('an African American')

        elif 'Myers-Briggs' in phrase:
            ind = phrase.index('(i.e., ')
            to_replace.append('an ' + phrase[ind-5:ind-1])
            to_replace.append('a ' + phrase[ind-5:ind-1])

        for phrase in to_replace:
            text = re.sub(r's {}'.format(phrase), 's a person', text, flags=re.IGNORECASE)

        to_delete = ['As a person living in America, ', 'As a person, ', '^As someone([^,]+), ', '^As an American([^,]+), ', '^As a ([^,]+), ', '^As an ([^,]+), ']
        for phrase in to_delete:
            text = re.sub(r'{}'.format(phrase), '', text, flags=re.IGNORECASE)

        names = np.concatenate([*oip.names.values()])
        for phrase in names:
            text = re.sub(r'I am {}'.format(phrase), '', text, flags=re.IGNORECASE)
            text = re.sub(r'As {}'.format(phrase), '', text, flags=re.IGNORECASE)
            text = re.sub(r'My name is {}'.format(phrase), '', text, flags=re.IGNORECASE)

        try:
            text = text[0].capitalize() + text[1:]
        except:
            pass

    if 'R3-1' in user_prompt:
        quotes = ['"'] 
        for quote in quotes:
            found_quotes = list(re.finditer(r'{}'.format(quote), text))
            if len(found_quotes) == 2:
                text = text[found_quotes[0].end():found_quotes[1].start()]
    return text


