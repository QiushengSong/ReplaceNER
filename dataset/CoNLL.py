import torch
import os
import numpy as np
import sys
sys.path.append("..")
# import faiss
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from exceptions import DatasetError
from abc import abstractmethod
from typing import List
# from simcse import SimCSE


task_description = "You are an excellent linguist. " \
                   "The task is 'named entity recognition',you are demanded to label entities in the given sentence." \
                   "In total,there are four types of entities including Location,Person,Organization and Miscellany." \
                   "Location entities are the name of politically or geographically defined locations such as cities, provinces, countries, international regions, bodies of water, mountains, etc." \
                   "Person entities are named persons or family." \
                   "Organization entities are limited to named corporate,governmental,or other organizational entities." \
                   "Miscellany entities include events, nationalities, products and works of art , etc . " \
                   "You need to repeat the given sentence, and use character '@@' and '##' to label entities as '@@entity##'."

prefix_sentence = "NER: Input a sentence and predict the named entities (LOC, PER, ORG, MISC) in the sentence:"

MRC_instruction = "Named Entity Recognition: Repeat the sentence and replace the named entities (LOC, PER, ORG, MISC) in the sentence."


def label_aligning(sentence, label, tokenizer):

    tokens = tokenizer.tokenize(sentence)
    aligned_label = ['O']
    label_type = " "
    index = 0
    for token in tokens:
        if token.startswith("Ġ"):
            aligned_label.append(label[index])

            if len(label[index]) != 1:
                label_type = label[index][1:]
            else:
                label_type = label[index]

            index += 1

        else:
            if label_type != "O":
                aligned_label.append('I' + label_type)
            else:
                aligned_label.append(label_type)
    aligned_label.append('O')

    return aligned_label


def label_mapping(label):
    map_dict = {"O": 0, "B-PER": 1, "I-PER": 2, "B-LOC": 3,
                "I-LOC": 4, "B-ORG": 5, "I-ORG": 6, "B-MISC": 7, "I-MISC": 8}
    new_label = []
  
    for key in label:
        new_label.append(map_dict[key])
    return new_label


def extract_entity(src_sentences, src_labels):
    """
    Transform label type into relevant type

    Args:
        src_sentences: Source sentences, it should be preprocessed primarily as shape[num, sentence]
        src_labels: Source labels, it should be preprocessed primarily as shape[num, sentence]
    """
    if np.shape(src_sentences) != np.shape(src_sentences):
        raise DatasetError("Sentences' shape mismatch labels' shape, please check you code")

    references = []
    for i, labels in enumerate(src_labels):
        reference = []
        for j, label in enumerate(labels):
            if label != "O":
                # label = "@@" + src_sentences[i, j] + "##"
                word = src_sentences[i, j]
                reference.append(word)
        references.append(reference)

    return references


def transform_into_labeled(sentence_word: List,
                           sentence_label: List,
                           ) -> str:
    """
    Transform entity into @@entity##.
    """

    for idx, (word, label) in enumerate(zip(sentence_word, sentence_label)):
        if label != 'O':
            sentence_word[idx] = "@@" + word + "##"
    labeled_sentence = "The labeled sentence:" + " ".join(sentence_word)

    return labeled_sentence


class CoNLL(Dataset):
    """
    Base class for CoNLL dataset.

    Args:
        path: The path to dataset.
        max_length: The max length of input tokens used in tokenizer.
    """
    def __init__(self,
                 path: str,
                 max_length: int,
                 ):
        tokenizer = "EleutherAI/gpt-neox-20b"
        self.path = path
        self.sentences = list()
        self.sentences_label = list()
        self.sentences_word = list()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = max_length

        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Not found target file, please check your code: {self.path}")
        with open(self.path, 'r') as datafile:
            all_contents = datafile.readlines()
            sentence = list()
            sentence_label = list()
            for line in all_contents:
                if not line.strip():
                    self.sentences.append(" ".join(sentence))
                    self.sentences_label.append(sentence_label)
                    self.sentences_word.append(sentence)
                    sentence = []
                    sentence_label = []
                else:
                    content = line.strip().split()
                    sentence.append(content[0])
                    sentence_label.append(content[-1])

    @abstractmethod
    def __getitem__(self, item):
        pass

    @abstractmethod
    def __len__(self):
        pass


class FineTuneCoNLL(CoNLL):

    def __init__(self, path, max_length):
        super().__init__(path, max_length)

    def __getitem__(self, item):
        tokens = self.tokenizer(self.sentences[item], add_special_tokens=True,
                                truncation=True, padding='max_length',
                                max_length=self.max_length, return_tensors='pt')
        input_ids = tokens['input_ids'].squeeze(0)
        label = label_mapping(label_aligning(self.sentences[item],
                                             self.sentences_label[item],
                                             tokenizer=self.tokenizer))
        if len(label) < self.max_length:
            num_padding = self.max_length - len(label)
            label.extend([0] * num_padding)
            # label += [0`] * num_padding

        elif len(label) > self.max_length:
            num_truncation = len(label) - self.max_length
            label = label[:-num_truncation]
            # del label[-num_truncation:]

        print(input_ids.shape)
        label = torch.as_tensor(label)
        return input_ids, label

    def __len__(self):
        return len(self.sentences)


class ZeroShotCoNLL(CoNLL):
    """
    CoNLL dataset for zero shot pattern.

    Args:
        path: The path to dataset.
        max_length: The max length of input tokens used in tokenizer
    """
    def __init__(self,
                 path,
                 max_length=400,
                 ):
        super().__init__(path, max_length)

        self.task_description = task_description
        self.task_description_token = 149
        self.references = extract_entity(self.sentences, self.sentences_label)

        for index in range(len(self.sentences)):
            self.sentences[index] = self.task_description + "The given sentence:" + self.sentences[index] + "The labeled sentence:"

    def __getitem__(self, item):
        tokens = self.tokenizer(self.sentences[item], add_special_tokens=True,
                                truncation=True, padding='max_length',
                                max_length=self.max_length, return_tensors='pt')

        input_ids = tokens['input_ids'].squeeze(0)

        reference = self.references[item]

        return input_ids, reference

    def __len__(self):
        return len(self.sentences)

    def __str__(self):
        print("A class defined zero shot CoNLL2003 dataset for causal language model")
        print(f"Data: {self.sentences[:, :]}")
        print(f"Labels: {self.references[:, :]}")

    def __repr__(self):
        print(f"Type of data: {type(self.sentences)}")
        print(self.sentences[:, :])
        print(f"Type of label: {type(self.references)}")
        print(self.references[:, :])


class FewShotCoNLL(CoNLL):
    """
    CoNLL dataset for zero shot pattern.

    Args:
        path: The path to dataset.
        max_length: The max length of input tokens used in tokenizer.
        batch_size: Batch size used in SimCSE.encode().
        top_k: The number of k how many nearest similarity sentence you want to search for.
    """

    def __init__(self,
                 path: str,
                 max_length: int,
                 batch_size: int,
                 top_k: int,
                 ):

        super().__init__(path, max_length)

        self.task_description = task_description
        self.task_description_token = 149
        self.references = extract_entity(self.sentences, self.sentences_label)

        sim_model = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased")
        embeddings = sim_model.encode(self.sentences, device='cpu', batch_size=batch_size, normalize_to_unit=True, return_numpy=True)

        # Vector library faiss
        quantizer = faiss.IndexFlatIP(embeddings.shape[1])
        quantizer.nprob = min(10, len(self.sentences))
        index = quantizer
        index.add(embeddings.astype(np.float32))
        start = 0
        while start < len(self.sentences):
            demonstration = ""
            top_distance, top_index = index.search(embeddings[start], top_k)
            for idx in top_index[0]:
                labeled_sentence = transform_into_labeled(self.sentences_word[idx], self.sentences_label[idx])
                demonstration += "The given sentence:" + self.sentences[idx] + labeled_sentence
            self.sentences[start] = self.task_description + demonstration + "The given sentence:" + self.sentences[start]
            start += 1

    def __getitem__(self, item):
        tokens = self.tokenizer(self.sentences[item], add_special_tokens=True,
                                truncation=True, padding='max_length',
                                max_length=self.max_length, return_tensors='pt')

        input_ids = tokens['input_ids'].squeeze(0)

        reference = self.references[item]

        return input_ids, reference

    def __len__(self):
        return len(self.sentences)

    def __str__(self):
        print("A class defined few shot CoNLL2003 dataset for causal language model")
        print(f"Data: {self.sentences[:, :]}")
        print(f"Labels: {self.references[:, :]}")

    def __repr__(self):
        print(f"Type of data: {type(self.sentences)}")
        print(self.sentences[:, :])
        print(f"Type of label: {type(self.references)}")
        print(self.references[:, :])


class MRCCoNLL(CoNLL):
    """

    """
    def __init__(self,
                 path: str,
                 max_length: int,
                 ):
        super().__init__(path, max_length)
        self.prompt = MRC_instruction
        self.references = self.sentences_word

        for i, sentence in enumerate(self.sentences):
            self.sentences[i] = self.prompt + " Input:" + sentence + " Output:"

        self.references = self.transformation(self.references, self.sentences_label)

    def __getitem__(self, item):
        tokens = self.tokenizer(self.sentences[item], add_special_tokens=True,
                                truncation=True, padding='max_length',
                                max_length=self.max_length, return_tensors='pt')

        input_ids = tokens['input_ids'].squeeze(0)

        reference = self.tokenizer.encode(self.references[item],  add_special_tokens=True,
                                          truncation=True, padding='max_length',
                                          max_length=self.max_length, return_tensors='pt')
        reference = reference.squeeze(0)

        return input_ids, reference

    def __len__(self):
        return len(self.sentences)

    @classmethod
    def transformation(cls,
                       references: List,
                       sentences_label: List,
                       ):

        for i, row in enumerate(references):
          for j, column in enumerate(references[i]):
            if len(sentences_label[i][j]) > 1:
              entity = sentences_label[i][j][2:]
              references[i][j] = entity`

        for idx, reference in enumerate(references):
            references[idx] = " ".join(reference)

        return references

    def __str__(self):
        print("A class defined MRC CoNLL2003 dataset for causal language model")
        print(f"Data: {self.sentences[:, :]}")
        print(f"Labels: {self.references[:, :]}")

    def __repr__(self):
        print(f"Type of data: {type(self.sentences)}")
        print(self.sentences[:, :])
        print(f"Type of label: {type(self.references)}")
        print(self.references[:, :])