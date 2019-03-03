# coding: utf-8
import logging
from collections import namedtuple

import numpy as np
import mxnet as mx
import gluonnlp as nlp

from utils_func import get_data_bio2, bio_bioes

TaggedToken = namedtuple('TaggedToken', ['text', 'tag'])

NULL_TAG = "X"


def bert_tokenize_sentence(sentence, bert_tokenizer):
    """Apply BERT tokenizer on a tagged sentence to break words into sub-words.
    This function assumes input tags are following IOBES, and outputs IOBES tags.

    Parameters
    ----------
    sentence: List[TaggedToken]
        List of tagged words
    bert_tokenizer: nlp.data.BertTokenizer
        BERT tokenizer

    Returns
    -------
    List[TaggedToken]: list of annotated sub-word tokens
    """
    ret = []
    for token in sentence:
        # break a word into sub-word tokens
        sub_token_texts = bert_tokenizer(token.text)
        # only the first token of a word is going to be tagged
        ret.append(TaggedToken(text=sub_token_texts[0], tag=token.tag))
        ret += [TaggedToken(text=sub_token_text, tag=NULL_TAG)
                for sub_token_text in sub_token_texts[1:]]
    return ret


def load_segment(file_path, bert_tokenizer):
    """Load CoNLL format NER datafile with BIO-scheme tags.

    Tagging scheme is converted into BIOES, and words are tokenized into wordpieces using `bert_tokenizer`

    Parameters
    ----------
    file_path: str
        Path of the file
    bert_tokenizer: nlp.data.BERTTokenizer

    Returns
    -------
    List[List[TaggedToken]]: List of sentences, each of which is the list of `TaggedToken`s.
    """
    logging.info("Loading sentences in {}...".format(file_path))
    # use existing utilities to read data
    word_tokens_list, tags_bio_list = get_data_bio2(file_path, use_lower_zero=False)
    # tags_bioes_list = [bio_bioes(tag) for tag in tags_bio_list]
    # TODO: temporary hack
    tags_bioes_list = tags_bio_list
    tagged_word_sentences = [[TaggedToken(text=word, tag=tag)
                              for word, tag in zip(word_tokens, tags) if word != "-DOCSTART-"]
                             for word_tokens, tags in zip(word_tokens_list, tags_bioes_list)]
    tagged_subword_sentences = [bert_tokenize_sentence(sentence, bert_tokenizer)
                                for sentence in tagged_word_sentences]

    logging.info("load {}, its max seq len: {}".format(
        file_path, max(len(sentence) for sentence in tagged_subword_sentences)))

    return tagged_subword_sentences


class BERTTaggingDataset(object):
    """

    Parameters
    ----------
    text_vocab: gluon.nlp.Vocab
        Vocabulary of text tokens/
    train_path: Optional[str]
        Path of the file to locate training data.
    dev_path: str
        Path of the file to locate development data.
    test_path: str
        Path of the file to locate test data.
    seq_len: int
        Length of the input sequence to BERT.
    is_cased: bool
        Whether to use cased model.
    """

    def __init__(self, text_vocab, train_path, dev_path, test_path, seq_len, is_cased, tag_vocab=None):
        self.text_vocab = text_vocab
        self.seq_len = seq_len

        self.bert_tokenizer = nlp.data.BERTTokenizer(vocab=text_vocab, lower=not is_cased)

        train_sentences = [] if train_path is None else load_segment(train_path, self.bert_tokenizer)
        dev_sentences = [] if dev_path is None else load_segment(dev_path, self.bert_tokenizer)
        test_sentences = [] if test_path is None else load_segment(test_path, self.bert_tokenizer)
        all_sentences = train_sentences + dev_sentences + test_sentences

        if tag_vocab is None:
            logging.info("Indexing tags...")
            tag_counter = nlp.data.count_tokens(token.tag for sentence in all_sentences for token in sentence)
            self.tag_vocab = nlp.Vocab(tag_counter, padding_token=NULL_TAG,
                                       bos_token=None, eos_token=None, unknown_token=None)
        else:
            self.tag_vocab = tag_vocab
        self.null_tag_index = self.tag_vocab[NULL_TAG]

        logging.info("example train sentences:")
        for i in range(10):
            logging.info("{}".format(train_sentences[i]))

        self.train_inputs = [self.encode_as_input(sentence) for sentence in train_sentences]
        self.dev_inputs = [self.encode_as_input(sentence) for sentence in dev_sentences]
        self.test_inputs = [self.encode_as_input(sentence) for sentence in test_sentences]

        logging.info("tag_vocab: {}".format(self.tag_vocab))

    def encode_as_input(self, sentence):
        """Enocde a single sentence into numpy arrays as input to the BERTTagger model.

        Parameters
        ----------
        sentence: List[TaggedToken]
            A sentence as a list of tagged tokens.

        Returns
        -------
        np.array: token text ids (batch_size, seq_len)
        np.array: token types (batch_size, seq_len), which is all zero because we have only one sentence for tagging
        np.array: valid_length (batch_size,) the number of tokens until [SEP] token
        np.array: tag_ids (batch_size, seq_len)
        np.array: flag_nonnull_tag (batch_size, seq_len), which is simply tag_ids != self.null_tag_index

        """
        # check whether the given sequence can be fit into `seq_len`.
        assert len(sentence) <= self.seq_len - 2, \
            "the number of tokens {} should not be larger than {} - 2. offending sentence: {}".format(
                len(sentence), self.seq_len, sentence)

        text_tokens = ([self.text_vocab.cls_token] + [token.text for token in sentence] +
                       [self.text_vocab.sep_token])
        padded_text_ids = (self.text_vocab.to_indices(text_tokens)
                           + [self.text_vocab[self.text_vocab.padding_token]] * (self.seq_len - len(text_tokens)))

        tags = [NULL_TAG] + [token.tag for token in sentence] + [NULL_TAG]
        padded_tag_ids = (self.tag_vocab.to_indices(tags)
                          + [self.tag_vocab[NULL_TAG]] * (self.seq_len - len(tags)))

        assert len(text_tokens) == len(tags)
        assert len(padded_text_ids) == len(padded_tag_ids)
        assert len(padded_text_ids) == self.seq_len

        valid_length = len(text_tokens)

        # in sequence tagging problems, only one sentence is given
        token_types = [0] * self.seq_len

        np_tag_ids = np.array(padded_tag_ids, dtype='int32')
        # gluon batchify cannot batchify numpy.bool? :(
        flag_nonnull_tag = (np_tag_ids != self.null_tag_index).astype('int32')

        return (np.array(padded_text_ids, dtype='int32'),
                np.array(token_types, dtype='int32'),
                np.array(valid_length, dtype='int32'),
                np_tag_ids,
                flag_nonnull_tag
                )

    def get_train_data_loader(self, batch_size):
        return mx.gluon.data.DataLoader(
            self.train_inputs,
            batch_size=batch_size,
            shuffle=True,
            last_batch='keep',
            # num_workers=4,
            # prefetch=True
        )

    def get_dev_data_loader(self, batch_size):
        return mx.gluon.data.DataLoader(
            self.dev_inputs,
            batch_size=batch_size,
            shuffle=False,
            last_batch='keep',
            # num_workers=4,
            # prefetch=True
        )

    def get_test_data_loader(self, batch_size):
        return mx.gluon.data.DataLoader(
            self.test_inputs,
            batch_size=batch_size,
            shuffle=False,
            last_batch='keep',
            # num_workers=4,
            # prefetch=True
        )

    @property
    def num_tag_types(self):
        """The number of unique tags.

        Returns
        -------
        int: number of tag types.
        """
        return len(self.tag_vocab)
