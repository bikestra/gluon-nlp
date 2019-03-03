#!/usr/bin/env python
# coding: utf-8
import argparse
import logging

import numpy as np
import mxnet as mx
import os
import pickle

import gluonnlp as nlp

from bert_common import *
from bert_data import BERTTaggingDataset, NULL_TAG
from bert_model import BERTTagger

# TODO: currently, our evaluation is dependent on this package. figure out whether to take actual dependency on it.
import seqeval.metrics

from gluonnlp.model.bert import *
from gluonnlp.vocab import BERTVocab


def str2bool(v: str):
    """
    Utility function for parsing boolean in argparse

    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    :param v: value of the argument
    :return:
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_context(gpu_index):
    """ This method gets context of execution"""
    context = None
    if gpu_index is None or gpu_index == '':
        context = mx.cpu()
    if isinstance(gpu_index, int):
        context = mx.gpu(gpu_index)
    return context


def find_model_file_from_checkpoint(checkpoint_prefix: str):
    dirname, file_prefix = os.path.split(checkpoint_prefix)
    # find checkpoint file names and sort by name to find the most recent one.
    checkpoint_filenames = ([f for f in os.listdir(dirname)
                             if f.startswith(file_prefix)
                             and f.endswith(os.path.extsep + "params")])
    last_checkpoint_filename = max(checkpoint_filenames)
    logging.info(f"found checkpoint filename: {last_checkpoint_filename}")
    last_checkpoint_path = os.path.join(dirname, last_checkpoint_filename)
    return last_checkpoint_path


def parse_args():
    """Parse command line arguments."""
    arg_parser = argparse.ArgumentParser(
        description="Train a BERT-based named entity recognition model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # data file paths
    arg_parser.add_argument("--test-path", type=str, required=True,
                            help="Path to the test data file")
    arg_parser.add_argument("--seq-len", type=int, default=200,
                            help="The length of the sequence input to BERT."
                                 " An exception will raised if this is not large enough.")
    arg_parser.add_argument("--load-checkpoint-prefix", type=str, required=False, default=None,
                            help="Prefix of model checkpoint file")

    arg_parser.add_argument("--gpu", type=int,
                            help='Number (index) of GPU to run on, e.g. 0.  If not specified, uses CPU.')
    arg_parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    args = arg_parser.parse_args()
    return args


def main(config):
    with open(metadata_file_path(config.load_checkpoint_prefix), "rb") as ifp:
        metadata = pickle.load(ifp)
    train_config = metadata["config"]

    # TODO: redundant code start
    if train_config.cased:
        bert_dataset_name = 'book_corpus_wiki_en_cased'
    else:
        bert_dataset_name = 'book_corpus_wiki_en_uncased'

    ctx = get_context(config.gpu)

    logging.info("Loading BERT model...")
    bert_model, text_vocab = nlp.model.get_bert_model(
        model_name=train_config.bert_model,
        dataset_name=bert_dataset_name,
        pretrained=True,
        ctx=ctx,
        use_pooler=False,
        use_decoder=False,
        use_classifier=False)

    dataset = BERTTaggingDataset(text_vocab, None, None, config.test_path,
                                 config.seq_len, train_config.cased, tag_vocab=metadata["tag_vocab"])

    test_data_loader = dataset.get_test_data_loader(config.batch_size)

    net = BERTTagger(bert_model, dataset.num_tag_types, train_config.dropout_prob)
    model_filename = find_model_file_from_checkpoint(config.load_checkpoint_prefix)
    # loaded = mx.nd.load(model_filename)
    # net.collect_params().initialize(loaded, ctx, True, force_reinit=True)
    net.load_parameters(model_filename, ctx=ctx)

    # TODO: mimicking https://github.com/dmlc/gluon-nlp/pull/493/files#diff-b5de13a7e0607d688f8a38256062d107R271
    # but is there a more natural way to initialize just the parameters added on top of BERT?
    # net.tag_classifier.initialize(init=mx.init.Normal(0.02), ctx=ctx)
    net.hybridize(static_alloc=True)

    loss_function = mx.gluon.loss.SoftmaxCrossEntropyLoss()
    loss_function.hybridize(static_alloc=True)

    def evaluate(data_loader):
        entries_list = []

        for batch_id, data in enumerate(data_loader):
            logging.info("evaluating on batch index: {}/{}".format(batch_id, len(data_loader)))
            text_ids, token_types, valid_length, tag_ids, flag_nonnull_tag = \
                [x.astype('float32').as_in_context(ctx) for x in data]
            out = net(text_ids, token_types, valid_length)

            np_valid_length = valid_length.astype('int32').asnumpy()
            np_pred_tags = out.argmax(axis=-1).asnumpy()
            np_tag_ids = tag_ids.asnumpy()
            np_text_ids = text_ids.astype('int32').asnumpy()

            for sample_index in range(np_valid_length.shape[0]):
                sample_len = np_valid_length[sample_index]
                entries = []
                for i in range(1, sample_len - 1):
                    token_text = text_vocab.idx_to_token[np_text_ids[sample_index, i]]
                    true_tag = dataset.tag_vocab.idx_to_token[int(np_tag_ids[sample_index, i])]
                    pred_tag = dataset.tag_vocab.idx_to_token[int(np_pred_tags[sample_index, i])]
                    if true_tag == NULL_TAG:
                        last_entry = entries[-1]
                        entries[-1] = (last_entry[0] + token_text, last_entry[1], last_entry[2])
                    else:
                        entries.append((token_text, true_tag, pred_tag))

                entries_list.append(entries)

        all_true_tags = [[entry[1] for entry in entries] for entries in entries_list]
        all_pred_tags = [[entry[2] for entry in entries] for entries in entries_list]
        seqeval_f1 = seqeval.metrics.f1_score(all_true_tags, all_pred_tags)
        return seqeval_f1

    test_f1 = evaluate(test_data_loader)
    logging.info("test f1: {:.3f}".format(test_f1))


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                        level=logging.DEBUG, datefmt='%Y-%m-%d %I:%M:%S')
    logging.getLogger().setLevel(logging.INFO)
    main(parse_args())
