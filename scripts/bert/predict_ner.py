#!/usr/bin/env python
# coding: utf-8
import argparse
import logging

import os

from ner_common import *
from ner_data import BERTTaggingDataset, convert_arrays_to_text
from ner_model import BERTTagger

# TODO: currently, our evaluation is dependent on this package. figure out whether to take actual dependency on it.
import seqeval.metrics


def _find_model_file_from_checkpoint(checkpoint_prefix: str):
    dirname, file_prefix = os.path.split(checkpoint_prefix)
    # find checkpoint file names and sort by name to find the most recent one.
    checkpoint_filenames = ([f for f in os.listdir(dirname)
                             if f.startswith(file_prefix)
                             and f.endswith(os.path.extsep + 'params')])
    last_checkpoint_filename = max(checkpoint_filenames)
    logging.info('found checkpoint filename: {:s}'.format(last_checkpoint_filename))
    last_checkpoint_path = os.path.join(dirname, last_checkpoint_filename)
    return last_checkpoint_path


def parse_args():
    """Parse command line arguments."""
    arg_parser = argparse.ArgumentParser(
        description='Predict on CoNLL format data using BERT-based named entity recognition model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # data file paths
    arg_parser.add_argument('--test-path', type=str, required=True,
                            help='Path to the test data file')
    arg_parser.add_argument('--seq-len', type=int, default=200,
                            help='The length of the sequence input to BERT.'
                                 ' An exception will raised if this is not large enough.')
    arg_parser.add_argument("--load-checkpoint-prefix", type=str, required=False, default=None,
                            help="Prefix of model checkpoint file")

    arg_parser.add_argument("--gpu", type=int,
                            help='Number (index) of GPU to run on, e.g. 0.  If not specified, uses CPU.')
    arg_parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    args = arg_parser.parse_args()
    return args


def main(config):
    """Main method for predicting BERT-based NER model on CoNLL-formatted test data."""
    train_config, tag_vocab = load_metadata(config.load_checkpoint_prefix)

    ctx = get_context(config.gpu)
    bert_model, text_vocab = get_bert_model(train_config.bert_model, train_config.cased, ctx, train_config.dropout_prob)

    dataset = BERTTaggingDataset(text_vocab, None, None, config.test_path,
                                 config.seq_len, train_config.cased, tag_vocab=tag_vocab)

    test_data_loader = dataset.get_test_data_loader(config.batch_size)

    net = BERTTagger(bert_model, dataset.num_tag_types, train_config.dropout_prob)
    model_filename = _find_model_file_from_checkpoint(config.load_checkpoint_prefix)
    net.load_parameters(model_filename, ctx=ctx)

    net.hybridize(static_alloc=True)

    loss_function = mx.gluon.loss.SoftmaxCrossEntropyLoss()
    loss_function.hybridize(static_alloc=True)

    # TODO: make it not redundant between train and predict
    def evaluate(data_loader):
        predictions = []

        for batch_id, data in enumerate(data_loader):
            logging.info('evaluating on batch index: {}/{}'.format(batch_id, len(data_loader)))
            text_ids, token_types, valid_length, tag_ids, flag_nonnull_tag = \
                [x.astype('float32').as_in_context(ctx) for x in data]
            out = net(text_ids, token_types, valid_length)

            # convert results to numpy arrays for easier access
            np_text_ids = text_ids.astype('int32').asnumpy()
            np_pred_tags = out.argmax(axis=-1).asnumpy()
            np_valid_length = valid_length.astype('int32').asnumpy()
            np_true_tags = tag_ids.asnumpy()

            predictions += convert_arrays_to_text(text_vocab, dataset.tag_vocab, np_text_ids,
                                                  np_true_tags, np_pred_tags, np_valid_length)

        all_true_tags = [[entry.true_tag for entry in entries] for entries in predictions]
        all_pred_tags = [[entry.pred_tag for entry in entries] for entries in predictions]
        seqeval_f1 = seqeval.metrics.f1_score(all_true_tags, all_pred_tags)
        return seqeval_f1

    test_f1 = evaluate(test_data_loader)
    logging.info('test f1: {:.3f}'.format(test_f1))


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                        level=logging.DEBUG, datefmt='%Y-%m-%d %I:%M:%S')
    logging.getLogger().setLevel(logging.INFO)
    main(parse_args())