#!/usr/bin/env python
# coding: utf-8
import argparse
import logging

import mxnet as mx

import gluonnlp as nlp

from bert_data import BERTTaggingDataset, NULL_TAG
from bert_model import BERTTagger

# TODO: currently, our evaluation is dependent on this package. figure out whether to take actual dependency on it.
import seqeval.metrics


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


def parse_args():
    """Parse command line arguments."""
    arg_parser = argparse.ArgumentParser(
        description="Train a BERT-based named entity recognition model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # data file paths
    arg_parser.add_argument("--train-path", type=str, required=True,
                            help="Path to the training data file")
    arg_parser.add_argument("--dev-path", type=str, required=True,
                            help="Path to the development data file")
    arg_parser.add_argument("--test-path", type=str, required=True,
                            help="Path to the test data file")

    # bert options
    arg_parser.add_argument("--bert-model", type=str, default="bert_12_768_12",
                            help="Name of the BERT model")
    arg_parser.add_argument("--cased", type=str2bool, default=True,
                            help="Path to the development data file")
    arg_parser.add_argument("--dropout-prob", type=float, default=0.5,
                            help="Dropout probability for the last layer")

    # optimization parameters
    arg_parser.add_argument("--seq-len", type=int, default=180,
                            help="The length of the sequence input to BERT."
                                 " An exception will raised if this is not large enough.")
    arg_parser.add_argument("--gpu", type=int,
                            help='Number (index) of GPU to run on, e.g. 0.  If not specified, uses CPU.')
    arg_parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    arg_parser.add_argument("--num-epochs", type=int, default=10, help="Number of epochs to train")
    arg_parser.add_argument("--optimizer", type=str, default="bertadam", help="Optimization algorithm to use")
    arg_parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate for optimization")
    args = arg_parser.parse_args()
    return args


def main(config):
    if config.cased:
        bert_dataset_name = 'book_corpus_wiki_en_cased'
    else:
        bert_dataset_name = 'book_corpus_wiki_en_uncased'

    ctx = get_context(config.gpu)

    logging.info("Loading BERT model...")
    bert_model, text_vocab = nlp.model.get_bert_model(
        model_name=config.bert_model,
        dataset_name=bert_dataset_name,
        pretrained=True,
        ctx=ctx,
        use_pooler=False,
        use_decoder=False,
        use_classifier=False)

    dataset = BERTTaggingDataset(text_vocab, config.train_path, config.dev_path, config.test_path,
                                 config.seq_len, config.cased)

    train_data_loader = dataset.get_train_data_loader(config.batch_size)
    dev_data_loader = dataset.get_dev_data_loader(config.batch_size)
    test_data_loader = dataset.get_test_data_loader(config.batch_size)

    net = BERTTagger(bert_model, dataset.num_tag_types, config.dropout_prob)
    # TODO: mimicking https://github.com/dmlc/gluon-nlp/pull/493/files#diff-b5de13a7e0607d688f8a38256062d107R271
    # but is there a more natural way to initialize just the parameters added on top of BERT?
    net.tag_classifier.initialize(init=mx.init.Normal(0.02), ctx=ctx)
    net.hybridize(static_alloc=True)

    loss_function = mx.gluon.loss.SoftmaxCrossEntropyLoss()
    loss_function.hybridize(static_alloc=True)

    optimizer_params = {'learning_rate': config.learning_rate}
    try:
        trainer = mx.gluon.Trainer(net.collect_params(), config.optimizer, optimizer_params)
    except ValueError as e:
        print(e)
        logging.warning('AdamW optimizer is not found. Please consider upgrading to '
                        'mxnet>=1.5.0. Now the original Adam optimizer is used instead.')
        trainer = mx.gluon.Trainer(net.collect_params(), 'adam', optimizer_params)

    # collect differentiable parameters
    logging.info("Collect params...")
    # do not apply weight decay on LayerNorm and bias terms
    for _, v in net.collect_params('.*beta|.*gamma|.*bias').items():
        v.wd_mult = 0.0
    params = [p for p in net.collect_params().values() if p.grad_req != 'null']

    def train(data_loader):
        for batch_id, data in enumerate(data_loader):
            logging.info("training on batch index: {}/{}".format(batch_id, len(data_loader)))

            text_ids, token_types, valid_length, tag_ids, flag_nonnull_tag = \
                [x.astype('float32').as_in_context(ctx) for x in data]

            with mx.autograd.record():
                out = net(text_ids, token_types, valid_length)
                loss_value = loss_function(out, tag_ids, flag_nonnull_tag.expand_dims(axis=2)).mean()

            loss_value.backward()
            nlp.utils.clip_grad_global_norm(params, 1)
            trainer.step(1)

            pred_tags = out.argmax(axis=-1)
            logging.info("loss_value: {:.6f}".format(loss_value.asscalar()))

            num_tag_preds = flag_nonnull_tag.sum().asscalar()
            logging.info(
                "accuracy: {:.6f}".format(((pred_tags == tag_ids) * flag_nonnull_tag).sum().asscalar() / num_tag_preds))

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

    best_dev_f1 = 0.0
    best_epoch = -1

    for epoch_index in range(config.num_epochs):
        train(train_data_loader)
        train_f1 = evaluate(train_data_loader)
        logging.info("train f1: {.3f}".format(train_f1))
        dev_f1 = evaluate(dev_data_loader)
        logging.info("dev f1: {.3f}, previous best dev f1: {.3f}".format(train_f1, best_dev_f1))
        if dev_f1 > best_dev_f1:
            logging.info("update the best dev f1 to be: {.3f}".format(best_dev_f1))
            best_dev_f1 = dev_f1
            best_epoch = epoch_index
            test_f1 = evaluate(test_data_loader)
            logging.info("test f1: {.3f}".format(test_f1))
        logging.info("current best epoch: {:d}".format(best_epoch))


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                        level=logging.DEBUG, datefmt='%Y-%m-%d %I:%M:%S')
    logging.getLogger().setLevel(logging.INFO)
    main(parse_args())
