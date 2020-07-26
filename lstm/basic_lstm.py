from typing import Optional, Dict

from allennlp.common.checks import check_dimensions_match
from allennlp.models import Model
from allennlp.data import Vocabulary, TextFieldTensors
from allennlp.modules import FeedForward, Seq2VecEncoder, TextFieldEmbedder
from allennlp.modules import InputVariationalDropout
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import CategoricalAccuracy, F1Measure
from allennlp.nn import util
from torch.nn.modules import Linear, Dropout
import torch
from overrides import overrides
import numpy

@Model.register("lstm_target_classifier")
class LSTMTargetClassifier(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 encoder: Seq2VecEncoder,
                 feedforward: Optional[FeedForward] = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 dropout: float = 0.0,
                 label_name: str = 'target-sentiment-labels') -> None:
        super().__init__(vocab, regularizer)
        '''
        :param vocab: A Vocabulary, required in order to compute sizes 
                      for input/output projections.
        :param embedder: Used to embed the text.
        :param encoder: Encodes the sentence/text. E.g. LSTM
        :param feedforward: An optional feed forward layer to apply after the 
                            encoder
        :param initializer: Used to initialize the model parameters.
        :param regularizer: If provided, will be used to calculate the 
                            regularization penalty during training.
        :param dropout: To apply dropout after each layer apart from the last 
                        layer. All dropout that is applied to timebased data 
                        will be `variational dropout`_ all else will be  
                        standard dropout.
        :param label_name: Name of the label name space.
        
        This is based on the LSTM model by 
        `Tang et al. 2016 <https://www.aclweb.org/anthology/C16-1311.pdf>`_
        
        '''
        self.label_name = label_name
        self.embedder = embedder
        self.encoder = encoder
        self.num_classes = self.vocab.get_vocab_size(self.label_name)
        self.feedforward = feedforward

        if feedforward is not None:
            output_dim = self.feedforward.get_output_dim()
        else:
            output_dim = self.encoder.get_output_dim()
        self.label_projection = Linear(output_dim, self.num_classes)
        
        self.metrics = {
                "accuracy": CategoricalAccuracy()
        }
        self.f1_metrics = {}
        # F1 Scores
        label_index_name = self.vocab.get_index_to_token_vocabulary(self.label_name)
        for label_index, _label_name in label_index_name.items():
            _label_name = f'F1_{_label_name.capitalize()}'
            self.f1_metrics[_label_name] = F1Measure(label_index)
        self._variational_dropout = InputVariationalDropout(dropout)
        self._naive_dropout = Dropout(dropout)
        check_dimensions_match(embedder.get_output_dim(),
                               encoder.get_input_dim(), 'Embedding',
                               'Encoder')
        if self.feedforward is not None:
            check_dimensions_match(encoder.get_output_dim(), 
                                   feedforward.get_input_dim(), 'Encoder', 
                                   'FeedForward')
        initializer(self)

    def forward(self, tokens: TextFieldTensors,
                targets: TextFieldTensors,
                target_sentiments: torch.LongTensor = None,
                metadata: torch.LongTensor = None, **kwargs
                ) -> Dict[str, torch.Tensor]:
        '''
        B = Batch
        NT = Number Targets
        TSL = Target Sequence Length
        '''
        targets_mask = util.get_text_field_mask(targets, num_wrapping_dims=1)
        b, nt, tsl = targets_mask.shape
        #b_nt = b * nt

        # Embedding text and getting mask for the text/context
        text_mask = util.get_text_field_mask(tokens)
        #text_mask.names = ('B', 'CSL')
        embedded_text = self.embedder(tokens)
        embedded_text = self._variational_dropout(embedded_text)
        #embedded_text.names = ('B', 'CSL', 'ET_D')

        encoded_text = self.encoder(embedded_text, text_mask)
        encoded_text = self._naive_dropout(encoded_text)
        logits = self.label_projection(encoded_text)
        # Make the same predictions for all targets in the same sentence
        logits = logits.unsqueeze(1).repeat(1,nt,1)
        label_mask = (targets_mask.sum(dim=-1) >= 1).type(torch.int64)
        # label_mask.names = ('B', 'NT')
        masked_class_probabilities = util.masked_softmax(logits, 
                                                         label_mask.unsqueeze(-1))

        output_dict = {"class_probabilities": masked_class_probabilities, 
                       "targets_mask": label_mask}
        # Convert it to bool tensor.
        label_mask = label_mask == 1

        if target_sentiments is not None:
            # gets the loss per target instance due to the average=`token`
            
            loss = util.sequence_cross_entropy_with_logits(logits, target_sentiments, 
                                                           label_mask, average='token')
            for metrics in [self.metrics, self.f1_metrics]:
                for metric in metrics.values():
                    metric(logits, target_sentiments, label_mask)
            output_dict["loss"] = loss

        if metadata is not None:
            words = []
            texts = []
            meta_targets = []
            target_words = []
            for sample in metadata:
                words.append(sample['text words'])
                texts.append(sample['text'])
                meta_targets.append(sample['targets'])
                target_words.append(sample['target words'])
            output_dict["words"] = words
            output_dict["text"] = texts
            output_dict["targets"] = meta_targets
            output_dict["target words"] = target_words
        return output_dict

    @overrides
    def make_output_human_readable(self, output_dict: Dict[str, torch.Tensor]
                                   ) -> Dict[str, torch.Tensor]:
        '''
        Adds the predicted label to the output dict, also removes any class 
        probabilities that do not have a target associated which is caused 
        through the batch prediction process and can be removed by using the 
        target mask.
        '''
        batch_target_predictions = output_dict['class_probabilities'].cpu().data.numpy()
        target_masks = output_dict['targets_mask'].cpu().data.numpy()
        # Should have the same batch size and max target nubers
        batch_size = batch_target_predictions.shape[0]
        max_number_targets = batch_target_predictions.shape[1]
        assert target_masks.shape[0] == batch_size
        assert target_masks.shape[1] == max_number_targets

        sentiments = []
        non_masked_class_probabilities = []
        for batch_index in range(batch_size):
            target_sentiments = []
            target_non_masked_class_probabilities = []

            target_predictions = batch_target_predictions[batch_index]
            target_mask = target_masks[batch_index]
            for index, target_prediction in enumerate(target_predictions):
                if target_mask[index] != 1:
                    continue
                label_index = numpy.argmax(target_prediction)
                label = self.vocab.get_token_from_index(label_index, 
                                                        namespace=self.label_name)
                target_sentiments.append(label)
                target_non_masked_class_probabilities.append(target_prediction)
            sentiments.append(target_sentiments)
            non_masked_class_probabilities.append(target_non_masked_class_probabilities)
        output_dict['sentiments'] = sentiments
        output_dict['class_probabilities'] = non_masked_class_probabilities
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        # Other scores
        metric_name_value = {}
        for metric_name, metric in self.metrics.items():
            metric_name_value[metric_name] = metric.get_metric(reset)
        # F1 scores
        all_f1_scores = []
        for metric_name, metric in self.f1_metrics.items():
            precision, recall, f1_measure = metric.get_metric(reset)
            all_f1_scores.append(f1_measure)
            metric_name_value[metric_name] = f1_measure
        metric_name_value['Macro_F1'] = sum(all_f1_scores) / len(self.f1_metrics)
        return metric_name_value