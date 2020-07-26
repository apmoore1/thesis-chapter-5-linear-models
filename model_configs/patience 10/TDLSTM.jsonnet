{
    "dataset_reader": {
      "type": "target_sentiment",
      "token_indexers": {
        "tokens": {
          "type": "single_id",
          "lowercase_tokens": true,
          "token_min_padding_length": 1
        }
      },
      "left_right_contexts": true,
      "reverse_right_context": true,
      "incl_target": true
    },
    "model": {
      "type": "split_contexts_classifier",
      "context_field_embedder": {
          "token_embedders" :{
          "tokens": {
            "type": "embedding",
            "embedding_dim": 10,
            "trainable": true
          }
        }
      },
      "left_text_encoder": {
        "type": "lstm",
        "input_size": 10,
        "hidden_size": 20,
        "bidirectional": false,
        "num_layers": 1
      },
      "right_text_encoder": {
        "type": "lstm",
        "input_size": 10,
        "hidden_size": 20,
        "bidirectional": false,
        "num_layers": 1
      }
    },
    "data_loader": {
    "batch_size": 32,
    "shuffle": true,
    "drop_last": false
    },
    "trainer": {
      "optimizer": {
        "type": "sgd",
        "lr": 0.01
      },
      "num_epochs": 300,
      "cuda_device": 0,
      "patience": 10,
      "validation_metric": "-loss"
    }
  }