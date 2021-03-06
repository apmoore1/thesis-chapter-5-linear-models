{
    "dataset_reader": {
      "type": "target_sentiment",
      "token_indexers": {
        "tokens": {
          "type": "single_id",
          "lowercase_tokens": true,
          "token_min_padding_length": 1
        }
      }
    },
    "model": {
      "type": "lstm_target_classifier",
      "embedder": {
          "token_embedders" :{
          "tokens": {
            "type": "embedding",
            "embedding_dim": 10,
            "trainable": true
          }
        }
      },
      "encoder": {
        "type": "lstm",
        "input_size": 10,
        "hidden_size": 10,
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
      "patience": 5,
      "validation_metric": "-loss"
    }
  }