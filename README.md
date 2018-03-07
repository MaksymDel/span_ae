# Span Based AutoEncoder for sentences
This repository implements Span Based AutoEncoder for text data. <br>
Note: WIP <br>
The model is structured as follows: <br>
<i> For each sentence we: </i> 
1) tokenize and embed source words
2) embed individual words using usual Embedding layer
3) compute contextualized word embeddings by feeding embedded sentence to bidirectional LSTM
4) extract all possible spans of width up to N from the encoder outputs (embeddings from step 3)
5) prune some spans based on FeedForward network scorer that trains end-to-end with the rest of the model <br>
(top K spans are left after this step)
6) try to reconstruct the sentence with decoder LSTM which attends to the remaining spans from the step 5

The code is written using [allennlp](https://github.com/allenai/allennlp) library. Follow the installation procedure from the allennlp website (available via pip) <br>
To run this code simply execute: <br>
`python -m allennlp.run train tests/fixtures/experiment.json --serialization-dir models/dry --include-package span_ae` <br>


