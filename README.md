# Span Based AutoEncoder for sentences
This repository implements Span Based AutoEncoder for text data. <br>
Note: WIP <br>
The model is structured as follows: <br>
<i> For each sentence we: </i> 
1) tokenize and embed source words
2) compute contextualized word embeddings by feeding embedded sentence to bidirectional LSTM
3) extract all possible spans of width up to N from the encoder outputs (embeddings from step 2)
4) prune some spans based on FeedForward network scorer that trains end-to-end with the rest of the model <br>
(top K spans are left after this step)
5) try to reconstruct the sentence with decoder LSTM which attends to the remaining spans from the step 4

The code is written using [allennlp](https://github.com/allenai/allennlp) library. Follow the installation procedure from the allennlp website (available via pip) <br>
To run this code simply execute: <br>
`python -m allennlp.run train configs/experiment_gpu.json --serialization-dir models/baseline --include-package span_ae` <br>


