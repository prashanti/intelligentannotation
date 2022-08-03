## Knowledge of the Ancestors
#### Intelligent Ontology-aware Annotation of Biological Literature using Semantic Similarity

The goal of intelligent concept annotation is to maximize accurate retrieval rate and subsequently maximize partial accuracy in cases where complete accuracy is not achieved, thereby increasing overall accuracy.

If a model predicts an exact ontology concept, then the retrieval is considered accurate. If the model predicts/ retrieves an alternate ontology concept which is semantically close to the true concept, the retrieval is considered partially accurate. If the predicted alternate ontology concept is not semantically similar to the true concept, the retrieval is considered inaccurate.

Here, we present intelligent deep learning architectures that are ontology-aware and use the hierarchies embedded in the ontology to improve concept prediction accuracy. We use the Colorado Richly Annotated Full Text Corpus (CRAFT) as a gold standard for training and testing. We enrich the dataset with parts-of-speech information and character sequences in each word in the training corpus.

**Our best model results in a 0.81 F1 and 0.84 semantic similarity**.
#### Deep Learning Architecture
[This](./data/model_output/new_architecture.svg) figure represents our deep learning architecture based on Bidirectional GRU (Bi-GRU) which consists of three major components:
1. Input pipelines
2. Embeddings/Latent Representations and
3. Sequence Modeler.


In order to replicate or re-run the experiments as described in [this](https://github.com/prashanti/intelligentannotation) paper, clone this repository.

#### To perform model training with available train and test dataset

1. Install the required python packages by running the command:

  ```
  pip install -r requirements.txt
  ```
2. `src` directory contains three python scripts for model training using CRAFT, GloVe and ELMo embeddings. The embedding type is defined in the filename itself. Additionally, there is a `notebook` directory inside `src` in case you prefer running .ipynb files instead of .py files.

3. Once the script starts, you will see a corresponding log file. When the script completes, you should see the results towards the end of the log.

4. The `result` directory includes a .txt file with the results from the paper.

**Note:** These experiments are carried out on single **Tesla V100-SXM2-16GB** GPU and **8 core** CPU with **51.01 GB RAM**.

### Support or Contact

Email: &emsp;p_devkota@uncg.edu
&emsp;&emsp;&emsp;&emsp;sdmohant@uncg.edu
&emsp;&emsp;&emsp;&emsp;p_manda@uncg.edu
