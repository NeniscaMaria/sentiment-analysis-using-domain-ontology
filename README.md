# Aspect-weighted sentiment analysis using domain ontology and deep neural network

We calculate aspect scores using two approaches:
1. conditional probability from dataset
2. a domain ontology

We incorporate these scores into our neural architecture to find the sentiment of a textual review. The scores are used to initialize a trainable layer of the neural architecture.

> **Domain:** Music
>
> To add any new domain, create folders inside dataset, models, and ontology folder with the name of the new domain (ex: movie/restaurant)

### Steps to run
1. Run `create_data.py` script under `dataset/music` folder.
2. Run the embedding scripts from the `embeddings` folder if corresponding pickle files are not created.
3. Run preprocessing.py if corresponding pickle files are not created.
4. Run model.py.
5. You change the experimental setup values to match the experiment you want to be conducted.

P.S: Read the commented introduction in each of the files mentioned to run the commands correctly.