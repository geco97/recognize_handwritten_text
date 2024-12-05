import os
import logging
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import argparse


def train_word2vec(corpus_file: str, model_output: str, vector_size: int = 100, window: int = 5, min_count: int = 1, epochs: int = 10):
    """
    Train a Word2Vec model on the given corpus.

    :param corpus_file: Path to the text corpus file.
    :param model_output: Path to save the trained model.
    :param vector_size: Size of the word vectors.
    :param window: Maximum distance between the current and predicted word.
    :param min_count: Ignores words with total frequency lower than this.
    :param epochs: Number of iterations over the corpus.
    """
    logging.info(f"Loading corpus from {corpus_file}...")
    sentences = LineSentence(corpus_file)

    logging.info("Training Word2Vec model...")
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4,
    )

    logging.info(f"Training for {epochs} epochs...")
    model.train(sentences, total_examples=model.corpus_count, epochs=epochs)

    logging.info(f"Saving model to {model_output}...")
    model.save(model_output)
    logging.info("Model training and saving complete!")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a Word2Vec model on a text corpus.")
    parser.add_argument("corpus", type=str, help="Path to the input text corpus file.")
    parser.add_argument("output", type=str, help="Path to save the trained Word2Vec model.")
    parser.add_argument("--vector_size", type=int, default=100, help="Size of the word vectors (default: 100).")
    parser.add_argument("--window", type=int, default=5, help="Window size for context (default: 5).")
    parser.add_argument("--min_count", type=int, default=1, help="Minimum frequency of words to include (default: 1).")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (default: 10).")
    return parser.parse_args()


def main():
    logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

    args = parse_arguments()
    if not os.path.exists(args.corpus):
        logging.error(f"Corpus file {args.corpus} does not exist!")
        return

    train_word2vec(
        corpus_file=args.corpus,
        model_output=args.output,
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count,
        epochs=args.epochs,
    )


if __name__ == "__main__":
    main()
