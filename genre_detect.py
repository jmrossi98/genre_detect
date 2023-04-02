import argparse

class GenreClassifier:

    def __init__(self):
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Give a path to an audio file...')
    parser.add_argument(
        '--path', '-p',
        help="Path to file you'd like to classify"
    )
    classifier = GenreClassifier()
