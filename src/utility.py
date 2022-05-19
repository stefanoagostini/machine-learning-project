import torch
from torch.autograd import Variable


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


def colorize(r, g, b, text):
    return "\033[38;2;{};{};{}m{}\033[38;2;255;255;255m".format(r, g, b, text)


""" utile per generare una funzione per colorare le parole di un testo di colori diversi"""
# define aliases to the color-codes


def test():
    t = "That was one hell of a show for a one man band!"
    utterances = t.split()

    if "one" in utterances:
        # modificare questa parte per cambiare il valore in base a un valore tra 0 e 1
        # figure out the list-indices of occurences of "one"
        idxs = [i for i, x in enumerate(utterances) if x == "one"]

        # modify the occurences by wrapping them in ANSI sequences
        for i in idxs:
            utterances[i] = colorize(0, 255, 0, utterances[i])

    # join the list back into a string and print
    utterances = " ".join(utterances)
    print(utterances)
