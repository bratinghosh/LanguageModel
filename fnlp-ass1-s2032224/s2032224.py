"""
Foundations of Natural Language Processing
Assignment 1: Corpora Analysis and Language Identification

Please complete functions, based on their doc_string description
and instructions of the assignment. 

To test your code run:

```
[hostname]s1234567 python3 s1234567.py
```

Before submittion executed your code with ``--answers`` flag
```
[hostname]s1234567 python3 s1234567.py --answers
```
include generated answers.py file and generated plots to
your submission.

Best of Luck!
"""

import numpy as np  # for np.mean() and np.std()
import nltk, sys, inspect
import nltk.corpus.util
from nltk.corpus import inaugural, brown  # import corpora
from nltk.corpus import stopwords  # stopword list
from nltk import FreqDist

# Import the Twitter corpus and LgramModel
try:
    from nltk_model import *  # See the README inside the nltk_model folder for more information
    from twitter import *
except ImportError:
    from .nltk_model import *  # Compatibility depending on how this script was run
    from .twitter import *


# Override default plotting function in matplotlib so that your plots would be saved during the execution
from matplotlib import pylab, pyplot
plot_enum = 0

def my_show(**kwargs):
    global plot_enum
    plot_enum += 1
    return pylab.savefig('plot_{}.png'.format(plot_enum))

pylab.show=my_show
pyplot.show=my_show

twitter_file_ids = "20100128.txt"
assert twitter_file_ids in xtwc.fileids()

# Helper function to get tokens of corpus containing multiple files
def get_corpus_tokens(corpus, list_of_files):
    '''Get the tokens from (part of) a corpus

    :type corpus: nltk.corpus.CorpusReader
    :param corpus: An NLTK corpus
    :type list_of_files: list(str) or str
    :param list_of_files: file(s) to read from
    :rtype: list(str)
    :return: the tokenised contents of the files'''

    # Get a list of all tokens in the corpus
    corpus_tokens = corpus.words(list_of_files)
    # Return the list of corpus tokens
    return corpus_tokens


# =======================================
# Section A: Corpora Analysis [45 marks]
# =======================================

# Question 1 [5 marks]
def avg_type_length(corpus, list_of_files):
    '''
    Compute the average word type length from (part of) a corpus
    specified in list_of_files

    :type corpus: nltk.corpus.CorpusReader
    :param corpus: An NLTK corpus
    :type list_of_files: list(str) or str
    :param list_of_files: file(s) to read from
    :rtype: float
    :return: the average word type length over all the files
    '''

    # Get a list of all tokens in the corpus
    tokens = get_corpus_tokens(corpus, list_of_files)

    # Construct a list that contains the token lengths for each DISTINCT token in the document
    distinct_token_lengths = [len(elem) for elem in set(token.lower() for token in tokens)]
    
    # Return the average type length of the document
    return  sum(distinct_token_lengths)/len(distinct_token_lengths)

# Question 2 [5 marks]
def open_question_1():
    '''
    Question: Why might the average type length be greater for twitter data?

    :rtype: str
    :return: your answer'''

    return inspect.cleandoc("""
    The twitter corpus consists of tweets in various languages. A set of multiple characters/sentences in other languages are read as a single word by nltk. We can also find casual text message sequences like 'hahahaha' or 'wasssssup' which have long word lengths.Add to that, website-links, long hashtags and user-tags cause an increase in the average word type length of the Twitter corpus.
    Example:
    http://ow.ly/16rdMR is read as a 19 character length word.
    """)[0:500]

# Question 3 [10 marks]
def plot_frequency(tokens, topK=50):
    '''
    Tabulate and plot the top x most frequently used word types
    and their counts from the specified list of tokens

    :type tokens: list(str) 
    :param tokens: List of tokens
    :type topK: int
    :param number of top tokens to plot and return as a result
    :rtype: list(tuple(string,int))
    :return: top K word types and their counts from the files
    '''

    # Construct a frequency distribution over the lowercased tokens in the document
    fd_doc_types = FreqDist(w.lower() for w in tokens)

    # Find the top topK most frequently used types in the document
    top_types = fd_doc_types.most_common(topK)

    # Produce a plot showing the top topK types and their frequencies
    fd_doc_types.plot(topK)
    # Return the top topK most frequently used types
    return top_types

# Question 4 [15 marks]
def clean_data(corpus_tokens):
    '''
    Clean a list of corpus tokens

    :type corpus_tokens: list(str)
    :param corpus_tokens: corpus tokens
    :rtype: list(str)
    :return: cleaned list of corpus tokens
    '''

    stops = list(stopwords.words("english"))
    clean_corpus_tokens = []
    # A token is 'clean' if it's alphanumeric and NOT in the list of stopwords
    for token in corpus_tokens:
        if token.isalnum() and (token.lower() not in stops):
            clean_corpus_tokens.append(token.lower())
    # Return a cleaned list of corpus tokens
    return clean_corpus_tokens

# Question 5 [10 marks]
def open_question_2():
    '''
    Problem: noise in Twitter data

    :rtype: str
    :return: your answer []
    '''
    return inspect.cleandoc("""
    The non-english words and characters should be removed to clean the corpus. This can be done by checking if the letters in the words are not english alphabets. Add to that, we can also check whether that words exists in the dictionary so as to make sure the words in the cleaned corpus are actual english words.
    Example:
    Dá will be eliminated as it contains á, which is not an english alphabet.
    jajajaja will be eliminated as it is not present in the english dictionary.
    """)[0:500]

# ==============================================
# Section B: Language Identification [45 marks]
# ==============================================

# Question 6 [15 marks]
def train_LM(corpus):
    '''
    Build a bigram letter language model using LgramModel
    based on the all-alpha subset the entire corpus

    :type corpus: nltk.corpus.CorpusReader
    :param corpus: An NLTK corpus
    :rtype: LgramModel
    :return: A padded letter bigram model based on nltk.model.NgramModel
    '''

    # subset the corpus to only include all-alpha tokens
    corpus_tokens= []
    for word in corpus.words():
        if word.isalpha():
            corpus_tokens.append(word.lower())

    # Return a smoothed padded bigram letter language model
    return LgramModel(2, corpus_tokens, pad_left=True, pad_right=True)

# Question 7 [15 marks]
def tweet_ent(file_name,bigram_model):
    '''
    Using a character bigram model, compute sentence entropies
    for a subset of the tweet corpus, removing all non-alpha tokens and
    tweets with less than 5 all-alpha tokens

    :type file_name: str
    :param file_name: twitter file to process
    :rtype: list(tuple(float,list(str)))
    :return: ordered list of average entropies and tweets
    '''

    # Clean up the tweet corpus to remove all non-alpha 
    # # tokens and tweets with less than 5 (remaining) tokens
    list_of_tweets = xtwc.sents(file_name)
    cleaned_list_of_tweets = []
    for tweet in list_of_tweets:
        clean_tweet = []
        for token in tweet:
            if token.isalpha():
                clean_tweet.append(token.lower())
        if len(clean_tweet) >= 5:
            cleaned_list_of_tweets.append(clean_tweet)

    # Construct a list of tuples of the form: (entropy,tweet)
    #  for each tweet in the cleaned corpus, where entropy is the
    #  average word for the tweet, and return the list of
    #  (entropy,tweet) tuples sorted by entropy
    
    entropy_tweet = []
    for tweet in cleaned_list_of_tweets:
        entropy = 0
        for token in tweet:
            entropy = entropy + bigram_model.entropy(token.lower(), pad_left=True, pad_right=True, perItem=True)
        avg_entropy = entropy/len(tweet)
        entropy_tweet.append((avg_entropy,tweet))
    entropy_tweet.sort(key=lambda tup: tup[0])
    return entropy_tweet
# Question 8 [10 marks]
def open_question_3():
    '''Question: What differentiates the beginning and end of the list
       of tweets and their entropies?

    :rtype: str
    :return: your answer [500 chars max]'''

    return inspect.cleandoc("""
    The tweets at the beginning of the list comprises of letters only. The tweets at the end of the list are long, consisting of many words which are several letters long. Add to that, the low entropy tweets primarily comprises of English letters while the high entropy tweets, at the end of the list are in various languages which are very different from English.
    """)[0:500]


# Question 9 [15 marks]
def tweet_filter(list_of_tweets_and_entropies):
    '''
    Compute entropy mean, standard deviation and using them,
    likely non-English tweets in the all-ascii subset of list 
    of tweetsand their biletter entropies

    :type list_of_tweets_and_entropies: list(tuple(float,list(str)))
    :param list_of_tweets_and_entropies: tweets and their
                                    english (brown) average biletter entropy
    :rtype: tuple(float, float, list(tuple(float,list(str)))
    :return: mean, standard deviation, ascii tweets and entropies,
             not-English tweets and entropies
    '''

    # Find the "ascii" tweets - those in the lowest-entropy 90%
    #  of list_of_tweets_and_entropies
    list_of_ascii_tweets_and_entropies = list_of_tweets_and_entropies[0:int(0.9*len(list_of_tweets_and_entropies))]
    # Extract a list of just the entropy values
    list_of_entropies = []
    for entropy, tweet in list_of_ascii_tweets_and_entropies:
        list_of_entropies.append(entropy)

    # Compute the mean of entropy values for "ascii" tweets
    mean = np.mean(list_of_entropies)

    # Compute their standard deviation
    standard_deviation = np.std(list_of_entropies)

    # Get a list of "probably not English" tweets, that is
    #  "ascii" tweets with an entropy greater than (mean + std_dev))
    threshold = mean + standard_deviation
    list_of_not_English_tweets_and_entropies = []
    for entropy, tweet in list_of_ascii_tweets_and_entropies:
        if entropy > threshold:
            list_of_not_English_tweets_and_entropies.append((entropy, tweet))
    # Return mean, standard_deviation,
    #  list_of_ascii_tweets_and_entropies,
    #  list_of_not_English_tweets_and_entropies
    return mean, standard_deviation, list_of_ascii_tweets_and_entropies, list_of_not_English_tweets_and_entropies

# Utility function
def ppEandT(eAndTs):
    '''
    Pretty print a list of entropy-tweet pairs

    :type eAndTs: list(tuple(float,list(str)))
    :param eAndTs: entropies and tweets
    :return: None
    '''

    for entropy,tweet in eAndTs:
        print("{:.3f} [{}]".format(entropy,", ".join(tweet)))

"""
Format the output of your submission for both development and automarking. 
DO NOT MODIFY THIS PART !
""" 
def answers():

    # Global variables for answers that will be used by automarker
    global avg_inaugural, avg_twitter, top_types_inaugural, top_types_twitter
    global tokens_inaugural, tokens_clean_inaugural, total_tokens_clean_inaugural
    global fst100_clean_inaugural, top_types_clean_inaugural
    global tokens_twitter, tokens_clean_twitter, total_tokens_clean_twitter
    global fst100_clean_twitter, top_types_clean_twitter, lm
    global best10_ents, worst10_ents, mean, std, best10_ascci_ents, worst10_ascci_ents
    global best10_non_eng_ents, worst10_non_eng_ents
    global answer_open_question_1, answer_open_question_2, answer_open_question_3

    # Question 1
    print("*** Question 1 ***")
    avg_inaugural = avg_type_length(inaugural, inaugural.fileids())
    avg_twitter = avg_type_length(xtwc, twitter_file_ids)
    print("Average token length for Inaugural corpus: {:.2f}".format(avg_inaugural))
    print("Average token length for Twitter corpus: {:.2f}".format(avg_twitter))
    
    # Question 2
    print("*** Question 2 ***")
    answer_open_question_1 = open_question_1()
    print(answer_open_question_1)

    # Question 3
    print("*** Question 3 ***")
    print("Most common 50 types for the Inaugural corpus:")
    tokens_inaugural = get_corpus_tokens(inaugural,inaugural.fileids())
    top_types_inaugural = plot_frequency(tokens_inaugural,50)
    print(top_types_inaugural)
    print("Most common 50 types for the Twitter corpus:")
    tokens_twitter = get_corpus_tokens(xtwc, twitter_file_ids)
    top_types_twitter = plot_frequency(tokens_twitter,50)
    print(top_types_twitter)

    # Question 4
    print("*** Question 4 ***")
    tokens_inaugural = get_corpus_tokens(inaugural,inaugural.fileids())
    tokens_clean_inaugural = clean_data(tokens_inaugural)
    total_tokens_inaugural = len(tokens_inaugural)
    total_tokens_clean_inaugural = len(tokens_clean_inaugural)
    print("Inaugural Corpus:")
    print("Number of tokens in original corpus: {}".format(total_tokens_inaugural))
    print("Number of tokens in cleaned corpus: {}".format(total_tokens_clean_inaugural))
    print("First 100 tokens in cleaned corpus:")
    fst100_clean_inaugural = tokens_clean_inaugural[:100]
    print(fst100_clean_inaugural)
    print("Most common 50 types for the cleaned corpus:")
    top_types_clean_inaugural = plot_frequency(tokens_clean_inaugural,50)
    print(top_types_clean_inaugural)

    print('-----------------------')
    
    tokens_twitter = get_corpus_tokens(xtwc,twitter_file_ids)
    tokens_clean_twitter = clean_data(tokens_twitter)
    total_tokens_twitter = len(tokens_twitter)
    total_tokens_clean_twitter = len(tokens_clean_twitter)
    print("Twitter Corpus:")
    print("Number of tokens in original corpus: {}".format(total_tokens_twitter))
    print("Number of tokens in cleaned corpus: {}".format(total_tokens_clean_twitter))
    print("First 100 tokens in cleaned corpus:")
    fst100_clean_twitter = tokens_clean_twitter[:100]
    print(fst100_clean_twitter)
    print("Most common 50 types for the cleaned corpus:")
    top_types_clean_twitter = plot_frequency(tokens_clean_twitter,50)
    print(top_types_clean_twitter)
    # Question 5
    print("*** Question 5 ***")
    answer_open_question_2 = open_question_2()
    print(answer_open_question_2)
    
    # Question 6
    print("*** Question 6 ***")
    print('Building brown bigram letter model ... ')
    lm = train_LM(brown)
    print('Letter model buid')

    # Question 7
    print("*** Question 7 ***")
    ents = tweet_ent(twitter_file_ids, lm)
    print("Best 10 english entropies:")
    best10_ents = ents[:10]
    ppEandT(best10_ents)
    print("Worst 10 english entropies:")
    worst10_ents = ents[-10:]
    ppEandT(worst10_ents)

    # Question 8
    print("*** Question 8 ***")
    answer_open_question_3 = open_question_3()
    print(answer_open_question_3)

    # Question 9
    print("*** Question 9 ***")
    mean, std, ascci_ents, non_eng_ents = tweet_filter(ents)
    print('Mean: {}'.format(mean))
    print('Standard Deviation: {}'.format(std))
    print('ASCII tweets ')
    print("Best 10 English entropies:")
    best10_ascci_ents = ascci_ents[:10]
    ppEandT(best10_ascci_ents)
    print("Worst 10 English entropies:")
    worst10_ascci_ents = ascci_ents[-10:]
    ppEandT(worst10_ascci_ents)
    print('--------')
    print('Non-English tweets ')
    print("Best 10 English entropies:")
    best10_non_eng_ents = non_eng_ents[:10]
    ppEandT(best10_non_eng_ents)
    print("Worst 10 English entropies:")
    worst10_non_eng_ents = non_eng_ents[-10:]
    ppEandT(worst10_non_eng_ents)

if __name__ == "__main__":

    answers()

    if len(sys.argv)>1 and sys.argv[1] == '--answers':
        ans=['avg_inaugural', 'avg_twitter',
             ('answer_open_question_1', repr(answer_open_question_1)),
             'top_types_inaugural', 'top_types_twitter',
             'total_tokens_clean_inaugural',
             'fst100_clean_inaugural', 'top_types_clean_inaugural',
             'total_tokens_clean_twitter',
             'fst100_clean_twitter','top_types_clean_twitter',
             ('answer_open_question_2',repr(answer_open_question_2)),
             ('lm_stats',[lm._N,
                         lm.prob('h','t'),
                         lm.prob('u','q'),
                         lm.prob('z','q'),
                         lm.prob('j',('<s>',),True),
                         lm.prob('</s>','e',True)]),
             'best10_ents','worst10_ents',
             ('answer_open_question_3', repr(answer_open_question_3)),
             'mean','std',
             'best10_ascci_ents', 'worst10_ascci_ents',
             'best10_non_eng_ents', 'worst10_non_eng_ents']

        ansd={}
        for an in ans:
            if type(an) is tuple:
                ansd[an[0]]=an[1]
            else:
                ansd[an]=eval(an)
        # dump for automarker
        with open('answers.py',"w") as f:
            for aname,aval in ansd.items():
                print("{}={}".format(aname,aval),file=f)
