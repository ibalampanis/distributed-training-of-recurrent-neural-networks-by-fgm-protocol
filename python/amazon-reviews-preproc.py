# Import modules that we need
from collections import Counter
from string import punctuation
from wordcloud import WordCloud
import numpy as np
import pandas as pd


# Let's create a vocabulary reduction function in case we want to reduce the vocabulary based on min frequency or
# polarity.
def vocabulary_reduction(reviews, min_freq=10, polarity_cut_off=0.1):
    pos_count = Counter()
    neg_count = Counter()
    tot_count = Counter()

    for i in range(len(reviews)):
        for word in reviews[i].split():
            tot_count[word] += 1
            if labels[i] == 1:
                pos_count[word] += 1
            else:
                neg_count[word] += 1

                # Identify words with frequency greater than min_freq
    vocab_freq = []
    for word in tot_count.keys():
        if tot_count[word] > min_freq:
            vocab_freq.append(word)

            # Use polarity to reduce vocab
    pos_neg_ratio = Counter()
    vocab_pos_neg = (set(pos_count.keys())).intersection(set(neg_count.keys()))
    for word in vocab_pos_neg:
        if tot_count[word] > 100:
            ratio = pos_count[word] / float(neg_count[word] + 1)
            if ratio > 1:
                pos_neg_ratio[word] = np.log(ratio)
            else:
                pos_neg_ratio[word] = -np.log(1 / (ratio + 0.01))

    mean_ratio = np.mean(list(pos_neg_ratio.values()))

    vocab_polarity = []
    for word in pos_neg_ratio.keys():
        if (pos_neg_ratio[word] < (mean_ratio - polarity_cut_off)) or (
                pos_neg_ratio[word] > (mean_ratio + polarity_cut_off)):
            vocab_polarity.append(word)

    vocab_rm_polarity = set(pos_neg_ratio.keys()).difference(vocab_polarity)
    vocab_reduced = (set(vocab_freq)).difference(set(vocab_rm_polarity))

    reviews_cleaned = []

    for review in reviews:
        review_temp = [word for word in review.split() if word in vocab_reduced]
        reviews_cleaned.append(' '.join(review_temp))

    return reviews_cleaned


# Once we have the vocabulary_to_int dictionary created, we can transform each review into a list of integers using
# the dictionary previously created.
def reviews_to_integers(reviews):
    reviews_to_int = []
    for i in range(len(reviews)):
        to_int = [vocabulary_to_int[word] for word in reviews[i].split()]
        reviews_to_int.append(to_int)
    return reviews_to_int


if __name__ == '__main__':

    data = pd.read_csv('../datasets/Reviews.csv')

    # The dataset contains 568,454 food reviews Amazon users left from October 1999 to October 2012.
    # What are the fields provided in the Amazon fine food reviews dataset?
    data.head()

    # The review for a specific product by a specific user is stored in the Text column. The Text column consists of an
    # unstructured text. In addition to the Text feature representing the review, we can identify the product using the
    # ProductId feature, the Amazon user with the UserID feature, the date the review was posted by the Time column,
    # and a brief summary of the review with the Summary variable. Additionally, HelpfulnessNumerator and
    # HelpfulnessDenominator, two variables representing how helpful a review is, are also provided in the review. More
    # importantly, for each review, we have a “Score” variable representing a rating from 1 to 5 (1 is a poor review,
    # and 5 is an excellent review).

    # ### Sentiment Score

    # As we can observe, the majority of Scores are equal to 4 and 5, and with an average score of 4.18. Because of the
    # distribution is very skewed to the left, we will make a binary prediction. We can consider a negative review will
    # have a Score between 1 and 3, and a positive review will have a Score equal to 4 or 5.
    data.ix[data.Score > 3, 'Sentiment'] = "POSITIVE"
    data.ix[data.Score <= 3, 'Sentiment'] = "NEGATIVE"

    # With this new classification, 78% of the fine food reviews are considered as positive and 22% of them are
    # considered as negative.
    # As we only need the Text field as an input for our model and the Sentiment (positive or negative) as an ouput,
    # we will store this information in a reviews and sentiment variables.
    reviews = data.Text.values
    labels = data.Sentiment.values

    # ### Exploratory Visualization
    # A common approach to distinguish positive from negative reviews is to look at the frequency of the words. We can
    # imagine that certain words, such as "excellent" or "very tasty," tend to occur more often in positive reviews than
    # negative reviews. Let's see if we can validate this theory.
    positive_reviews = [reviews[i] for i in range(len(reviews)) if labels[i] == "POSITIVE"]
    negative_reviews = [reviews[i] for i in range(len(reviews)) if labels[i] == "NEGATIVE"]

    cnt_positve = Counter()

    for row in positive_reviews:
        cnt_positve.update(row.split(" "))

    cnt_negative = Counter()

    for row in negative_reviews:
        cnt_negative.update(row.split(" "))

    cnt_total = Counter()

    for row in reviews:
        cnt_total.update(row.split(" "))

    pos_neg_ratio = Counter()
    vocab_pos_neg = (set(cnt_positve.keys())).intersection(set(cnt_negative.keys()))

    for word in vocab_pos_neg:
        if cnt_total[word] > 100:
            ratio = cnt_positve[word] / float(cnt_negative[word] + 1)
            if ratio > 1:
                pos_neg_ratio[word] = np.log(ratio)
            else:
                pos_neg_ratio[word] = -np.log(1 / (ratio + 0.01))

    positive_dict = {}
    for word, cnt in pos_neg_ratio.items():
        if cnt > 1:
            positive_dict[word] = cnt

    wordcloud = WordCloud()
    wordcloud.generate_from_frequencies(frequencies=positive_dict)

    negative_dict = {}
    for word, cnt in pos_neg_ratio.items():
        if (cnt < 1) & (cnt > 0):
            negative_dict[word] = -np.log(cnt)

    wordcloud = WordCloud()
    wordcloud.generate_from_frequencies(frequencies=negative_dict)

    # ## Data Preprocessing

    # In order to train our model, we had to transform the reviews into the right format.
    # We performed the following steps:
    reviews = data.Text.values
    labels = np.array([1 if s == "POSITIVE" else 0 for s in data.Sentiment.values])

    # First we need to remove punctuations and transform all the characters into a list of integers.
    reviews_cleaned = []
    for i in range(len(reviews)):
        reviews_cleaned.append(''.join([c.lower() for c in reviews[i] if c not in punctuation]))

    # What's the size of the vocabulary after removing punctuations and transform characters to lower case?
    vocabulary = set(' '.join(reviews_cleaned).split())

    # ## Noise reduction by reducing vocabulary

    # Let's create a vocabulary reduction function in case we want to reduce the vocabulary based on min frequency or
    # polarity.
    reviews_cleaned = vocabulary_reduction(reviews_cleaned, min_freq=0, polarity_cut_off=0)

    # Then we need to transform each review into a list of integers. First we can create a dictionary to map each word
    # contained in vocabulary of the reviews to an integer.
    # Store all the text from each review in a text variable
    text = ' '.join(reviews_cleaned)

    # List all the vocabulary contained in the reviews
    vocabulary = set(text.split(' '))

    # Map each word to an integer
    vocabulary_to_int = {word: i for i, word in enumerate(vocabulary, 0)}

    # Once we have the vocabulary_to_int dictionary created, we can transform each review into a list of integers using
    # the dictionary previously created.
    reviews_to_int = reviews_to_integers(reviews_cleaned)

    # To train the RNN on the review dataset we have to create an array with 200 columns,
    # and enough rows to fit one review per row. Then store each review consisting of integers in the array.
    # If the number of words in the review is less than 200 words, we can pad the list with extra zeros.
    # If the number of words in the review are more than 200 words then we can limit the review to the first 200 words.
    review_lengths = [len(review) for review in reviews_to_int]

    pd.DataFrame(review_lengths).describe()

    # The mean and the third quartile of the review length is equal to 79 and 97, respectively.  Therefore if we limit
    # the review length to 200 words, we shouldn't lose too much information.
    max_length = 200
    features = np.zeros(shape=(len(reviews_to_int), max_length), dtype=int)

    for i in range(len(reviews_to_int)):
        nb_words = len(reviews_to_int[i])
        features[i] = [0] * (max_length - nb_words) + reviews_to_int[i][:200]

    # Export to csv format
    np.savetxt("../datasets/Amazon_Features.csv", features, delimiter=",")
    np.savetxt("../datasets/Amazon_Labels.csv", labels, delimiter=",")
