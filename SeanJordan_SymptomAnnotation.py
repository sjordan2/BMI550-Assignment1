import pandas as pd
import sys
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import ngrams
from fuzzywuzzy import fuzz


def is_in_blacklist(blacklist_list, ngram_length, ngram_position):
    checked_span = list(range(ngram_position, ngram_position + ngram_length))
    for curr_tuple in blacklist_list:
        curr_span = list(range(curr_tuple[1], curr_tuple[1] + curr_tuple[0]))
        for pos in checked_span:
            if pos in curr_span:
                return True
    return False


def symptom_exists(symptom_cui, cui_list, neg_list):
    for row_index in range(len(cui_list)):
        # If the symptom has already been added to the post's dataframe
        if cui_list[row_index] == symptom_cui:
            # We also need to check if it was previously reported as positive and not negative
            if neg_list[row_index] == "0":
                return True
    return False


def get_negation(tokenized_words, negation_word_list):
    for word in tokenized_words:
        # If there is a negation in this phrase, then all symptoms in that phrase will be negated
        if word in negation_word_list:
            # print("Negated sentence:", " ".join(tokenized_words))
            return "1"
    return "0"


def is_overlapping(ngram_len, ngram_pos, sentence_dataframe):
    checked_span = list(range(ngram_pos, ngram_pos + ngram_len))
    for row_index in range(len(sentence_dataframe.index)):
        curr_span = list(range(sentence_dataframe.iloc[row_index, 1], sentence_dataframe.iloc[row_index, 1]
                               + sentence_dataframe.iloc[row_index, 0]))
        for pos in checked_span:
            if pos in curr_span:
                return True
    return False


def compute_ngram_position(ngram, tokenized_words):
    # Basically, iterate through the tokenized words of a sentence until we reach a sequence of words that matches
    # the query ngram and then return the index
    for index, sublist in enumerate((tokenized_words[index:index + len(ngram)]
                                     for index in range(len(tokenized_words)))):
        if list(ngram) == sublist:
            return index
    return None  # This should never be reached, as the ngram should *always* be found


def add_sep(the_list):
    return "$$$" + "$$$".join(the_list) + "$$$"


def annotate_post(post_id, post, symptoms, neg_list):
    # Replace new lines, semicolons, and parentheses/brackets with periods, since they are basically new sentences
    post = post.replace("\n", ".")
    post = post.replace(";", ".")
    post = post.replace("(", ". ")
    post = post.replace(")", ". ")
    post = post.replace(" [", ". ")
    post = post.replace("] ", ". ")

    # Replace forward slashes with spaces since they are connected with previous phrases
    post = post.replace("/", " ")
    # Remove apostrophes, since they don't change the meaning of the sentence and are just for grammar
    post = post.replace("'", "")
    # Lowercase everything
    post = post.lower()
    # Split sentences at transition words since they usually switch connotations of phrases
    post = post.replace("but", ".")
    post = post.replace("however", ".")
    post = post.replace("although", ".")
    post = post.replace("other than", ".")
    post = post.replace("aside from", ".")
    post = post.replace("besides", ".")
    post = post.replace("except", ".")

    symptoms_list = []
    CUIs_list = []
    negations_list = []
    sentences = sent_tokenize(post)
    for sentence in sentences:
        blacklist_2d_list = []
        sentence_df = pd.DataFrame(columns=["Ngram Length", "Start Position", "Symptom", "CUI", "Negation"])
        words = word_tokenize(sentence)
        # Remove commas and periods to avoid them clogging up ngrams
        new_words = []
        for word in words:
            if word not in [".", ","]:
                new_words.append(word)
        for ngram_length in range(4, 0, -1):
            sentence_ngrams = list(ngrams(new_words, ngram_length))
            for ngram_instance in sentence_ngrams:
                for symptom in symptoms.keys():
                    combined_ngram = " ".join(ngram_instance)
                    # Do fuzzy matching to catch typos or to catch rearranged words
                    # e.g. "sore throat" and "throat is sore" have a sort ratio of 88
                    if fuzz.token_sort_ratio(combined_ngram, symptom) > 85:
                        # Remove smaller overlaps
                        # Some longer ngrams could overshadow smaller ones (e.g. "sore throat" overshadows "sore")
                        # We want to prune our computed ngram positions to remove the smaller ones,
                        # which means we must remove overlaps
                        if not is_overlapping(ngram_length, compute_ngram_position(ngram_instance, new_words),
                                              sentence_df):

                            # If a symptom is reported to be present, but is later reported to be negated,
                            # then do *not* add it, since the patient has already had said symptom
                            negation_result = get_negation(words, neg_list)
                            # print("////////")
                            # print("Ngram:", ngram_instance)
                            # print("Found symptom:", symptom)
                            # print("Blacklist:", blacklist_2d_list)
                            # print("Symptom:", symptom)
                            # print("Ngram length:", ngram_length)
                            # print("Ngram pos:", compute_ngram_position(ngram_instance, new_words))
                            # print("Symptom Exists already?", symptom_exists(symptoms[symptom],
                            # CUIs_list, negations_list))
                            # print("Is in blacklist?", is_in_blacklist(blacklist_2d_list, ngram_length,
                            #                             compute_ngram_position(ngram_instance, new_words)))
                            if not is_in_blacklist(blacklist_2d_list, ngram_length,
                                                   compute_ngram_position(ngram_instance, new_words)):
                                if negation_result == "0":
                                    # If a symptom passes these tests, add it to the post df for later processing
                                    new_row = (ngram_length, compute_ngram_position(ngram_instance, new_words),
                                               symptom, symptoms[symptom], get_negation(words, neg_list))
                                    sentence_df.loc[len(sentence_df.index)] = new_row
                                    blacklist_2d_list.append(
                                        (ngram_length, compute_ngram_position(ngram_instance, new_words)))
                                else:
                                    if not symptom_exists(symptoms[symptom], CUIs_list, negations_list):
                                        # If a symptom passes these tests, add it to the post df
                                        new_row = (ngram_length, compute_ngram_position(ngram_instance, new_words),
                                                   symptom, symptoms[symptom], get_negation(words, neg_list))
                                        sentence_df.loc[len(sentence_df.index)] = new_row
                                blacklist_2d_list.append(
                                    (ngram_length, compute_ngram_position(ngram_instance, new_words)))
                                # print(blacklist_2d_list)

        # sentence_df = remove_overlaps(sentence_df)
        # print("================")
        # print(post_id)
        # print(sentence)
        # print(list(sentence_df["Symptom"]))
        # print(list(sentence_df["CUI"]))
        # print(list(sentence_df["Negation"]))
        # print("================")

        symptoms_list.extend(list(sentence_df["Symptom"]))
        CUIs_list.extend(list(sentence_df["CUI"]))
        negations_list.extend(list(sentence_df["Negation"]))

    annotation_list = (post_id, add_sep(symptoms_list), add_sep(CUIs_list), add_sep(negations_list))
    print(annotation_list)
    return annotation_list


# Assuming that the filename is provided as a command line argument
reddit_posts_df = None
try:
    source_file = sys.argv[1]
    reddit_posts_df = pd.read_excel(source_file)
except (IndexError, FileNotFoundError):
    print("ERROR: You must provide a valid file name as an argument! Example: `python3 SeanJordan_SymptomAnnotation.py "
          "UnlabeledSet.xlsx`")
    sys.exit(1)

reddit_posts_df = reddit_posts_df.dropna()
num_posts = len(reddit_posts_df.index)
final_annotation_df = pd.DataFrame(columns=["ID", "Symptom Expression", "Symptom CUIs", "Negation Flag"])

# Using symptom loading code from Dr. Sarker's solution since I would probably use Pandas instead of doing this
# and this is a lot better lol
symptom_dict = {}
infile = open('./COVID-Twitter-Symptom-Lexicon.txt')
for line in infile:
    items = line.split('\t')
    symptom_dict[str.strip(items[-1].lower())] = str.strip(items[1])
infile.close()

# Standard negation list
negation_list = ["no", "not", "without", "absence of", "cannot", "couldn't", "could not", "didn't", "did not", "denied",
                 "denies", "free of", "negative for", "never had", "resolved", "exclude", "with no", "rule out", "free",
                 "went away"]

# new_rows = annotate_post(reddit_posts_df.iloc[11, 0], reddit_posts_df.iloc[11, 1], symptom_dict, negation_list)
# new_row_df = pd.DataFrame(new_rows, columns=["ID", "Symptom Expression", "Symptom CUIs", "Negation Flag"])
# final_annotation_df = pd.concat([final_annotation_df, new_row_df], ignore_index=True)
# print(final_annotation_df)

# Find the position of the ID and text columns
id_col = 0
text_col = 0
for column_num in range(len(list(reddit_posts_df.columns))):
    if list(reddit_posts_df.columns)[column_num] == "ID":
        id_col = column_num
    if list(reddit_posts_df.columns)[column_num] == "TEXT":
        text_col = column_num

for post_index in range(num_posts):
    print("Annotating post " + str(post_index + 1) + " out of " + str(num_posts) + "...")
    curr_post_id = reddit_posts_df.iloc[post_index, id_col]
    curr_post = reddit_posts_df.iloc[post_index, text_col]

    new_rows = annotate_post(curr_post_id, curr_post, symptom_dict, negation_list)
    final_annotation_df.loc[len(final_annotation_df.index)] = new_rows
    # new_row_df = pd.DataFrame(new_rows, columns=["ID", "Symptom Expression", "Symptom CUIs", "Negation Flag"])
    # final_annotation_df = pd.concat([final_annotation_df, new_rows], ignore_index=True)
    # print(final_annotation_df)

final_annotation_df.to_excel("result.xlsx")
