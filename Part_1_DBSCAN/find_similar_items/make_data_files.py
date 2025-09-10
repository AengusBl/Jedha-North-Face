def main():
    import spacy
    import pandas as pd
    from spacy.lang.en.stop_words import STOP_WORDS
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import PCA
    from sklearn.cluster import DBSCAN

    print("Downloading the model for English...")
    try:
        nlp = spacy.load("en_core_web_md")
    except OSError:
        from spacy.cli.download import download
        download("en_core_web_md")
        nlp = spacy.load("en_core_web_md")

    try:
        data_df = pd.read_csv("Data/sample-data.csv")
    except:
        print("Cannot find the base dataset. Please make sure you've downloaded it from https://www.kaggle.com/datasets/cclark/product-item-data and stored "
              "it in the same folder as this .py file.")
        return

    print("Processing the data...")

    df = data_df.copy()
    df["clean_docs"] = df["description"].str.replace(r"<[^>]*>", " ", regex=True)\
                                        .str.replace(r"[^a-zA-Z0-9']+", " ", regex=True)\
                                        .apply(lambda desc: nlp(desc.lower()))\
                                        .apply(lambda doc: [token.lemma_ for token in doc if token.text not in STOP_WORDS])\
                                        .apply(lambda ls: " ".join(ls))


    # I am not using a max or min document frequency for terms because I want to decide which cols to keep after the n-grams are made
    vectoriser = TfidfVectorizer(stop_words="english", ngram_range=(1, 4))
    X = vectoriser.fit_transform(df["clean_docs"])
    dense = X.todense()
    tfidf_df = pd.DataFrame(dense, 
                            columns=vectoriser.vocabulary_, 
                            index=[f"doc_{x}" for x in range(1, dense.shape[0]+1)])

    cols_more_than_four = [term for term in vectoriser.vocabulary_ if len(tfidf_df[tfidf_df[term] != 0.0]) > 4] # I checked that no elements were negative, but I used `!=` just to be sure.
    denser_df = tfidf_df[cols_more_than_four]


    _500_pca = PCA(random_state=444719) # The maximum number of features is already 500 because it's min(n_samples, n_features)
    _500_for_dbscan_PC = _500_pca.fit_transform(denser_df)


    # The one I will present first. It follows all the instructions given for the project, including not having any clusters with fewer than 6 elements so that I can suggest
    # five more in the python script, and the categories are well-balanced. It has 48 outliers, however.
    db = DBSCAN(eps=0.7192, min_samples=7, metric="cosine", algorithm="brute")
    db.fit(_500_for_dbscan_PC)
    labels = db.labels_
    best_labels_df = data_df.copy()
    best_labels_df["labels"] = labels
    best_labels_df.to_csv("best_labels_data.csv")

    # This one has very few outliers and the labels are a lot more balanced than many others I've obtained.
    # However, it has 23 labels, which is more than the maximum given in the project instructions, and it has a few labels that only occur twice,
    # which doesn't allow me to provide the user with 5 other suggestions, as described in the project instructions.
    db = DBSCAN(eps=0.7016, min_samples=2, metric="cosine", algorithm="brute")
    db.fit(_500_for_dbscan_PC)
    labels = db.labels_
    few_outliers_df = data_df.copy()
    few_outliers_df["labels"] = labels
    few_outliers_df.to_csv("few_outliers_data.csv")

    print("Done!")


if __name__ == "__main__":
    main()