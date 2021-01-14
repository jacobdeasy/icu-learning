from googletrans import Translator
from tqdm import tqdm_notebook


translator = Translator()


def translate_df(df, cols=None):
    if cols is None:
        cols = df.columns.to_list()

    print("TRANSLATING DATAFRAME")
    translations = {}
    for col in tqdm_notebook(df[cols].columns):
        for unique in df[col].unique():
            if isinstance(unique, str):
                translations[unique] = translator.translate(unique).text

    return df.replace(translations), translations
