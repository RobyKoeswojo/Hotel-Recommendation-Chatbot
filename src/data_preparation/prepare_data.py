import os, re, logging
import pandas as pd
import numpy as np

from abc import abstractmethod

np.random.seed(0)

DATA_DIR = os.path.join(os.path.dirname(os.getcwd()), 'data')


class Data:
    def __init__(self, raw_file_name):
        self.raw_file_name = raw_file_name
        self.processed_data_name = ""
        self.data = None

    @abstractmethod
    def create_processed_data(self, country):
        pass

    @abstractmethod
    def load_processed_data(self, country):
        pass


class CSVData(Data):
    def __init__(self, raw_file_name):
        super().__init__(raw_file_name)

    def check_processed_data(self):
        data_path = os.path.join(
            DATA_DIR, 'processed', self.processed_data_name)
        self.data = pd.read_csv(data_path)
        if isinstance(self.data, pd.DataFrame):
            print(f"Processed data already exists at {data_path}.")

    def create_processed_data(self, country):
        self.processed_data_name = f"{country}_processed_df.csv"
        data_path = os.path.join(
            DATA_DIR, 'processed', self.processed_data_name)
        if os.path.isfile(data_path):
            self.check_processed_data()
        else:
            print("Creating processed data.")
            df = pd.read_csv(os.path.join(DATA_DIR, 'raw',
                                          f"{self.raw_file_name}.csv"))
            # new columns
            df['Clean_Tags'] = df['Tags'].apply(self.clean_tag)
            df['Postal_Code'] = df['Hotel_Address'].apply(
                lambda x: " ".join(x.split(" ")[-4:-2]))
            df['City'] = df['Hotel_Address'].apply(lambda x: x.split(" ")[-5])
            # filter
            df = df[
                (df['Hotel_Address'].str.contains(country))
                & (df['Reviewer_Nationality'].str.contains(country))
            ]
            df.reset_index(drop=True, inplace=True)
            df = df[df['Reviewer_Score'] >= 8]

            take_cols = [
                'Hotel_Name', 'Average_Score',
                'Positive_Review', 'Negative_Review', 'Review_Date',
                'Hotel_Address', 'Postal_Code', 'City', 'lat', 'lng',
                'Reviewer_Score', 'Clean_Tags',
            ]
            df = df[take_cols]

            # aggregate
            agg_df = df.groupby('Hotel_Name').agg(
                {
                    'Average_Score': 'first',
                    'Hotel_Address': 'first',
                    'Review_Date': 'first',
                    'Postal_Code': 'first',
                    'City': 'first',
                    'lat': 'first',
                    'lng': 'first',
                    'Clean_Tags': 'first',
                }
            ).reset_index()

            review_dct = dict(Hotel_Name=[],
                              Positive_Review=[],
                              Negative_Review=[])

            excl_reviews = ["no negative",
                            "no positive",
                            "none",
                            "nothing",
                            "n a",
                            "na"]
            n_review = 3
            for hotel_name in agg_df['Hotel_Name']:
                review_dct['Hotel_Name'].append(hotel_name)
                for col in ['Positive_Review', 'Negative_Review']:
                    reviews = df[df['Hotel_Name'] == hotel_name][col].values
                    review = []
                    for text in reviews:
                        if text.lower() in excl_reviews: continue
                        review.append(text.strip())

                        if len(review) == n_review: break

                    review_dct[col].append(", ".join(review))

            review_df = pd.DataFrame(review_dct)

            final_df = pd.merge(
                left=agg_df, right=review_df, on='Hotel_Name', how='left')

            final_df.to_csv(os.path.join(DATA_DIR,
                                         'processed',
                                         self.processed_data_name),
                            index=False)

    @staticmethod
    def clean_tag(tag):
        pattern = r"[\'\[\]\,]"
        clean_tag = re.sub(pattern, "", tag).strip(" ")
        clean_tag = re.sub(r" {3}", ",", clean_tag)
        return ", ".join(clean_tag.split(','))
