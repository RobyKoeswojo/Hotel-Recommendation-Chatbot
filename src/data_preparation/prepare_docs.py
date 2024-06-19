import os, re
import pandas as pd
import numpy as np

from abc import abstractmethod

np.random.seed(0)

DATA_DIR = os.path.join(os.path.dirname(os.getcwd()), 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')


class Data:
    """
    A base class to prepare the dataset to be used as the documents for
    retriever tool.
    """
    def __init__(self, raw_file_name: str):
        """
        Initialize a new Data class.

        Parameters
        ----------
        raw_file_name: str
            The name of the raw data.
        """
        self.raw_file_name = raw_file_name
        self.raw_data_name = ""
        self.processed_data_name = ""
        self.data = None
        self.raw_data_name = f"{self.raw_file_name}.csv"
        self._check_raw_data()

    @abstractmethod
    def create_processed_data(self, country: str):
        """
        Create the processed data and save to the processed data directory.

        Parameters
        ----------
        country: str
            The country to be the focus of the dataset.
        """
        pass

    @abstractmethod
    def _check_raw_data(self) -> bool:
        """
        Check whether the raw data exists.

        Returns
        -------
        bool
            If True, the raw data exists. Otherwise, return False.
        """
        pass

    @abstractmethod
    def _check_processed_data(self) -> bool:
        """
        Check whether the processed data has already been created.

        Returns
        -------
        bool
            If True, the processed data has already been created.

        Raises
        ------
        NotImplementedError
            If the desired country has not been implemented.
        """
        pass


class CSVData(Data):
    def __init__(self, raw_file_name):
        super().__init__(raw_file_name)

    def create_processed_data(self, country):
        self.processed_data_name = f"{country}_processed_df.csv"
        data_path = os.path.join(PROCESSED_DATA_DIR, self.processed_data_name)
        if os.path.isfile(data_path):
            self._check_processed_data()
        else:
            print("[INFO] Creating processed data.")
            if not os.path.exists(RAW_DATA_DIR):
                os.mkdir(RAW_DATA_DIR)
            if not os.path.exists(PROCESSED_DATA_DIR):
                os.mkdir(PROCESSED_DATA_DIR)
            df = pd.read_csv(os.path.join(RAW_DATA_DIR,
                                          self.raw_data_name))
            # create new columns
            len_country = len(country.split())
            df['Clean_Tags'] = df['Tags'].apply(self._clean_tag)
            if country == "United Kingdom":
                df['Postal_Code'] = df['Hotel_Address'].apply(
                    lambda x: " ".join(
                        x.split(" ")[-(len_country + 2):-len_country]))
                df['City'] = df['Hotel_Address'].apply(
                    lambda x: x.split(" ")[-(len_country + 3)])
            else:
                raise NotImplementedError(
                    "Need to implement for other countries.")

            # filter based on the desired country
            df = df[
                (df['Hotel_Address'].str.contains(country))
                & (df['Reviewer_Nationality'].str.contains(country))
            ]
            df.reset_index(drop=True, inplace=True)
            df = df[df['Reviewer_Score'] >= 8]  # only take from reviewer > 8
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

            # reorganize the reviews
            review_dct = dict(Hotel_Name=[],
                              Positive_Review=[],
                              Negative_Review=[])
            excl_reviews = [  # list of words to be ignored from the reviews
                "no negative",
                "no positive",
                "none",
                "nothing",
                "n a",
                "na"]
            n_review = 3  # collect only last 3 reviews
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

            # merge
            final_df = pd.merge(
                left=agg_df, right=review_df, on='Hotel_Name', how='left')
            # save to the directory
            final_df.to_csv(os.path.join(DATA_DIR,
                                         'processed',
                                         self.processed_data_name),
                            index=False)

    def _check_raw_data(self):
        data_path = os.path.join(
            RAW_DATA_DIR, self.raw_data_name)
        if os.path.isfile(data_path):
            print(f"[INFO] Raw data exists: {data_path}.")
        else:
            exc_msg = f"""
            Raw data does not exist. 
            Please download the 'Hotel_Reviews.csv' from:
            https://www.kaggle.com/datasets/jiashenliu/515k-hotel-reviews-data-in-europe
            """
            raise Exception(exc_msg)

    def _check_processed_data(self):
        data_path = os.path.join(
            PROCESSED_DATA_DIR, self.processed_data_name)
        self.data = pd.read_csv(data_path)
        if isinstance(self.data, pd.DataFrame):
            print(f"[INFO] Processed data already exists at {data_path}.")
        else:
            raise TypeError("Processed data is not a CSV file.")

    @staticmethod
    def _clean_tag(tag) -> str:
        """
        Cleaning the tag of the hotels.

        Parameters
        ----------
        tag: str
            The tag of the hotel from the raw data.

        Returns
        -------
        str
            The tag that has been rearranged.
        """
        pattern = r"[\'\[\]\,]"
        clean_tag = re.sub(pattern, "", tag).strip(" ")
        clean_tag = re.sub(r" {3}", ",", clean_tag)
        return ", ".join(clean_tag.split(','))
