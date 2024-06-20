import unittest, os

from src.data_preparation import CSVData


class TestCSVData(unittest.TestCase):
    def test_create_processed_data(self):
        country = "United Kingdom"
        CSVData("Hotel_Reviews").create_processed_data(country)

        processed_data_dir = os.path.join(
            os.path.dirname(os.getcwd()), 'data', 'processed'
        )
        self.assertTrue(os.path.exists(processed_data_dir))
        self.assertTrue(os.path.isfile(os.path.join(
            processed_data_dir, f"{country}_processed_df.csv"
        )))


if __name__ == "__main__":
    unittest.main()
