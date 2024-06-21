import unittest, os
from src.models import Models, ModelType
from dotenv import load_dotenv

ENV_DIR = os.path.join(os.path.dirname(os.getcwd()), ".env")
load_dotenv(ENV_DIR)


class TestModels(unittest.TestCase):
    def test_get(self):
        llm = Models.get(ModelType.CHATGPTSTANDARD)
        self.assertIsNotNone(llm, ValueError)
        print(llm)


if __name__ == "__main__":
    unittest.main()
