import unittest
from src.prompts import ReactPrompt, RAGPrompt


class TestReactPrompt(unittest.TestCase):
    def test_react_prompt_with_memory_get(self):
        prompt = ReactPrompt(conversation_history=True).get()
        self.assertIsNotNone(prompt, ValueError)
        self.assertIn("Use the following format:", prompt.template)
        self.assertIn("Previous conversation history:", prompt.template)
        print(f"Prompt for react with memory:\n{prompt.template}")

    def test_react_prompt_no_memory_get(self):
        prompt = ReactPrompt(conversation_history=False).get()
        self.assertIsNotNone(prompt, ValueError)
        self.assertIn("Use the following format:", prompt.template)
        self.assertNotIn("Previous conversation history:", prompt.template)
        print(f"Prompt for react with no memory:\n{prompt.template}")


class TestRAGPrompt(unittest.TestCase):
    def test_rag_prompt(self):
        prompt = RAGPrompt().get()
        self.assertIsNotNone(prompt, ValueError)
        self.assertNotIn("Use the following format:", prompt.template)
        print(f"Prompt for RAG: {prompt.template}")


if __name__ == "__main__":
    unittest.main()
