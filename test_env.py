# This will test the environment to ensure that the .env file is set up
# correctly and that the Ollama and Neo4j connections are working.
import os
import unittest

from dotenv import load_dotenv, find_dotenv
load_dotenv()

class TestEnvironment(unittest.TestCase):

    skip_env_variable_tests = False
    skip_ollama_test = False
    skip_neo4j_test = False

    def test_env_file_exists(self):
        env_file_exists = True if find_dotenv() > "" else False
        if env_file_exists:
            TestEnvironment.skip_env_variable_tests = False
        self.assertTrue(env_file_exists, ".env file not found.")

    def env_variable_exists(self, variable_name):
        self.assertIsNotNone(
            os.getenv(variable_name),
            f"{variable_name} not found in .env file")

    def test_ollama_variables(self):
        if TestEnvironment.skip_env_variable_tests:
            self.skipTest("Skipping Ollama env variable test")

        self.env_variable_exists('OLLAMA_SERVER')
        TestEnvironment.skip_openai_test = False

    def test_neo4j_variables(self):
        if TestEnvironment.skip_env_variable_tests:
            self.skipTest("Skipping Neo4j env variables test")

        self.env_variable_exists('NEO4J_URI')
        self.env_variable_exists('NEO4J_USERNAME')
        self.env_variable_exists('NEO4J_PASSWORD')
        TestEnvironment.skip_neo4j_test = False

    def test_ollama_connection(self):
        if TestEnvironment.skip_ollama_test:
            self.skipTest("Skipping OpenAI test")

        from openai import OpenAI, AuthenticationError

        llm = OpenAI(base_url=os.getenv('OLLAMA_SERVER'),api_key='ollama')

        try:
            models = llm.models.list()
        except AuthenticationError as e:
            models = None
        self.assertIsNotNone(
            models,
            "Ollama connection failed. Check the OLLAMA_SERVER key in .env file.")

    def test_neo4j_connection(self):
        if TestEnvironment.skip_neo4j_test:
            self.skipTest("Skipping Neo4j connection test")

        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(
            os.getenv('NEO4J_URI'),
            auth=(os.getenv('NEO4J_USERNAME'),
                  os.getenv('NEO4J_PASSWORD'))
        )
        try:
            driver.verify_connectivity()
            connected = True
        except Exception as e:
            connected = False

        driver.close()

        self.assertTrue(
            connected,
            "Neo4j connection failed. Check the NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD values in .env file."
        )

def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestEnvironment('test_env_file_exists'))
    suite.addTest(TestEnvironment('test_ollama_variables'))
    suite.addTest(TestEnvironment('test_neo4j_variables'))
    suite.addTest(TestEnvironment('test_ollama_connection'))
    suite.addTest(TestEnvironment('test_neo4j_connection'))
    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
