import random
import unittest

import numpy as np

from testit import run_util
from testit.testit import TestItEnv


class CampaignSeedTests(unittest.TestCase):
    def test_none_values_disable_seed(self):
        self.assertIsNone(run_util._normalize_campaign_seed(None))
        self.assertIsNone(run_util._normalize_campaign_seed("None"))
        self.assertIsNone(run_util._normalize_campaign_seed("null"))
        self.assertIsNone(run_util._normalize_campaign_seed(""))

    def test_integer_values_enable_seed(self):
        self.assertEqual(run_util._normalize_campaign_seed(123), 123)
        self.assertEqual(run_util._normalize_campaign_seed("123"), 123)
        self.assertEqual(run_util._normalize_campaign_seed("0x10"), 16)

    def test_invalid_seed_values_are_rejected(self):
        for seed in (-1, 2**32, True, "abc"):
            with self.subTest(seed=seed):
                with self.assertRaises(ValueError):
                    run_util._normalize_campaign_seed(seed)

    def test_seed_campaign_restarts_python_and_numpy_random_sequences(self):
        env = TestItEnv({"target": {"seed": 123}})

        env.seed_campaign()
        first_random_value = random.randint(1, 100)
        first_numpy_values = np.random.randint(0, 100, size=4).tolist()

        env.seed_campaign()
        second_random_value = random.randint(1, 100)
        second_numpy_values = np.random.randint(0, 100, size=4).tolist()

        self.assertEqual(first_random_value, second_random_value)
        self.assertEqual(first_numpy_values, second_numpy_values)


if __name__ == "__main__":
    unittest.main()
