import torch
import unittest

import utils


class TestUtil(unittest.TestCase):
    def setUp(self):
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

    def test_temporal_coding_zeros(self):
        input_data = torch.zeros((2, 1, 3, 3))
        expected_windows = torch.zeros((2, 12, 3))
        new_windows = utils.encode_data(
            input_data, (1, 3), (1, 3), self.device, encoding="temporal", time_per_window=4)
        self.assertTrue(torch.all(torch.eq(new_windows, expected_windows)))

    def test_temporal_coding_same_values(self):
        input_data = torch.ones((2, 1, 3, 3))
        expected_windows = torch.zeros((2, 12, 3))
        expected_windows[:, [3, 7, 11], :] = 1.0
        new_windows = utils.encode_data(
            input_data, (1, 3), (1, 3), self.device, encoding="temporal", time_per_window=4)
        self.assertTrue(torch.all(torch.eq(new_windows, expected_windows)))

    def test_temporal_coding_simple_input(self):
        input_data = torch.tensor([[[[0.0, 0.4],
                                     [0.8, 0.0]]]])
        expected_windows = torch.tensor([[0.0, 1.0],
                                         [0.0, 0.0],
                                         [0.0, 0.0],
                                         [1.0, 0.0]])

        new_windows = utils.encode_data(
            input_data, (1, 2), (1, 2), self.device, encoding="temporal", time_per_window=2)
        self.assertTrue(torch.all(torch.eq(new_windows, expected_windows)))

    def test_temporal_coding_check_sum(self):
        # check that every value different than 0.0 gets a spike in the encoded data
        input_data = torch.rand((2, 1, 3, 3))
        expected_values = torch.squeeze(torch.sum(input_data > 0.0, dim=2))
        new_windows = utils.encode_data(
            input_data, (1, 3), (1, 3), self.device, encoding="temporal", time_per_window=4)
        self.assertTrue(
            torch.all(torch.eq(expected_values, torch.sum(new_windows.int(), dim=1))))

if __name__ == '__main__':
    unittest.main()
