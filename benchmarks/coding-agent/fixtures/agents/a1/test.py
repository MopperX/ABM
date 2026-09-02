from stats import median


def test_even_length_median():
    assert median([1, 2, 3, 4]) == 2.5


if __name__ == "__main__":
    test_even_length_median()
    print("1 public test passed")
