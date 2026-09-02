from duration import parse_duration


def test_milliseconds_are_not_seconds():
    assert parse_duration("250ms") == 250


if __name__ == "__main__":
    test_milliseconds_are_not_seconds()
    print("1 public test passed")
