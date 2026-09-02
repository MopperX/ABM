from src.routing import normalize_route


def test_contract_normalization():
    assert normalize_route(" /API//V2/Items/ ") == "/API/V2/Items"


if __name__ == "__main__":
    test_contract_normalization()
    print("1 public test passed")
