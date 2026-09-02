from projects import Project, list_projects


def test_status_filter_keeps_alphabetical_order():
    rows = [Project(1, "Zulu", "archived"), Project(2, "Beta", "active"), Project(3, "Alpha", "active")]
    assert [project.name for project in list_projects(rows, status="active")] == ["Alpha", "Beta"]


if __name__ == "__main__":
    test_status_filter_keeps_alphabetical_order()
    print("1 public test passed")
