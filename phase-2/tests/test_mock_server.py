"""The mock HR system enforces the same rules the bot does (defence in depth)."""
H = {"X-Demo-User": "EMP-0002"}


def test_needs_a_known_active_user(mock_client):
    assert mock_client.get("/api/v1/employees/me").status_code == 401
    assert mock_client.get("/api/v1/employees/me", headers={"X-Demo-User": "EMP-9999"}).status_code == 401
    assert mock_client.get("/api/v1/employees/me", headers={"X-Demo-User": "EMP-0004"}).status_code == 403   # inactive account


def test_only_own_data_can_be_read(mock_client):
    assert mock_client.get("/api/v1/leave/balance", params={"employee": "EMP-0003"}, headers=H).status_code == 403
    assert mock_client.get("/api/v1/leave/balance", params={"employee": "me"}, headers=H).json()["cl"] == 9


def test_payroll_items_and_search_are_hr_only(mock_client):
    assert mock_client.get("/api/v1/payroll/items", params={"month": "2026-09"}, headers=H).status_code == 403
    hr = {"X-Demo-User": "EMP-0011"}
    assert mock_client.get("/api/v1/payroll/items", params={"month": "2026-09"}, headers=hr).json()["count"] == 3
    assert mock_client.get("/api/v1/employees/search", params={"name": "Priya Nair"}, headers=H).status_code == 403


def test_two_people_with_the_same_name_come_back_as_a_list_to_choose_from(mock_client):
    r = mock_client.get("/api/v1/employees/search", params={"name": "Priya Nair"}, headers={"X-Demo-User": "EMP-0011"}).json()
    assert len(r["matches"]) == 2 and {m["department_name"] for m in r["matches"]} == {"Customer Support", "Engineering"}
