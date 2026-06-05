import os
import time
import uuid

import pytest
import requests


BASE_URL = "http://127.0.0.1:8000"


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_API_FLOW_TESTS") != "1",
    reason="API flow tests require a running server; set RUN_API_FLOW_TESTS=1 to enable.",
)


def unique_name(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def register_user(username: str, password: str):
    response = requests.post(
        f"{BASE_URL}/api/auth/register",
        json={
            "username": username,
            "password": password,
        },
        timeout=20,
    )
    assert response.status_code in [200, 400]
    return response


def login(username: str, password: str) -> str:
    response = requests.post(
        f"{BASE_URL}/api/auth/login",
        data={
            "username": username,
            "password": password,
        },
        timeout=20,
    )

    assert response.status_code == 200, response.text

    data = response.json()
    assert "access_token" in data

    return data["access_token"]


def auth_headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
    }


def test_health_check():
    response = requests.get(f"{BASE_URL}/health", timeout=10)

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_user_register_login_and_me():
    username = unique_name("testuser")
    password = "123456"

    response = register_user(username, password)
    assert response.status_code == 200

    token = login(username, password)

    response = requests.get(
        f"{BASE_URL}/api/users/me",
        headers=auth_headers(token),
        timeout=20,
    )

    assert response.status_code == 200
    data = response.json()

    assert data["username"] == username
    assert data["role"] == "user"


def test_admin_create_order_and_user_refund_flow():
    """
    测试完整流程：

    1. 注册普通用户
    2. 管理员登录
    3. 管理员给普通用户创建订单
    4. 普通用户登录
    5. 普通用户申请退款
    6. 系统创建退款工单
    """

    username = unique_name("refunduser")
    password = "123456"

    register_user(username, password)

    admin_token = login("admin", "admin123")
    user_token = login(username, password)

    order_id = f"ORDER-{uuid.uuid4().hex[:8].upper()}"

    create_order_response = requests.post(
        f"{BASE_URL}/api/orders",
        headers=auth_headers(admin_token),
        json={
            "username": username,
            "order_id": order_id,
            "product_name": "自动化测试订单",
            "amount": 99.0,
            "status": "paid",
            "refundable": True,
        },
        timeout=20,
    )

    assert create_order_response.status_code == 200, create_order_response.text

    order_data = create_order_response.json()
    assert order_data["order_id"] == order_id
    assert order_data["status"] == "paid"
    assert order_data["refundable"] is True

    chat_response = requests.post(
        f"{BASE_URL}/api/chat",
        headers={
            **auth_headers(user_token),
            "Content-Type": "application/json",
        },
        json={
            "message": f"我想申请退款，订单号：{order_id}",
        },
        timeout=60,
    )

    assert chat_response.status_code == 200, chat_response.text

    chat_data = chat_response.json()

    assert chat_data["compliance_passed"] is True
    assert "工单已创建成功" in chat_data["response"]
    assert "工单号" in chat_data["response"]


def test_user_cannot_see_other_user_order():
    """
    普通用户不能查看别人的订单。
    """

    user_a = unique_name("usera")
    user_b = unique_name("userb")
    password = "123456"

    register_user(user_a, password)
    register_user(user_b, password)

    admin_token = login("admin", "admin123")
    user_b_token = login(user_b, password)

    order_id = f"ORDER-{uuid.uuid4().hex[:8].upper()}"

    response = requests.post(
        f"{BASE_URL}/api/orders",
        headers=auth_headers(admin_token),
        json={
            "username": user_a,
            "order_id": order_id,
            "product_name": "权限测试订单",
            "amount": 88.0,
            "status": "paid",
            "refundable": True,
        },
        timeout=20,
    )

    assert response.status_code == 200

    response = requests.get(
        f"{BASE_URL}/api/orders/{order_id}",
        headers=auth_headers(user_b_token),
        timeout=20,
    )

    assert response.status_code == 403
    assert "无权查看" in response.text


def test_normal_user_cannot_update_ticket_status():
    """
    普通用户不能修改工单状态。
    """

    username = unique_name("ticketuser")
    password = "123456"

    register_user(username, password)

    admin_token = login("admin", "admin123")
    user_token = login(username, password)

    order_id = f"ORDER-{uuid.uuid4().hex[:8].upper()}"

    response = requests.post(
        f"{BASE_URL}/api/orders",
        headers=auth_headers(admin_token),
        json={
            "username": username,
            "order_id": order_id,
            "product_name": "工单权限测试订单",
            "amount": 66.0,
            "status": "paid",
            "refundable": True,
        },
        timeout=20,
    )

    assert response.status_code == 200

    response = requests.post(
        f"{BASE_URL}/api/chat",
        headers={
            **auth_headers(user_token),
            "Content-Type": "application/json",
        },
        json={
            "message": f"我想申请退款，订单号：{order_id}",
        },
        timeout=60,
    )

    assert response.status_code == 200

    text = response.json()["response"]

    # 从回复里提取工单号
    ticket_id = None
    for line in text.splitlines():
        if line.startswith("工单号："):
            ticket_id = line.replace("工单号：", "").strip()
            break

    assert ticket_id is not None

    response = requests.patch(
        f"{BASE_URL}/api/tickets/{ticket_id}/status",
        headers={
            **auth_headers(user_token),
            "Content-Type": "application/json",
        },
        json={
            "status": "processing",
        },
        timeout=20,
    )

    assert response.status_code == 403
