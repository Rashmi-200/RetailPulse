import os
from unittest.mock import patch
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError
import src.api.control_plane as cp


def test_m14d_stage2_api_auth_error_sanitization_and_rate_limit():
    cp.reset_rate_limiter()
    client = TestClient(cp.app, raise_server_exceptions=False)

    with patch.dict(
        os.environ,
        {
            "APP_ENV": "production",
            "RETAILPULSE_API_KEY": "rp_prod_secret_token_9876543210",
            "API_RATE_LIMIT_PER_MIN": "100",
        },
    ):
        cp.reset_rate_limiter()

        # 1. Public health endpoint remains accessible without credentials
        r_health = client.get("/api/v1/health")
        assert r_health.status_code == 200
        assert r_health.json()["status"] == "HEALTHY"

        # 2. Unauthenticated requests to all protected endpoints are rejected (401)
        protected = [
            ("GET", "/api/v1/runs", None),
            ("GET", "/api/v1/quality", None),
            ("GET", "/api/v1/incidents", None),
            ("POST", "/api/v1/incidents/inc_001/analyze", None),
            ("GET", "/api/v1/lineage/gold/factorders", None),
            ("GET", "/api/v1/forecast", None),
            ("POST", "/api/v1/copilot/ask", {"question": "How many orders failed?"}),
        ]
        for method, path, body in protected:
            resp = client.request(method, path, json=body)
            assert resp.status_code == 401, f"Expected 401 for {method} {path}, got {resp.status_code}"

        # 3. Invalid credentials rejected (401)
        assert client.get("/api/v1/runs", headers={"X-API-Key": "wrong_key"}).status_code == 401
        assert client.get("/api/v1/runs", headers={"Authorization": "Bearer wrong_key"}).status_code == 401

        # 4. Valid credentials succeed via X-API-Key and Authorization: Bearer
        r_ok_key = client.get("/api/v1/runs", headers={"X-API-Key": "rp_prod_secret_token_9876543210"})
        assert r_ok_key.status_code == 200 and "runs" in r_ok_key.json()

        r_ok_bearer = client.get("/api/v1/quality", headers={"Authorization": "Bearer rp_prod_secret_token_9876543210"})
        assert r_ok_bearer.status_code == 200 and "datasets" in r_ok_bearer.json()

        r_copilot = client.post(
            "/api/v1/copilot/ask",
            headers={"X-API-Key": "rp_prod_secret_token_9876543210"},
            json={"question": "Show recent pipeline quality status"},
        )
        assert r_copilot.status_code == 200

        # 5. Request validation check
        r_invalid_limit = client.get(
            "/api/v1/runs?limit=99999",
            headers={"X-API-Key": "rp_prod_secret_token_9876543210"},
        )
        assert r_invalid_limit.status_code == 422

        # 6. Safe error response test: verify no SQL details, credentials, stack traces, or paths leak
        def broken_connect():
            raise OperationalError(
                "SELECT * FROM secret_table WHERE password='leaked_db_pass'",
                {},
                Exception("D:\\RetailPulse\\src\\secret_internal.py traceback"),
            )

        with patch.object(cp.engine, "connect", side_effect=broken_connect):
            r_err = client.get("/api/v1/runs", headers={"X-API-Key": "rp_prod_secret_token_9876543210"})
            assert r_err.status_code == 503
            err_text = r_err.text
            for forbidden in ["SELECT", "leaked_db_pass", "RetailPulse", "Traceback", "OperationalError"]:
                assert forbidden not in err_text

    # 7. Fail-closed check when production mode has missing/insecure API key
    with patch.dict(os.environ, {"APP_ENV": "production", "RETAILPULSE_API_KEY": "change_me_api_key"}):
        cp.reset_rate_limiter()
        r_fail_closed = client.get("/api/v1/runs", headers={"X-API-Key": "change_me_api_key"})
        assert r_fail_closed.status_code == 503

    # 8. Rate limiting check
    with patch.dict(
        os.environ,
        {
            "APP_ENV": "production",
            "RETAILPULSE_API_KEY": "rp_prod_secret_token_9876543210",
            "API_RATE_LIMIT_PER_MIN": "2",
        },
    ):
        cp.reset_rate_limiter()
        h = {"X-API-Key": "rp_prod_secret_token_9876543210"}
        assert client.get("/api/v1/runs", headers=h).status_code == 200
        assert client.get("/api/v1/runs", headers=h).status_code == 200
        assert client.get("/api/v1/runs", headers=h).status_code == 429

    cp.reset_rate_limiter()
