"""End-to-end API tests over the real app (mock LLM, isolated SQLite).

With the TestClient, FastAPI background tasks complete before the response
returns — so after POST /source the shortlist already exists.
"""
from __future__ import annotations

CARBON_BIOS = [
    {"raw_bio": "Dr. Lena Park is a Senior Climate Economist at Meridian. 12 years on "
                "voluntary carbon markets and emissions trading. Based in Geneva.",
     "source": "synthetic"},
    {"raw_bio": "Amara Nwosu is a Lead Carbon Analyst at OffsetWorks, on carbon credit "
                "verification. 13 years, based in Nairobi.", "source": "synthetic"},
    {"raw_bio": "Raj Patel is a Senior Security Architect at Sentinel Systems. 11 years "
                "in appsec. Based in Bangalore.", "source": "synthetic"},
]


def test_health_is_open(client):
    r = client.get("/health", headers={"X-API-Key": ""})
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_auth_required(client):
    r = client.get("/experts/search", headers={"X-API-Key": "wrong-key"})
    assert r.status_code == 401


def test_ingest_then_fetch_and_search(client):
    r = client.post("/experts/ingest", json={"bios": CARBON_BIOS})
    assert r.status_code == 200
    experts = r.json()
    assert len(experts) == 3
    first_id = experts[0]["id"]

    got = client.get(f"/experts/{first_id}")
    assert got.status_code == 200
    assert got.json()["name"] == "Lena Park"

    # org memory: filter by domain
    found = client.get("/experts/search", params={"domain": "carbon markets"})
    assert found.status_code == 200
    names = {e["name"] for e in found.json()}
    assert "Lena Park" in names and "Amara Nwosu" in names
    assert "Raj Patel" not in names  # cybersecurity, different domain


def test_unknown_expert_404(client):
    import uuid
    r = client.get(f"/experts/{uuid.uuid4()}")
    assert r.status_code == 404


def test_full_sourcing_loop(client):
    client.post("/experts/ingest", json={"bios": CARBON_BIOS})

    brief = {
        "title": "Carbon markets advisory",
        "description": "Need experts on voluntary carbon markets and MRV.",
        "required_domains": ["carbon markets"],
        "min_seniority": "senior",
        "num_experts_needed": 2,
    }
    created = client.post("/projects", json=brief)
    assert created.status_code == 201
    project_id = created.json()["id"]

    run = client.post(f"/projects/{project_id}/source")
    assert run.status_code == 202
    assert "run_id" in run.json()

    matches = client.get(f"/projects/{project_id}/matches")
    assert matches.status_code == 200
    body = matches.json()
    assert len(body) == 2  # num_experts_needed

    # ranked descending, scores normalised, carbon experts on top
    scores = [m["overall_score"] for m in body]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= s <= 1.0 for s in scores)
    top_names = {m["expert"]["name"] for m in body}
    assert "Lena Park" in top_names
    assert body[0]["outreach_draft"]  # outreach generated for the shortlist


def test_regenerate_outreach(client):
    client.post("/experts/ingest", json={"bios": CARBON_BIOS})
    pid = client.post("/projects", json={
        "title": "X", "description": "carbon", "required_domains": ["carbon markets"],
        "num_experts_needed": 1,
    }).json()["id"]
    client.post(f"/projects/{pid}/source")
    match_id = client.get(f"/projects/{pid}/matches").json()[0]["id"]

    r = client.post(f"/matches/{match_id}/outreach")
    assert r.status_code == 200
    assert r.json()["outreach_draft"]


def test_runs_observability(client):
    client.post("/experts/ingest", json={"bios": CARBON_BIOS})
    pid = client.post("/projects", json={
        "title": "X", "description": "carbon", "required_domains": ["carbon markets"],
        "num_experts_needed": 2,
    }).json()["id"]
    client.post(f"/projects/{pid}/source")

    runs = client.get("/runs").json()
    assert len(runs) > 0
    stages = {r["stage"] for r in runs}
    assert {"extract", "classify"} <= stages

    classify_runs = client.get("/runs", params={"stage": "classify"}).json()
    assert all(r["stage"] == "classify" for r in classify_runs)
    assert all(r["latency_ms"] is not None for r in classify_runs)
