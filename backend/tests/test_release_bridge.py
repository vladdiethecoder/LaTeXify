from backend.app.services import release_bridge


def test_demo_plan_contains_sections():
    plan = release_bridge.build_demo_plan()
    assert plan["sections"], "Demo plan should contain sections"
    jobs = list(release_bridge.plan_jobs(plan))
    assert jobs, "Plan jobs should be derived from sections"
    assert all(job["id"] for job in jobs)
