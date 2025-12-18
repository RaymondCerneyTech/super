from behaviors.aggregate import AggregatePassages
from behaviors.web_search import WebSearch


def test_aggregate_requires_two_domains() -> None:
    behavior = AggregatePassages()
    ctx = {
        "data": {
            "goal_flags": ["grounded", "cited"],
            "passages": [
                {
                    "doc_id": "1",
                    "text": "Summary from example.com.",
                    "meta": {"title": "Example A", "url": "https://example.com/a"},
                }
            ],
        }
    }

    first = behavior.run(ctx)
    assert first["ok"] is True
    assert "grounded" not in first["effects"]
    assert ctx["data"]["needs_more_sources"] is True

    ctx["data"]["passages"] = ctx["data"]["passages"] + [
        {
            "doc_id": "2",
            "text": "Summary from example.org.",
            "meta": {"title": "Example B", "url": "https://example.org/b"},
        }
    ]

    second = behavior.run(ctx)
    assert second["ok"] is True
    assert "grounded" in second["effects"]
    domains = ctx["data"]["source_domains"]
    assert any("example.com" in domain for domain in domains)
    assert any("example.org" in domain for domain in domains)


def test_web_search_whitelist_match() -> None:
    behavior = WebSearch()
    whitelist = ["wikipedia.org", ".gov"]
    assert behavior._matches_whitelist("https://en.wikipedia.org/wiki/Riemann", whitelist) is True
    assert behavior._matches_whitelist("https://nasa.gov/mission", whitelist) is True
    assert behavior._matches_whitelist("https://example.com/page", whitelist) is False
