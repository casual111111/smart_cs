from app.evaluation.rag.retrieval_metrics import (
    evaluate_retrieval_cases,
    hit_at_k,
    mrr_at_k,
    precision_at_k,
)
from app.evaluation.rag.runner import load_rag_eval_cases


def test_hit_at_k():
    results = [
        {
            "source": "refund.md",
            "content": "退款一般 3-5 个工作日到账",
            "score": 0.9,
        },
        {
            "source": "shipping.md",
            "content": "物流一般 2-3 天",
            "score": 0.5,
        },
    ]

    assert hit_at_k(
        results,
        expected_sources=["refund.md"],
        expected_keywords=["3-5 个工作日"],
        k=1,
    ) == 1.0


def test_precision_at_k():
    results = [
        {
            "source": "refund.md",
            "content": "退款一般 3-5 个工作日到账",
            "score": 0.9,
        },
        {
            "source": "shipping.md",
            "content": "物流一般 2-3 天",
            "score": 0.5,
        },
    ]

    assert precision_at_k(
        results,
        expected_sources=["refund.md"],
        expected_keywords=["3-5 个工作日"],
        k=2,
    ) == 0.5


def test_mrr_at_k():
    results = [
        {
            "source": "shipping.md",
            "content": "物流一般 2-3 天",
            "score": 0.9,
        },
        {
            "source": "refund.md",
            "content": "退款一般 3-5 个工作日到账",
            "score": 0.8,
        },
    ]

    assert mrr_at_k(
        results,
        expected_sources=["refund.md"],
        expected_keywords=["3-5 个工作日"],
        k=5,
    ) == 0.5


def test_evaluate_retrieval_cases():
    cases = [
        {
            "id": "case-1",
            "query": "退款多久到账",
            "expected_sources": ["refund.md"],
            "expected_keywords": ["3-5 个工作日"],
        },
        {
            "id": "case-2",
            "query": "物流多久",
            "expected_sources": ["shipping.md"],
            "expected_keywords": ["2-3 天"],
        },
    ]
    retrieval_results = {
        "case-1": [
            {"source": "refund.md", "content": "退款一般 3-5 个工作日到账", "score": 0.9},
            {"source": "shipping.md", "content": "物流一般 2-3 天", "score": 0.5},
        ],
        "case-2": [
            {"source": "other.md", "content": "会员积分规则", "score": 0.7},
        ],
    }

    report = evaluate_retrieval_cases(cases, retrieval_results)

    assert report["case_count"] == 2
    assert report["metrics"]["hit@5"] == 0.5
    assert report["metrics"]["precision@5"] == 0.1
    assert report["metrics"]["mrr@5"] == 0.5


def test_load_rag_eval_cases():
    cases = load_rag_eval_cases()

    assert 10 <= len(cases) <= 20
    for case in cases:
        assert case["id"]
        assert case["query"]
        assert isinstance(case["expected_sources"], list)
        assert case["expected_sources"]
        assert isinstance(case["expected_keywords"], list)
        assert case["expected_keywords"]
