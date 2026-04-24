from src.showcase_rag import ShowcaseTechniques


def test_showcase_techniques_defaults():
    techniques = ShowcaseTechniques()

    assert techniques.parser == "mineru"
    assert techniques.query_mode == "mix"
    assert techniques.query_optimization_strategy == "all"
    assert techniques.top_k == 10
