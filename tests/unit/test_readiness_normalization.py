from agents.readiness import _normalize


def test_normalize_coerces_scalar_optional_string_fields():
    claims = {
        "framework_integrations": [],
        "additional_dependencies": [],
        "maturity_level": "research_only",
        "maturity_reasoning": "No deployable artifact found.",
        "min_gpu_requirement": 80,
        "estimated_inference_cost": 106,
    }
    verified = {"verified_github": None, "verified_hf_model": None}

    readiness, error = _normalize(
        claims=claims,
        verified=verified,
        hf={},
        framework_mentions=[],
    )

    assert error is None
    assert readiness is not None
    assert readiness.min_gpu_requirement == "80"
    assert readiness.estimated_inference_cost == "106"
