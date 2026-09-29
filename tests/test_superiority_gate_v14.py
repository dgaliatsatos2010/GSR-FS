import pandas as pd
from gsrfs.publication import (
    SuperiorityGateConfig,
    evaluate_superiority_gate,
)


def _rows(candidate_better=True):
    rows=[]
    for i in range(30):
        c=0.9 if candidate_better else 0.5
        for method,val,src in [
            ('GSR-FS',c,'native_or_local'),
            ('CAN-A',0.6,'canonical:author'),
            ('CAN-B',0.58,'author_code:repo'),
            ('LOCAL',0.55,'native_or_local'),
        ]:
            rows.append({'dataset':f'd{i}', 'method':method, 'balanced_accuracy':val,
                         'implementation_source':src, 'error':''})
    return pd.DataFrame(rows)


def test_superiority_gate_can_pass_on_clear_separation():
    report=evaluate_superiority_gate(
        _rows(True),
        config=SuperiorityGateConfig(min_complete_datasets=25, min_significant_canonical_wins=1),
        publication_gate_report={'passed':True},
    )
    assert report['passed'] is True


def test_superiority_gate_fails_when_evidence_gate_fails():
    report=evaluate_superiority_gate(
        _rows(True), publication_gate_report={'passed':False}
    )
    assert report['passed'] is False
    assert report['checks'][0]['passed'] is False


def test_superiority_gate_fails_without_candidate_advantage():
    report=evaluate_superiority_gate(
        _rows(False), publication_gate_report={'passed':True}
    )
    assert report['passed'] is False
