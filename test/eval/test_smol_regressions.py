import math

import pytest

from lmms_eval.tasks.SmolInstruct_v2 import smiles_canonicalization as canonical
from lmms_eval.tasks.SmolInstruct_v2.metrics import calculate_fingerprint_similarity


@pytest.mark.parametrize('smiles', ['CCO', 'c1ccccc1', 'C[C@H](O)C(=O)O'])
def test_valid_smiles_survive_canonicalization(smiles):
    result = canonical.canonicalize_molecule_smiles(smiles)
    assert result is not None
    assert canonical.Chem.MolFromSmiles(result) is not None


def test_programming_errors_are_not_mislabeled_as_invalid_molecules(monkeypatch):
    def broken(*args, **kwargs):
        raise NameError('implementation regression')

    monkeypatch.setattr(canonical, 'canonicalize', broken)
    with pytest.raises(NameError, match='implementation regression'):
        canonical.canonicalize_molecule_smiles('CCO')


def test_invalid_smiles_are_still_marked_invalid():
    assert canonical.canonicalize_molecule_smiles('not_a_smiles') is None


def test_no_valid_fingerprint_predictions_have_finite_zero_score():
    scores = calculate_fingerprint_similarity([], [])
    assert all(math.isfinite(score) and score == 0 for score in scores)


def test_exact_fingerprint_match_scores_one():
    mol = canonical.Chem.MolFromSmiles('CCO')
    assert calculate_fingerprint_similarity([mol], [[mol]]) == (1.0, 1.0, 1.0)


def test_aggregate_rejects_metric_failure_instead_of_successful_zero(monkeypatch):
    from lmms_eval.tasks.SmolInstruct_v2 import utils
    def broken(*args, **kwargs):
        raise NameError('metric implementation broken')
    monkeypatch.setattr(utils, 'calculate_smiles_metrics', broken)
    with pytest.raises(RuntimeError, match='forward_synthesis'):
        utils.smol_aggregate_results([dict(task='forward_synthesis', pred='CCO', gold='CCO')])


def test_aggregate_rejects_nonfinite_score(monkeypatch):
    from lmms_eval.tasks.SmolInstruct_v2 import utils
    monkeypatch.setattr(utils, 'calculate_smiles_metrics', lambda *a, **k: {'t1_morgan_fps': float('nan')})
    with pytest.raises(RuntimeError, match='non-finite'):
        utils.smol_aggregate_results([dict(task='forward_synthesis', pred='CCO', gold='CCO')])


def test_aggregate_rejects_mixed_tasks_instead_of_returning_last():
    from lmms_eval.tasks.SmolInstruct_v2 import utils
    with pytest.raises(ValueError, match='one task'):
        utils.smol_aggregate_results([dict(task=t, pred='CCO', gold='CCO') for t in ('forward_synthesis', 'molecule_generation')])


def test_retrosynthesis_subset_is_not_intersection(monkeypatch):
    from lmms_eval.tasks.SmolInstruct_v2 import metrics
    monkeypatch.setattr(metrics, 'canonicalize_molecule_smiles_parallel', lambda xs, **kw: xs)
    result = metrics.calculate_smiles_metrics([['CCO.O']], [['CCO.N']], metrics=('multiple_match',))
    assert result['num_t1_intersection'] == 1
    assert result['num_t1_subset'] == 0


def test_smiles_metrics_normalize_benchmark_reactant_separator(monkeypatch):
    from lmms_eval.tasks.SmolInstruct_v2 import metrics
    monkeypatch.setattr(metrics, 'canonicalize_molecule_smiles_parallel',
        lambda xs, **kw: [canonical.canonicalize_molecule_smiles(x) for x in xs])
    gold = [['CCO;O']]
    result = metrics.calculate_smiles_metrics([['CCO.O']], gold)
    assert result['num_t1_exact_match'] == 1
    assert result['t1_morgan_fps'] == 1.0
    assert gold == [['CCO;O']]
