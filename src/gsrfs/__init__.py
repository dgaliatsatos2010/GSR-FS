from .selector import GSRSelector
from .partition import GSRPartitionSelector, persistent_balanced_gap_evidence
from .concordance import GSRConcordanceSelector, jackknife_geometry_concordance
from .equivalence import GSRSubstitutionSelector, contextual_geometry_substitution, substitution_aware_jaccard
from .groups import GSRStructuralGroupSelector
from .baselines import (
    MaxVarianceSelector, LaplacianScoreSelector, SPECSelector, MCFSSelector,
    ExternalRankingSelector, estimate_spectral_cluster_count,
)
from .recent import FGMRWAuthorSelector, FGMRW_COMMIT, FGMRW_FILES
from .canonical import (
    ScikitFeatureLaplacianSelector, ScikitFeatureSPECSelector,
    ScikitFeatureMCFSSelector,
)
from .stability import subsample_stability, substitution_aware_stability
from .audit import label_blindness_audit
from .repro import environment_manifest, stable_json_hash, write_manifest
from .external import (
    array_fingerprint, build_external_ranking_record, validate_external_ranking_record,
    write_external_ranking_record, load_external_ranking_record, selector_from_external_ranking,
)
from .realdata import load_offline_real_panel, load_statsmodels_extension_benchmarks, downstream_k
from .downstream import run_foldwise_downstream_validation, summarize_foldwise_downstream
from .publication import (
    PublicationGateConfig, evaluate_publication_gate, write_publication_gate_report,
    SuperiorityGateConfig, evaluate_superiority_gate, write_superiority_gate_report,
)

__all__ = [
    "GSRSelector", "GSRSubstitutionSelector", "GSRStructuralGroupSelector", "contextual_geometry_substitution", "substitution_aware_jaccard", "GSRPartitionSelector", "persistent_balanced_gap_evidence", "GSRConcordanceSelector", "jackknife_geometry_concordance", "MaxVarianceSelector", "LaplacianScoreSelector",
    "SPECSelector", "MCFSSelector", "ExternalRankingSelector",
    "estimate_spectral_cluster_count", "FGMRWAuthorSelector", "FGMRW_COMMIT", "FGMRW_FILES", "ScikitFeatureLaplacianSelector",
    "ScikitFeatureSPECSelector", "ScikitFeatureMCFSSelector", "subsample_stability", "substitution_aware_stability",
    "label_blindness_audit", "environment_manifest", "stable_json_hash", "write_manifest",
    "array_fingerprint", "build_external_ranking_record", "validate_external_ranking_record",
    "write_external_ranking_record", "load_external_ranking_record", "selector_from_external_ranking",
    "load_offline_real_panel", "load_statsmodels_extension_benchmarks", "downstream_k",
    "run_foldwise_downstream_validation", "summarize_foldwise_downstream",
    "PublicationGateConfig", "evaluate_publication_gate", "write_publication_gate_report",
    "SuperiorityGateConfig", "evaluate_superiority_gate", "write_superiority_gate_report",
    "SourceIntegrityRecord", "build_source_integrity_record", "audit_source_integrity_record",
    "audit_source_integrity_manifest", "eligible_datasets_from_source_audit",
    "write_source_integrity_audit", "file_sha256",
]

from .source_integrity import (
    SourceIntegrityRecord,
    build_source_integrity_record,
    audit_source_integrity_record,
    audit_source_integrity_manifest,
    eligible_datasets_from_source_audit,
    write_source_integrity_audit,
    file_sha256,
)
__version__ = "0.16.0"
