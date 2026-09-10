"""Vendored OrcaLab runtime for Microduck inference.

Only the private batched GPU runtime (:mod:`orcalab_rslrl._internal.runtime`)
and the OrcaLab batch renderer (:mod:`orcalab_rslrl.orcalab_batch_render`) are
used by ``run_duck.py`` / ``run_duck_multi_policy.py``. The training/task API
was removed to keep this folder minimal.
"""

__version__ = "0.2.0"
