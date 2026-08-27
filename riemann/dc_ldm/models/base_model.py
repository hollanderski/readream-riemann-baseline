"""base_model, de-Lightning'd.

ORIGINAL PRESERVED at base_model.py.orig (pl.LightningModule version).

Why this exists: EEGNet_Embedding_version.py does `from dc_ldm.models.base_model
import base_model` and the upstream class subclasses pl.LightningModule. We drive
training with tuning_p10_v3.py's own train()/test() loop, not a Lightning Trainer,
so every Lightning hook below was dead code AND pytorch_lightning became a hard
import dependency for no benefit.

Diff vs .orig, exactly six changes, no behaviour change for our loop:
  1. pl.LightningModule -> nn.Module
  2. dropped `import pytorch_lightning as pl`
  3. dropped save_hyperparameters()  (Lightning-only)
  4. dropped configure_optimizers()  (was already fully commented out upstream)
  5. dropped training_step / validation_step / test_step  (Trainer-only hooks;
     they call self.log(), which does not exist outside a Trainer)
  6. num_classes made a constructor argument instead of hardcoded 6

Every attribute the subclasses actually touch is preserved: loss, acc, config,
one_cycle_lr, predictions, ground_truth. EEGNet_Embedding overrides self.loss with
NLLLoss and sets self.one_cycle_lr itself, so it depends on almost none of this.

Ninon confirmed 2026-08-27 that no converted copy exists anywhere, including
/orcd/scratch/orcd/010/ninon/gao_lane/ where it was believed to live. Verified absent.
"""
from torch import nn

try:
    from torchmetrics.classification import Accuracy
    _HAS_TORCHMETRICS = True
except ImportError:  # torchmetrics is optional once Lightning is gone
    _HAS_TORCHMETRICS = False


class base_model(nn.Module):
    def __init__(self, num_classes: int = 6):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()
        self.acc = (Accuracy(task="multiclass", num_classes=num_classes)
                    if _HAS_TORCHMETRICS else None)
        self.config = None
        self.one_cycle_lr = True
        self.predictions = []
        self.ground_truth = []
