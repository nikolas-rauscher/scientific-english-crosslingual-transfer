# Import original as default to avoid disrupting running jobs
from .t5_datamodule import T5DataModule  # noqa: F401 - Original V1
from .t5_datamodule_v2 import T5DataModule as T5DataModuleV2  # noqa: F401 - V2 with dual mode
