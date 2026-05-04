import glob
import os
import shutil
import uuid
from datetime import datetime

# 1. Nuclear cleanup
def nuclear_cleanup():
    for pattern in ["./chroma_*", "./experiment_chroma_db", "./results_experiment", "./logs"]:
        for match in glob.glob(pattern, recursive=True):
            if os.path.exists(match):
                shutil.rmtree(match, ignore_errors=True)



if __name__ == "__main__":
    # 2. Unique directories per run
    RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]
    AUG_CHROMA_BASE = os.path.join(os.getcwd(), f"./chroma_extra_dim_experiment_{RUN_ID}")
    META_CHROMA_BASE = os.path.join(os.getcwd(), f"./chroma_meta_db_experiment_{RUN_ID}")
    ROTATED_CHROMA = os.path.join(os.getcwd(), f"./chroma_rotated_db_log_{RUN_ID}")

    # Call at the START of your script
    nuclear_cleanup()