import json
import os
import shutil


def test_dump(bundle, listing):
    shutil.copytree(bundle, os.environ["REPRO_OUT"] + "/bundle", dirs_exist_ok=True)
    open(os.environ["REPRO_OUT"] + "/listing.json", "w").write(json.dumps(listing))
