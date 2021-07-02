import os
import sys
import tempfile
from io import StringIO
import unittest
from mdgo.forcefield import *

test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_files")


class FFcrawlerTest(unittest.TestCase):
    def test_chrome(self):
        saved_stdout = sys.stdout
        download_dir = tempfile.mkdtemp()
        try:
            out = StringIO()
            sys.stdout = out

            lpg = FFcrawler(download_dir)
            lpg.data_from_pdb(os.path.join(test_dir, "EMC.pdb"))
            self.assertEqual(
                out.getvalue(),
                "LigParGen server connected.\n"
                "Structure info uploaded. Rendering force field...\n"
                "Force field file downloaded.\n"
                "Force field file saved.\n",
            )
        finally:
            sys.stdout = saved_stdout
            shutil.rmtree(download_dir)


if __name__ == "__main__":
    unittest.main()
