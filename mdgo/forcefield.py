# coding: utf-8
# Copyright (c) Tingzheng Hou.
# Distributed under the terms of the MIT License.

"""
This module implements two core class FFcrawler and MaestroRunner
for generating LAMMPS/GROMACS data files from molecule structure using
the LigParGen web server and Maestro, respectively.


For using the FFcrawler class, download the ChromeDriver executable that
matches your Chrome version via https://chromedriver.chromium.org/downloads

For using the MaestroRunner:
    1. download a free Maestro via https://www.schrodinger.com/freemaestro
    2. install the package and set the environment variable $SCHRODINGER
(e.g. 'export SCHRODINGER=/opt/schrodinger/suites2020-4', please check
https://www.schrodinger.com/kb/446296 or https://www.schrodinger.com/kb/1842
for details.

"""

from pymatgen.io.lammps.data import LammpsData
from mdgo.util import mass_to_name, ff_parser
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from string import Template
from io import StringIO
import pandas as pd
import time
import os
import re
import shutil
import signal
import subprocess

__author__ = "Tingzheng Hou"
__version__ = "1.0"
__maintainer__ = "Tingzheng Hou"
__email__ = "tingzheng_hou@berkeley.edu"
__date__ = "Feb 9, 2021"

MAESTRO = "$SCHRODINGER/maestro -console -nosplash"
FFLD = "$SCHRODINGER/utilities/ffld_server -imae {} " \
       "-version 14 -print_parameters -out_file {}"


class FFcrawler:
    """
    Web scrapper that can automatically upload structure to the LigParGen
    server and download LAMMPS/GROMACS data file.

    Examples:
    >>> LPG = FFcrawler('/path/to/work/dir', '/path/to/chromedriver')
    >>> LPG.data_from_pdb("/path/to/pdb")
    """

    def __init__(
            self,
            write_dir,
            chromedriver_dir,
            headless=True,
            xyz=False,
            gromacs=False
    ):
        """
        Base constructor.
        Args:
            write_dir (str): Directory for writing output.
            chromedriver_dir (str): Directory to the ChromeDriver executable.
            headless (bool): Whether to run Chrome in headless (silent) mode.
                Default to True.
            xyz (bool): Whether to write the structure in the LigParGen
                generated data file as .xyz. Default to False. This is useful
                because the order and the name of the atoms could be
                different from the initial input.)
            gromacs (bool): Whether to save GROMACS format data files.
                Default to False.
        """
        self.write_dir = write_dir
        self.xyz = xyz
        self.gromacs = gromacs
        self.preferences = {"download.default_directory": write_dir,
                            "safebrowsing.enabled": "false",
                            "profile.managed_default_content_settings.images":
                                2}
        self.options = webdriver.ChromeOptions()
        self.options.add_argument('user-agent="Mozilla/5.0 '
                                  '(Macintosh; Intel Mac OS X 10_14_6) '
                                  'AppleWebKit/537.36 (KHTML, like Gecko) '
                                  'Chrome/88.0.4324.146 Safari/537.36"')
        self.options.add_argument("--window-size=1920,1080")
        if headless:
            self.options.add_argument('--headless')
        self.options.add_experimental_option("prefs", self.preferences)
        self.options.add_experimental_option('excludeSwitches',
                                             ['enable-automation'])
        self.web = webdriver.Chrome(chromedriver_dir, options=self.options)
        self.wait = WebDriverWait(self.web, 10)
        self.web.get("http://zarbi.chem.yale.edu/ligpargen/")
        time.sleep(1)
        print("LigParGen server connected.")

    def data_from_pdb(self, pdb_dir):
        upload = self.web.find_element_by_xpath('//*[@id="exampleMOLFile"]')
        upload.send_keys(pdb_dir)
        submit = self.web.find_element_by_xpath(
            '/html/body/div[2]/div/div[2]/form/button[1]')
        submit.click()
        pdb_filename = os.path.basename(pdb_dir)
        try:
            self.download_data(os.path.splitext(pdb_filename)[0] + ".lmp")
        except TimeoutException:
            print(
                "Timeout! Web server no response for 30s, file download failed!"
            )
        finally:
            self.web.quit()

    def data_from_smiles(self, smiles_code):
        smile = self.web.find_element_by_xpath('//*[@id="smiles"]')
        smile.send_keys(smiles_code)
        submit = self.web.find_element_by_xpath(
            '/html/body/div[2]/div/div[2]/form/button[1]')
        submit.click()
        try:
            self.download_data(smiles_code + '.lmp')
        except TimeoutException:
            print(
                "Timeout! Web server no response for 30s, file download failed!"
            )
        finally:
            self.web.quit()

    def download_data(self, lmp_name):
        print("Structure info uploaded. Rendering force field...")
        self.wait.until(
            EC.presence_of_element_located((By.NAME, 'go'))
        )
        data_lmp = self.web.find_element_by_xpath(
            "/html/body/div[2]/div[2]/div[1]/div/div[14]/form/input[1]"
        )
        data_lmp.click()
        print("Force field file downloaded.")
        time.sleep(1)
        lmp_file = max(
            [self.write_dir + "/" + f for f
             in os.listdir(self.write_dir)
             if os.path.splitext(f)[1] == ".lmp"],
            key=os.path.getctime)
        if self.xyz:
            data = LammpsData.from_file(lmp_file)
            element_id_dict = mass_to_name(data.masses)
            coords = data.atoms[['type', 'x', 'y', 'z']]
            lines = list()
            lines.append(str(len(coords.index)))
            lines.append("")
            for _, r in coords.iterrows():
                line = element_id_dict.get(int(r['type'])) + ' ' + ' '.join(
                    str(r[loc]) for loc in ["x", "y", "z"])
                lines.append(line)

            with open(os.path.join(self.write_dir, lmp_name + ".xyz"),
                      "w") as xyz_file:
                xyz_file.write("\n".join(lines))
            print(".xyz file saved.")
        if self.gromacs:
            data_gro = self.web.find_element_by_xpath(
                "/html/body/div[2]/div[2]/div[1]/div/div[8]/form/input[1]"
            )
            data_itp = self.web.find_element_by_xpath(
                "/html/body/div[2]/div[2]/div[1]/div/div[9]/form/input[1]"
            )
            data_gro.click()
            data_itp.click()
            time.sleep(1)
            gro_file = max(
                [self.write_dir + "/" + f for f
                 in os.listdir(self.write_dir)
                 if os.path.splitext(f)[1] == ".gro"],
                key=os.path.getctime)
            itp_file = max(
                [self.write_dir + "/" + f for f
                 in os.listdir(self.write_dir)
                 if os.path.splitext(f)[1] == ".itp"],
                key=os.path.getctime)
            shutil.move(
                gro_file,
                os.path.join(self.write_dir, lmp_name[:-4] + ".gro")
            )
            shutil.move(
                itp_file,
                os.path.join(self.write_dir, lmp_name[:-4] + ".itp")
            )
        shutil.move(lmp_file, os.path.join(self.write_dir, lmp_name))
        print("Force field file saved.")


class MaestroRunner:

    """
    Wrapper for the Maestro software that can be used to generate the OPLS_2005
    force field parameter for a molecule.

    The OPLS_2005 parameters are described in

    Banks, J.L.; Beard, H.S.; Cao, Y.; Cho, A.E.; Damm, W.; Farid, R.;
    Felts, A.K.; Halgren, T.A.; Mainz, D.T.; Maple, J.R.; Murphy, R.;
    Philipp, D.M.; Repasky, M.P.; Zhang, L.Y.; Berne, B.J.; Friesner, R.A.;
    Gallicchio, E.; Levy. R.M. Integrated Modeling Program, Applied Chemical
    Theory (IMPACT). J. Comp. Chem. 2005, 26, 1752.

    The OPLS_2005 parameters are located in

    $SCHRODINGER/mmshare-vversion/data/f14/

    Examples:
    >>> MR = MaestroRunner('/path/to/structure', '/path/to/working/dir')
    >>> MR.get_mae()
    >>> MR.get_ff()
    """

    template_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "templates",
        "mae_cmd.txt"
    )

    def __init__(
            self,
            structure_dir,
            working_dir,
            out="lmp",
            cmd_template=None):
        """
        Base constructor.
        Args:
            structure_dir (str): Directory of the structure file.
            working_dir (str): Directory for writing intermediate
                and final output.
            cmd_template (str): String template for input script
                with placeholders. Default to None, i.e., using
                the default template.
        """
        self.structure = structure_dir
        self.out = out
        self.structure_format = os.path.splitext(self.structure)[1][1:]
        self.name = os.path.splitext(os.path.split(self.structure)[-1])[0]
        print("Input format:", self.structure_format)
        self.work = working_dir
        self.cmd = os.path.join(self.work, "maetro_script.cmd")
        self.mae = os.path.join(self.work, self.name)
        self.ff = os.path.join(self.work, self.name + ".out")
        self.xyz = os.path.join(self.work, self.name + ".xyz")
        if cmd_template:
            self.cmd_template = cmd_template
        else:
            with open(self.template_path, "r") as f:
                cmd_template = f.read()
            self.cmd_template = cmd_template

    def get_mae(self):
        """Write a Maestro command script and execute it to generate a
        maestro file containing all the info needed."""
        with open(self.cmd, 'w') as f:
            cmd_template = Template(self.cmd_template)
            cmd_script = cmd_template.substitute(
                format=self.structure_format,
                file=self.structure,
                mae=self.mae,
                xyz=self.xyz
            )
            f.write(cmd_script)
        try:
            p = subprocess.Popen(
                f'{MAESTRO} -c {self.cmd}',
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid
            )

            counter = 0
            while not os.path.isfile(self.mae + ".mae"):
                time.sleep(1)
                counter += 1
                if counter > 30:
                    raise TimeoutError(
                        "Failed to generate Maestro file in 30 secs!"
                    )
            print("Maestro file generated.")

        except subprocess.CalledProcessError as e:
            raise ValueError(
                "Maestro failed with errorcode {}  and stderr: {}".format(
                    e.returncode, e.stderr
                )
            )
        finally:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)

    def get_ff(self):
        """Read out and save the force field from the Maestro file"""
        try:
            p = subprocess.run(
                FFLD.format(self.mae + ".mae", self.ff),
                check=True,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
        except subprocess.CalledProcessError as e:
            raise ValueError(
                "Maestro failed with errorcode {} and stderr: {}".format(
                    e.returncode, e.stderr
                )
            )
        print("Force field file generated.")
        if self.out:
            with open(
                    os.path.join(self.work, self.name + "." + self.out), 'w'
            ) as f:
                f.write(ff_parser(self.ff, self.xyz))


def main():
    """
    LPG = FFcrawler(
        "/Users/th/Downloads/test_selenium",
        "/Users/th/Downloads/package/chromedriver/chromedriver",
        xyz=True,
        gromacs=True
    )
    LPG.data_from_pdb("/Users/th/Downloads/test_selenium/EMC.pdb")
    """
    MR = MaestroRunner("/Users/th/Downloads/test_mr/EMC.pdb",
                       "/Users/th/Downloads/test_mr")
    MR.get_mae()
    MR.get_ff()



if __name__ == "__main__":
    main()
