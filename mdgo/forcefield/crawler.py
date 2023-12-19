# coding: utf-8
# Copyright (c) Tingzheng Hou.
# Distributed under the terms of the MIT License.

"""
This module implements two core class FFcrawler for generating
LAMMPS/GROMACS data files from molecule structure using
the LigParGen web server.

For using the FFcrawler class:

  * Download the ChromeDriver executable that
    matches your Chrome version via https://chromedriver.chromium.org/downloads
"""

import os
import shutil
import time
from typing import Optional


from pymatgen.io.lammps.data import LammpsData
from selenium import webdriver
from selenium.common.exceptions import (
    TimeoutException,
    WebDriverException,
)
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from mdgo.util.dict_utils import lmp_mass_to_name


class FFcrawler:
    """
    Web scrapper that can automatically upload structure to the LigParGen
    server and download LAMMPS/GROMACS data file.

    Args:
        write_dir: Directory for writing output.
        chromedriver_dir: Directory to the ChromeDriver executable.
        headless: Whether to run Chrome in headless (silent) mode.
            Default to True.
        xyz: Whether to write the structure in the LigParGen
            generated data file as .xyz. Default to False. This is useful
            because the order and the name of the atoms could be
            different from the initial input.)
        gromacs: Whether to save GROMACS format data files.
            Default to False.

    Examples:

        >>> lpg = FFcrawler('/path/to/work/dir', '/path/to/chromedriver')
        >>> lpg.data_from_pdb("/path/to/pdb")
    """

    def __init__(
        self,
        write_dir: str,
        chromedriver_dir: Optional[str] = None,
        headless: bool = True,
        xyz: bool = False,
        gromacs: bool = False,
    ):
        """Base constructor."""
        self.write_dir = write_dir
        self.xyz = xyz
        self.gromacs = gromacs
        self.preferences = {
            "download.default_directory": write_dir,
            "safebrowsing.enabled": "false",
            "profile.managed_default_content_settings.images": 2,
        }
        self.options = webdriver.ChromeOptions()
        self.server = webdriver.ChromeService(chromedriver_dir)
        self.options.add_argument(
            'user-agent="Mozilla/5.0 '
            "(Macintosh; Intel Mac OS X 10_14_6) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            'Chrome/88.0.4324.146 Safari/537.36"'
        )
        self.options.add_argument("--window-size=1920,1080")
        self.options.add_argument('ignore-certificate-errors')
        if headless:
            self.options.add_argument("--headless")
        self.options.add_experimental_option("prefs", self.preferences)
        self.options.add_experimental_option("excludeSwitches", ["enable-automation"])
        if chromedriver_dir is None:
            self.web = webdriver.Chrome(options=self.options)
        else:
            self.web = webdriver.Chrome(service=self.server, options=self.options)
        self.wait = WebDriverWait(self.web, 10)
        self.web.get("http://traken.chem.yale.edu/ligpargen/")
        time.sleep(1)
        print("LigParGen server connected.")

    def quit(self):
        """
        Method for quiting ChromeDriver.

        """
        self.web.quit()

    def data_from_pdb(self, pdb_dir: str):
        """
        Use the LigParGen server to generate a LAMMPS data file from a pdb file.
        Write out a LAMMPS data file.

        Args:
            pdb_dir: The path to the input pdb structure file.
        """
        self.web.get("http://traken.chem.yale.edu/ligpargen/")
        upload_xpath = '//*[@id="exampleMOLFile"]'
        time.sleep(1)
        self.wait.until(EC.presence_of_element_located((By.XPATH, upload_xpath)))
        upload = self.web.find_element(By.XPATH, upload_xpath)
        try:
            upload.send_keys(pdb_dir)
            submit = self.web.find_element(By.XPATH, "/html/body/div[2]/div/div[2]/form/button[1]")
            submit.click()
            pdb_filename = os.path.basename(pdb_dir)
            self.download_data(os.path.splitext(pdb_filename)[0] + ".lmp")
        except TimeoutException:
            print("Timeout! Web server no response for 10s, file download failed!")
        except WebDriverException as e:
            print(e)
        finally:
            self.quit()

    def data_from_smiles(self, smiles_code):
        """
        Use the LigParGen server to generate a LAMMPS data file from a SMILES code.
        Write out a LAMMPS data file.

        Args:
            smiles_code: The SMILES code for the LigParGen input.
        """
        self.web.get("http://traken.chem.yale.edu/ligpargen/")
        time.sleep(1)
        smile = self.web.find_element(By.XPATH, '//*[@id="smiles"]')
        smile.send_keys(smiles_code)
        submit = self.web.find_element(By.XPATH, "/html/body/div[2]/div/div[2]/form/button[1]")
        submit.click()
        try:
            self.download_data(smiles_code + ".lmp")
        except TimeoutException:
            print("Timeout! Web server no response for 10s, file download failed!")
        finally:
            self.quit()

    def download_data(self, lmp_name: str):
        """
        Helper function that download and write out the LAMMPS data file.

        Args:
            lmp_name: Name of the LAMMPS data file.
        """
        print("Structure info uploaded. Rendering force field...")
        lmp_xpath = "/html/body/div[2]/div[2]/div[1]/div/div[14]/form/input[1]"
        jmol_xpath = "/html/body/div[2]/div[2]/div[2]"
        self.wait.until(EC.presence_of_element_located((By.XPATH, jmol_xpath)))
        self.wait.until(EC.presence_of_element_located((By.XPATH, lmp_xpath)))
        self.web.execute_script("arguments[0].remove();", jmol_xpath)
        self.wait.until(EC.element_to_be_clickable((By.XPATH, lmp_xpath)))
        data_lmp = self.web.find_element(By.XPATH, lmp_xpath)
        num_file = len([f for f in os.listdir(self.write_dir) if os.path.splitext(f)[1] == ".lmp"]) + 1
        data_lmp.click()
        while True:
            files = sorted(
                [
                    os.path.join(self.write_dir, f)
                    for f in os.listdir(self.write_dir)
                    if os.path.splitext(f)[1] == ".lmp"
                ],
                key=os.path.getmtime,
            )
            # wait for file to finish download
            if len(files) < num_file:
                time.sleep(1)
                print("waiting for download to be initiated")
            else:
                newest = files[-1]
                if ".crdownload" in newest:
                    time.sleep(1)
                    print("waiting for download to complete")
                else:
                    break
        print("Force field file downloaded.")
        lmp_file = newest
        if self.xyz:
            data_obj = LammpsData.from_file(lmp_file)
            element_id_dict = lmp_mass_to_name(data_obj.masses)
            coords = data_obj.atoms[["type", "x", "y", "z"]]
            lines = []
            lines.append(str(len(coords.index)))
            lines.append("")
            for _, r in coords.iterrows():
                element_name = element_id_dict.get(int(r["type"]))
                assert element_name is not None
                line = element_name + " " + " ".join(str(r[loc]) for loc in ["x", "y", "z"])
                lines.append(line)

            with open(os.path.join(self.write_dir, lmp_name + ".xyz"), "w") as xyz_file:
                xyz_file.write("\n".join(lines))
            print(".xyz file saved.")
        if self.gromacs:
            data_gro = self.web.find_element(By.XPATH, "/html/body/div[2]/div[2]/div[1]/div/div[8]/form/input[1]")
            data_itp = self.web.find_element(By.XPATH, "/html/body/div[2]/div[2]/div[1]/div/div[9]/form/input[1]")
            data_gro.click()
            data_itp.click()
            time.sleep(1)
            gro_file = max(
                [self.write_dir + "/" + f for f in os.listdir(self.write_dir) if os.path.splitext(f)[1] == ".gro"],
                key=os.path.getctime,
            )
            itp_file = max(
                [self.write_dir + "/" + f for f in os.listdir(self.write_dir) if os.path.splitext(f)[1] == ".itp"],
                key=os.path.getctime,
            )
            shutil.move(gro_file, os.path.join(self.write_dir, lmp_name[:-4] + ".gro"))
            shutil.move(itp_file, os.path.join(self.write_dir, lmp_name[:-4] + ".itp"))
        shutil.move(lmp_file, os.path.join(self.write_dir, lmp_name))
        print("Force field file saved.")
