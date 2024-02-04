# Copyright (c) Tingzheng Hou.
# Distributed under the terms of the MIT License.

"""
This module implements a core class PubChemRunner for accessing PubChem data that
can be used to retrieve compound structure and information.
"""

from __future__ import annotations

import os
import time
from typing import Final
from urllib.parse import quote

import pubchempy as pcp
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from mdgo.util.reformat import sdf_to_pdb

MAESTRO: Final[str] = "$SCHRODINGER/maestro -console -nosplash"
FFLD: Final[str] = "$SCHRODINGER/utilities/ffld_server -imae {} -version 14 -print_parameters -out_file {}"
MolecularWeight: Final[str] = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{}/property/MolecularWeight/txt"
MODULE_DIR: Final[str] = os.path.dirname(os.path.abspath(__file__))
DATA_DIR: Final[str] = os.path.join(MODULE_DIR, "data")


class PubChemRunner:
    """Wrapper for accessing PubChem data that can be used to retriving compound
    structure and information.

    Args:
        write_dir: Directory for writing output.
        chromedriver_dir: Directory to the ChromeDriver executable.
        api: Whether to use the PUG REST web interface for accessing
            PubChem data. If None, then all search/download will be
            performed via web browser mode. Default to True.
        headless: Whether to run Chrome in headless (silent) mode.
            Default to False.

    Examples:
        >>> web = PubChemRunner('/path/to/work/dir', '/path/to/chromedriver')
        >>> long_name, short_name = "ethylene carbonate", "PC"
        >>> cid = web.obtain_entry(long_name, short_name)
    """

    def __init__(
        self,
        write_dir: str,
        chromedriver_dir: str,
        api: bool = True,
        headless: bool = False,
    ):
        """Base constructor."""
        self.write_dir = write_dir
        self.api = api
        if not self.api:
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
            if headless:
                self.options.add_argument("--headless")
            self.options.add_experimental_option("prefs", self.preferences)
            self.options.add_experimental_option("excludeSwitches", ["enable-automation"])
            self.web = webdriver.Chrome(options=self.options, service=self.server)
            self.wait = WebDriverWait(self.web, 10)
            self.web.get("https://pubchem.ncbi.nlm.nih.gov/")
            time.sleep(1)
            print("PubChem server connected.")

    def quit(self):
        """
        Method for quiting ChromeDriver.

        """
        if not self.api:
            self.web.quit()

    def obtain_entry(self, search_text: str, name: str, output_format: str = "sdf") -> str | None:
        """
        Search the PubChem database with a text entry and save the
        structure in desired format.

        Args:
            search_text: The text to use as a search query.
            name: The short name for the molecule.
            output_format: The output format of the structure.
                Default to sdf.
        """
        if self.api:
            return self._obtain_entry_api(search_text, name, output_format=output_format)
        return self._obtain_entry_web(search_text, name, output_format=output_format)

    def smiles_to_pdb(self, smiles: str):
        """
        Obtain pdf file based on SMILES code.

        Args:
            smiles: SMILES code.

        Returns:

        """
        convertor_url = "https://cactus.nci.nih.gov/translate/"
        input_xpath = "/html/body/div/div[2]/div[1]/form/table[1]/tbody/tr[2]/td[1]/input[1]"
        pdb_xpath = "/html/body/div/div[2]/div[1]/form/table[1]/tbody/tr[2]/td[2]/div/input[4]"
        translate_xpath = "/html/body/div/div[2]/div[1]/form/table[2]/tbody/tr/td/input[2]"
        download_xpath = "/html/body/center/b/a"
        self.web.get(convertor_url)
        self.web.find_element(By.XPATH, input_xpath).clear()
        self.web.find_element(By.XPATH, input_xpath).send_keys(smiles)
        self.web.find_element(By.XPATH, pdb_xpath).click()
        self.web.find_element(By.XPATH, translate_xpath).click()
        time.sleep(1)
        self.web.find_element(By.XPATH, download_xpath).click()
        print("Waiting for downloads.", end="")
        time.sleep(1)
        while any(filename.endswith(".crdownload") for filename in os.listdir(self.write_dir)):
            time.sleep(1)
            print(".", end="")
        print("\nStructure file saved.")

    def _obtain_entry_web(self, search_text: str, name: str, output_format: str) -> str | None:
        cid = None

        try:
            query = quote(search_text)
            url = "https://pubchem.ncbi.nlm.nih.gov/#query=" + query
            self.web.get(url)
            loaded_element_path = '//*[@id="main-results"]/div[1]/div/ul'
            self.wait.until(EC.presence_of_element_located((By.XPATH, loaded_element_path)))
            best_xpath = '//*[@id="featured-results"]/div/div[2]' "/div/div[1]/div[2]/div[1]/a/span/span"
            relevant_xpath = (
                '//*[@id="collection-results-container"]'
                "/div/div/div[2]/ul/li[1]/div/div/div[1]"
                "/div[2]/div[1]/a/span/span"
            )
            if EC.presence_of_element_located((By.XPATH, best_xpath)):
                match = self.web.find_element(By.XPATH, best_xpath)
            else:
                match = self.web.find_element(By.XPATH, relevant_xpath)
            self.wait.until(EC.element_to_be_clickable(match))
            match.click()
            time.sleep(1)
            # density_locator = '//*[@id="Density"]/div[2]/div[1]/p'
            cid_locator = '//*[@id="Title-and-Summary"]/div/div/div/div[1]/div[2]'
            smiles_locator = '//*[@id="Canonical-SMILES"]/div[2]/div[1]'
            self.wait.until(EC.presence_of_element_located((By.XPATH, cid_locator)))
            cid = self.web.find_element(By.XPATH, cid_locator).text
            smiles = self.web.find_element(By.XPATH, smiles_locator).text
            print("Best match found, PubChem ID:", cid)
            if output_format.lower() == "smiles":
                print("SMILES code:", smiles)
            elif output_format.lower() == "pdb":
                self.smiles_to_pdb(smiles)
            else:
                self.web.get(
                    f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/CID/"
                    f"{cid}/record/{output_format.upper()}/?record_type=3d&"
                    f'response_type=save&response_basename={name + "_" + cid}'
                )
                print("Waiting for downloads.", end="")
                time.sleep(1)
                while any(filename.endswith(".crdownload") for filename in os.listdir(self.write_dir)):
                    time.sleep(1)
                    print(".", end="")
                print("\nStructure file saved.")
        except TimeoutException:
            print("Timeout! Web server no response for 10s, file download failed!")
        except NoSuchElementException:
            print(
                "The download link was not correctly generated, "
                "file download failed!\n"
                "Please try another search text or output format."
            )
        finally:
            self.quit()
        return cid

    def _obtain_entry_api(self, search_text, name, output_format) -> str | None:
        cid = None
        cids = pcp.get_cids(search_text, "name", record_type="3d")
        if len(cids) == 0:
            print("No exact match found, please try the web search")
        else:
            cid = str(cids[0])
            if output_format.lower() == "smiles":
                compound = pcp.Compound.from_cid(int(cid))
                print("SMILES code:", compound.canonical_smiles)
            elif output_format.lower() == "pdb":
                sdf_file = os.path.join(self.write_dir, name + "_" + cid + ".sdf")
                pdb_file = os.path.join(self.write_dir, name + "_" + cid + ".pdb")
                pcp.download("SDF", sdf_file, cid, record_type="3d", overwrite=True)
                sdf_to_pdb(sdf_file, pdb_file)
            else:
                pcp.download(
                    output_format.upper(),
                    os.path.join(self.write_dir, name + "_" + cid + "." + output_format.lower()),
                    cid,
                    record_type="3d",
                    overwrite=True,
                )
        return cid
