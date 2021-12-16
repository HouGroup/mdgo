# coding: utf-8
# Copyright (c) Tingzheng Hou.
# Distributed under the terms of the MIT License.

"""
This module implements two core class FFcrawler and MaestroRunner
for generating LAMMPS/GROMACS data files from molecule structure using
the LigParGen web server and Maestro, respectively.

For using the FFcrawler class:

  * Download the ChromeDriver executable that
    matches your Chrome version via https://chromedriver.chromium.org/downloads

For using the MaestroRunner class:

  * Download a free Maestro via https://www.schrodinger.com/freemaestro

  * Install the package and set the environment variable $SCHRODINGER
    (e.g. 'export SCHRODINGER=/opt/schrodinger/suites2021-4', please
    check https://www.schrodinger.com/kb/446296 or
    https://www.schrodinger.com/kb/1842 for details.

"""
import os
import re
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from string import Template
from typing import Optional, Union
from urllib.parse import quote

import numpy as np
import pubchempy as pcp
from monty.json import MSONable
from monty.serialization import loadfn
from pymatgen.core import Lattice, Structure
from pymatgen.core.ion import Ion
from pymatgen.io.lammps.data import ForceField, LammpsData, Topology, lattice_2_lmpbox
from selenium import webdriver
from selenium.common.exceptions import (
    NoSuchElementException,
    TimeoutException,
    WebDriverException,
)
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from typing_extensions import Final, Literal

from mdgo.util import ff_parser, lmp_mass_to_name, sdf_to_pdb

__author__ = "Tingzheng Hou"
__version__ = "1.0"
__maintainer__ = "Tingzheng Hou"
__email__ = "tingzheng_hou@berkeley.edu"
__date__ = "Feb 9, 2021"

MAESTRO: Final[str] = "$SCHRODINGER/maestro -console -nosplash"
FFLD: Final[str] = "$SCHRODINGER/utilities/ffld_server -imae {} -version 14 -print_parameters -out_file {}"
MolecularWeight: Final[str] = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{}/property/MolecularWeight/txt"
MODULE_DIR: Final[str] = os.path.dirname(os.path.abspath(__file__))
DATA_DIR: Final[str] = os.path.join(MODULE_DIR, "data")
DATA_MODELS: Final[dict] = {
    "water": {
        "spc": "water_spc.lmp",
        "spce": "water_spce.lmp",
        "tip3pew": "water_tip3p_ew.lmp",
        "tip3pfb": "water_tip3p_fb.lmp",
        "opc3": "water_opc3.lmp",
        "tip4p2005": "water_tip4p_2005.lmp",
        "tip4pew": "water_tip4p_ew.lmp",
        "tip4pfb": "water_tip4p_fb.lmp",
        "opc": "water_opc.lmp",
    },
}
WATER_SIGMA: Final[dict] = {
    "spc": 3.16557,
    "spce": 3.16557,
    "tip3p": 3.1507,
    "tip3pew": 3.188,
    "tip3pfb": 3.178,
    "opc": 3.16655,
    "opc3": 3.17427,
    "tip4p": 3.1536,
    "tip4p2005": 3.1589,
    "tip4pew": 3.16435,
    "tip4pfb": 3.1655,
}


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
        if chromedriver_dir is None:
            self.web = webdriver.Chrome(options=self.options)
        else:
            self.web = webdriver.Chrome(chromedriver_dir, options=self.options)
        self.wait = WebDriverWait(self.web, 10)
        self.web.get("http://zarbi.chem.yale.edu/ligpargen/")
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
        self.web.get("http://zarbi.chem.yale.edu/ligpargen/")
        time.sleep(1)
        upload = self.web.find_element_by_xpath('//*[@id="exampleMOLFile"]')
        try:
            upload.send_keys(pdb_dir)
            submit = self.web.find_element_by_xpath("/html/body/div[2]/div/div[2]/form/button[1]")
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
        self.web.get("http://zarbi.chem.yale.edu/ligpargen/")
        time.sleep(1)
        smile = self.web.find_element_by_xpath('//*[@id="smiles"]')
        smile.send_keys(smiles_code)
        submit = self.web.find_element_by_xpath("/html/body/div[2]/div/div[2]/form/button[1]")
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
        self.wait.until(EC.presence_of_element_located((By.NAME, "go")))
        data_lmp = self.web.find_element_by_xpath("/html/body/div[2]/div[2]/div[1]/div/div[14]/form/input[1]")
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
            data_gro = self.web.find_element_by_xpath("/html/body/div[2]/div[2]/div[1]/div/div[8]/form/input[1]")
            data_itp = self.web.find_element_by_xpath("/html/body/div[2]/div[2]/div[1]/div/div[9]/form/input[1]")
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


class MaestroRunner:
    """
    Wrapper for the Maestro software that can be used to generate the OPLS_2005
    force field parameter for a molecule.

    Args:
        structure_dir: Path to the structure file.
            Supported input format please check
            https://www.schrodinger.com/kb/1278
        working_dir: Directory for writing intermediate
            and final output.
        out: Force field output form. Default to "lmp",
            the data file for LAMMPS. Other supported formats
            are under development.
        cmd_template: String template for input script
            with placeholders. Default to None, i.e., using
            the default template.
        assign_bond: Whether to assign bond to the input
            structure. Default to None.

    Supported input format please check https://www.schrodinger.com/kb/1278

    The OPLS_2005 parameters are described in

    Banks, J.L.; Beard, H.S.; Cao, Y.; Cho, A.E.; Damm, W.; Farid, R.;
    Felts, A.K.; Halgren, T.A.; Mainz, D.T.; Maple, J.R.; Murphy, R.;
    Philipp, D.M.; Repasky, M.P.; Zhang, L.Y.; Berne, B.J.; Friesner, R.A.;
    Gallicchio, E.; Levy. R.M. Integrated Modeling Program, Applied Chemical
    Theory (IMPACT). J. Comp. Chem. 2005, 26, 1752.

    The OPLS_2005 parameters are located in

    $SCHRODINGER/mmshare-vversion/data/f14/

    Examples:

        >>> mr = MaestroRunner('/path/to/structure', '/path/to/working/dir')
        >>> mr.get_mae()
        >>> mr.get_ff()
    """

    template_assignbond = os.path.join(MODULE_DIR, "templates", "mae_cmd_assignbond.txt")

    template_noassignbond = os.path.join(MODULE_DIR, "templates", "mae_cmd_noassignbond.txt")

    def __init__(
        self,
        structure_dir: str,
        working_dir: str,
        out: str = "lmp",
        cmd_template: Optional[str] = None,
        assign_bond: bool = False,
    ):
        """Base constructor."""
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
            if assign_bond:
                with open(self.template_assignbond, "r") as f:
                    cmd_template = f.read()
                self.cmd_template = cmd_template
            else:
                with open(self.template_noassignbond, "r") as f:
                    cmd_template = f.read()
                self.cmd_template = cmd_template

    def get_mae(self, wait: float = 30):
        """Write a Maestro command script and execute it to generate a
        maestro file containing all the info needed.

        Args:
            wait: The time waiting for Maestro execution in seconds. Default to 30.
        """
        with open(self.cmd, "w") as f:
            cmd_template = Template(self.cmd_template)
            cmd_script = cmd_template.substitute(file=self.structure, mae=self.mae, xyz=self.xyz)
            f.write(cmd_script)
        try:
            p = subprocess.Popen(  # pylint: disable=consider-using-with
                f"{MAESTRO} -c {self.cmd}",
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid,
            )
        except subprocess.CalledProcessError as e:
            raise ValueError(f"Maestro failed with errorcode {e.returncode}  and stderr: {e.stderr}") from e

        counter = 0
        while not os.path.isfile(self.mae + ".mae"):
            time.sleep(1)
            counter += 1
            if counter > wait:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
                raise TimeoutError(f"Failed to generate Maestro file in {wait} secs!")
        print("Maestro file generated!")
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)

    def get_ff(self):
        """Read the Maestro file and save the force field as LAMMPS data file."""
        try:
            subprocess.run(
                FFLD.format(self.mae + ".mae", self.ff),
                check=True,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except subprocess.CalledProcessError as e:
            raise ValueError(f"Maestro failed with errorcode {e.returncode} and stderr: {e.stderr}") from e
        print("Maestro force field file generated.")
        if self.out:
            if self.out == "lmp":
                with open(os.path.join(self.work, self.name + "." + self.out), "w") as f:
                    f.write(ff_parser(self.ff, self.xyz))
                print("LAMMPS data file generated.")
            else:
                print("Output format not supported, ff format not converted.")


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
            self.web = webdriver.Chrome(chromedriver_dir, options=self.options)
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

    def obtain_entry(self, search_text: str, name: str, output_format: str = "sdf") -> Optional[str]:
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
        self.web.find_element_by_xpath(input_xpath).clear()
        self.web.find_element_by_xpath(input_xpath).send_keys(smiles)
        self.web.find_element_by_xpath(pdb_xpath).click()
        self.web.find_element_by_xpath(translate_xpath).click()
        time.sleep(1)
        self.web.find_element_by_xpath(download_xpath).click()
        print("Waiting for downloads.", end="")
        time.sleep(1)
        while any(filename.endswith(".crdownload") for filename in os.listdir(self.write_dir)):
            time.sleep(1)
            print(".", end="")
        print("\nStructure file saved.")

    def _obtain_entry_web(self, search_text: str, name: str, output_format: str) -> Optional[str]:
        cid = None

        try:
            query = quote(search_text)
            url = "https://pubchem.ncbi.nlm.nih.gov/#query=" + query
            self.web.get(url)
            time.sleep(1)
            best_xpath = '//*[@id="featured-results"]/div/div[2]' "/div/div[1]/div[2]/div[1]/a/span/span"
            relevant_xpath = (
                '//*[@id="collection-results-container"]'
                "/div/div/div[2]/ul/li[1]/div/div/div[1]"
                "/div[2]/div[1]/a/span/span"
            )
            if EC.presence_of_element_located((By.XPATH, best_xpath)):
                match = self.web.find_element_by_xpath(best_xpath)
            else:
                match = self.web.find_element_by_xpath(relevant_xpath)
            match.click()
            # density_locator = '//*[@id="Density"]/div[2]/div[1]/p'
            cid_locator = '//*[@id="main-content"]/div/div/div[1]/' "div[3]/div/table/tbody/tr[1]/td"
            smiles_locator = '//*[@id="Canonical-SMILES"]/div[2]/div[1]/p'
            self.wait.until(EC.presence_of_element_located((By.XPATH, cid_locator)))
            cid = self.web.find_element_by_xpath(cid_locator).text
            smiles = self.web.find_element_by_xpath(smiles_locator).text
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

    def _obtain_entry_api(self, search_text, name, output_format) -> Optional[str]:
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


@dataclass
class IonLJData(MSONable):
    """
    A lightweight dataclass for storing ion force field parameters. The data
    file ion_lj_params.json distributed with mdgo is a serialized list of these
    objects.

    Attributes:
        name: The name of the parameter set
        formula: formula of the ion, e.g. 'Li+'
        combining_rule: the method used to compute pairwise interaction parameters
            from single atom parameters. 'geometric' or 'LB' for Lorentz-Berthelot
        water_model: The water model for which the ion parameters were optimized.
        sigma: The Lennard Jones sigma value, in Å
        epsilon: The Lennard Jones epsilon value, in kcal/mol
    """

    name: Literal["jensen_jorgensen", "joung_cheatham", "li_merz"]
    formula: str
    combining_rule: Literal["geometric", "LB"]
    water_model: Literal[
        "spc",
        "spce",
        "tip3p",
        "tip3pew",
        "tip3pfb",
        "opc3",
        "tip4p2005",
        "tip4p",
        "tip4pew",
        "tip4pfb",
        "opc",
    ]
    sigma: float
    epsilon: float


class Aqueous:
    """
    A class for retreiving water and ion force field parameters.

    Available water models are:
        1. SPC
        2. SPC/E
        3. TIP3P-EW
        4. TIP3P-FB
        5. OPC3
        6. TIP4P-EW
        7. TIP4P-2005
        8. TIP4P-FB
        9. OPC

    Multiple sets of Lennard Jones parameters for ions are available as well.
    Not every set is available for every water model. The parameter sets included
    are:
        1. Jensen and Jorgensen, 2006 (abbreviation 'jj')
        2. Joung and Cheatham, 2008 (abbreviation 'jc')
        3. Li and Merz group, 2020 (abbreviation, 'lm')

    Examples:
        Retreive SPC/E water model:
        >>> spce_data = Aqueous.get_water()
        Retreive Li+ ion by Jensen and Jorgensen:
        >>> li_data = Aqueous.get_ion(model="jj", ion="li+")
        Retreive a customized water data file:
        >>> spce_data = Aqueous.get_ion(file_name="path/to/data/file")
    """

    @staticmethod
    def get_water(model: str = "spce") -> LammpsData:
        """
        Retrieve water model parameters.

        Args:
            model: Water model to use. Valid choices are "spc", "spce", "opc3",
                "tip3pew", "tip3pfb", "tip4p2005", "tip4pew", "tip4pfb", and "opc".
                (Default: "spce")
        Returns:
            LammpsData: Force field parameters for the chosen water model.
                If you specify an invalid water model, None is returned.
        """
        signature = "".join(re.split(r"[\W|_]+", model)).lower()
        if DATA_MODELS["water"].get(signature):
            return LammpsData.from_file(os.path.join(DATA_DIR, "water", DATA_MODELS["water"].get(signature)))
        raise ValueError("Water model not found. Please specify a customized data path or try another water model.\n")

    @staticmethod
    def get_ion(
        ion: Union[Ion, str],
        parameter_set: str = "auto",
        water_model: str = "auto",
        mixing_rule: Optional[str] = None,
    ) -> LammpsData:
        """
        Retrieve force field parameters for an ion in water.

        Args:
            ion: Formula of the ion (e.g., "Li+"). Not case sensitive. May be
                passed as either a string or an Ion object.
            parameter_set: Force field parameters to use for ions.
                Valid choices are:
                    1. "jj" for the Jensen and Jorgensen parameters (2006)"
                    2. "jc" for Joung-Cheatham parameters (2008)
                    3. "lm" for the Li and Merz group parameters (2020-2021)"
                The default parameter set is "auto", which assigns a recommended
                parameter set that is compatible with the chosen water model.
            water_model: Water model to use. Models must be given as a string
                (not case sensitive). "-" and "/" are ignored. Hence "tip3pfb"
                and "TIP3P-FB" are both valid inputs for the TIP3P-FB water model.
                Available water models are:
                    1. SPC
                    2. SPC/E
                    3. TIP3P-EW
                    4. TIP3P-FB
                    5. OPC3
                    6. TIP4P-EW
                    7. TIP4P-2005
                    8. TIP4P-FB
                    9. OPC
                The default water model is "auto", which assigns a recommended
                water model that is compatible with the chosen ion parameters. Other
                combinations are possible at your own risk. See documentation.

            When both the parameter_set and water_model are set to "auto", the function returns the
            Joung-Cheatham parameters for the SPC/E water model.

                For a systematic comparison of the performance of different water models, refer to

                    Sachini et al., Systematic Comparison of the Structural and Dynamic Properties of
                    Commonly Used Water Models for Molecular Dynamics Simulations. J. Chem. Inf. Model.
                    2021, 61, 9, 4521–4536. https://doi.org/10.1021/acs.jcim.1c00794

            mixing_rule: The mixing rule to use for the ion parameter. Default to None, which does not
                change the original mixing rule of the parameter set. Available choices are 'LB'
                (Lorentz-Berthelot or arithmetic) and 'geometric'. If the specified mixing rule does not
                match the default mixing rule of the parameter set, the output parameter will be converted
                accordingly.


        Returns:
            Force field parameters for the chosen water model.
        """
        alias = {"aq": "aqvist", "jj": "jensen_jorgensen", "jc": "joung_cheatham", "lm": "li_merz"}
        default_sets = {
            "spc": "N/A",
            "spce": "jc",
            "tip3p": "jc",
            "tip3pew": "N/A",
            "tip3pfb": "lm",
            "opc3": "lm",
            "tip4p2005": "N/A",
            "tip4p": "jj",
            "tip4pew": "jc",
            "tip4pfb": "lm",
            "opc": "lm",
            "jj": "tip4p",
            "jc": "spce",
            "lm": "tip4pfb",
        }
        water_model = water_model.replace("-", "").replace("/", "").lower()
        parameter_set = parameter_set.lower()

        if water_model == "auto" and parameter_set == "auto":
            water_model = "spce"
            parameter_set = "jc"
        elif parameter_set == "auto":
            parameter_set = default_sets.get(water_model, parameter_set)
            if parameter_set == "N/A":
                raise ValueError(
                    f"The {water_model} water model has no specifically parameterized ion parameter sets"
                    "Please try a different water model."
                )
        elif water_model == "auto":
            water_model = default_sets.get(parameter_set, water_model)

        parameter_set = alias.get(parameter_set, parameter_set)

        # Make the Ion object to get mass and charge
        if isinstance(ion, Ion):
            ion_obj = ion
        else:
            ion_obj = Ion.from_formula(ion.capitalize())

        # load ion data as a list of IonLJData objects
        ion_data = loadfn(os.path.join(DATA_DIR, "ion_lj_params.json"))

        # make sure the ion is in the DataFrame
        key = ion_obj.reduced_formula
        filtered_data = [d for d in ion_data if d.formula == key]
        if len(filtered_data) == 0:
            raise ValueError(f"Ion {key} not found in database. Please try a different ion.")

        # make sure the parameter set is in the DataFrame
        filtered_data = [d for d in filtered_data if d.name == parameter_set and d.water_model == water_model]
        if len(filtered_data) == 0:
            raise ValueError(
                f"No {parameter_set} parameters for water model {water_model} for ion {key}. "
                "See documentation and try a different combination."
            )

        if len(filtered_data) != 1:
            raise ValueError(
                f"Something is wrong: multiple ion data entries for {key}, {parameter_set}, and {water_model}"
            )

        # we only consider monatomic ions at present
        # construct a cubic LammpsBox from a lattice
        lat = Lattice([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        box = lattice_2_lmpbox(lat)[0]
        # put it in the center of a cubic Structure
        struct = Structure(lat, ion_obj, [[0.5, 0.5, 0.5]])
        # construct Topology with the ion centered in the box
        topo = Topology(struct, charges=[ion_obj.charge])

        # retrieve Lennard-Jones parameters
        # construct ForceField object
        sigma = filtered_data[0].sigma
        epsilon = filtered_data[0].epsilon
        if mixing_rule is None:
            pass
        else:
            default_mixing = filtered_data[0].combining_rule
            water_sigma = WATER_SIGMA.get(filtered_data[0].water_model)
            if mixing_rule.lower() in ["lb", "arithmetic", "lorentz-berthelot", "lorentz berthelot"]:
                mixing_rule = "LB"
            elif mixing_rule.lower() == "geometric":
                mixing_rule = "geometric"
            else:
                raise ValueError("Invalid mixing rule. Supported mixing rules are 'LB'(arithmetic) and 'geometric'. ")
            if default_mixing == mixing_rule:
                pass
            elif default_mixing == "LB" and mixing_rule == "geometric":
                sigma = ((water_sigma + sigma) / 2) ** 2 / water_sigma
                print(
                    "The parameter mixing rule has been converted from the original 'LB' to 'geometric'.\n"
                    "Please use the parameter set with caution!"
                )
            else:
                sigma = 2 * ((water_sigma * sigma) ** (1 / 2)) - water_sigma
                print(
                    "The parameter mixing rule has been converted from the original 'geometric' to 'LB'.\n"
                    "Please use the parameter set with caution!"
                )
        ff = ForceField([(str(e), e) for e in ion_obj.elements], nonbond_coeffs=[[epsilon, sigma]])

        return LammpsData.from_ff_and_topologies(box, ff, [topo], atom_style="full")


class ChargeWriter:
    """
    A class for write, overwrite, scale charges of a LammpsData object.
    TODO: Auto determine number of significant figures of charges
    TODO: write to obj or write separate charge file
    TODO: Read LammpsData or path

    Args:
        data: The provided LammpsData obj.
        precision: Number of significant figures.
    """

    def __init__(self, data: LammpsData, precision: int = 10):
        """Base constructor."""
        self.data = data
        self.precision = precision

    def scale(self, factor: float) -> LammpsData:
        """
        Scales the charge in of the in self.data and returns a new one. TODO: check if non-destructive

        Args:
            factor: The charge scaling factor

        Returns:
            A recreated LammpsData obj
        """
        items = {}
        items["box"] = self.data.box
        items["masses"] = self.data.masses
        atoms = self.data.atoms.copy(deep=True)
        atoms["q"] = atoms["q"] * factor
        assert np.around(atoms.q.sum(), decimals=self.precision) == np.around(
            self.data.atoms.q.sum() * factor, decimals=self.precision
        )
        digit_count = 0
        for q in atoms["q"]:
            rounded = self.count_significant_figures(q)
            if rounded > digit_count:
                digit_count = rounded
        print("No. of significant figures to output for charges: ", digit_count)
        items["atoms"] = atoms
        items["atom_style"] = self.data.atom_style
        items["velocities"] = self.data.velocities
        items["force_field"] = self.data.force_field
        items["topology"] = self.data.topology
        return LammpsData(**items)

    def count_significant_figures(self, number: float) -> int:
        """
        Count significant figures in a float.

        Args:
            number: The number to count.

        Returns:
            The number of significant figures.
        """
        number_str = repr(float(number))
        tokens = number_str.split(".")
        if len(tokens) > 2:
            raise ValueError(f"Invalid number '{number}' only 1 decimal allowed")
        if len(tokens) == 2:
            decimal_num = tokens[1][: self.precision].rstrip("0")
            return len(decimal_num)
        return 0


if __name__ == "__main__":
    # w = pcp.get_properties('MolecularWeight', 7303,)[0].get("MolecularWeight")
    # print(w)

    """
    pcr = PubChemRunner(
        "/Users/th/Downloads/test_pc/",
        "/Users/th/Downloads/package/chromedriver/chromedriver",
        api=True
    )
    long_name = "ethylene carbonate"
    short_name = "EC"
    cid = pcr.obtain_entry(long_name, short_name, "sdf")


    LPG = FFcrawler(
        "/Users/th/Downloads/test_selenium",
        "/Users/th/Downloads/package/chromedriver/chromedriver",
        xyz=True,
        gromacs=True
    )
    LPG.data_from_pdb("/Users/th/Downloads/test_selenium/EMC.pdb")

    MR = MaestroRunner("/Users/th/Downloads/test_mr/EC.sdf",
                       "/Users/th/Downloads/test_mr")
    MR.get_mae()
    MR.get_ff()

    pcr = PubChemCrawler(
        "/Users/th/Downloads/test_pc/",
        "/Users/th/Downloads/package/chromedriver/chromedriver",
        headless=True
    )
    long_name = "Propylene Carbonate"
    short_name = "PC"
    cid = pcr.obtain_entry(long_name, short_name)
    MR = MaestroRunner(
        f"/Users/th/Downloads/test_pc/{short_name}_{cid}.sdf",
        "/Users/th/Downloads/test_pc")
    MR.get_mae()
    MR.get_ff()

    pcr = PubChemRunner(
        "/Users/th/Downloads/test_pc/",
        "/Users/th/Downloads/package/chromedriver/chromedriver",
        api=True
    )
    long_name = "Ethyl Methyl Carbonate"
    short_name = "EMC"
    cid = pcr.obtain_entry(long_name, short_name)
    MR = MaestroRunner(
        f"/Users/th/Downloads/test_pc/{short_name}_{cid}.sdf",
        "/Users/th/Downloads/test_pc")
    MR.get_mae()
    MR.get_ff()

    pcr = PubChemRunner(
        "/Users/th/Downloads/test_mdgo/",
        "/Users/th/Downloads/package/chromedriver/chromedriver",
        api=True
    )
    long_name = "Ethyl Methyl Carbonate"
    short_name = "EMC"
    cid = pcr.obtain_entry(long_name, short_name, "pdb")
    """
    # lmp_data = Aqueous().get_ion(model="aq", ion="Na+")
    # print(lmp_data.get_string())
    pcr = PubChemRunner(
        "/Users/th/Downloads/test_mdgo/",
        "/Users/th/Downloads/package/chromedriver/chromedriver",
        api=True,
    )
    long_name = "Ethyl Methyl Carbonate"
    short_name = "EMC"
    obtained_cid = pcr.obtain_entry(long_name, short_name, "pdb")
