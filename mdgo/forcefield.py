from pymatgen.io.lammps.data import LammpsData
from mdgo.util import mass_to_name
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import os
import shutil

CHROME = "/Users/th/Downloads/package/chromedriver/chromedriver"


class FFcrawler:
    def __init__(self, write_path, chromedriver_path, headless=True, xyz=False):
        self.write_path = write_path
        self.xyz = xyz
        self.preferences = {"download.default_directory": write_path,
                            "safebrowsing.enabled": "false",
                            "profile.managed_default_content_settings.images": 2}
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
        self.web = webdriver.Chrome(chromedriver_path, options=self.options)
        self.wait = WebDriverWait(self.web, 10)
        self.web.get("http://zarbi.chem.yale.edu/ligpargen/")
        time.sleep(1)
        print("LigParGen server opened.")

    def data_from_pdb(self, pdb_path):
        upload = self.web.find_element_by_xpath('//*[@id="exampleMOLFile"]')
        upload.send_keys(pdb_path)
        submit = self.web.find_element_by_xpath(
            '/html/body/div[2]/div/div[2]/form/button[1]')
        submit.click()
        pdb_filename = os.path.basename(pdb_path)
        self.download_lmp(os.path.splitext(pdb_filename)[0] + ".lmp")
        self.web.quit()

    def data_from_smiles(self, smiles_code):
        smile = self.web.find_element_by_xpath('//*[@id="smiles"]')
        smile.send_keys(smiles_code)
        submit = self.web.find_element_by_xpath(
            '/html/body/div[2]/div/div[2]/form/button[1]')
        submit.click()
        self.download_lmp(smiles_code + '.lmp')
        self.web.quit()

    def download_lmp(self, lmp_name):
        print("Structure info uploaded. Rendering force field...")
        self.wait.until(
            EC.presence_of_element_located((By.NAME, 'go')))
        data = self.web.find_element_by_xpath('/html/body/div[2]/div[2]/div[1]/'
                                              'div/div[14]/form/input[1]')
        data.click()
        print("Force field file downloaded.")
        time.sleep(1)
        data_file = max(
            [self.write_path + "/" + f for f
             in os.listdir(self.write_path)
             if os.path.splitext(f)[1] == ".lmp"],
            key=os.path.getctime)
        if self.xyz:
            data = LammpsData.from_file(data_file)
            element_id_dict = mass_to_name(data.masses)
            coords = data.atoms[['type', 'x', 'y', 'z']]
            lines = list()
            lines.append(str(len(coords.index)))
            lines.append("")
            for _, r in coords.iterrows():
                line = element_id_dict.get(int(r['type'])) + ' ' + ' '.join(
                    str(r[loc]) for loc in ["x", "y", "z"])
                lines.append(line)

            with open(os.path.join(self.write_path, lmp_name + ".xyz"),
                      "w") as xyz_file:
                xyz_file.write("\n".join(lines))
            print(".xyz file saved.")
        shutil.move(data_file, os.path.join(self.write_path, lmp_name))
        print("Force field file saved.")


def main():
    instance = FFcrawler("/Users/th/Downloads/test_selenium", CHROME, xyz=True)
    instance.data_from_pdb("/Users/th/Downloads/test_selenium/EMC.pdb")


if __name__ == "__main__":
    main()
