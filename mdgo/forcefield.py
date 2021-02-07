from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import os
import shutil


DOWNLOAD = "/Users/th/Downloads/test_selenium"
PDB = "/Users/th/Downloads/test_selenium/EMC.pdb"
CHROME = "/Users/th/Downloads/package/chromedriver/chromedriver"


class FFcrawler:
    def __init__(self, download_path, chromedriver_path, headless=True):
        self.download_path = download_path
        self.preferences = {"download.default_directory": download_path,
                            "safebrowsing.enabled": "false",
                            "profile.managed_default_content_settings.images": 2}
        self.options = webdriver.ChromeOptions()
        self.options.add_argument("--window-size=1920,1080")
        if headless:
            self.options.add_argument('--headless')
        self.options.add_experimental_option("prefs", self.preferences)
        self.options.add_experimental_option('excludeSwitches',
                                             ['enable-automation'])
        self.web = webdriver.Chrome(chromedriver_path, options=self.options)
        self.wait = WebDriverWait(self.web, 10)

    def data_from_pdb(self, pdb_path):
        self.web.get("http://zarbi.chem.yale.edu/ligpargen/")
        time.sleep(1)
        upload = self.web.find_element_by_xpath('//*[@id="exampleMOLFile"]')
        upload.send_keys(pdb_path)
        submit = self.web.find_element_by_xpath(
            '/html/body/div[2]/div/div[2]/form/button[1]')
        submit.click()
        pdb_filename = os.path.basename(pdb_path)
        self.download_lmp(os.path.splitext(pdb_filename)[0] + ".lmp")
        self.web.quit()

    def data_from_smiles(self, smiles_code):
        self.web.get("http://zarbi.chem.yale.edu/ligpargen/")
        time.sleep(1)
        smile = self.web.find_element_by_xpath('//*[@id="smiles"]')
        smile.send_keys(smiles_code)
        submit = self.web.find_element_by_xpath(
            '/html/body/div[2]/div/div[2]/form/button[1]')
        submit.click()
        self.download_lmp(smiles_code + '.lmp')
        self.web.quit()

    def download_lmp(self, lmp_name):
        self.wait.until(
            EC.presence_of_element_located((By.NAME, 'go')))
        data = self.web.find_element_by_xpath('/html/body/div[2]/div[2]/div[1]/'
                                              'div/div[14]/form/input[1]')
        data.click()
        time.sleep(1)
        data_file = max(
            [self.download_path + "/" + f for f
             in os.listdir(self.download_path)
             if os.path.splitext(f)[1] == ".lmp"],
            key=os.path.getctime)
        shutil.move(data_file, os.path.join(self.download_path, lmp_name))


instance = FFcrawler(DOWNLOAD, CHROME, headless=False)
instance.data_from_pdb(PDB)


