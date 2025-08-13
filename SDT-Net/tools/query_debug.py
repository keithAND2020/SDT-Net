import pdb
from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver import Edge,Chrome,ChromeOptions
from selenium.webdriver.chrome.options import Options

chrome_options = Options()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')

driver=Chrome(executable_path='/mnt/workspace/luyan/Fast_detection/tools/chromedriver',options=chrome_options)
url='http://simbad.cds.unistra.fr/simbad/sim-fcoo'
driver.get(url)
driver.maximize_window()

query_info = '14 30 37.42174998160 +44 22 58.0036271396'
query_bar=driver.find_element_by_name('Coord')
query_bar.send_keys(query_info)
submit_button=driver.find_element_by_name('submit')
submit_button.click()
try: #find basic infomation
    basic_info=driver.find_element_by_id('basic_data')
    target_name, target_kind = basic_info.text.split('\n')[0].split(' -- ')
except:
    target_name, target_kind = None, 'Not known'
if target_name is None:
    try:
        datatable = driver.find_elements_by_class_name("datatable")
        dist_asec = driver.find_elements_by_class_name("computed")[1].text
        target_kind = 'Otype-'+datatable[1].text.split(dist_asec)[1].split(' ')[1]
    except:
        pass

try: #find full screen button
    full_screen_button=driver.find_element_by_xpath('//*[@title="Full screen"]')
    image_searched_pred = True
except:
    image_searched_pred = False
if image_searched_pred:
    full_screen_button.click()
    #driver.save_screenshot(os.path.join('query',target_kind,save_name))
    driver.save_screenshot('tmp_query_pred.png')
    full_screen_button=driver.find_element_by_xpath('//*[@title="Restore original size"]')
    full_screen_button.click()
clear_button=driver.find_element_by_id('CLEAR')
clear_button.click()

driver.close()