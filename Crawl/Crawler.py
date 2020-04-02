# -*- coding: utf-8 -*-
"""
Created on Fri Feb 3 12:40:12 2019

@author: Sidong Feng
"""
import os
import argparse
import urllib.request
from time import perf_counter 
from selenium import webdriver

parser = argparse.ArgumentParser(description='Dribbble Crawler')
parser.add_argument('--chromedriver', type=str, default='/usr/local/bin/chromedriver',
                    help='Chrome Driver Location (default: /usr/local/bin/chromedriver)')
parser.add_argument('--ipath', type=str, default='./images/',
                    help='Image Save Location (default: ./images/)')
parser.add_argument('--mpath', type=str, default='./Metadata.csv',
                    help='Metadata Save Location (default: ./Metadata.csv)')
parser.add_argument('--headless', action='store_false',
                    help='Chrome headless (default: true)')
args = parser.parse_args()

IMG_PATH = args.ipath
METADATA_PATH = args.mpath
CHROMEDRIVER_PATH = args.chromedriver

def Crawl():
    print("#"*20)
    print("Start Crawling ...")
    print("#"*20)
    # start from backup
    f = open(METADATA_PATH,'r')
    lines = f.readlines()
    f.close()
    if len(lines) == 1:
        no = 0
    else:
        no = int(lines[-1].split(',')[0])+1
    # web option
    options = webdriver.ChromeOptions()
    if args.headless:
        options.add_argument("--headless")
    options.add_argument("--window-size=1920,1080")
    driver = webdriver.Chrome(CHROMEDRIVER_PATH, options=options)
    # crawl information
    t1_start = perf_counter()
    t1_stop = 0
    while t1_stop - t1_start < 1000:
        url = 'https://dribbble.com/shots/'+str(no)
        driver.get(url)
        # error page
        if len(driver.find_elements_by_xpath("//*[contains(text(), 'Whoops, that page is gone.')]"))>0:
            no += 1
            t1_stop = perf_counter() 
            continue
        new_frame = str(no) + ','
        # title
        title = driver.find_element_by_class_name('shot-title').text.replace(",","")
        new_frame += title + ","
        # by and by_href
        try:
            by = driver.find_element_by_xpath("//span[@class='shot-byline-user']/a").text.replace(",","")
            by_href = driver.find_element_by_xpath("//span[@class='shot-byline-user']/a").get_attribute("href")
        except:
            by = driver.find_element_by_class_name('shot-byline-user').text.replace(",","").replace("by ","")
            by_href = " "
        new_frame += by + ',' + by_href + ','
        # for and for_href
        try:
            fo = driver.find_element_by_xpath("//span[@class='shot-byline-team']/a").text.replace(",","")
            fo_href = driver.find_element_by_xpath("//span[@class='shot-byline-team']/a").get_attribute("href")
        except:
            fo = " "
            fo_href = " "
        new_frame += fo + ',' + fo_href + ','
        # tags
        tags = [driver.find_element_by_xpath("//div[@class='shot-tags']/ol/li[@id='"+el.get_attribute('id')+"']/a").text for el in driver.find_elements_by_xpath("//div[@class='shot-tags']/ol/li[contains(@id,'tag-li')]")]
        tags = "      ".join(tags)
        new_frame += tags + ','
        # color
        colors = driver.find_elements_by_xpath("//div[@class='shot-colors']/ul/li")
        colors = [x.text for x in colors]
        colors = "+".join(colors)
        new_frame += colors + ','
        # view, like, save, date
        try:
            views = driver.find_element_by_class_name('shot-views').text.replace(",","")
        except:
            views = " "
        try:
            likes = driver.find_element_by_xpath("//div[@class='shot-likes']/a").text.replace(",","")
        except:
            likes = " "
        try:
            saves = driver.find_element_by_xpath("//div[@class='shot-saves']/a").text.replace(",","")
        except:
            saves = " "
        date = driver.find_element_by_xpath("//div[@class='shot-date']/a").text.replace(",","")
        new_frame += views + ',' + likes + ',' + saves + ',' + date + '\n'
        # save image
        image_elements = driver.find_element_by_class_name("detail-shot").get_attribute('data-img-src')
        image_format = image_elements.rsplit(".")[-1]
        urllib.request.urlretrieve(image_elements, IMG_PATH+str(no)+'.'+image_format)
        # write file
        f = open(METADATA_PATH, 'a')
        f.write(new_frame)
        f.close()
        # iterate
        no += 1
        t1_stop = perf_counter() 
    print("#"*20)
    print("Error: Internet Dead or Timeout!")
    print("Totally Crawl: ", no)
    print("#"*20)
    driver.close()

if __name__ == '__main__': 
    print('-'*10)
    print(args)
    print('-'*10)
    """ Intial """
    if not os.path.exists(IMG_PATH):
        os.mkdir(IMG_PATH)
    if not os.path.exists(METADATA_PATH):
        f = open(METADATA_PATH,'w')
        f.write('id,title,by,by_href,for,for_href,tags,colors,views,likes,saves,date\n')
        f.close

    """ Crawl Image and Metadata from Dribbble"""
    Crawl()