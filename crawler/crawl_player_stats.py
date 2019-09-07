import csv
import time
import unicodedata

import numpy as np
from selenium import webdriver


def next_page(driver):
    next_button = driver.find_element_by_xpath(next_path)
    next_button.click()


def get_table(driver, writer):
    # table = driver.find_element_by_xpath(table_path)
    for row_number in range(1, 51):
        table = driver.find_element_by_xpath(table_path + '/div[{0}]'.format(str(row_number)))
        tb_txt = table.text.split('\n')
        player_name = unicodedata.normalize('NFKD', tb_txt[1]).encode('ascii', 'ignore')
        team = unicodedata.normalize('NFKD', tb_txt[2]).encode('ascii', 'ignore')
        position = unicodedata.normalize('NFKD', tb_txt[3]).encode('ascii', 'ignore')
        game = unicodedata.normalize('NFKD', tb_txt[4]).encode('ascii', 'ignore')
        game_play = unicodedata.normalize('NFKD', tb_txt[5]).encode('ascii', 'ignore')
        goals = unicodedata.normalize('NFKD', tb_txt[6]).encode('ascii', 'ignore')
        assists = unicodedata.normalize('NFKD', tb_txt[7]).encode('ascii', 'ignore')
        points = unicodedata.normalize('NFKD', tb_txt[8]).encode('ascii', 'ignore')
        pm = unicodedata.normalize('NFKD', tb_txt[9]).encode('ascii', 'ignore')
        pim = unicodedata.normalize('NFKD', tb_txt[10]).encode('ascii', 'ignore')
        ppg = unicodedata.normalize('NFKD', tb_txt[11]).encode('ascii', 'ignore')
        ppp = unicodedata.normalize('NFKD', tb_txt[12]).encode('ascii', 'ignore')
        shg = unicodedata.normalize('NFKD', tb_txt[13]).encode('ascii', 'ignore')
        shp = unicodedata.normalize('NFKD', tb_txt[14]).encode('ascii', 'ignore')
        gwg = unicodedata.normalize('NFKD', tb_txt[15]).encode('ascii', 'ignore')
        otg = unicodedata.normalize('NFKD', tb_txt[16]).encode('ascii', 'ignore')
        s = unicodedata.normalize('NFKD', tb_txt[17]).encode('ascii', 'ignore')
        s_p = unicodedata.normalize('NFKD', tb_txt[18]).encode('ascii', 'ignore')
        toi_pg = int(tb_txt[19].split(':')[0]) * 60 + int(tb_txt[19].split(':')[1])
        shift_pg = unicodedata.normalize('NFKD', tb_txt[20]).encode('ascii', 'ignore')
        fow = unicodedata.normalize('NFKD', tb_txt[21]).encode('ascii', 'ignore')
        writer.writerow(
            {'Player': player_name, 'Team': team, 'Pos': position, 'Game': game, 'GP': game_play, 'G': goals,
             'A': assists, 'P': points, '+/-': pm, 'PIM': pim, 'PPG': ppg, "PPP": ppp, 'SHG': shg, 'SHP': shp,
             'GWG': gwg, 'OTG': otg, 'S': s, 'S%': s_p, 'TOI/GP': toi_pg, 'Shifts/GP': shift_pg, 'FOW%': fow})


if __name__ == '__main__':
    # summary
    # table_path = '//*[@id="stats-page-body"]/div[2]/div[2]/div[1]'
    table_path = '//*[@id="stats-page-body"]/div[2]/div[2]/div[1]/div[3]'
    # '//*[@id="stats-page-body"]/div[2]/div[2]/div[1]/div[3]/div[1]'
    next_path = '//*[@id="stats-page-body"]/div[2]/div[2]/div[2]/div/div[3]/button'
    driver = webdriver.Chrome('./driver_dir/chromedriver')
    driver.get(
        "http://www.nhl.com/stats/player?reportType=game&dateFrom=2018-10-03&dateTo=2019-06-12"
        "&gameType=2&filter=gamesPlayed,gte,1&sort=points,goals,assists")
    time.sleep(8)
    with open('NHL_game_by_game_stats.csv', 'w') as csvfile:
        fieldnames = ['Player', 'Team', 'Pos', 'Game', 'GP', 'G', 'A', 'P', '+/-', 'PIM', 'PPG',
                      'PPP', 'SHG', 'SHP', 'GWG', 'OTG', 'S', 'S%', 'TOI/GP', 'Shifts/GP', 'FOW%']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        page_num = 916
        for i in range(page_num):
            print('working on page {0}'.format(str(i)))
            get_table(driver, writer)
            next_page(driver)
            time.sleep(2)
