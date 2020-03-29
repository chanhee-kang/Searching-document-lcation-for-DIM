# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 12:15:05 2019

@author: Administrator
"""
### Packages ###
import pandas as pd
import numpy as np
import pickle
from collections import defaultdict
from gensim.corpora.dictionary import Dictionary
from gensim.models.wrappers import DtmModel
from gensim.matutils import corpus2csc
from scipy.spatial.distance import cosine
from scipy.stats import linregress
import csv
import numpy as np
import seaborn

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
plt.rc('font', family='Malgun Gothic')

import pickle
import csv 
import numpy as np

#영문 논문#
EN_paper_all = pd.read_csv("/Users/chanhee.kang/Desktop/DIM_test/EN_paper_all_remove_des.csv")
EN_paper_all['Contents']
EN_paper_all.sort_values('PY',inplace=True) #날짜순 정렬
EN_paper_all.reset_index(drop=True,inplace=True)
EN_paper_all.to_csv("/Users/chanhee.kang/Desktop/DIM_test/EN_paper_all_remove_des2.csv")   

time_slice = EN_paper_all.Contents.groupby(EN_paper_all.PY).size().values

# DIM #
DIM_EN_paper = DtmModel.load('/Users/chanhee.kang/Desktop/DIM_test/EN_Paper_DIM.model')

DIM_EN_paper.influences_time # [time_slice] [document_no] [topic_no]

if __name__ == "__main__":
    output_filename = "output.csv"
    documents = "EN_paper_all_remove_des2.csv"
    show_data = 5 # 상위 몇개의 데이터를 출력할 것인지 
    data = DIM_EN_paper.influences_time

    if data:
        en_papers = open(documents, 'r', encoding='utf-8')
        en_papers = list(csv.reader(en_papers))
        
        out = open(output_filename, 'w', newline='')
        wr = csv.writer(out)    

        en_papers_header = ['PT', 'AU', 'BA', 'BE', 'GP', 'AF', 'BF', 'CA', 'TI', 'SO', 'SE', 'BS', 'LA', 'DT', 'CT', 'CY', 'CL', 'SP', 'HO', 'DE', 'ID', 'AB', 'C1', 'RP', 'EM', 'RI', 'OI', 'FU', 'FX', 'CR', 'NR', 'TC', 'Z9', 'U1', 'U2', 'PU', 'PI', 'PA', 'SN', 'EI', 'BN', 'J9', 'JI', 'PD', 'PY', 'VL', 'IS', 'PN', 'SU', 'SI', 'MA', 'BP', 'EP', 'AR', 'DI', 'D2', 'EA', 'EY', 'PG', 'WC', 'SC', 'GA', 'UT', 'PM', 'OA', 'HC', 'HP', 'DA', 'Query', 'Contents']
        all_data = [['time', 'topic', 'rank'] + en_papers_header]
        idx_cumulator = 1  # header 제외
        for i, time in enumerate(data):
            time = time.transpose()  # 2차원 배열을 열과행 대칭변환
            time_data = []
            for j, topic in enumerate(time, 1):
                sorted_docs = sorted(topic, reverse=True) # True=높은순, False=낮은순으로 정렬
                ranked_docs = [en_papers[idx_cumulator + list(topic).index(x)] for x in sorted_docs[:show_data]]
                line = ["201%d " % i, "topic %d" % j]
                # 내림차순 정렬된 데이터의 상위 n개 데이터를 결과로 출력
                ranked_docs = [line + [i] + doc for i, doc in enumerate(ranked_docs, 1)]
                time_data.extend(ranked_docs)
                
            all_data.extend(time_data)
            idx_cumulator += len(data[i-1])

        for row in all_data:
            wr.writerow(row)

        out.close()
    else:
        print("에러발생")