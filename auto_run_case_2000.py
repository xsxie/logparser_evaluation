#coding = UTF-8
#This python file uses the following encoding: utf-8

"""
	Case study
	Function: (1)计算Non-conformal measure
	          (2)计算p_value
	          (3)计算credibility 和 confidence
	Author: Xie xueshuo
	Date: 2019.01.13
"""

import math
import decimal
import Levenshtein
import time
import csv
import pandas as pd
import numpy as np
import heapq
import multiprocessing


"""计算两个字符串的相似性，Non-conformal measure，score function"""
def string_similar(s1, s2):
    sim = Levenshtein.distance(s1, s2)
    return sim
#the edit distance of two logs
def editDistOfSeq(wordList1,wordList2):
    v = math.floor(len(wordList1) / len(wordList2))
    print "The parameter v is: %d" % v
    m = len(wordList1)+1
    n = len(wordList2)+1
    d = []
    t=s=0
    for i in range(m):
        d.append([t])
        #t+=1/(math.exp(i-v)+1)
        t += 1 / (decimal.Decimal(i-v).exp() + 1)
    del d[0][0]
    for j in range(n):
        d[0].append(s)
        #s+=1/(math.exp(j-v)+1)
        s += 1 / (decimal.Decimal(j-v).exp() + 1)
    for i in range(1,m):
        for j in range(1,n):
            if wordList1[i-1]==wordList2[j-1]:
                d[i].insert(j,d[i-1][j-1])
            else:
                #weight=1.0/(math.exp(i-1-v)+1)
                weight = 1 / (decimal.Decimal(i - 1 - v).exp() + 1)
                minimum = min(d[i-1][j]+weight, d[i][j-1]+weight, d[i-1][j-1]+2*weight)
                d[i].insert(j, minimum)
    return  d[-1][-1]

"""计算日志消息的content部分与每一个模板的score"""
def score(Input_filepath1,Input_filepath2,Output_filepath, name):
    with open(Input_filepath1, 'rb') as csvfile:
        reader = csv.DictReader(csvfile)
        column1 = [row['Content'] for row in reader]
    with open(Input_filepath2, 'rb') as csvfile:
        reader = csv.DictReader(csvfile)
        column2 = [row['EventTemplate'] for row in reader]
    L1 = len(column1)
    L2 = len(column2)
    number = L1 * L2
    count = 0.0
    distList=[]
    distMat=np.zeros((L1,L2))
    for i in range(len(column1)):
        for j in range(len(column2)):
            start = time.time()
            sim = editDistOfSeq(column1[i], column2[j])
            distMat[i][j] = sim
            distList.append(sim)
            end = time.time()
            count = count + 1.0
            print "%s Time：%f" % (name, (end - start))
            print "%s, %s/%s, Process: %f" % (name, count, number, (count / number))
    distArray=np.array(distList)
    np.savetxt(Output_filepath,distMat,delimiter=',')
    return Output_filepath

"""将score按照模板个数进行排序"""
def p_value_file(Inputfilepath, Intput_filepath1, Output_filepath):

    df = pd.DataFrame(pd.read_csv(Intput_filepath1, header=None, index_col=False))
    row_length = df.iloc[:, 0].size
    column_length = df.columns.size
    distList = []
    distMat = np.zeros((row_length, column_length))
    for j in range(column_length):
        column = df[j]
        column_sort = column.sort_values(ascending=True)
        score_array = np.array(column_sort)
        for k in range(len(score_array)):
            distMat[k][j] = score_array[k]
    distList.append(score_array)
    distArray = np.array(distList)
    np.savetxt(Output_filepath, distMat, delimiter=',')
    return Output_filepath

"""p_value 计算函数"""
def p_value(value,list):
    for i in range(len(list)):
        if list[i] >= value:
            break
    p_value = float(1.0 - ((i + 1) / 2000.0))
    return p_value

"""计算每条日志与每个模板的p_value"""
def p_value_all(Input_filepath, Input_filepath1, Output_filepath, name):
    df = pd.DataFrame(pd.read_csv(Input_filepath, header=None, index_col=False))
    df1 = pd.DataFrame(pd.read_csv(Input_filepath1, header=None, index_col=False))
    row_length = df.iloc[:, 0].size
    column_length = df.columns.size
    distList = []
    number = column_length * row_length
    count = 0.0
    distMat = np.zeros((row_length, column_length))
    for j in range(column_length):
        for i in range(row_length):
            start = time.time()
            column = df1[j]
            Pvalue = p_value(df.loc[i][j], column)
            distMat[i][j] = Pvalue
            end = time.time()
            count = count + 1.0
            print "%s Time：%f" % (name, (end - start))
            print "%s, %s/%s, Process: %f" % (name, count, number, (count / number))
        distList.append(Pvalue)
    distArray = np.array(distList)
    np.savetxt(Output_filepath, distMat, delimiter=',')
    return Output_filepath

"""计算每条日志对应的confidence和credibility"""
def confidence(Input_filepath, Output_filepath):
    df = pd.DataFrame(pd.read_csv(Input_filepath, header=None, index_col=False))
    row_length = df.iloc[:, 0].size
    distMat = np.zeros((row_length, 2))
    for i in range(row_length):
        row_list = df.loc[i].tolist()
        row_sen = heapq.nlargest(2, row_list)
        cre = row_sen[0]
        con = 1.0 - row_sen[1]
        distMat[i][0] = cre
        distMat[i][1] = con
    np.savetxt(Output_filepath, distMat, delimiter=',')
    return Output_filepath

def cre_file(Input_filepath1, Iutput_filepath2, Output_filepath):
    df1 = pd.DataFrame(pd.read_csv(Input_filepath1, header=None, index_col=False))
    df2 = pd.DataFrame(pd.read_csv(Iutput_filepath2, header=None, index_col=False))
    df3 = df2.drop(columns=1)
    result_file = pd.concat([df1, df3], axis=1)
    np.savetxt(Output_filepath, result_file, delimiter=',')

def con_file(Input_filepath1, Iutput_filepath2, Output_filepath):
    df1 = pd.DataFrame(pd.read_csv(Input_filepath1, header=None, index_col=False))
    df2 = pd.DataFrame(pd.read_csv(Iutput_filepath2, header=None, index_col=False))
    df3 = df2.drop(columns=0)
    result_file = pd.concat([df1, df3], axis=1)
    np.savetxt(Output_filepath, result_file, delimiter=',')

def file_result(Input_filepath, Output_filepath):
    df = pd.DataFrame(pd.read_csv(Input_filepath, header=None, index_col=False))
    row_length = df.iloc[:, 0].size
    column_length = df.columns.size
    distList = []
    for i in range(row_length):
        row_list = df.loc[i].tolist()
        row_position = row_list.index(min(row_list))
        distList.append(row_position)
    np.savetxt(Output_filepath, distList, delimiter=',')
    return Output_filepath

def file_combine(Input_filepath1, Iutput_filepath2, Output_filepath):
    df1 = pd.DataFrame(pd.read_csv(Input_filepath1, header=None, index_col=False))
    df2 = pd.DataFrame(pd.read_csv(Iutput_filepath2, header=None, index_col=False))
    result_file = pd.concat([df1, df2], axis=1)
    np.savetxt(Output_filepath, result_file, delimiter=',')
    return Output_filepath

def combine_file(namelogdata, Outputfilepath, namelist):

    df = pd.DataFrame()

    for al in namelist:
        file = './p_value/%s/%s_cre_con.csv' % (namelogdata, al)

        data = pd.read_csv(file, header=None, index_col=False)
        cren = al + '_cre'
        conn = al + '_con'
        df[cren] = data[0]
        df[conn] = data[1]

    df.to_csv(Outputfilepath)
    return Outputfilepath

def combine_average(namelogdata, Outputfilepath, namelist):

    df = pd.DataFrame()
    distList_cre = []
    distList_con = []
    for al in namelist:
        file = './p_value/%s/%s_cre_con.csv' % (namelogdata, al)

        data = pd.read_csv(file, header=None, index_col=False)
        cre = np.mean(data[0])
        con = np.mean(data[1])
        distList_cre.append(cre)
        distList_con.append(con)
    df['Credibility'] = distList_cre
    df['Confidence'] = distList_con
    df.to_csv(Outputfilepath)
    return Outputfilepath

def main(name_logdata, name_algorithm):
    Input_filepath_structured = './input/%s/%s_result/%s_2k.log_structured.csv' % (
    name_logdata, name_algorithm, name_logdata)
    Input_filepath_template = './input/%s/%s_result/%s_2k.log_templates.csv' % (
    name_logdata, name_algorithm, name_logdata)
    Output_filepath1 = './result/result/%s/%s_result/%s_score.csv' % (name_logdata, name_algorithm, name_logdata)
    Output_filepath2 = './result/result/%s/%s_result/%s_result_p_value.csv' % (
    name_logdata, name_algorithm, name_logdata)
    Output_filepath3 = './result/result/%s/%s_result/%s_score_snort.csv' % (name_logdata, name_algorithm, name_logdata)
    Output_filepath4 = './result/result/%s/%s_result/%s_count.csv' % (name_logdata, name_algorithm, name_logdata)
    Output_filepath_p_value = './result/p_value/%s/%s_result_combine.csv' % (name_logdata, name_algorithm)
    Output_filepath_count = './result/count/%s/%s_result_combine.csv' % (name_logdata, name_algorithm)
    Output_filepath_p_value_all = './p_value/%s/%s_result_p_value.csv' % (name_logdata, name_algorithm)
    Output_filepath_cre_con = './p_value/%s/%s_cre_con.csv' % (name_logdata, name_algorithm)
    Output_filepath_cre = './result/p_value/%s/%s_result_cre.csv' % (name_logdata, name_algorithm)
    Output_filepath_con = './result/p_value/%s/%s_result_con.csv' % (name_logdata, name_algorithm)

    print name_algorithm
    Input_filepath_score = score(Input_filepath_structured, Input_filepath_template, Output_filepath1, name_algorithm)
    Output_filepath_score_snort = p_value_file(Input_filepath_template, Input_filepath_score, Output_filepath3)

    file_result(Input_filepath_score, Output_filepath4)
    file_combine(Input_filepath_score, Output_filepath4, Output_filepath_count)
    p_value_filepath = p_value_all(Output_filepath1, Output_filepath3,
                                   Output_filepath_p_value_all, name_algorithm)
    cre_con_filepath = confidence(Output_filepath_p_value_all, Output_filepath_cre_con)

    cre_file(Output_filepath1, cre_con_filepath, Output_filepath_cre)
    con_file(Output_filepath1, cre_con_filepath, Output_filepath_con)

    Output_filepath_all = './result/cre_con/%s/%s_all.csv' % (name_logdata, name_logdata)
    Output_filepath_average = './result/cre_con/%s/Cred&Conf_%s.csv' % (name_logdata, name_logdata)
    Output_filepath_cre_con = combine_file(name_logdata, Output_filepath_all, namelist_algorithm)
    Output_filepath_ave = combine_average(name_logdata, Output_filepath_average, namelist_algorithm)
    print name_logdata

if __name__ == '__main__':

    namelist_logdata = ['HDFS']

    namelist_algorithm = ['Lenma','LogCluster', 'LogMine']

    for name_logdata in namelist_logdata:
        for name_algorithm in namelist_algorithm:
            main(name_logdata, name_algorithm,)

    '''
    """多进程并行计算"""
    pool = multiprocessing.Pool(processes=3)
    for i in xrange(4):
        for name_logdata in namelist_logdata:
            for name_algorithm in namelist_algorithm:
                pool.apply_async(main, (name_logdata,name_algorithm,))
    pool.close()
    pool.join()
    '''
