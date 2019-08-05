#coding = UTF-8
#This python file uses the following encoding: utf-8

"""
	Case study
	Function: (1)计算tsne坐标
	          (2)plot tsne图（label）
	          (3)plot grid图（credibility和confidence）
	          (4)plot credibility 和confidence的箱体图
	          (5)plot credibility和confidence的均值柱状图
	Author: Xie xueshuo
	Date: 2019.01.13
"""

import pylab as pl
import csv
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn import preprocessing
import seaborn as sns

'''计算tsne坐标，与label一起保存'''
def Caltsne_label(InputFilePath, OutputFilePath):

    data = pd.read_csv(InputFilePath, header=None, index_col=False)
    cols = data.shape[1]
    labels = data[data.columns[-1]]
    X = data.iloc[:,0:cols-1]
    X = preprocessing.normalize(X, norm='l2')
    tsne = TSNE(n_components=2)
    tsne.fit_transform(X)
    tsne = pd.DataFrame(tsne.embedding_)
    tsne["Label"] = labels
    tsne.to_csv(OutputFilePath)
    return tsne

'''读tsne坐标，分label画图'''
def draw_tsne(InputFilePath, name):

    tsne = pd.read_csv(InputFilePath)
    labels = tsne['Label']
    plt.scatter(tsne['0'], tsne['1'], c=labels, s=0.5, alpha=0.5)
    plt.title('tSNE for %s' % name)
    plt.tight_layout()
    #plt.show()

'''计算tsne坐标，与credibility和confidence一起保存'''
def Caltsne_pv(InputFilePath, OutputFilePath):

    data = pd.read_csv(InputFilePath, header=None, index_col=False)
    cols = data.shape[1]
    labels = data[data.columns[-1]]
    X = data.iloc[:,0:cols-1]
    X = preprocessing.normalize(X, norm='l2')
    tsne = TSNE(n_components=2)
    tsne.fit_transform(X)
    tsne = pd.DataFrame(tsne.embedding_)
    tsne["pv"] = labels
    tsne.to_csv(OutputFilePath)
    return tsne

'''计算每个格子的pv和包含多少点，返回dataframe'''
def grid(tsne,L):
    print 'gb'
    pv = np.zeros(L*L)#每个格子的p-value总和
    n = np.zeros(L*L) #每个格子里多少个点
    rowR = np.linspace(tsne['0'].min(), tsne['0'].max(), L+1)
    colR = np.linspace(tsne['1'].min(), tsne['1'].max(), L+1)
    rowU = rowR[1]-rowR[0]
    colU = colR[1]-colR[0]
    for index,ax in tsne.iterrows():
        a = (ax['0']-rowR[0])/rowU
        a = int(a) + 1
        b = (ax['1']-colR[0])/colU
        b = int(b) + 1
        if a>L:
            a = L
        if b>L:
            b= L
        x = (a-1)*L + (b-1)
        pv[x] = pv[x] + ax['pv']
        n[x]  =  n[x] + 1
    res = []

    for row in rowR:
        for col in colR:
            if row!=rowR[L] and col!=colR[L]:
                res.append((row, col))
    res = pd.DataFrame(res)
    res['pv_sum'] = pv
    res['pv_mean'] = pv/n
    res['number'] = n
    res.fillna(0)

    print 'ge'
    return res

'''画出带有颜色的图 pv-grid'''
def draw_pv(name1,name2,res,L):

    data2 = res[res['pv_mean']>=0.9]  #取pv_mean大于0.9的各自单独显示
    data2 = data2.values
    res  =res.values
    nearest = lambda x,s: s[np.argmin(np.abs(s-x))]
    f = np.arange(0, 1.0, 0.05)
    res[:,4] = [nearest(x,f) for x in res[:,4]]
    cm = pl.cm.get_cmap('Reds')
    pl.scatter(res[:, 1], res[:, 2], c=res[:, 4], marker=',', cmap=cm, vmin=0, vmax=1)
    pl.colorbar()

    plt.title('%s grid for %s' % (name1,name2))
    plt.tight_layout()


def draw_boxplot(Inputfilepath, Outputfilepath1, Outputfilepath2, namelogdata):

    plt.figure(figsize=(20, 10))
    df = pd.read_csv(Inputfilepath)
    df1 =  df.iloc[:,1:]
    sns.set(style = 'whitegrid')
    sns.boxplot(data = df1, width = 0.5)

    plt.xticks(rotation=45)
    font1 = {'family':'Times New Roman','weight':'normal','size':40}
    plt.legend(prop=font1)
    plt.title('Credibility and Confidence on %s' % namelogdata)
    plt.subplots_adjust(top=0.919, bottom=0.206, left=0.082, right=0.963, hspace=0.235, wspace=0.189)
    plt.savefig(Outputfilepath1)
    plt.savefig(Outputfilepath2)
    plt.show()

def draw_bar (Inputfilepath, Outfilepath1, Outfilepath2, name_list1 ,name_list2):
    with open(Inputfilepath, 'rb') as csvfile:
        reader = csv.DictReader(csvfile)
        column1 = [row['Credibility'] for row in reader]

    with open(Inputfilepath, 'rb') as csvfile:
        reader = csv.DictReader(csvfile)
        column2 = [row['Confidence'] for row in reader]

    plt.figure(figsize=(20, 10))
    num_list = column1[0:14]
    num_list = [float(each) for each in num_list]
    num_list1 = column2[0:14]
    num_list1 = [float(each) for each in num_list1]

    x = list(range(len(num_list)))
    total_width, n = 0.4, 2
    width = total_width / n

    plt.bar(x, num_list, width=width, label='credibility', tick_label=name_list1, color='rosybrown',
            alpha=1.0, edgecolor='salmon', linestyle='--', hatch='o')
    for i in range(len(x)):
        x[i] = x[i] + 1.2 * width
    plt.bar(x, num_list1, width=width, label='confidence', color='lightcoral', alpha=1.0,
            edgecolor='lightsalmon', linestyle='-', hatch='.')
    plt.title('Credibility and Confidence on %s' % name_list2)
    plt.ylim((0.0, 1.0))
    plt.legend(['credibility', 'confidence'], loc=9, ncol=2)
    plt.subplots_adjust(top=0.963, bottom=0.041, left=0.028, right=0.989, hspace=0.235, wspace=0.189)
    plt.savefig(Outfilepath1)
    plt.savefig(Outfilepath2)
    plt.show()

if __name__ == '__main__':
    '''
    namelist_logdata = ['Android', 'Apache', 'BGL', 'Hadoop', 'HDFS',
                        'HealthApp', 'HPC', 'Linux', 'Mac',
                        'OpenSSH', 'OpenStack', 'Proxifier', 'Spark', 'Thunderbird',
                        'Windows', 'Zookeeper']
    '''
    namelist_logdata = ['HDFS']

    namelist_algorithm = ['AEL', 'Drain', 'IPLoM', 'Lenma', 'LFA',
                          'LKE', 'LogCluster', 'LogMine',
                          'LogSig', 'MoLFI', 'SHISO', 'SLCT', 'Spell',
                          'Standard']
    '''
    #计算tsne坐标
    for name_logdata in namelist_logdata:
        for name_algorithm in namelist_algorithm:

            print name_algorithm

            Input_filepath_cre = './result/p_value/%s/%s_result_cre.csv' % (name_logdata, name_algorithm)
            Input_filepath_con = './result/p_value/%s/%s_result_con.csv' % (name_logdata, name_algorithm)
            Input_filepath_label = './result/count/%s/%s_result_combine.csv' % (name_logdata, name_algorithm)

            Output_filepath_tsne_cre = './plot/tsne/%s/%s_tsne_cre.csv' % (name_logdata, name_algorithm)
            Output_filepath_tsne_con = './plot/tsne/%s/%s_tsne_con.csv' % (name_logdata, name_algorithm)
            Output_filepath_tsne_label = './plot/tsne/%s/%s_tsne_label.csv' % (name_logdata, name_algorithm)

            Output_filepath_res = './plot/tsne/%s/%s_res.csv' % (name_logdata, name_algorithm)

            Caltsne_pv(Input_filepath_cre, Output_filepath_tsne_cre)
            print "cre"
            Caltsne_pv(Input_filepath_con, Output_filepath_tsne_con)
            print "con"
            Caltsne_label(Input_filepath_label, Output_filepath_tsne_label)
            print "label"

            print name_logdata

   #plot带label的tsne图
    for name_logdata in namelist_logdata:
        count = 1
        plt.figure(figsize=(20, 10))
        for name_algorithm in namelist_algorithm:
            print name_logdata
            Tsne_file =  './plot/tsne/%s/%s_tsne_label.csv' % (name_logdata, name_algorithm)
            plt.subplot(3, 5, count)
            draw_tsne(Tsne_file, name_algorithm)
            count = count + 1
            print name_algorithm
        plt.subplots_adjust(top=0.963, bottom=0.041, left=0.028, right=0.99, hspace=0.235, wspace=0.193)
        plt.savefig('./plot/plot/tsne/eps/%s/%s.eps' % (name_logdata, name_logdata))
        plt.savefig('./plot/plot/tsne/png/%s/%s.png' % (name_logdata, name_logdata))
        plt.show()

   #划分格子后，计算每个格子的pv和点的数量,保存
    L = 50  # 将坐标划分成L*L的格子
    for name_logdata in namelist_logdata:
        for name_algorithm in namelist_algorithm:
            print name_logdata
            tsnefile1 = './plot/tsne/%s/%s_tsne_cre.csv' % (name_logdata, name_algorithm)
            tsne = pd.read_csv(tsnefile1)
            res = grid(tsne, L)
            res.to_csv('./plot/res/%s/%s_res_cre.csv' % (name_logdata, name_algorithm))
            print name_algorithm

    #画图，credibility带颜色
    L = 50
    name_cre = 'Credibility'
    for name_logdata in namelist_logdata:
        count = 1
        plt.figure(figsize=(20, 10))
        for name_algorithm in namelist_algorithm:
            print name_logdata
            resfile2 = './plot/res/%s/%s_res_cre.csv' % (name_logdata, name_algorithm)
            res = pd.read_csv(resfile2)
            plt.subplot(3, 5, count)
            draw_pv(name_cre, name_algorithm, res, L)
            count = count + 1
            print name_algorithm
        plt.subplots_adjust(top=0.963, bottom=0.041, left=0.028, right=0.989, hspace=0.235, wspace=0.189)
        plt.savefig('./plot/plot/pv-grid/eps/%s/Credibility_%s.eps' % (name_logdata, name_logdata))
        plt.savefig('./plot/plot/pv-grid/png/%s/Credibility_%s.png' % (name_logdata, name_logdata))
        plt.show()

    #划分格子后，计算每个格子的pv和点的数量,保存
    L = 50  # 将坐标划分成L*L的格子
    for name_logdata in namelist_logdata:
            for name_algorithm in namelist_algorithm:
                print name_logdata
                tsnefile1 = './plot/tsne/%s/%s_tsne_con.csv' % (name_logdata, name_algorithm)
                tsne = pd.read_csv(tsnefile1)
                res = grid(tsne, L)
                res.to_csv('./plot/res/%s/%s_res_con.csv' % (name_logdata, name_algorithm))
                print name_algorithm

    #画图，confidence带颜色
    L = 50
    name_con = 'Confidence'
    for name_logdata in namelist_logdata:
            count = 1
            plt.figure(figsize=(20, 10))
            for name_algorithm in namelist_algorithm:
                print name_logdata
                resfile2 = './plot/res/%s/%s_res_con.csv' % (name_logdata, name_algorithm)
                res = pd.read_csv(resfile2)
                plt.subplot(3, 5, count)
                draw_pv(name_con, name_algorithm, res, L)
                count = count + 1
                print name_algorithm
            plt.subplots_adjust(top=0.963, bottom=0.041, left=0.028, right=0.989, hspace=0.235, wspace=0.189)
            plt.savefig('./plot/plot/pv-grid/eps/%s/Confidence_%s.eps' % (name_logdata, name_logdata))
            plt.savefig('./plot/plot/pv-grid/png/%s/Confidence_%s.png' % (name_logdata, name_logdata))
            plt.show()

    for name_logdata in namelist_logdata:

        Input_filepath = './result/cre_con/%s/%s_all.csv' % (name_logdata, name_logdata)
        Output_filepath_eps = './plot/plot/cre_con/eps/%s/%s_boxplot.eps' % (name_logdata, name_logdata)
        Output_filepath_png = './plot/plot/cre_con/png/%s/%s_boxplot.png' % (name_logdata, name_logdata)

        draw_boxplot(Input_filepath, Output_filepath_eps, Output_filepath_png, name_logdata)
'''
    for name_logdata in namelist_logdata:

        Input_filepath = './result/cre_con/%s/Cred&Conf_%s.csv' % (name_logdata, name_logdata)
        Output_filepath_eps = './plot/plot/cre_con/eps/%s/%s_Average.eps' % (name_logdata, name_logdata)
        Output_filepath_png = './plot/plot/cre_con/png/%s/%s_Average.png' % (name_logdata, name_logdata)

        draw_bar(Input_filepath, Output_filepath_eps, Output_filepath_png, namelist_algorithm, name_logdata)
