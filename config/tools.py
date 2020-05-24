# -*-coding:utf-8 -*-
# Author   : zzp
# Date     : 2020/5/20 23:53
# Email AD ：2410520561@qq.com
# SoftWare : PyCharm
# Project Name   : Recsys
# Python Version : 3.6

'''The validation module contains the cross_validate function, inspired from
the mighty scikit learn.'''

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import time
import numpy as np
from joblib import Parallel
from joblib import delayed
from six import iteritems
import os
from surprise.model_selection.split import  get_cv
from surprise import accuracy
import time as T
import pandas as pd


# create a log file
def create_result_file(filename):
    log_file_path='./log'
    if os.path.exists(log_file_path):
        pass
    else:
        os.makedirs(log_file_path)
    date= T.asctime().split()           #获取时间
    year=date[-1]
    day=date[-3]
    time=date[-2].replace(':','')
    filename=filename+'算法+'+year+day+time+'.txt'
    file=open('./log/'+filename,'w')
    file.close()
    return filename

# 交叉验证
def cross_validate(algo, data, measures=['rmse', 'mae'], cv=None,
                   return_train_measures=False, n_jobs=1,
                   pre_dispatch='2*n_jobs', verbose=False):
    measures = [m.lower() for m in measures]

    cv = get_cv(cv)

    delayed_list = (delayed(fit_and_score)(algo, trainset, testset, measures,
                                           return_train_measures)
                    for (trainset, testset) in cv.split(data))
    out = Parallel(n_jobs=n_jobs, pre_dispatch=pre_dispatch)(delayed_list)

    (test_measures_dicts,
     train_measures_dicts,
     fit_times,
     test_times) = zip(*out)

    test_measures = dict()
    train_measures = dict()
    ret = dict()
    for m in measures:
        # transform list of dicts into dict of lists
        # Same as in GridSearchCV.fit()
        test_measures[m] = np.asarray([d[m] for d in test_measures_dicts])
        ret['test_' + m] = test_measures[m]
        if return_train_measures:
            train_measures[m] = np.asarray([d[m] for d in
                                            train_measures_dicts])
            ret['train_' + m] = train_measures[m]

    ret['fit_time'] = fit_times
    ret['test_time'] = test_times

    #保存日志
    filename=save_model_result(algo, measures, test_measures, train_measures, fit_times,
                      test_times, cv.n_splits)

    if verbose:
        print_summary(algo, measures, test_measures, train_measures, fit_times,
                      test_times, cv.n_splits)

    return ret,filename


# 模型训练分数
def fit_and_score(algo, trainset, testset, measures,
                  return_train_measures=False):
    start_fit = time.time()
    algo.fit(trainset)
    fit_time = time.time() - start_fit
    start_test = time.time()
    predictions = algo.test(testset)
    test_time = time.time() - start_test

    if return_train_measures:
        train_predictions = algo.test(trainset.build_testset())

    test_measures = dict()
    train_measures = dict()
    for m in measures:
        f = getattr(accuracy, m.lower())
        test_measures[m] = f(predictions, verbose=0)
        if return_train_measures:
            train_measures[m] = f(train_predictions, verbose=0)

    return test_measures, train_measures, fit_time, test_time


#  打印摘要
def print_summary(algo, measures, test_measures, train_measures, fit_times,
                  test_times, n_splits):
    '''Helper for printing the result of cross_validate.'''
    print('Evaluating {0} of algorithm {1} on {2} split(s).'.format(
          ', '.join((m.upper() for m in measures)),
          algo.__class__.__name__, n_splits))
    print()

    row_format = '{:<18}' + '{:<8}' * (n_splits + 2)
    s = row_format.format(
        '',
        *['Fold {0}'.format(i + 1) for i in range(n_splits)] + ['Mean'] +
        ['Std'])
    s += '\n'
    s += '\n'.join(row_format.format(
        key.upper() + ' (testset)',
        *['{:1.4f}'.format(v) for v in vals] +
        ['{:1.4f}'.format(np.mean(vals))] +
        ['{:1.4f}'.format(np.std(vals))])
        for (key, vals) in iteritems(test_measures))
    if train_measures:
        s += '\n'
        s += '\n'.join(row_format.format(
            key.upper() + ' (trainset)',
            *['{:1.4f}'.format(v) for v in vals] +
            ['{:1.4f}'.format(np.mean(vals))] +
            ['{:1.4f}'.format(np.std(vals))])
            for (key, vals) in iteritems(train_measures))
    s += '\n'
    s += row_format.format('Fit time',
                           *['{:.2f}'.format(t) for t in fit_times] +
                           ['{:.2f}'.format(np.mean(fit_times))] +
                           ['{:.2f}'.format(np.std(fit_times))])
    s += '\n'
    s += row_format.format('Test time',
                           *['{:.2f}'.format(t) for t in test_times] +
                           ['{:.2f}'.format(np.mean(test_times))] +
                           ['{:.2f}'.format(np.std(test_times))])
    print(s)


# 保存模型结果
def save_model_result(algo, measures, test_measures, train_measures, fit_times,
                  test_times, n_splits):
    filename=create_result_file(algo.__class__.__name__)    # 创建文件并返回文件名字

    # 具体描述
    details='Evaluating {0} of algorithm {1} on {2} split(s).'.format(
    ', '.join((m.upper() for m in measures)),
    algo.__class__.__name__, n_splits)

    # result
    row_format = '{:<18}' + '{:<8}' * (n_splits + 2)
    s = row_format.format(
        '',
        *['Fold {0}'.format(i + 1) for i in range(n_splits)] + ['Mean'] +
        ['Std'])
    s += '\n'
    s += '\n'.join(row_format.format(
        key.upper() + ' (testset)',
        *['{:1.4f}'.format(v) for v in vals] +
        ['{:1.4f}'.format(np.mean(vals))] +
        ['{:1.4f}'.format(np.std(vals))])
        for (key, vals) in iteritems(test_measures))
    if train_measures:
        s += '\n'
        s += '\n'.join(row_format.format(
            key.upper() + ' (trainset)',
            *['{:1.4f}'.format(v) for v in vals] +
            ['{:1.4f}'.format(np.mean(vals))] +
            ['{:1.4f}'.format(np.std(vals))])
            for (key, vals) in iteritems(train_measures))
    s += '\n'
    s += row_format.format('Fit time',
                           *['{:.2f}'.format(t) for t in fit_times] +
                           ['{:.2f}'.format(np.mean(fit_times))] +
                           ['{:.2f}'.format(np.std(fit_times))])
    s += '\n'
    s += row_format.format('Test time',
                           *['{:.2f}'.format(t) for t in test_times] +
                           ['{:.2f}'.format(np.mean(test_times))] +
                           ['{:.2f}'.format(np.std(test_times))])

    with open('./log/' + filename, 'w') as file:
        file.writelines(details+'\n')
        file.writelines(s)
    return filename


# the set of itemid
def diff_item_set(uid):
    data=pd.read_csv('./dataset/data.csv',usecols=[0,1])
    # 数据集商品总集合
    item_set=set(data.iloc[:,1])
    # 用户已购买的商品集合
    user_item=set(data[data.iloc[:,0]==int(uid)].iloc[:,1])
    diff_set=item_set-user_item
    return diff_set

# 将数据转换成特定格式
def convert_to_csv(excel_file_path):
    excel_file = pd.read_excel(excel_file_path)
    excel_file.drop(index=0,inplace=False)
    # excel_name_list=excel_file.columns
    # column_num=len(excel_name_list)
    # if column_num<3:
    #     raise Exception("Error,The column is not enough!")
    excel_file.to_csv('./dataset/data.csv', encoding='utf-8',index=False,header=False,)


# 从文件路径里面导入数据文件
def import_data_from_file(file_path):
    print('导入数据...')
    if os.path.exists(file_path):           #判断文件是否存在
        file_name,file_type=os.path.splitext(file_path)
        if file_type=='.xlsx' or file_type=='.xls': # 转化问csv格式
            convert_to_csv(file_path)
            print('导入数据完毕!')
        elif file_type=='.csv':
            csv_file=pd.read_csv(file_path)
            csv_file.drop(index=0, inplace=True)
            csv_file.reset_index(drop=True, inplace=True)
            csv_file.to_csv('./dataset/data.csv', encoding='utf-8',index=False,header=False)    #去除序号列和首行
            print('导入数据完毕!')
        else:
            raise Exception('The file type is wrong!')
    else:
        raise Exception('The file is no exists!')


def cut_dataset(dataset_path):
    totalnum = len(open(dataset_path).readlines())
    print('数据量: ', totalnum)
    if totalnum>1:
        data=pd.read_csv(dataset_path,skiprows=[rownum for rownum in range(0,totalnum,2)])
        data.to_csv(dataset_path,encoding='utf-8',index=False)
    totalnum = len(open(dataset_path).readlines())
    print('裁剪后数据量: ', totalnum)
    return totalnum

if __name__ == '__main__':
    pass


