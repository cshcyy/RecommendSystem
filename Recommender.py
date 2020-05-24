# -*-coding:utf-8 -*-
# Author   : zzp
# Date     : 2020/5/22 17:40
# Email AD ：2410520561@qq.com
# SoftWare : PyCharm
# Project Name   : Recsys
# Python Version : 3.6

from PyQt5.QtWidgets import QApplication,QMainWindow,QFileDialog
from config.Ui_RecommendFast import Ui_MainWindow
import sys,os,shutil,time
import pandas as pd
from config.tools import import_data_from_file,cross_validate,cut_dataset,diff_item_set
from surprise import Reader,Dataset
from surprise.prediction_algorithms.matrix_factorization import SVD,SVDpp,NMF
from surprise.prediction_algorithms.slope_one import SlopeOne
from surprise.prediction_algorithms.knns import KNNBasic,KNNWithMeans,KNNBaseline
from surprise.prediction_algorithms.co_clustering import CoClustering
from surprise.prediction_algorithms.baseline_only import BaselineOnly
from surprise.prediction_algorithms.random_pred import NormalPredictor


class Window(QMainWindow,Ui_MainWindow):
    # 初始化
    def __init__(self):
        super(Window,self).__init__()
        self.setupUi(self)
        self.connect_slot_function()
        self.current_path = os.getcwd()
        self.dataset_path='./dataset/data.csv'
        self.result_path = './result/pre_result.txt'
        self.help_file_path='./help/help.txt'
        self.max_totalnum=10000
        self.cut_num=0
        self.algo = SVD()
        self.display_process_label.append('初始化加载SVD模型.')
        self.algo_change_flag = False
        self.algo_trained_flag=False
        self.init_dir()



    # 检查配置文件路径是否完备
    def init_dir(self):
        if os.path.exists('./log') is False:
            os.mkdir('./log')
        if os.path.exists('./result') is False:
            os.mkdir('./result')
        if os.path.exists('./dataset') is False:
            os.mkdir('./dataset')


    # 连接槽函数
    def connect_slot_function(self):
        self.import_data_btn.triggered.connect(self.slot_import_data_btn)
        self.select_model_btn.triggered.connect(self.slot_select_model_btn)
        self.select_algo_comboBox.currentIndexChanged.connect(self.slot_select_algo_combobox)
        self.clean_context_btn.triggered.connect(self.slot_clean_context_btn)
        self.save_result_btn.triggered.connect(self.slot_save_result_btn)
        self.need_help_btn.triggered.connect(self.slot_need_help_btn)
        self.config_ok_btn.clicked.connect(self.slot_cut_dataset)
        self.recommed_result_btn.triggered.connect(self.slot_recommed_result_btn)

    '''
        触发导入数据功能:
        1、选择要导入的数据文件路径，可处理的文件类型包括:xlsx、xls、csv；
        2、得到数据文件路径，将数据文件类型统一转化为cvs格式；
        3、将csv 文件统一保存到dataset文件夹下，文件名为data.csv;
    '''
    def slot_import_data_btn(self):
        # 获取数据文件路径
        file_path,file_type = QFileDialog.getOpenFileName(self,'请选择您要加载的数据文件',self.current_path,'*.xlsx *.xls *.csv')
        self.display_process_label.append('正在导入数据...')
        if file_path:
            self.display_process_label.append('已选择数据文件路径: ' + file_path)
            self.display_process_label.append('开始导入数据...')
            import_data_from_file(file_path)
            self.display_process_label.append('数据导入完毕!')
            totalnum=len(open(self.dataset_path).readlines())
            self.display_process_label.append('成功读取数据量: '+str(totalnum)+'行.')
            if totalnum>self.max_totalnum:
                self.display_process_label.append('读入的数据量过大,建议采用隔行读取!!!')
        else:
            self.display_process_label.append('没有导入任何数据文件!!!')


    '''下来选项改变触发算法选择
        1、选择推荐算法
    '''
    def slot_select_algo_combobox(self):
        self.algo_change_flag=True
        self.algo_trained_flag=False
        algo_name=self.select_algo_comboBox.currentText()
        if algo_name=='SVD':
            self.algo=SVD()
            self.display_process_label.append('加载SVD模型...')
        elif algo_name=='SVD++':
            self.algo = SVDpp()
            self.display_process_label.append('加载SVD++模型...')
        elif algo_name == 'NMF':
            self.algo = NMF()
            self.display_process_label.append('加载NMF模型...')
        elif algo_name == 'Slope One':
            self.algo = SlopeOne()
            self.display_process_label.append('加载Slope One模型...')
        elif algo_name == 'k-NN':
            self.algo = KNNBasic()
            self.display_process_label.append('加载k-NN模型...')
        elif algo_name == 'Centered k-NN':
            self.algo = KNNWithMeans()
            self.display_process_label.append('加载Centered k-NN模型...')
        elif algo_name == 'k-NN Baseline':
            self.algo = KNNBaseline()
            self.display_process_label.append('加载k-NN Baseline模型...')
        elif algo_name == 'Co-Clustering':
            self.algo = CoClustering()
            self.display_process_label.append('加载Co-Clustering模型...')
        elif algo_name == 'Baseline':
            self.algo = BaselineOnly()
            self.display_process_label.append('加载Baseline模型...')
        elif algo_name == 'Random':
            self.algo = NormalPredictor()
            self.display_process_label.append('加载Random模型...')

    '''
        根据界面传入值，减少数据量
    '''
    def slot_cut_dataset(self):
        # 获取传入值
        self.cut_num=self.row_read_times_spinBox.value()


    '''
        检查复选框有没有被选择
    '''
    def check_measure(self):
        self.measure_list=[]
        if self.rmse_checkBox.isChecked():
            self.measure_list.append('RMSE')
        if self.mae_checkBox.isChecked():
            self.measure_list.append('MAE')
        if self.fcp_checkBox.isChecked():
            self.measure_list.append('FCP')
        if self.mse_checkBox.isChecked():
            self.measure_list.append('MSE')



    '''触发选择模型功能：
        1、选择要采用的模型，保存日志到log目录;
        2、判断是否隔行读取;
        3、将训练好算法模型保留下来;
    '''
    def slot_select_model_btn(self):
        self.display_process_label.append('开始生成模型...')
        # 选择评价标准
        self.check_measure()
        print(self.measure_list)
        # 数据格式
        reader = Reader(line_format='user item rating timestamp', sep=',')
        if os.path.exists('./dataset/data.csv') is True:
            try:
                self.display_process_label.append('加载数据集中...')
                data = Dataset.load_from_file(self.dataset_path, reader=reader)
                self.display_process_label.append('加载完毕...')
            except ValueError:
               self.display_process_label.append('传入的数据集不规范,请重新修改并导入数据！！！')
            self.display_process_label.append('正在进行交叉验证...')
            result=cross_validate(algo=self.algo,data=data,measures=self.measure_list,verbose=True)
            self.display_process_label.append('交叉验证验证完毕!')
            # 获取日志文件名
            log_path='./log/'+result[-1]
            print(log_path)
            #将结果显示出来
            self.display_result_label.append('模型创建时间: '+str(time.asctime()))
            accracy_context=open(log_path,'r')
            for line in accracy_context:
                self.display_result_label.append(line)
            accracy_context.close()
            self.display_result_label.append('------------------------------------------------------------------------------')
            self.algo_change_flag=False
            self.algo_trained_flag=True
        else:
            self.display_process_label.append('没有导入数据,请执行导入数据操作!')

    '''
       触发生成推荐数据文件，主要算法对象
    '''
    def slot_recommed_result_btn(self):
        self.display_process_label.append('开始生成推荐数据...')
        if self.algo_change_flag==False and self.algo_trained_flag==True:
            # 进行数据裁剪
            if self.cut_num>0:
                totalnum = len(open(self.dataset_path).readlines())
                self.display_process_label.append('原始数据集的数据量有：'+str(totalnum)+'行.')
                for i in range(self.cut_num):
                    dataset_after_cut=cut_dataset(self.dataset_path)
                    self.display_process_label.append('裁剪后数据量有：' + str(dataset_after_cut) + '行.')
                    if dataset_after_cut<100:
                        self.display_process_label.append('数据量过少!数据量过少!')
                self.row_read_times_spinBox.setValue(0)
            # 生成推荐数据
            self.display_process_label.append('正在生成推荐数据...')
            rating_tab = pd.read_csv(self.dataset_path)
            self.display_process_label.append('正在生成用户集...')
            user_set = set(rating_tab.iloc[:, 0])
            user_set_num = len(user_set)
            count = 0
            rec_list = []
            file = open('./result/pre_result.txt', 'w')
            file.close()
            for user in user_set:
                diff_set = diff_item_set(user)
                rec_dict = {'user': user}
                count += 1
                for item in diff_set:
                    pre_result = self.algo.predict(str(user), str(item))
                    pre_result = list(pre_result)[3]
                    pre_result = float(pre_result)
                    pre_result = round(pre_result, 2)
                    rec_dict.update({str(item): pre_result})
                rec_list.append(rec_dict)
                self.display_process_label.append('用户ID: '+str(user)+'预测完毕,'+'还剩下'+str(user_set_num-count)+'个用户.')
                for rec in rec_list:
                    file = open('./result/pre_result.txt', 'a')
                    file.writelines(str(rec))
                    file.writelines('\n')
                    file.close()
                rec_list = []  # 清空列表
            self.display_process_label.append('Congratulations! 所有用户的推荐数据生成完毕!')
            # 显示前一百个推荐
            self.display_process_label.append('Congratulations! 所有用户的推荐数据生成完毕!')
            recommend_result=open(self.result_path,'r')
            self.display_result_label.append('-----------------------------------展示前一百个用户的推荐数据-----------------------------------')
            count=1
            for line in recommend_result:
                if count<=100:
                    self.display_result_label.append('第'+str(count)+'用户: '+line.replace('{','').replace('}',''))
                else:
                    break
                count+=1
            self.display_result_label.append('----------------------------------------------结束--------------------------------------------')
        else:
            self.display_process_label.append('模型已经改变或者模型未生成,请执行*生成模型*步骤！！！')

    '''
        显示帮助文档，留下邮箱地址
    '''
    def slot_need_help_btn(self):
        help_context=open(self.help_file_path,'r')
        for line in help_context:
            self.display_result_label.append(line)
        help_context.close()


    '''
        触发保存推荐数据文件
    '''
    def slot_save_result_btn(self):
        if os.path.exists(self.result_path):
            save_path, file_type = QFileDialog.getSaveFileName(self,'请选择您要保存的路径',self.current_path,"Text Files (*.txt)")
            # self.display_process_label('保存路径:'+str(save_path))
            shutil.move(self.result_path,save_path)
            self.display_process_label.append('文件保存成功!')
        else:
            self.display_process_label.append('没有生成任何推荐文件!')


    '''
        清除结果显示框内容
    '''
    def slot_clean_context_btn(self):
        self.display_result_label.clear()


if __name__ == '__main__':
    app=QApplication(sys.argv)
    window=Window()
    window.show()
    sys.exit(app.exec_())
        
        
        



