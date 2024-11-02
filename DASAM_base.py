import numpy as np
import geatpy as ea
import random
from scipy.stats import poisson
import pandas as pd
import csv
import copy
from numpy import exp
import gc
import os

#import warnings
# 将所有警告升级为异常
#warnings.simplefilter('error')

global essential_type#提供方本质类型，列表
global g#利他主义程度，列表
global per#对不同类型评论做出提升型回应的概率，嵌套列表，内部先好评后差评
global sp_limit#实际服务质量上下限，价格上下限
global o_limit#夸张宣传上下限
global pci_limit#好评激励上下限
global u0#固定效用
global alpha#评论偏好的均值
global b#不同类型评论对于潜在消费者的单位影响，好评无回应、差评无回应、价值共创类型评论、评论端价值下降评论、回应端价值下降评论、价值共毁类型评论
global a#不同类型评论的情感收益
global Po_val#好评阈值的均值
global Po_val_std#好评阈值的标准差
global c_rate#平台佣金率
global room_number#房间数量，列表
global p_reply#提供方回应概率，嵌套列表

global b_weight
global a_weight
b_weight=10
a_weight=0.1

essential_type=[0,0]#研究变量,0是利己主义、1是存在利他主义偏好
g=[0,0]#0表示完全利己主义、1表示完全利他主义
per=[[0.939,0.486],[0.939,0.486]]#对好评做出提升型回应的概率，对差评做出提升型回应的概率，研究变量
#默认值0.486，实验取值0.1，0.5，0.9
sp_limit=[0,1]#价格质量上下限
o_limit=[0,0.3]#夸大宣传上下限
pci_limit=[0,0.3]#好评激励上下限
u0=0.3
alpha=0.65#评论偏好的均值
b=b_weight*np.array([0.11,-0.15,0.17,0,0,-0.23])#评论对潜在消费者的单位影响
#比例固定，但是大小还没固定
a=a_weight*np.array([1,1.5,-1.5,-1.5])#回应对消费者满意度的影响，内部分别是四类评论

Po_val=-0.2
Po_val_std=0.1
c_rate=0.15
room_number=[17,17]
p_reply=[[0.198,0.378],[0.198,0.378]]#提供方做出回应的概率，内部先好评后差评


class Platform:  # 平台
    def __init__(self):
        self.c_rate = c_rate
        self.income = 0


class Provider:
    def __init__(self, num, initial_flag_i, g_i, per_i, room_number_i, p_reply_i):
        self.num = num  # 商家编号,0或1
        self.initial_flag_i = initial_flag_i  # 商家本质类型
        self.g_i = g_i  # 商家利他程度
        self.price_i = 0.5  # 价格
        self.service_i = 0.5  # 质量
        self.o_i = 0  # 夸大宣传程度
        self.pci_i = 0  # 好评激励
        self.per_i = per_i  # 做出提升型回应的概率
        self.p_reply_i = p_reply_i  # 回应概率
        self.room_number_i = room_number_i  # 房间总数量

        self.occupy_num = 0  # 房间被占用数量
        self.service_c_list = []  # 正在入住的消费者对象列表
        self.comment_sum = [0, 0, 0, 0, 0, 0]  # 累计各类评论数量
        self.service_num = 0  # 本期服务过的顾客数量
        self.order_num = 0  # 本期办理入住的顾客数量
        self.c_revenue_sum = 0  # 本期服务过的消费者的收益之和
        self.c_satisf_sum = 0  # 本期服务过的消费者的满意度之和
        self.c_surplus_sum = 0  # 本期服务过的消费者的满意度之和
        self.income = 0  # 提供方本期收益
        self.s_welfare_i = 0  # 本提供方创造的社会总福利
        self.incentive_num = 0  # 被激励好评人数
        self.recieve_pci_num = 0  # 接受好评激励的人数


class Customer:
    def __init__(self, alpha_j, occupancy_days, Po_val_j):
        self.alpha_j = alpha_j  # 评论偏好，受评论影响程度
        self.occupancy_days = occupancy_days  # 入住天数
        self.Po_val_j = Po_val_j  # 好评阈值
        self.perceived_provider = -1  # 实际入住的提供方,0,1
        self.attend_time = 0  # 到达时间
        self.comment = [0, 0, 0, 0, 0, 0]  # 评论
        self.perceived_service = 0  # 提供方宣称的服务质量
        self.expected_surplus = 0  # 消费者的预期剩余
        self.price = 0  # 住宿时的价格
        self.service = 0  # 住宿时的实际服务质量


# 初始化对象
def instantiate_platform():
    platform = Platform()
    return platform


def instantiate_providers():
    providers = []
    for i in range(2):
        provider_i = Provider(i, essential_type[i], g[i], per[i], room_number[i], p_reply[i])
        providers.append(provider_i)
    return providers


def instantiate_customer():
    alpha_j = np.random.normal(alpha, np.min([alpha, 1 - alpha]) / 3, 1)[0]
    occupancy_days = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], size=1,
                                      p=[0.1797, 0.2364, 0.2315, 0.1486, 0.0666, 0.0330, 0.0740, 0.0099, 0.0072, 0.0097,
                                         0.0034])[0]
    Po_val_j = np.random.normal(Po_val, Po_val_std, 1)[0]
    customer = Customer(alpha_j, occupancy_days, Po_val_j)
    return customer


def customer_choice(customer, providers, time_step):
    degree = []  # 消费者对可选提供方的预期剩余估计
    alternative = []  # 可选提供方
    U_list = []
    for provider in providers:
        if provider.occupy_num == provider.room_number_i:  # 无空房
            continue
        else:
            c_alpha_j = customer.alpha_j * (
                        1 - np.exp(-0.1 * sum(provider.comment_sum)))  # 消费者的实际评论偏好受到评论数量的影响，防止消费者前期误判，即过度相信少量的评论
            Ur = np.dot(np.array(provider.comment_sum) / (sum(provider.comment_sum) + 1), b.T)  # 评论中感知的消费者剩余
            #Ur=(Ur+0.23)/0.4#归一化
            Usp = u0 + (1 + provider.o_i) * (provider.service_i) - provider.price_i  # 提供方展示信息中的消费者剩余感知
            U = (1 - c_alpha_j) * Usp + c_alpha_j * (Ur) * Usp  # 预期剩余
            # print('消费者从评论和宣称信息中感知的净效用：',(Ur)*Usp,Usp,U)
            if U >= 0:  # 预期剩余大于0才会购买
                degree.append(np.exp(U))
                U_list.append(U)
                alternative.append(provider.num)
    if len(alternative) == 0:  # 说明没有空余房间
        return False  # 表明消费者未预订
    elif len(alternative) == 1:  # 说明只有一个提供方可选
        choice = alternative[0]  # 只能选择该提供方
        customer.expected_surplus = U_list[0]
    else:  # 两个提供方均可选
        if time_step <= 30:  # 第一期，消费者随机选择，避免仿真受初始值的影响
            choice = np.random.choice(alternative, size=1)[0]
            customer.expected_surplus = U_list[choice]
        else:  # 选择预期剩余最大的提供方购买
            degree = np.array(degree)
            choice = np.where(degree == max(degree))[0]
            choice = np.random.choice(choice, size=1)[0]
            customer.expected_surplus = U_list[choice]
    # 消费者信息更新，剩下消费者评论退房时更新
    customer.perceived_provider = choice
    customer.attend_time = time_step
    customer.price = providers[choice].price_i
    customer.service = providers[choice].service_i
    customer.perceived_service = (1 + providers[choice].o_i) * providers[choice].service_i
    providers[choice].occupy_num += 1  # 房间占用数+1
    providers[choice].service_c_list.append(customer)  # 添加在住消费者
    providers[choice].order_num += 1  # 办理入住数量
    # 当月未退房消费者不算在本月服务过的消费者数量中
    return True  # 表明消费者成功预订


def checkout(providers, time_step, platform):  # 检验到期消费者，办理退房
    for provider in providers:
        remove_list = []
        for j in range(len(provider.service_c_list)):
            customer = provider.service_c_list[j]
            if customer.attend_time + customer.occupancy_days == time_step:  # 房间到期，办理退住
                Comment_produce(customer, provider, platform)
                provider.occupy_num -= 1  # 房间占用减一
                remove_list.append(customer)
                provider.service_num += 1  # 本期服务消费者数量加一
        for customer in remove_list:
            provider.service_c_list.remove(customer)  # 在住消费者列表剔除
    return 0


def Comment_produce(customer, provider, platform):  # 评论和回复的产生以及消费者收益的计算,商家收益的累计
    Val = u0 + customer.service - customer.price  # 实际剩余
    # print(f'消费者实际剩余与预期剩余：{Val,customer.service,customer.price}，{customer.expected_surplus}')
    satisfaction = (Val - customer.expected_surplus) / customer.expected_surplus  # customer.expected_surplus始终>0
    if satisfaction >= 0:  # 当实际大于预期，感知净效用提高
        percevied_Val = Val * (1 + satisfaction)
    else:  # 当实际小于预期，感知净效用下降
        if Val >= 0:
            percevied_Val = Val * (1 + satisfaction)  # 使正的减小，satisfaction在-1到0之间
        else:  # 这里可以结合容忍区理论，当实际净效用小于0时，感知不确认的影响更大
            percevied_Val = Val * (1 + abs(satisfaction))  # 使负的更多，satisfaction<0
    flag = 0  # 被激励好评标志
    receive_flag = 0  # 获得好评激励标志

    review_p = min((1 + abs(satisfaction)) * 0.5, 1)  # 评论概率，根据满意度的程度计算
    review_flag = np.random.choice([0, 1], size=1, p=[1 - review_p, review_p])[0]  # 是否评论

    if review_flag == 1:  # 愿意评论
        if satisfaction >= customer.Po_val_j:  # 自发好评
            reply_flag = np.random.choice([0, 1], size=1, p=[1 - provider.p_reply_i[0], provider.p_reply_i[0]])[
                0]  # 是否回应,.p_reply_i[0]表示对好评进行回应的概率
            receive_flag = 1
            provider.recieve_pci_num += 1  # 接受好评激励的人数
            if reply_flag == 1:  # 提供方回应
                reply_choice = np.random.choice([0, 1], size=1, p=[1 - provider.per_i[0], provider.per_i[0]])[
                    0]  # 回复的类型,per_i[0]表示对好评做出提升回应的概率，0表示共毁型回复、1表示提升型回复
                if reply_choice == 1:  # 提升型回应
                    customer.comment = [0, 0, 1, 0, 0, 0]  # 价值共创类评论
                else:  # 破坏型回应
                    customer.comment = [0, 0, 0, 0, 1, 0]  # 回应端价值下降类评论
            else:
                customer.comment = [1, 0, 0, 0, 0, 0]  # 好评无回应
        else:  # 差评
            satisfaction = (Val + provider.pci_i - customer.expected_surplus) / customer.expected_surplus
            if satisfaction >= customer.Po_val_j:  # 被激励好评
                reply_flag = np.random.choice([0, 1], size=1, p=[1 - provider.p_reply_i[0], provider.p_reply_i[0]])[
                    0]  # 是否回应,.p_reply_i[0]表示对好评进行回应的概率
                flag = 1
                receive_flag = 1
                provider.incentive_num += 1  # 被激励好评人数
                provider.recieve_pci_num += 1  # 接受好评激励的人数
                if reply_flag == 1:
                    reply_choice = np.random.choice([0, 1], size=1, p=[1 - provider.per_i[0], provider.per_i[0]])[
                        0]  # 回复的类型,per_i[0]表示对好评做出提升回应的概率，0表示共毁型回复、1表示提升型回复
                    if reply_choice == 1:  # 提升型回应
                        customer.comment = [0, 0, 1, 0, 0, 0]  # 价值共创类评论
                    else:  # 破坏型回应
                        customer.comment = [0, 0, 0, 0, 1, 0]  # 回应端价值下降类评论
                else:
                    customer.comment = [1, 0, 0, 0, 0, 0]  # 好评无回应
            else:  # 保持差评
                reply_flag = np.random.choice([0, 1], size=1, p=[1 - provider.p_reply_i[1], provider.p_reply_i[1]])[
                    0]  # 是否回应,.p_reply_i[1]表示对差评进行回应的概率
                if reply_flag == 1:
                    reply_choice = np.random.choice([0, 1], size=1, p=[1 - provider.per_i[1], provider.per_i[1]])[
                        0]  # 回复的类型,per_i[1]表示对差评做出提升回应的概率，0表示共毁型回复、1表示提升型回复
                    if reply_choice == 1:  # 提升型回应
                        customer.comment = [0, 0, 0, 1, 0, 0]  # 评论端价值下降类评论
                    else:  # 破坏型回应
                        customer.comment = [0, 0, 0, 0, 0, 1]  # 价值共毁类评论
                else:
                    customer.comment = [0, 1, 0, 0, 0, 0]  # 差评无回应
    else:  # 不评论
        customer.comment = [0, 0, 0, 0, 0, 0]
    # 消费者收益计算
    c_revenue = percevied_Val * customer.occupancy_days + receive_flag * provider.pci_i + np.dot(
        np.array(customer.comment[2:]), a.T)  # 感知+好评激励+回应的作用
    provider.c_revenue_sum += c_revenue
    satisfaction = ((Val - customer.expected_surplus) / customer.expected_surplus) * customer.occupancy_days
    provider.c_satisf_sum += satisfaction
    c_surplus = Val * customer.occupancy_days
    provider.c_surplus_sum += c_surplus
    # 商家评论和收益积累
    provider_income = ((1 - platform.c_rate) * customer.price - customer.service) * customer.occupancy_days - receive_flag * provider.pci_i
    provider.income += provider_income
    provider.comment_sum = [x + y for x, y in zip(provider.comment_sum, customer.comment)]
    # 平台抽成
    platform_income = platform.c_rate * customer.price * customer.occupancy_days
    platform.income += platform_income
    # 创造的社会福利累计
    provider.s_welfare_i += c_revenue + provider_income + platform_income
    return 0


class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self, providers, platform, time_step, M=2):
        self.providers = providers
        self.platform = platform
        self.time_step = time_step
        name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
        Dim = 8  # 初始化Dim（决策变量维数）
        maxormins = [-1] * M  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        varTypes = [0] * 8  # 初始化varTypes（决策变量的类型，0：实数；1：整数）,价格、质量、夸大程度、好评奖励
        lb = [sp_limit[0], sp_limit[0], o_limit[0], pci_limit[0]] + [sp_limit[0], sp_limit[0], o_limit[0], pci_limit[0]]  # 决策变量下界
        ub = [sp_limit[1], sp_limit[1], o_limit[1], pci_limit[1]] + [sp_limit[1], sp_limit[1], o_limit[1], pci_limit[1]]  # 决策变量上界
        lbin = [1] * Dim  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def evalVars(self, Vars):  # 目标函数
        F = []
        CV = []  # CV矩阵
        for X in Vars:
            f = []
            providers_test = copy.deepcopy(self.providers)  # 提供方
            platform_test = copy.deepcopy(self.platform)  # 平台
            i = 0
            for provider in providers_test:  # 变量值赋给对应变量
                provider.price_i = X[0 + 4 * i]
                provider.service_i = X[1 + 4 * i]
                provider.o_i = X[2 + 4 * i]
                provider.pci_i = X[3 + 4 * i]
                i += 1

            for t in range(self.time_step, self.time_step + 30):  # 模拟一期后能够取得的收益
                checkout(providers_test, t, platform_test)  # 办理退房
                # 消费者到达和入住
                attend_num = int(poisson.rvs(mu=9.38))  # 实际到达顾客数
                for j in range(attend_num):
                    customer = instantiate_customer()
                    customer_choice(customer, providers_test, t)

            for provider in providers_test:
                if provider.initial_flag_i == 0:  # 利己
                    f_n = provider.income
                else:  # 利他
                    f_n = provider.g_i * provider.c_revenue_sum + (1 - provider.g_i) * provider.income
                f.append(f_n)
            F.append(f)
            cv1 = -providers_test[0].income
            cv2 = -providers_test[1].income
            # cv3=-(social_welfare)
            cv4 = -(providers_test[0].c_revenue_sum)
            cv5 = -(providers_test[1].c_revenue_sum)
            CV.append([cv1, cv2, cv4, cv5])  # ,cv3,cv4,cv5])
        del providers_test
        del platform_test
        gc.collect()
        F = np.array(F)
        # 利用可行性法则处理约束条件
        CV = np.array(CV)
        return F, CV


def Provider_decision(providers, platform, time_step):
    problem = MyProblem(providers, platform, time_step)  # 实例化问题对象
    # 构建算法
    algorithm = ea.moea_NSGA2_templet(problem, ea.Population(Encoding='BG', NIND=30), MAXGEN=30,
                                      logTras=1)  # 最大进化代数  #表示每隔多少代记录一次日志信息，0表示不记录。
    # algorithm.mutOper.Pm = 0.2  # 修改变异算子的变异概率
    # algorithm.recOper.XOVR = 0.9  # 修改交叉算子的交叉概率
    # 求解
    res = ea.optimize(algorithm, seed=time_step, verbose=False, drawing=0, outputMsg=False, drawLog=False,
                      saveFlag=False)
    if res['success'] == False:  # 优化未成功，保持原决策
        return 0
    else:
        choice = np.random.choice(range(len(res['CV'])), size=1)[0]  # 从帕累托最优解中随机选择一个
        d_base = res['CV'][choice]
        d_var = res['Vars'][choice]
        print('寻优成功,CV值和决策变量值：', d_base, d_var)
        i = 0
        for provider in providers:
            provider.price_i = d_var[0 + 4 * i]
            provider.service_i = d_var[1 + 4 * i]
            provider.o_i = d_var[2 + 4 * i]
            provider.pci_i = d_var[3 + 4 * i]
            i += 1
        return 0


def data_update(providers, platform):
    for provider in providers:
        provider.service_num = 0
        provider.order_num = 0
        provider.c_revenue_sum = 0
        provider.c_satisf_sum = 0
        provider.c_surplus_sum = 0
        provider.income = 0
        provider.s_welfare_i = 0
        provider.incentive_num = 0  # 被激励好评人数
        provider.recieve_pci_num = 0  # 接受好评激励的人数
    platform.income = 0
    return 0


def safe_division(numerator, denominator):  # 除法
    try:
        result = numerator / denominator
    except ZeroDivisionError:
        result = None
    return result


def record_data(providers, platform, filepath, attend_num_sum, month_step):
    social_welfare = platform.income + sum([provider.income for provider in providers]) + sum(
        [provider.c_revenue_sum for provider in providers])
    pr_percentage = [safe_division(provider.comment_sum[0] + provider.comment_sum[2] + provider.comment_sum[4],
                                   sum(provider.comment_sum)) for provider in providers]  # 好评占比
    pr_sum = sum(
        [provider.comment_sum[0] + provider.comment_sum[2] + provider.comment_sum[4] for provider in providers])
    r_sum = sum([sum(provider.comment_sum) for provider in providers])
    pr_sum_percentage = safe_division(pr_sum, r_sum)
    # 总好评数量占比
    row = [month_step] + [provider.initial_flag_i for provider in providers] + [provider.g_i for provider in
                                                                                providers] + [provider.o_i for provider
                                                                                              in providers] + \
          [provider.pci_i for provider in providers] + [provider.price_i for provider in providers] + [
              provider.service_i for provider in providers] + \
          [provider.service_num for provider in providers] + [sum([provider.service_num for provider in providers])] + [
              provider.comment_sum for provider in providers] + \
          [social_welfare] + [platform.income] + [provider.income for provider in providers] + [
              sum([provider.income for provider in providers])] + \
          [provider.c_revenue_sum for provider in providers] + [
              sum([provider.c_revenue_sum for provider in providers])] + [provider.c_satisf_sum for provider in providers] + [
              sum([provider.c_satisf_sum for provider in providers])] + [provider.c_surplus_sum for provider in providers] + [
              sum([provider.c_surplus_sum for provider in providers])] + [attend_num_sum] + \
          [provider.order_num for provider in providers] + [sum([provider.order_num for provider in providers])] + \
          [provider.incentive_num for provider in providers] + [provider.recieve_pci_num for provider in providers] + \
          pr_percentage + [pr_sum_percentage]
    new_row_df = pd.DataFrame([row])
    new_row_df.to_csv(filepath, mode='a', header=False, index=False, encoding='utf-8-sig')
    return 0


def set_global_dct(v_name, v_value):  # 用于修改全局变量的值
    glb_dct = globals()
    glb_dct[v_name] = v_value


def main(n):
    random.seed(n)  # 设置随机数种子
    np.random.seed(n)
    columns = ['step', '提供方1类型', '提供方2类型', '提供方1利他程度', '提供方2利他程度', '夸大宣传程度1',
               '夸大宣传程度2', '好评激励1', '好评激励2',
               '价格1', '价格2', '服务质量1', '服务质量2', '服务人数1', '服务人数2', '总服务人数', '评论1', '评论2',
               '社会总福利', '平台收益', '提供方收益1', '提供方收益2', '提供方总收益', '消费者收益1', '消费者收益2',
               '消费者总收益', '消费者满意度1', '消费者满意度2', '消费者总满意度','消费者剩余1', '消费者剩余2', '消费者总剩余',
               '实际到达人数', '办理入住数1', '办理入住数2', '办理入住总数', '被激励好评数量1', '被激励好评数量2',
               '获得激励数量1', '获得激励数量2',
               '好评占比1', '好评占比2', '总好评数量占比']
    data = pd.DataFrame(columns=columns)  # 要记录哪些数据
    folder_path = '数据\实验数据\实验名\实验取值'
    # 检查目录是否存在，如果不存在则创建
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    filepath = folder_path + f'\指标结果{n}.csv'
    data.to_csv(filepath, encoding='utf-8-sig', index=None)
    # 实例化对象
    attend_num_sum = 0
    platform = instantiate_platform()
    providers = instantiate_providers()
    time_step = 1
    month_step = 1
    while time_step <= 3000:
        checkout(providers, time_step, platform)  # 办理退房
        # 消费者到达和入住
        attend_num = int(poisson.rvs(mu=9.38))  # 实际到达顾客数
        attend_num_sum += attend_num
        for j in range(attend_num):
            customer = instantiate_customer()
            customer_choice(customer, providers, time_step)
        if time_step % 30 == 0:
            print(f'第{month_step}期')
            record_data(providers, platform, filepath, attend_num_sum, month_step)
            data_update(providers, platform)
            Provider_decision(providers, platform, time_step)
            attend_num_sum = 0
            month_step += 1
        time_step += 1
    return n

if __name__ == '__main__':
    main(0)
