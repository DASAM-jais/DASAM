import multiprocessing as mp
from base import *
import itertools


def main(n, x, y, z1, z2, ):
    print(n, x, y, z1, z2, )
    set_global_dct('essential_type', x)
    set_global_dct('g', y)
    set_global_dct('per', [[0.939, z1], [0.939, z2]])
    random.seed(n)  # 设置随机数种子
    np.random.seed(n)
    columns = ['step', '提供方1类型', '提供方2类型', '提供方1利他程度', '提供方2利他程度', '夸大宣传程度1',
               '夸大宣传程度2', '好评激励1', '好评激励2',
               '价格1', '价格2', '服务质量1', '服务质量2', '服务人数1', '服务人数2', '总服务人数', '评论1', '评论2',
               '社会总福利', '平台收益', '提供方收益1', '提供方收益2', '提供方总收益', '消费者收益1', '消费者收益2',
               '消费者总收益', '消费者满意度1', '消费者满意度2', '消费者总满意度', '消费者剩余1', '消费者剩余2',
               '消费者总剩余',
               '实际到达人数', '办理入住数1', '办理入住数2', '办理入住总数', '被激励好评数量1', '被激励好评数量2',
               '获得激励数量1', '获得激励数量2',
               '好评占比1', '好评占比2', '总好评数量占比']
    data = pd.DataFrame(columns=columns)  # 要记录哪些数据
    folder_path = r'数据\实验数据\基本实验3\类型{}-类型{}-回应偏好{}-回应偏好{}'.format(x[0], x[1], z1, z2)
    # 检查目录是否存在，如果不存在则创建
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    filepath = folder_path + f'\指标结果{n}.csv'
    if os.path.exists(filepath):
        print('跳过')
        return n
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


def init_lock(l):
    global lock
    lock = l


if __name__ == '__main__':
    essential_types = [[0, 0], [0, 1], [1, 1]]
    gs = [[0, 0], [0, 0.5], [0.5, 0.5]]
    pers1 = [0.1, 0.5, 0.9]
    pers2 = [0.1, 0.5, 0.9]
    num_cores = int(mp.cpu_count())
    l = mp.Lock()
    pool = mp.Pool(processes=num_cores - 2, initializer=init_lock, initargs=(l,))
    results = [pool.apply_async(main, args=(n, x, y, z1, z2,)) for n, (x, y), z1, z2 in
               itertools.product(range(0, 5), zip(essential_types, gs), pers1, pers2)]
    results = [p.get() for p in results]
    pool.terminate()
    pool.join()
    print(results)

