import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# 遗传算法参数
POP_SIZE = 100  # 种群大小
GEN_MAX = 50  # 最大迭代次数
MUTATION_RATE = 0.07  # 变异率
CROSS_RATE = 0.8  # 交叉率

# BP神经网络参数
INPUT_SIZE = 31  # 输入层节点数（31个指标）
HIDDEN_SIZE = 10  # 隐藏层节点数
OUTPUT_SIZE = 1  # 输出层节点数
LEARNING_RATE = 0.1
EPOCHS = 1000
ERROR_THRESHOLD = 0.01


# 数据预处理（加载数据、归一化、插值）
def load_data(file_path):
    # 读取数据
    data = pd.read_excel(file_path)

    # 提取特征和标签
    X = data.iloc[:, 2:].values  # 第3列往后是31个指标
    y = data.iloc[:, -1].values.reshape(-1, 1)  # 假设最后一列是标签（韧性指数）
    """
    这里索引是有问题的，因为模拟的数据里面没有计算熵值要计算过放到最后在跑这个模型
    """

    # 归一化处理
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_normalized = scaler_X.fit_transform(X)
    y_normalized = scaler_y.fit_transform(y)

    # 插值生成更多样本
    def interpolate_data(data, num_interpolations=499):
        interpolated_data = []
        for i in range(len(data) - 1):
            for t in range(num_interpolations + 1):
                interpolated_point = data[i] + (data[i + 1] - data[i]) * (t / (num_interpolations + 1))
                interpolated_data.append(interpolated_point)
        return np.array(interpolated_data)

    # 生成2001组数据
    X_interpolated = interpolate_data(X_normalized)
    y_interpolated = interpolate_data(y_normalized)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_interpolated, y_interpolated, test_size=0.1, random_state=42)

    return X_train, X_test, y_train, y_test, scaler_y


# BP神经网络类

class BPNN:
    def __init__(self, input_size, hidden_size, output_size):
        # He初始化（适配ReLU）
        self.w1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros(hidden_size)
        # Xavier初始化输出层
        self.w2 = np.random.randn(hidden_size, output_size) * np.sqrt(1.0 / hidden_size)
        self.b2 = np.zeros(output_size)

    def relu(self, z):
        """带数值保护的ReLU函数"""
        return np.maximum(0, z)

    def forward(self, x):
        # 隐藏层使用ReLU激活
        z = x @ self.w1 + self.b1
        self.hidden = self.relu(z)
        # 输出层保持线性
        output = self.hidden @ self.w2 + self.b2
        return output

    def train(self, x, y, lr):
        # 前向传播
        output = self.forward(x)

        # 误差计算（添加数值保护）
        error = y - output
        error = np.clip(error, -1e10, 1e10)  # 防止极端误差值

        # --- 反向传播 ---
        # 输出层梯度
        d_output = error * 1.0
        d_w2 = self.hidden.T @ d_output

        # 隐藏层梯度（ReLU导数）
        d_hidden = (d_output @ self.w2.T) * (self.hidden > 0).astype(float)
        d_w1 = x.T @ d_hidden

        # 梯度裁剪（逐参数裁剪）
        def clip_grad(grad, clip_value=1.0):
            return np.clip(grad, -clip_value, clip_value)

        d_w2 = clip_grad(d_w2)
        d_w1 = clip_grad(d_w1)
        d_b2 = clip_grad(np.sum(d_output, axis=0))
        d_b1 = clip_grad(np.sum(d_hidden, axis=0))

        # 参数更新（带学习率衰减）
        self.w1 += lr * 0.95 ** epoch * d_w1  # 指数衰减学习率
        self.b1 += lr * 0.95 ** epoch * d_b1
        self.w2 += lr * 0.95 ** epoch * d_w2
        self.b2 += lr * 0.95 ** epoch * d_b2

        # 参数数值保护
        self.w1 = np.clip(self.w1, -1e5, 1e5)
        self.w2 = np.clip(self.w2, -1e5, 1e5)

        return np.mean(np.abs(error))


# 遗传算法优化器
class GAOptimizer:
    def __init__(self, bpnn):
        self.bpnn = bpnn
        self.population = []

    def init_population(self):
        for _ in range(POP_SIZE):
            individual = np.concatenate([
                self.bpnn.w1.flatten(),
                self.bpnn.w2.flatten(),
                self.bpnn.b1.flatten(),
                self.bpnn.b2.flatten()
            ])
            self.population.append(individual)

    def fitness(self, individual):
        # 计算适应度（误差绝对值）
        self.bpnn.w1 = individual[:INPUT_SIZE * HIDDEN_SIZE].reshape(INPUT_SIZE, HIDDEN_SIZE)
        self.bpnn.w2 = individual[
                       INPUT_SIZE * HIDDEN_SIZE:INPUT_SIZE * HIDDEN_SIZE + HIDDEN_SIZE * OUTPUT_SIZE].reshape(
            HIDDEN_SIZE, OUTPUT_SIZE)
        self.bpnn.b1 = individual[
                       INPUT_SIZE * HIDDEN_SIZE + HIDDEN_SIZE * OUTPUT_SIZE:INPUT_SIZE * HIDDEN_SIZE + HIDDEN_SIZE * OUTPUT_SIZE + HIDDEN_SIZE]
        self.bpnn.b2 = individual[-OUTPUT_SIZE:]
        pred = self.bpnn.forward(X_train)
        return np.mean(np.abs(y_train - pred))

    def select(self):
        # 轮盘赌选择
        fitnesses = [1 / self.fitness(ind) for ind in self.population]
        probs = np.array(fitnesses) / np.sum(fitnesses)
        # 确保 population 是1维列表（每个元素是一个个体的一维数组）
        return [self.population[i] for i in np.random.choice(len(self.population), size=POP_SIZE, p=probs)]

    def crossover(self, parent1, parent2):
        # 单点交叉
        if np.random.rand() < CROSS_RATE:
            idx = np.random.randint(len(parent1))
            return np.concatenate([parent1[:idx], parent2[idx:]]), np.concatenate([parent2[:idx], parent1[idx:]])
        return parent1, parent2

    def mutate(self, individual):
        # 变异操作
        for i in range(len(individual)):
            if np.random.rand() < MUTATION_RATE:
                individual[i] += np.random.normal(0, 0.1)
        return individual

    def evolve(self):
        new_pop = []
        parents = self.select()
        for i in range(0, POP_SIZE, 2):
            child1, child2 = self.crossover(parents[i], parents[i + 1])
            new_pop.append(self.mutate(child1))
            new_pop.append(self.mutate(child2))
        self.population = new_pop


# 主程序
if __name__ == "__main__":
    # 加载数据
    file_path = "./data/bpga_data.xlsx"  # 数据集路径
    X_train, X_test, y_train, y_test, scaler_y = load_data(file_path)

    # 初始化BP神经网络
    bpnn = BPNN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)

    # 遗传算法优化
    ga = GAOptimizer(bpnn)
    ga.init_population()

    # 进化过程
    for gen in range(GEN_MAX):
        ga.evolve()
        best_idx = np.argmin([ga.fitness(ind) for ind in ga.population])
        best_ind = ga.population[best_idx]
        print(f"Generation {gen}, Best Fitness: {ga.fitness(best_ind)}")

    # 使用优化后的参数训练BPNN
    best_bpnn = BPNN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    best_bpnn.w1 = best_ind[:INPUT_SIZE * HIDDEN_SIZE].reshape(INPUT_SIZE, HIDDEN_SIZE)
    best_bpnn.w2 = best_ind[INPUT_SIZE * HIDDEN_SIZE:INPUT_SIZE * HIDDEN_SIZE + HIDDEN_SIZE * OUTPUT_SIZE].reshape(
        HIDDEN_SIZE, OUTPUT_SIZE)
    best_bpnn.b1 = best_ind[
                   INPUT_SIZE * HIDDEN_SIZE + HIDDEN_SIZE * OUTPUT_SIZE:INPUT_SIZE * HIDDEN_SIZE + HIDDEN_SIZE * OUTPUT_SIZE + HIDDEN_SIZE]
    best_bpnn.b2 = best_ind[-OUTPUT_SIZE:]

    # 训练网络
    for epoch in range(EPOCHS):
        try:
            error = best_bpnn.train(X_train, y_train, LEARNING_RATE)
            if np.isnan(error) or np.isinf(error):
                print(f"Early stopping at epoch {epoch} due to numerical instability")
                break
            if error < ERROR_THRESHOLD:
                print(f"Converged at epoch {epoch}")
                break
            print(f"Epoch {epoch}, Error: {error}")
        except FloatingPointError:
            print(f"Numerical error detected at epoch {epoch}")
            break


    # 测试并输出韧性指数
    test_pred = best_bpnn.forward(X_test)
    test_pred_original = scaler_y.inverse_transform(test_pred)  # 反归一化
    y_test_original = scaler_y.inverse_transform(y_test)  # 反归一化

    # 输出前10个样本的预测值和真实值
    print("\n=== 韧性指数预测结果 ===")
    for i in range(10):
        print(f"样本 {i + 1} - 预测值: {test_pred_original[i][0]:.4f}, 真实值: {y_test_original[i][0]:.4f}")

    # 计算平均绝对误差
    test_error = np.mean(np.abs(y_test_original - test_pred_original))
    print(f"\n测试集平均绝对误差: {test_error:.4f}")

"""
该代码是能够运行的但它好像是计算整一个产业韧性，并没有分上下游和抵抗恢复能力，
需要在此基础上为每个子系统独立实现遗传算法优化，
计算并输出各省份各年的韧性指数并保证结果在0-1范围内
"""
