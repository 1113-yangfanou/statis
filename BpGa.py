from tqdm import tqdm

from model import get_model
import numpy as np
import torch

def get_params(model):
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1).cpu().numpy())
    return np.concatenate(params)

def set_params(model, params):
    ptr = 0
    for param in model.parameters():
        size = param.data.numel()
        param_shape = param.data.shape
        param.data = torch.from_numpy(params[ptr:ptr+size]).view(param_shape)
        ptr += size


def evaluate_fitness(individual, train_loader, val_loader, epochs=1):
    try:
        model = get_model()
        set_params(model, individual)
        optimizer = torch.optim.Adam(model.parameters())
        criterion = torch.nn.L1Loss()
        model.cuda()
        # 训练模型
        model.train()
        for _ in range(epochs):
            for data, target in train_loader:
                data = data.cuda()
                target = target.cuda()
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

        # 计算验证准确率
        model.eval()
        loss = 0
        total_samples = 0  # 防止除零错误
        with torch.no_grad():
            for data, target in val_loader:
                data = data.cuda()
                target = target.cuda()
                output = model(data)
                loss += criterion(output, target).item()

        return -1 * loss / len(val_loader)  # 负值表示越小越好
    except Exception as e:
        print(f"个体评估失败: {str(e)}")
        return 0.0  # 保底返回值

class GeneticAlgorithm:
    def __init__(self, param_size, pop_size, mutation_rate, crossover_rate, elite=2):
        self.param_size = param_size
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite = elite  # 保留的精英个体数

    def initialize_population(self):
        return [np.random.randn(self.param_size) * np.sqrt(2.0/(self.param_size))
            for _ in range(self.pop_size)]

    def rank_selection(self, fitness):
        ranked = np.argsort(fitness)[::-1]
        return ranked[:self.pop_size]

    def crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_rate:
            split = np.random.randint(1, self.param_size)
            child = np.concatenate([parent1[:split], parent2[split:]])
            return child
        return parent1.copy()

    def mutate(self, individual):
        mask = np.random.rand(self.param_size) < self.mutation_rate
        individual[mask] += np.random.normal(0, 0.1, size=np.sum(mask))
        return individual

    def evolve(self, population, fitness):
        # 保留精英
        elite_indices = np.argsort(fitness)[-self.elite:]
        new_population = [population[i].copy() for i in elite_indices]

        # 处理适应度值
        fitness = np.array(fitness)

        # 1. 替换无效值
        fitness = np.nan_to_num(fitness, nan=0.0, posinf=0.0, neginf=0.0)

        # 2. 确保非负
        fitness = fitness - np.min(fitness)

        # 3. 处理全零情况
        if np.sum(fitness) == 0:
            fitness = np.ones_like(fitness)

        # 4. 归一化概率
        prob = fitness / np.sum(fitness)

        # 5. 验证维度一致性
        assert len(prob) == len(population), f"概率维度{len(prob)}≠种群维度{len(population)}"

        # 生成后代
        for _ in range((self.pop_size - self.elite) // 2):
            parent_indices = np.random.choice(len(population), size=2, p=prob)
            parent1 = population[parent_indices[0]]
            parent2 = population[parent_indices[1]]
            child1 = self.crossover(parent1, parent2)
            child2 = self.crossover(parent2, parent1)
            new_population.extend([self.mutate(child1), self.mutate(child2)])

        return new_population[:self.pop_size]


def ga_optimization(model, train_loader, val_loader, generations=50, pop_size=30):
    param_size = len(get_params(model))
    ga = GeneticAlgorithm(param_size, pop_size, mutation_rate=0.01, crossover_rate=0.8)
    population = ga.initialize_population()

    # 初始化全局最优
    best_fitness = -np.inf
    best_individual = None

    # 预评估初始种群
    initial_fitness = [evaluate_fitness(ind, train_loader, val_loader) for ind in population]
    best_idx = np.argmax(initial_fitness)
    best_fitness = initial_fitness[best_idx]
    best_individual = population[best_idx].copy()

    # 主循环
    for gen in tqdm(range(generations)):
        fitness = []
        for individual in population:
            fit = evaluate_fitness(individual, train_loader, val_loader)
            fitness.append(fit)

        # 更新全局最优
        current_best = np.max(fitness)
        if current_best < best_fitness:
            best_fitness = current_best
            best_individual = population[np.argmax(fitness)].copy()

        # 进化种群
        population = ga.evolve(population, np.array(fitness))

    return best_individual