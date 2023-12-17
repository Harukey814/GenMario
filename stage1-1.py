# マリオ関連のimport
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros

# プロット関連のimport
# import matplotlib.pyplot as plt
# from matplotlib import animation, rc

# 数値関連のimport
import math
import numpy as np
import numpy.random as rnd

# 警告関連のimport
import warnings

# マルチプロセス関連のimport
from concurrent.futures import ProcessPoolExecutor

# シード値を設定
rnd.seed(1704034800)

# 警告を非表示
warnings.filterwarnings("ignore", category=UserWarning, module="gym.envs.registration")

# アクションパターン
MOVEMENT = [
    ['right', 'B'],
    ['right', 'A', 'B'],
]

# 定数設定
MAX_WORKERS = 10       # 最大プロセス数
MAX_GENERATIONS = 100  # 最大世代数
NUM_MARIOS = 50        # 個体数
LEN_CHROMOSOME = 300   # ゲノムの長さ
CROSS_RATE = 0.8       # 交叉率
MUTATION_RATE = 0.1    # 突然変異率
FRAME_INTERVAL = 10    # 行動するフレーム間隔


def create_generation():
    """初期世代を作成する関数"""
    return rnd.randint(len(MOVEMENT), size=(NUM_MARIOS, LEN_CHROMOSOME))


def cross(parent1, parent2):
    """交叉する関数"""
    cross_points = rnd.choice(LEN_CHROMOSOME, 1, replace=False)
    child1 = np.concatenate([parent1[:cross_points[0]], parent2[cross_points[0]:]])
    child2 = np.concatenate([parent2[:cross_points[0]], parent1[cross_points[0]:]])
    return child1, child2


def mutation(mario):
    """突然変異する関数"""
    if rnd.random() < MUTATION_RATE:
        mutated_indexes = rnd.choice(LEN_CHROMOSOME, math.ceil(0.05 * LEN_CHROMOSOME), replace=False)
        for index in mutated_indexes:
            mario[index] = rnd.randint(len(MOVEMENT))

    return mario


def sorts(fitnesses, generation):
    """ゲノムを並び替える関数"""
    return zip(*sorted(zip(fitnesses, generation), key=lambda x: x[0], reverse=True))


def print_fitness(fitnesses, current_generation):
    """適応度を出力する関数"""
    max = fitnesses[0]
    min = fitnesses[NUM_MARIOS - 1]
    avg = int(sum(fitnesses) / NUM_MARIOS)
    print("{:<3}   max: {:<4}   min: {:<4}   avg: {:<4}".format(current_generation, max, min, avg))


def roulette_selection(fitnesses, generation):
    """ルーレット選択を行う関数"""
    selection_rates = fitnesses / np.sum(fitnesses)
    parent_indexes = rnd.choice(NUM_MARIOS, 2, p=selection_rates, replace=False)
    return generation[parent_indexes[0]], generation[parent_indexes[1]]


def evaluate(mario):
    """評価関数"""
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
    env = JoypadSpace(env, MOVEMENT)
    env.reset()
    end_flag = False
    for action in mario:
        for _ in range(FRAME_INTERVAL):
            observation, reward, done, info = env.step(action)
            if done:
                end_flag = True
                break

        if end_flag:
            break

    fitness = info["x_pos"]
    env.close()
    return fitness


if __name__ == "__main__":
    generation = create_generation()
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for current_generation in range(MAX_GENERATIONS):
            fitnesses = []
            evaluations = list(executor.map(evaluate, generation))
            fitnesses.extend(evaluations)
            fitnesses, generation = sorts(fitnesses, generation)
            print_fitness(fitnesses, current_generation)
            next_generation = []
            num_elite = math.ceil(NUM_MARIOS * (1 - CROSS_RATE))
            next_generation.extend(generation[:num_elite])
            while len(next_generation) < NUM_MARIOS:
                parent1, parent2 = roulette_selection(fitnesses, generation)
                child1, child2 = cross(parent1, parent2)
                next_generation.extend([mutation(child1), mutation(child2)])

            generation = next_generation[:NUM_MARIOS]
