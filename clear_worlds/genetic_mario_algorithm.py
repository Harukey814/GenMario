# マリオ関連のimport
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros

# プロット関連のimport
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from IPython.display import display

# 数値関連のimport
import math
import numpy as np
import numpy.random as rnd

# 警告関連のimport
from warnings import filterwarnings

# マルチプロセス関連のimport
from concurrent.futures import ProcessPoolExecutor

# アニメーションのサイズを拡張
plt.rcParams['animation.embed_limit'] = 200

# シード値を設定（Time stamp of 1/1/2024）
rnd.seed(1704034800)

# 警告を非表示
filterwarnings("ignore", category=UserWarning)
filterwarnings("ignore", category=DeprecationWarning)
filterwarnings("ignore", category=RuntimeWarning)


class GeneticMarioAlgorithm:
    """
    遺伝的アルゴリズムでマリオを成長させる。

    Attributes
    ----------
    stage : str
        ステージ名。
    movement : [[str]]
        アクションパターン。
    max_workers : int
        最大プロセス数。
    max_generations : int
        最大世代数。
    num_marios : int
        個体数。
    len_chromosome : int
        染色体の長さ。
    cross_rate : float
        交叉率。
    mutation_rate : float
        突然変異率。
    mutation_points_rate : float
        突然変異点率。
    frame_interval : int
        行動するフレーム間隔。
    stop_actions : int
        停滞と判断するアクション数。
    """

    def __init__(
            self,
            stage,
            movement,
            max_workers,
            max_generations,
            num_marios,
            len_chromosome,
            cross_rate,
            mutation_rate,
            mutation_points_rate,
            frame_interval,
            stop_actions):
        self.stage = stage
        self.movement = movement
        self.max_workers = max_workers
        self.max_generations = max_generations
        self.num_marios = num_marios
        self.len_chromosome = len_chromosome
        self.cross_rate = cross_rate
        self.mutation_rate = mutation_rate
        self.mutation_points_rate = mutation_points_rate
        self.frame_interval = frame_interval
        self.stop_actions = stop_actions

    def create_generation(self):
        """
        初期世代を作成する。

        Returns
        -------
        generation : [[int]]
            初期世代のマリオの配列。
        """
        return rnd.randint(len(self.movement), size=(self.num_marios, self.len_chromosome))

    def cross(self, parent1, parent2):
        """"
        交叉を行う。

        Parameters
        ----------
        parent1 : [int]
            親その1。
        parent2 : [int]
            親その2。

        Returns
        -------
        child1 : [int]
            子供その1。
        child2 : [int]
            子供その2。
        """
        # 一点交叉をするポイントを決定
        cross_point = rnd.randint(self.len_chromosome)

        # 子供達を作成
        child1 = np.concatenate([parent1[:cross_point], parent2[cross_point:]])
        child2 = np.concatenate([parent2[:cross_point], parent1[cross_point:]])

        return child1, child2

    def mutation(self, mario):
        """
        突然変異を行う。

        Parameters
        ----------
        mario : [int]
            突然変異する前のマリオ。

        Returns
        -------
        mario : [int]
            突然変異した後のマリオ。
        """
        if rnd.random() < self.mutation_rate:
            # 突然変異するポイントを決定
            num_mutation_points = math.ceil(self.mutation_points_rate * self.len_chromosome)
            mutated_points = rnd.choice(self.len_chromosome, num_mutation_points, replace=False)

            # 突然変異を実行
            for mutated_point in mutated_points:
                mario[mutated_point] = rnd.randint(len(self.movement))

        return mario

    def sorts(self, fitnesses, generation, images):
        """
        適応度・マリオ・最終コマをセットで並び替える。

        Parameters
        ----------
        fitnesses : [int]
            並び替える前の適応度の配列。
        generation : [[int]]
            並び替える前のマリオの配列。
        images : [np.ndarray]
            並び替える前の最終コマの配列。

        Returns
        -------
        fitnesses_generation_images : [[int, [int], np.ndarray]]
            並び替えた後の適応度・マリオ・最終コマのセットの配列。
        """
        return zip(*sorted(zip(fitnesses, generation, images), key=lambda x: x[0], reverse=True))

    def print_fitness(self, fitnesses, current_generation):
        """
        適応度(最大・最小・平均)を表示・出力する。

        Parameters
        ----------
        fitnesses : [int]
            並び替えた後の適応度の配列。
        current_generation : int
            現在の世代番号。

        Returns
        -------
        max_fitness : int
            最大適応度
        min_fitness : int
            最小適応度
        avg_fitness : int
            平均適応度
        """
        # 適応度(最大・最小・平均)を計算
        max_fitness = fitnesses[0]
        min_fitness = fitnesses[self.num_marios - 1]
        avg_fitness = int(sum(fitnesses) / self.num_marios)

        # 適応度(最大・最小・平均)を表示
        print("{:<3}   max: {:<4}   min: {:<4}   avg: {:<4}".format(
            current_generation, max_fitness, min_fitness, avg_fitness
        ))

        return max_fitness, min_fitness, avg_fitness

    def roulette_selection(self, fitnesses, generation):
        """
        ルーレット選択で親を選ぶ。

        Parameters
        ----------
        fitnesses : [int]
            並び替えた後の適応度の配列。
        generation : [[int]]
            並び替えた後のマリオの配列。

        Returns
        -------
        paren1 : [int]
            選択された親その1。
        paren2 : [int]
            選択された親その2。
        """
        # ルーレット選択の確率を計算
        selection_rates = fitnesses / np.sum(fitnesses)

        # 親達(番号)を選択
        parent_indexes = rnd.choice(self.num_marios, 2, p=selection_rates, replace=False)

        return generation[parent_indexes[0]], generation[parent_indexes[1]]

    def evaluate(self, mario):
        """
        マリオを評価する。

        Parameters
        ----------
        mario : [int]
            評価するマリオ。

        Returns
        -------
        fitness : int
            適応度。
        image : np.ndarray
            最終コマ。
        flag : bool
            ゴール到達を意味するフラッグ。
        """
        # 環境設定
        env = gym_super_mario_bros.make(self.stage)
        env = JoypadSpace(env, self.movement)
        env.reset()

        # ゲーム本番
        breaker = False
        positions = []
        for action in mario:
            for _ in range(self.frame_interval):
                observation, reward, done, info = env.step(action)
                if done:
                    breaker = True
                    break

            if breaker:
                break

            # 座標を保存
            positions.append(info["x_pos"])

            # 停滞(定めたアクション数の間、座標が一定)の場合は終了
            if len(positions) >= self.stop_actions and len(set(positions[-self.stop_actions:])) == 1:
                break

        # 適応度計算（進んだ距離）
        fitness = info["x_pos"]

        return fitness, env.render(mode='rgb_array'), info["flag_get"]

    def run(self):
        """
        遺伝的アルゴリズムを実行する。

        Returns
        -------
        super_marios : [[int]]
            各世代の最優秀マリオの配列。
        generations_fitnesses : [[int]]
            各世代の適応度(最大・最小・平均)の配列。
        """
        # GA本番（マルチプロセス）
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # 世代ごとの最優秀マリオ
            super_marios = []

            # 世代ごとの適応度(最大・最小・平均)
            generations_fitnesses = []

            # 初期世代
            generation = self.create_generation()

            # 前世代の最大適応度
            previous_max_fitness = 0

            for current_generation in range(1, self.max_generations + 1):
                # 評価
                evaluations = list(executor.map(self.evaluate, generation))
                fitnesses = [evaluation[0] for evaluation in evaluations]
                images = [evaluation[1] for evaluation in evaluations]

                # 並び替え・表示
                fitnesses, generation, images = self.sorts(fitnesses, generation, images)
                if previous_max_fitness < fitnesses[0]:
                    plt.figure(figsize=(3, 3))
                    plt.imshow(images[0])
                    plt.show()

                generation_fitnesses = self.print_fitness(fitnesses, current_generation)

                # 保存
                generations_fitnesses.append(generation_fitnesses)
                super_marios.append(generation[0])

                # ゴール到達なら終了
                if any(evaluation[2] for evaluation in evaluations):
                    break

                # 世代交代
                num_elite = math.ceil(self.num_marios * (1 - self.cross_rate))
                next_generation = list(generation[:num_elite])
                while len(next_generation) < self.num_marios:
                    parent1, parent2 = self.roulette_selection(fitnesses, generation)
                    child1, child2 = self.cross(parent1, parent2)
                    next_generation.extend([self.mutation(child1), self.mutation(child2)])

                previous_max_fitness = fitnesses[0]
                generation = next_generation[:self.num_marios]

            return super_marios, generations_fitnesses

    def mario_animation(self, mario):
        """
        マリオのアニメーションを作成する関数

        Parameters
        ----------
        mario : [int]
            アニメーションを作成するマリオ。
        """
        # 環境設定
        env = gym_super_mario_bros.make(self.stage)
        env = JoypadSpace(env, self.movement)
        env.reset()

        # コマ数を確認
        def count_frames():
            returner = False
            for count, action in enumerate(mario):
                for _ in range(self.frame_interval):
                    observation, reward, done, info = env.step(action)
                    if done:
                        returner = True
                        break

                if returner:
                    env.reset()
                    return count

        # 初期化関数
        def init():
            pass

        # 描画を更新
        def update(frame):
            for _ in range(self.frame_interval):
                observation, reward, done, info = env.step(mario[frame])
                image.set_data(env.render(mode='rgb_array'))
                if done:
                    env.reset()

        # アニメーションの準備
        fig, ax = plt.subplots(figsize=(5, 5))
        image = ax.imshow(env.render(mode='rgb_array'))

        # アニメーション作成（FPS of Super Mario Bros is 20）
        anime = animation.FuncAnimation(fig, update, init_func=init, frames=range(count_frames()), interval=1000 * self.frame_interval / 20)

        # アニメーションを表示
        rc('animation', html='jshtml')
        display(anime)

    def plot_fitnesses(self, fitnesses):
        """
        適応度(最大・最小・平均)の推移をプロットする関数

        Parameters
        ----------
        fitnesses : [[int]]
            各世代の適応度(最大・最小・平均)の配列。
        """
        # 分解
        max, min, avg = zip(*fitnesses)

        # 横軸
        x_values = list(range(1, len(max) + 1))

        # 描画
        plt.plot(x_values, max, label='max')
        plt.plot(x_values, min, label='min')
        plt.plot(x_values, avg, label='avg')

        # 凡例
        plt.legend()

        # ラベル
        plt.xlabel('Generation')
        plt.ylabel('Fitness')

        # 表示
        plt.show()
