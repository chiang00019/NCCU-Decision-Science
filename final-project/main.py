import numpy as np
import pandas as pd
import copy
# -------------------------
# Step 1. Person 類別
# -------------------------
class Person:
    """
    - person_id: 全域唯一編號
    - budget: 剩餘預算
    - capacity: 剩餘可喝量 (ml)
    - person_type: 'A' / 'B' / 'C'
    - price_sensitive: True / False
    - favorite_list: 若 B/C 型，紀錄可接受的酒款；A 型可為 None
    - beta: 需求函數中對價格的敏感度
    """
    def __init__(self, person_id, budget, capacity, person_type, price_sensitive, favorite_list, beta):
        self.person_id = person_id
        self.budget = budget
        self.capacity = capacity
        self.person_type = person_type
        self.price_sensitive = price_sensitive
        self.favorite_list = favorite_list
        self.beta = beta

# -------------------------
# Step 2. 全域參數
# -------------------------
NUM_BEERS = 20
INIT_PRICE = 1.0
UPPER_BOUND = 1.2
LOWER_BOUND = 0.5
PRICE_INCREASE = 0.005
PRICE_DECREASE = 0.04
TIME_SLOTS = 26
DAYS = 7

ALPHA = 500   # 基礎需求量
NOISE_STD = 100  # 需求量的雜訊

# 三種類型的比例 (A / B / C)
P_A = 0.4
P_B = 0.3
P_C = 0.3

P_SENSITIVE = 0.5  # 50% 價格敏感

# -------------------------
# (新增) 熱門酒款設定
# -------------------------
HOT_BEERS = [0, 1, 2, 3, 4]  # 假設前 5 款為熱門
# 欲使熱門酒款更容易被抽中，可以提高其權重
# 例如：熱門酒款的權重=3，非熱門酒款權重=1
beer_weights = [3 if i in HOT_BEERS else 1 for i in range(NUM_BEERS)]
weight_sum = sum(beer_weights)

def sample_hot_beers(k):
    """
    從 20 款啤酒中，依照 beer_weights 的權重，抽 k 個 (不重複)。
    回傳: list of beer indices
    """
    # 使用 np.random.choice 時，可傳入 p=加權後的機率
    # 首先計算各酒的機率
    probs = np.array(beer_weights) / weight_sum
    # 不重複抽 k 個
    chosen = np.random.choice(
        np.arange(NUM_BEERS),
        size=k,
        replace=False,
        p=probs
    )
    return chosen.tolist()

def sample_one_hot_beer():
    """
    從 20 款啤酒中，用權重抽 1 個
    """
    probs = np.array(beer_weights) / weight_sum
    return np.random.choice(np.arange(NUM_BEERS), p=probs)

# -------------------------
# Step 3. 需求函數
# -------------------------
def demand_function(price, beta):
    noise = np.random.normal(0, NOISE_STD)
    q = ALPHA - beta * price + noise
    return max(0, q)

# -------------------------
# Step 4. 選酒邏輯 (A/B/C + 價格敏感)
# -------------------------
def choose_beer(person, beer_prices):
    ptype = person.person_type
    sensitive = person.price_sensitive
    flist = person.favorite_list
    all_indices = np.arange(NUM_BEERS)

    # B 型: 單一款
    if ptype == 'B':
        return flist[0]

    # C 型: 多款清單
    if ptype == 'C':
        if sensitive:
            # 挑清單中最便宜
            possible_prices = [beer_prices[b] for b in flist]
            best_idx = flist[np.argmin(possible_prices)]
            return best_idx
        else:
            # 在清單中隨機
            return np.random.choice(flist)

    # A 型: 大雜燴 (可能所有酒都喝)
    if sensitive:
        # (1/price) 為權重
        inv_prices = 1 / beer_prices
        w = inv_prices / inv_prices.sum()
        return np.random.choice(all_indices, p=w)
    else:
        # 純隨機
        return np.random.randint(0, NUM_BEERS)

# -------------------------
# Step 5. 產生新客人 (含 A/B/C + 敏感度)
#    於此步驟中，引入「熱門酒款」的機制
# -------------------------
def generate_new_people(base_person_id, num_people):
    persons = []
    current_id = base_person_id

    for _ in range(num_people):
        # 決定類型 (A/B/C)
        r = np.random.rand()
        if r < P_A:
            p_type = 'A'
        elif r < P_A + P_B:
            p_type = 'B'
        else:
            p_type = 'C'

        # 價格敏感
        is_sensitive = (np.random.rand() < P_SENSITIVE)

        # favorite_list # NOTE
        if p_type == 'A':
            # A 型: 不限定任何固定清單 => 之後選酒才決定
            fav_list = None
        elif p_type == 'B':
            # B 型: 單一款 => 從熱門機制抽 1 (較容易抽到熱門)
            fav_list = [sample_one_hot_beer()]
        else:
            # C 型: 多款清單 => 從熱門機制抽 2~5 款
            L = np.random.randint(2, 6)  # 2~5
            fav_list = sample_hot_beers(L)

        # budget
        while True:
            budget = np.random.normal(1000, 200)
            if budget > 0:
                break

        # capacity
        capacity = np.random.triangular(400, 800, 1200)

        # beta
        if is_sensitive:
            beta_ = 40
        else:
            beta_ = 20

        p = Person(
            person_id=current_id,
            budget=budget,
            capacity=capacity,
            person_type=p_type,
            price_sensitive=is_sensitive,
            favorite_list=fav_list,
            beta=beta_
        )
        persons.append(p)
        current_id += 1

    return persons, current_id


# -------------------------
# (新) Step 5-1. 用來取得 (day, timeslot) 的 lambda (客流量)
# -------------------------
def get_lambda(day, timeslot):
    """
    day=1: 星期一, day=2: 星期二, ...
    day=5: 星期五, day=6: 星期六, day=7: 星期日
    early slot: 1~15
    late slot: 16~26
    """
    is_weekend = (day == 5 or day == 6)
    is_morning = (1 <= timeslot <= 15)

    if is_weekend:
        if is_morning:
            return 25
        else:
            return 40
    else:
        if is_morning:
            return 20
        else:
            return 25

# -------------------------
# Step 6. 靜態定價
# -------------------------
def simulate_static_pricing(run_id=1, days=DAYS):
    records = []
    beer_prices = np.array([INIT_PRICE] * NUM_BEERS)

    person_id_counter = 1
    active_persons = []

    for d in range(1, days+1):
        for t in range(1, TIME_SLOTS+1):
            lam = get_lambda(d, t)
            num_new = np.random.poisson(lam)
            new_persons, person_id_counter = generate_new_people(person_id_counter, num_new)
            active_persons.extend(new_persons)

            still_active = []
            for p in active_persons:
                if p.budget <= 0 or p.capacity <= 0:
                    continue

                beer_idx = choose_beer(p, beer_prices)
                price = beer_prices[beer_idx]

                budget_before = p.budget
                capacity_before = p.capacity

                q_demand = demand_function(price, p.beta)
                purchased_volume = min(q_demand, p.capacity, p.budget / price)

                p.capacity -= purchased_volume
                p.budget -= purchased_volume * price

                row_data = {
                    "run_id": run_id,
                    "strategy": "STATIC",
                    "day": d,
                    "timeslot": t,
                    "person_id": p.person_id,
                    "person_type": p.person_type,
                    "price_sensitive": p.price_sensitive,
                    "favorite_list": str(p.favorite_list),
                    "beer_idx": beer_idx,
                    "price": price,
                    "demand": q_demand,
                    "purchased_volume": purchased_volume,
                    "budget_before": budget_before,
                    "capacity_before": capacity_before,
                    "budget_after": p.budget,
                    "capacity_after": p.capacity
                }
                records.append(row_data)

                if p.budget > 0 and p.capacity > 0:
                    still_active.append(p)
            active_persons = still_active

        # 不保留跨日
        active_persons = []

    df = pd.DataFrame(records)
    return df

# -------------------------
# Step 7. 動態定價
# -------------------------
def simulate_dynamic_pricing(run_id=1, days=DAYS):
    records = []
    person_id_counter = 1
    active_persons = []

    # 1) 只在 函式最初 初始化一次
    beer_prices = np.array([INIT_PRICE] * NUM_BEERS)

    for d in range(1, days+1):
        # NOTE: 這裡 **不** 重設 beer_prices = [1.0]*20
        # 我們直接沿用上一天(或上一時段)的價格
        # 紀錄該天的啤酒價格
        daily_prices = copy.deepcopy(beer_prices)
        
        for t in range(1, TIME_SLOTS+1):
            beer_volume_sold = np.zeros(NUM_BEERS)

            lam = get_lambda(d, t)
            new_persons, person_id_counter = generate_new_people(person_id_counter, np.random.poisson(lam))
            active_persons.extend(new_persons)

            still_active = []
            for p in active_persons:
                if p.budget <= 0 or p.capacity <= 0:
                    continue
                beer_idx = choose_beer(p, beer_prices)
                price = beer_prices[beer_idx]
                budget_before = p.budget
                capacity_before = p.capacity

                q_demand = demand_function(price, p.beta)
                purchased_volume = min(q_demand, p.capacity, p.budget / price)
                p.capacity -= purchased_volume
                p.budget -= purchased_volume * price

                beer_volume_sold[beer_idx] += purchased_volume

                row_data = {
                    "run_id": run_id,
                    "strategy": "DYNAMIC",
                    "day": d,
                    "timeslot": t,
                    "person_id": p.person_id,
                    "person_type": p.person_type,
                    "price_sensitive": p.price_sensitive,
                    "favorite_list": str(p.favorite_list),
                    "beer_idx": beer_idx,
                    "price": price,
                    "demand": q_demand,
                    "purchased_volume": purchased_volume,
                    "budget_before": budget_before,
                    "capacity_before": capacity_before,
                    "budget_after": p.budget,
                    "capacity_after": p.capacity
                }
                records.append(row_data)
                if p.budget > 0 and p.capacity > 0:
                    still_active.append(p)

            # 價格調整：支持多次漲幅
            for i in range(NUM_BEERS):
                vol = beer_volume_sold[i]
                increments = int(vol // 500)
                if increments > 0:
                    beer_prices[i] *= (1 + PRICE_INCREASE) ** increments
                elif vol == 0:
                    beer_prices[i] *= (1 - PRICE_DECREASE)

                # 檢查漲停 / 跌停
                if beer_prices[i] > daily_prices[i] * UPPER_BOUND:
                    print(f"啤酒 {i} 漲停！")
                    beer_prices[i] = daily_prices[i] * UPPER_BOUND
                if beer_prices[i] < daily_prices[i] * LOWER_BOUND:
                    print(f"啤酒 {i} 跌停！")
                    beer_prices[i] = daily_prices[i] * LOWER_BOUND

            active_persons = still_active

        # 不清空 active_persons ? 依需求而定
        # 若要客人留到隔天, 可保留. 或許每天結束後清空?
        # 這裡示範: 每天結束後就清空, 只保留價格
        active_persons = []

    df = pd.DataFrame(records)
    return df


# -------------------------
# Step 8. 主程式: 進行多次模擬 + 比較利潤 + 輸出 CSV
# -------------------------
if __name__ == "__main__":
    np.random.seed(42)

    N_SIM = 10
    all_runs = []

    total_static_profit = 0.0
    total_dynamic_profit = 0.0

    for run_id in range(1, N_SIM+1):
        # (A) 靜態定價
        df_static = simulate_static_pricing(run_id=run_id, days=DAYS)
        static_revenue = (df_static["price"] * df_static["purchased_volume"]).sum()
        total_static_profit += static_revenue

        # (B) 動態定價
        df_dynamic = simulate_dynamic_pricing(run_id=run_id, days=DAYS)
        dynamic_revenue = (df_dynamic["price"] * df_dynamic["purchased_volume"]).sum()
        total_dynamic_profit += dynamic_revenue

        all_runs.append(df_static)
        all_runs.append(df_dynamic)

    final_df = pd.concat(all_runs, ignore_index=True)
    final_df.to_csv("simulation_details.csv", index=False)

    print("模擬完成，詳細資料已儲存為 simulation_details.csv")
    print(f"在 {N_SIM} 次模擬後：")
    print(f"靜態定價總利潤(營收): {total_static_profit:.2f}")
    print(f"動態定價總利潤(營收): {total_dynamic_profit:.2f}")

    if total_dynamic_profit > total_static_profit:
        print(">> 動態定價 效益更好！")
    elif total_dynamic_profit < total_static_profit:
        print(">> 靜態定價 效益更好！")
    else:
        print(">> 靜態/動態定價 利潤相同！")
