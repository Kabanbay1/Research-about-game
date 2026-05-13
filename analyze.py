import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# ╔══════════════════════════════════════════════╗
# ║              НАСТРОЙКИ                       ║
# ╚══════════════════════════════════════════════╝

INPUT_CSV   = "steam_reviews.csv"
TOP_N_TAGS  = 20   # сколько топ-тегов включить в регрессию

# ══════════════════════════════════════════════


# ── Загрузка данных ────────────────────────────────────────────────────────

print("Загружаем данные...")
cols = [
    "appid", "game_name", "game_positive", "game_negative", "game_score",
    "genres", "tags", "recommendationid", "steamid",
    "playtime_forever_min", "num_games_owned",
    "language", "voted_up", "votes_up", "votes_funny", "weighted_vote_score",
    "timestamp_created", "review",
]

df = pd.read_csv(
    INPUT_CSV,
    sep=",",                 # ВАЖНО: запятая!
    quotechar='"',           # учитывать кавычки
    engine="python",
    encoding="utf-8",
    on_bad_lines="skip"
)


# Одна строка на игру (для анализа тегов и регрессии)
games = df.drop_duplicates("appid").copy()
games["tags_list"] = games["tags"].fillna("").str.split("|")
games["tags_list"] = games["tags_list"].apply(
    lambda x: [t for t in x if t != ""]  # убрать пустые строки
)
print(f"Игр: {len(games)}, отзывов: {len(df)}\n")


# ══════════════════════════════════════════════════════════════════════════
# ШАГ 1 — Explode: статистика по каждому тегу
# ══════════════════════════════════════════════════════════════════════════

print("=" * 55)
print("ШАГ 1: Анализ тегов через explode")
print("=" * 55)

tags_exploded = games[["appid", "game_score", "tags_list"]].copy()
tags_exploded = tags_exploded.explode("tags_list").rename(columns={"tags_list": "tag"})
tags_exploded = tags_exploded[tags_exploded["tag"].notna() & (tags_exploded["tag"] != "")]

# Статистика по каждому тегу
tag_stats = (
    tags_exploded
    .groupby("tag")
    .agg(
        count=("appid", "count"),           # сколько игр с этим тегом
        mean_score=("game_score", "mean"),  # средний рейтинг
        median_score=("game_score", "median"),
    )
    .sort_values("count", ascending=False)
)

print("\nТоп-20 тегов по количеству игр:")
print(tag_stats.head(20).to_string())

print("\nТоп-20 тегов по среднему рейтингу (минимум 10 игр):")
print(
    tag_stats[tag_stats["count"] >= 10]
    .sort_values("mean_score", ascending=False)
    .head(20)
    .to_string()
)

# Отбираем топ-N тегов по частоте для следующего шага
top_tags = tag_stats.head(TOP_N_TAGS).index.tolist()
print(f"\nТоп-{TOP_N_TAGS} тегов для регрессии: {top_tags}")

# ══════════════════════════════════════════════════════════════════════════
# ДОПОЛНИТЕЛЬНО — Анализ тегов успешных игр (через lift)
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 55)
print("АНАЛИЗ: Какие теги чаще встречаются у успешных игр")
print("=" * 55)

# 1. Определяем успешные игры (например, топ 25%)
threshold = games["game_score"].quantile(0.75)
successful_games = games[games["game_score"] >= threshold]

print(f"Порог успешности (75-й перцентиль): {threshold:.3f}")
print(f"Успешных игр: {len(successful_games)} из {len(games)}")

# 2. Частота тегов среди успешных игр
success_tags = successful_games.explode("tags_list")
success_tags = success_tags[success_tags["tags_list"].notna() & (success_tags["tags_list"] != "")]
success_freq = success_tags["tags_list"].value_counts(normalize=True)

# 3. Частота тегов среди всех игр
all_tags = games.explode("tags_list")
all_tags = all_tags[all_tags["tags_list"].notna() & (all_tags["tags_list"] != "")]
all_freq = all_tags["tags_list"].value_counts(normalize=True)

# 4. Убираем редкие теги (очень важно)
min_count = 10
tag_counts = all_tags["tags_list"].value_counts()
valid_tags = tag_counts[tag_counts >= min_count].index

# 5. Lift (насколько чаще тег встречается у успешных игр)
lift = (success_freq / all_freq)
lift = lift[lift.index.isin(valid_tags)].dropna().sort_values(ascending=False)

print("\nТоп-20 тегов, наиболее характерных для успешных игр (lift):")
print(lift.head(20).to_string())

print("\nАнти-теги (реже встречаются у успешных игр):")
print(lift.tail(20).to_string())


# ══════════════════════════════════════════════════════════════════════════
# ШАГ 2 — One-hot encoding топ тегов + регрессия
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 55)
print("ШАГ 2: One-hot encoding и регрессия")
print("=" * 55)

# One-hot только для топ-N тегов
mlb = MultiLabelBinarizer(classes=top_tags)
tags_encoded = pd.DataFrame(
    mlb.fit_transform(games["tags_list"]),
    columns=mlb.classes_,
    index=games.index,
)

# Итоговый датафрейм для регрессии
model_df = pd.concat([
    games[["appid", "game_score", "game_positive", "game_negative"]].reset_index(drop=True),
    tags_encoded.reset_index(drop=True),
], axis=1).dropna(subset=["game_score"])

# Признаки и целевая переменная
feature_cols = top_tags  # только теги; добавьте другие числовые признаки если нужно
X = model_df[feature_cols]
y = model_df["game_score"]

# Разбивка на train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Линейная регрессия
reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

r2 = r2_score(y_test, y_pred)
print(f"\nR² модели на тестовой выборке: {r2:.3f}")
print("(1.0 = идеально, 0.0 = модель не лучше среднего)")

# Коэффициенты — какие теги влияют на score больше всего
coef_df = (
    pd.DataFrame({"tag": feature_cols, "coefficient": reg.coef_})
    .sort_values("coefficient", ascending=False)
)

print("\nВлияние тегов на game_score (регрессионные коэффициенты):")
print(coef_df.to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════
# ШАГ 3 — Визуализация
# ══════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Анализ факторов успеха инди игр", fontsize=14)

# График 1: средний score по топ тегам
top_by_score = (
    tag_stats[tag_stats["count"] >= 10]
    .sort_values("mean_score", ascending=False)
    .head(20)
)
axes[0].barh(top_by_score.index, top_by_score["mean_score"], color="steelblue")
axes[0].set_xlabel("Средний Wilson Score")
axes[0].set_title("Топ-20 тегов по среднему рейтингу\n(минимум 10 игр)")
axes[0].invert_yaxis()

# График 2: коэффициенты регрессии
colors = ["tomato" if c < 0 else "steelblue" for c in coef_df["coefficient"]]
axes[1].barh(coef_df["tag"], coef_df["coefficient"], color=colors)
axes[1].axvline(0, color="black", linewidth=0.8)
axes[1].set_xlabel("Коэффициент регрессии")
axes[1].set_title(f"Влияние тегов на game_score\n(R²={r2:.3f})")
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig("tag_analysis.png", dpi=150, bbox_inches="tight")
print("\n✓ График сохранён: tag_analysis.png")
plt.show()
