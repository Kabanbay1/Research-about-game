import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ╔══════════════════════════════════════════════╗
# ║              НАСТРОЙКИ                       ║
# ╚══════════════════════════════════════════════╝

INPUT_CSV         = "steam_reviews.csv"


THEMES = {
    "Баги и технические проблемы": [
        "bug", "bugs", "crash", "crashes", "broken", "fix", "glitch",
        "glitches", "error", "lag", "freeze", "freezes", "unstable"
    ],
    "Сюжет и нарратив": [
        "story", "narrative", "plot", "characters", "character", "writing",
        "dialogue", "lore", "ending", "immersive", "world"
    ],
    "Геймплей и механики": [
        "gameplay", "mechanics", "controls", "combat",
        "satisfying", "balanced", "progression", "difficulty", "challenge"
    ],
    "Цена и ценность": [
        "price", "worth", "cheap", "expensive", "refund", "value",
        "money", "cost", "overpriced", "sale", "free"
    ],
    "Графика и арт": [
        "art", "graphics", "visual", "visuals", "beautiful", "style",
        "aesthetic", "animation", "animations", "design", "pixel"
    ],
    "Музыка и звук": [
        "music", "soundtrack", "sound", "audio", "ost", "ambient",
        "effects", "voice", "acting"
    ],
    "Контент и продолжительность": [
        "content", "hours", "short", "long", "replayability", "replay",
        "ending", "complete", "finished", "length", "playtime"
    ],
    "Поддержка разработчика": [
        "developer", "dev", "devs", "support", "update", "updates",
        "patch", "community", "response", "abandoned", "active"
    ],
}

# ══════════════════════════════════════════════


# ── Загрузка ───────────────────────────────────────────────────────────────

print("Загружаем данные...")
df = pd.read_csv(
    INPUT_CSV,
    encoding="utf-8",
    encoding_errors="replace",
    on_bad_lines="skip",
    engine="python",
)

df = df.dropna(subset=["review", "game_score"])
df["review"]     = df["review"].astype(str)
df["game_score"] = pd.to_numeric(df["game_score"], errors="coerce")
df = df.dropna(subset=["game_score"])

games      = df.drop_duplicates("appid")[["appid", "game_name", "game_score"]]
median_score = games["game_score"].median()
print(f"Медиана game_score: {median_score:.3f}")

successful = set(games[games["game_score"] >= median_score]["appid"])
failed     = set(games[games["game_score"] <  median_score]["appid"])

df_success = df[df["appid"].isin(successful)]
df_fail    = df[df["appid"].isin(failed)]

print(f"Загружено: {len(df)} отзывов, {df['appid'].nunique()} игр")
print(f"Успешные:   {len(successful)} игр ({len(df_success)} отзывов)")
print(f"Неуспешные: {len(failed)} игр ({len(df_fail)} отзывов)\n")


# ── Sentiment analysis ─────────────────────────────────────────────────────

print("Считаем sentiment...")
analyzer = SentimentIntensityAnalyzer()
df["sentiment"] = df["review"].apply(
    lambda x: analyzer.polarity_scores(x)["compound"]
)

game_sentiment = (
    df.groupby("appid")["sentiment"]
    .mean()
    .reset_index()
    .rename(columns={"sentiment": "avg_sentiment"})
    .merge(games, on="appid")
)

corr = game_sentiment["avg_sentiment"].corr(game_sentiment["game_score"])
print(f"Корреляция avg_sentiment с game_score: {corr:.3f}\n")


# ── Статистические функции ─────────────────────────────────────────────────

def mention_vector(df, keywords):
    """Бинарный вектор: 1 если отзыв упоминает хотя бы одно слово из темы."""
    return df["review"].apply(
        lambda x: int(any(kw in str(x).lower() for kw in keywords))
    )

def chi_square_test(vec_success, vec_fail):
    """
    Chi-square тест на независимость.
    H0: частота упоминания одинакова в обеих группах.
    p < 0.05 = отвергаем H0, разница статистически значима.
    """
    contingency = [
        [vec_success.sum(), len(vec_success) - vec_success.sum()],
        [vec_fail.sum(),    len(vec_fail)    - vec_fail.sum()],
    ]
    chi2, p, dof, _ = stats.chi2_contingency(contingency)
    return round(chi2, 3), round(p, 4)

def cohens_d(vec_success, vec_fail):
    """
    Размер эффекта Cohen's d.
    Интерпретация: 0.2 = малый, 0.5 = средний, 0.8+ = большой.
    Показывает насколько БОЛЬШАЯ разница, а не только значима ли она.
    """
    diff       = vec_success.mean() - vec_fail.mean()
    pooled_std = np.sqrt((vec_success.std()**2 + vec_fail.std()**2) / 2)
    if pooled_std == 0:
        return 0.0
    return round(abs(diff / pooled_std), 3)

def effect_label(d):
    if d >= 0.8:  return "большой"
    if d >= 0.5:  return "средний"
    if d >= 0.2:  return "малый"
    return "пренебрежимый"


# ══════════════════════════════════════════════════════════════════════════
# Тематический анализ + статистика
# ══════════════════════════════════════════════════════════════════════════

print("=" * 75)
print("ТЕМАТИЧЕСКИЙ АНАЛИЗ С СТАТИСТИЧЕСКИМИ ТЕСТАМИ")
print("=" * 75)

results = []
for theme, keywords in THEMES.items():
    vs = mention_vector(df_success, keywords)
    vf = mention_vector(df_fail,    keywords)

    freq_s = round(vs.mean() * 100, 1)
    freq_f = round(vf.mean() * 100, 1)
    diff   = round(freq_s - freq_f, 1)

    chi2, p  = chi_square_test(vs, vf)
    d        = cohens_d(vs, vf)
    sig      = "✓" if p < 0.05 else "✗"
    direction = "чаще в успешных" if diff > 0 else "чаще в неуспешных"

    results.append({
        "theme":     theme,
        "success_%": freq_s,
        "fail_%":    freq_f,
        "diff_%":    diff,
        "chi2":      chi2,
        "p_value":   p,
        "cohens_d":  d,
        "effect":    effect_label(d),
        "sig":       sig,
        "direction": direction,
    })

results_df = pd.DataFrame(results).sort_values("diff_%", ascending=False)

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import statsmodels.api as sm

# ══════════════════════════════════════════════════════════════════════════
# РЕГРЕССИОННЫЙ АНАЛИЗ
# ══════════════════════════════════════════════════════════════════════════

print("=" * 75)
print("РЕГРЕССИОННЫЙ АНАЛИЗ")
print("=" * 75)

# Для каждой игры считаем долю отзывов упоминающих каждую тему
print("\nПодготовка матрицы признаков...")
game_theme_matrix = {}

for theme, keywords in THEMES.items():
    game_theme_matrix[theme] = (
        df.groupby("appid")["review"]
        .apply(lambda reviews: reviews.apply(
            lambda x: int(any(kw in str(x).lower() for kw in keywords))
        ).mean())
    )

# Собираем датафрейм
reg_df = pd.DataFrame(game_theme_matrix)
reg_df["game_score"]    = games.set_index("appid")["game_score"]
reg_df["avg_sentiment"] = df.groupby("appid")["sentiment"].mean()
reg_df = reg_df.dropna()

print(f"Игр в регрессии: {len(reg_df)}")

# Признаки и целевая переменная
feature_cols = list(THEMES.keys()) + ["avg_sentiment"]
X = reg_df[feature_cols]
y = reg_df["game_score"]

# ── Statsmodels — для p-value коэффициентов ────────────────────────────
X_const = sm.add_constant(X)
ols = sm.OLS(y, X_const).fit()

print("\nРезультаты OLS регрессии:")
print(f"R² = {ols.rsquared:.3f}")
print(f"Adjusted R² = {ols.rsquared_adj:.3f}")
print(f"F-statistic p-value = {ols.f_pvalue:.2e}")

print(f"\n{'Признак':<35} {'Коэф.':>8} {'Std Err':>8} {'t':>8} {'p-value':>10} {'Знач':>6}")
print("-" * 80)

for var in feature_cols:
    coef  = ols.params[var]
    se    = ols.bse[var]
    t     = ols.tvalues[var]
    p     = ols.pvalues[var]
    sig   = "✓" if p < 0.05 else "✗"
    p_str = f"{p:.2e}" if p < 0.001 else f"{p:.4f}"
    print(f"{var:<35} {coef:>8.4f} {se:>8.4f} {t:>8.3f} {p_str:>10} {sig:>6}")

# ── Cross-validation R² ────────────────────────────────────────────────
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

lr = LinearRegression()
cv_scores = cross_val_score(lr, X, y, cv=5, scoring="r2")
print(f"\nCross-validation R² (5-fold): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
print("(показывает насколько модель обобщается на новые данные)")

# ── Визуализация коэффициентов ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

coefs = pd.Series(ols.params[feature_cols], index=feature_cols)
colors = ["steelblue" if c > 0 else "tomato" for c in coefs]
coefs.sort_values().plot(kind="barh", ax=ax, color=colors[::-1])

ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Коэффициент регрессии")
ax.set_title(f"Вклад каждого фактора в game_score\n(R²={ols.rsquared:.3f}, Adj.R²={ols.rsquared_adj:.3f})")

# Отмечаем значимые коэффициенты звёздочкой
for i, var in enumerate(coefs.sort_values().index):
    p = ols.pvalues[var]
    if p < 0.05:
        ax.text(coefs[var], i, " *", va="center", fontsize=12)

plt.tight_layout()
plt.savefig("regression_analysis.png", dpi=150, bbox_inches="tight")
print("\n✓ График сохранён: regression_analysis.png")
plt.show()

# Вывод таблицы
header = f"\n{'Тема':<35} {'Усп%':>6} {'Неусп%':>7} {'Разн%':>7} {'chi2':>8} {'p-value':>9} {'d':>6} {'Эффект':>14} {'Знач':>5}"
print(header)
print("-" * 100)
for _, r in results_df.iterrows():
    sign = "+" if r["diff_%"] > 0 else ""
    print(
        f"{r['theme']:<35} {r['success_%']:>5}% {r['fail_%']:>6}% "
        f"{sign}{r['diff_%']:>5}% {r['chi2']:>8} {r['p_value']:>9} "
        f"{r['cohens_d']:>6} {r['effect']:>14} {r['sig']:>5}"
    )

print("\n✓ = p < 0.05 (статистически значимо)")
print("✗ = p ≥ 0.05 (возможно случайная разница)")


# ── Детальный анализ по словам ─────────────────────────────────────────────

print("\n" + "=" * 75)
print("ДЕТАЛЬНЫЙ АНАЛИЗ ПО СЛОВАМ (топ-5 по размеру эффекта)")
print("=" * 75)

for theme, keywords in THEMES.items():
    print(f"\n{theme}:")
    word_results = []
    for kw in keywords:
        vs_kw = df_success["review"].str.lower().str.contains(kw, regex=False).astype(int)
        vf_kw = df_fail["review"].str.lower().str.contains(kw, regex=False).astype(int)

        fs   = round(vs_kw.mean() * 100, 1)
        ff   = round(vf_kw.mean() * 100, 1)
        diff = round(fs - ff, 1)
        d    = cohens_d(vs_kw, vf_kw)
        _, p = chi_square_test(vs_kw, vf_kw)
        sig  = "✓" if p < 0.05 else "✗"

        word_results.append((kw, fs, ff, diff, d, p, sig))

    word_results.sort(key=lambda x: x[4], reverse=True)
    print(f"  {'Слово':<20} {'Усп%':>6} {'Неусп%':>8} {'Разн%':>7} {'d':>6} {'p':>8} {'Знач':>5}")
    print("  " + "-" * 65)
    for kw, fs, ff, diff, d, p, sig in word_results[:5]:
        sign = "+" if diff > 0 else ""
        print(f"  {kw:<20} {fs:>5}% {ff:>7}% {sign}{diff:>5}% {d:>6} {p:>8} {sig:>5}")


# ── Итоговые выводы ────────────────────────────────────────────────────────

print("\n" + "=" * 75)
print("ИТОГОВЫЕ ВЫВОДЫ")
print("=" * 75)

sig_results = results_df[results_df["p_value"] < 0.05].copy()
positive    = sig_results[sig_results["diff_%"] > 0]
negative    = sig_results[sig_results["diff_%"] < 0]

print("\nФакторы УСПЕХА (значимо чаще в успешных играх):")
for _, r in positive.iterrows():
    print(f"  + {r['theme']}: +{r['diff_%']}% (d={r['cohens_d']}, {r['effect']} эффект)")

print("\nФакторы ПРОВАЛА (значимо чаще в неуспешных играх):")
for _, r in negative.iterrows():
    print(f"  - {r['theme']}: {r['diff_%']}% (d={r['cohens_d']}, {r['effect']} эффект)")

insignificant = results_df[results_df["p_value"] >= 0.05]
if len(insignificant) > 0:
    print("\nНезначимые темы (p ≥ 0.05, возможно случайная разница):")
    for _, r in insignificant.iterrows():
        print(f"  ~ {r['theme']}: {r['diff_%']}% (p={r['p_value']})")


# ── Визуализация ───────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle("Тематический анализ отзывов инди игр (с статистическими тестами)", fontsize=13)

# График 1: частота тем по группам
x = np.arange(len(results_df))
w = 0.35
bars_s = axes[0].barh(x + w/2, results_df["success_%"], w, label="Успешные", color="steelblue")
bars_f = axes[0].barh(x - w/2, results_df["fail_%"],    w, label="Неуспешные", color="tomato")
axes[0].set_yticks(x)
axes[0].set_yticklabels(results_df["theme"], fontsize=9)
axes[0].set_xlabel("% отзывов упоминающих тему")
axes[0].set_title("Частота тем в отзывах")
axes[0].legend()
axes[0].invert_yaxis()

# Звёздочки для значимых различий
for i, (_, r) in enumerate(results_df.iterrows()):
    if r["p_value"] < 0.05:
        max_val = max(r["success_%"], r["fail_%"])
        axes[0].text(max_val + 0.5, i, "*", fontsize=12, color="black", va="center")

# График 2: разница с размером эффекта
colors = ["steelblue" if d > 0 else "tomato" for d in results_df["diff_%"]]
bars = axes[1].barh(results_df["theme"], results_df["diff_%"], color=colors, alpha=0.8)
axes[1].axvline(0, color="black", linewidth=0.8)
axes[1].set_xlabel("Разница в % (успешные − неуспешные)")
axes[1].set_title("Разница с размером эффекта (Cohen's d)\n* = p < 0.05")
axes[1].invert_yaxis()

# Подписи Cohen's d
for i, (_, r) in enumerate(results_df.iterrows()):
    x_pos = r["diff_%"] + (0.3 if r["diff_%"] > 0 else -0.3)
    label = f"d={r['cohens_d']}"
    if r["p_value"] < 0.05:
        label += "*"
    axes[1].text(x_pos, i, label, va="center", fontsize=8,
                 ha="left" if r["diff_%"] > 0 else "right")

plt.tight_layout()
plt.savefig("theme_analysis_stats.png", dpi=150, bbox_inches="tight")
print("\n✓ График сохранён: theme_analysis_stats.png")
plt.show()
