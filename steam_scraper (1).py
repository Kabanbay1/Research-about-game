import requests
import csv
import time
import json
import math
import random
from pathlib import Path

# ╔══════════════════════════════════════════════╗
# ║              НАСТРОЙКИ                       ║
# ╚══════════════════════════════════════════════╝

TARGET_GAMES        = 1500        # сколько игр нужно найти
REVIEWS_PER_GAME    = 100         # отзывов на каждую игру
REVIEW_LANGUAGE     = "english"   # "russian" / "english" / "all"
OUTPUT_CSV          = "steam_reviews.csv"
CHECKPOINT_FILE     = "checkpoint.json"
REQUEST_DELAY       = 1.0         # секунды между запросами
MIN_REVIEWS         = 50          # минимум отзывов у игры (фильтр мусора)

# Фильтр по жанрам — игра должна иметь хотя бы один из этих жанров
# Оставьте пустым set() если жанр не важен
REQUIRED_GENRES     = {"Indie"}

# ══════════════════════════════════════════════


def wilson_score(positive: int, total: int) -> float:
    """
    Нижняя граница доверительного интервала Вильсона.
    Используется как метрика успеха в CSV — не для сортировки.
    """
    if total == 0:
        return 0.0
    z = 1.96
    p = positive / total
    return (
        (p + z*z / (2*total) - z * math.sqrt((p*(1-p) + z*z/(4*total)) / total))
        / (1 + z*z / total)
    )


def fetch_candidates(genre: str) -> list[dict]:
    """
    Загружает игры по жанру из Steam Spy и перемешивает случайно.
    Случайная выборка даёт репрезентативный датасет для исследования —
    game_score используется как переменная внутри датасета, а не как критерий отбора.
    """
    print(f"Загружаем {genre} игры из Steam Spy...")
    candidates = []

    try:
        resp = requests.get(
            "https://steamspy.com/api.php",
            params={"request": "genre", "genre": genre},
            timeout=30,
        )
        batch = resp.json()

        for appid, info in batch.items():
            positive = info.get("positive", 0)
            negative = info.get("negative", 0)
            total    = positive + negative

            if total < MIN_REVIEWS:
                continue  # фильтруем мусор с почти нулём отзывов

            candidates.append({
                "appid":    int(appid),
                "name":     info.get("name", ""),
                "positive": positive,
                "negative": negative,
                "total":    total,
                "score":    wilson_score(positive, total),
            })

        print(f"Загружено {len(candidates)} игр (после фильтра минимум {MIN_REVIEWS} отзывов)")

    except Exception as e:
        print(f"Ошибка при загрузке кандидатов: {e}")

    # Случайное перемешивание — репрезентативная выборка
    random.shuffle(candidates)
    return candidates


def fetch_genres(appid: int) -> set[str] | None:
    """
    Получает жанры игры через Steam Store API.
    Возвращает None если это не игра (DLC, саундтрек и т.д.) или ошибка.
    """
    try:
        resp = requests.get(
            "https://store.steampowered.com/api/appdetails",
            params={"appids": appid, "l": "english"},
            timeout=30,
        )
        data = resp.json().get(str(appid), {})
        if not data.get("success"):
            return None
        d = data["data"]
        if d.get("type") != "game":
            return None
        return {g["description"] for g in d.get("genres", [])}
    except Exception as e:
        print(f"    Ошибка fetch_genres: {e}")
        return None


def fetch_tags(appid: int) -> set[str]:
    """Возвращает set пользовательских тегов из Steam Spy."""
    try:
        resp = requests.get(
            "https://steamspy.com/api.php",
            params={"request": "appdetails", "appid": appid},
            timeout=15,
        )
        return set(resp.json().get("tags", {}).keys())
    except Exception:
        return set()


def passes_filters(genres: set[str]) -> bool:
    """Проверяет жанровый фильтр. Теги не фильтруем — они идут в CSV как данные."""
    if REQUIRED_GENRES and not REQUIRED_GENRES.intersection(genres):
        return False
    return True


def fetch_reviews(appid: int, max_reviews: int = REVIEWS_PER_GAME) -> list[dict]:
    url = f"https://store.steampowered.com/appreviews/{appid}"
    reviews = []
    cursor = "*"

    while len(reviews) < max_reviews:
        need = min(100, max_reviews - len(reviews))
        try:
            resp = requests.get(url, params={
                "json": 1,
                "language": REVIEW_LANGUAGE,
                "cursor": cursor,
                "num_per_page": need,
                "review_type": "all",
                "purchase_type": "all",
                "filter": "recent",
            }, timeout=15)
            data = resp.json()
        except Exception as e:
            print(f"    Ошибка при получении отзывов: {e}")
            break

        if data.get("success") != 1:
            break

        batch = data.get("reviews", [])
        if not batch:
            break

        reviews.extend(batch)
        cursor = data.get("cursor", "")
        if not cursor or len(batch) < need:
            break

        time.sleep(REQUEST_DELAY)

    return reviews


CSV_FIELDS = [
    "appid", "game_name", "game_positive", "game_negative", "game_score",
    "genres", "tags",
    "recommendationid", "steamid",
    "playtime_forever_min", "num_games_owned",
    "language", "voted_up",
    "votes_up", "votes_funny", "weighted_vote_score",
    "timestamp_created", "review",
]

def flatten(review: dict, candidate: dict, genres: set, tags: set) -> dict:
    a = review.get("author", {})
    return {
        "appid":                candidate["appid"],
        "game_name":            candidate["name"],
        "game_positive":        candidate["positive"],
        "game_negative":        candidate["negative"],
        "game_score":           round(candidate["score"], 4),
        "genres":               "|".join(sorted(genres)),
        "tags":                 "|".join(sorted(tags)),
        "recommendationid":     review.get("recommendationid"),
        "steamid":              a.get("steamid"),
        "playtime_forever_min": a.get("playtime_forever"),
        "num_games_owned":      a.get("num_games_owned"),
        "language":             review.get("language"),
        "voted_up":             review.get("voted_up"),
        "votes_up":             review.get("votes_up"),
        "votes_funny":          review.get("votes_funny"),
        "weighted_vote_score":  review.get("weighted_vote_score"),
        "timestamp_created":    review.get("timestamp_created"),
        "review":               review.get("review", "").replace("\n", " ").replace("\r", ""),
    }


def load_checkpoint() -> dict:
    if Path(CHECKPOINT_FILE).exists():
        with open(CHECKPOINT_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {"done_appids": [], "games_collected": 0}

def save_checkpoint(done_appids: list, games_collected: int):
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump({"done_appids": done_appids, "games_collected": games_collected}, f)


def main():
    # 1. Загрузить кандидатов (случайный порядок)
    genre = next(iter(REQUIRED_GENRES)) if REQUIRED_GENRES else "Indie"
    candidates = fetch_candidates(genre)

    # 2. Загрузить checkpoint
    cp = load_checkpoint()
    done_set        = set(cp["done_appids"])
    games_collected = cp["games_collected"]

    remaining = [c for c in candidates if c["appid"] not in done_set]
    print(f"Прогресс: {games_collected}/{TARGET_GAMES} игр уже собрано")
    print(f"Осталось проверить кандидатов: {len(remaining)}\n")

    # 3. Открыть CSV
    file_exists = Path(OUTPUT_CSV).exists() and games_collected > 0
    with open(OUTPUT_CSV, "a" if file_exists else "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not file_exists:
            writer.writeheader()

        for candidate in remaining:
            if games_collected >= TARGET_GAMES:
                break

            appid = candidate["appid"]
            print(f"Проверяю appid={appid} ({candidate['name']})...")

            # Проверка жанра через Steam Store API
            genres = fetch_genres(appid)
            time.sleep(REQUEST_DELAY)
            if genres is None:
                done_set.add(appid)
                continue

            if not passes_filters(genres):
                done_set.add(appid)
                continue

            # Теги — не фильтруем, просто сохраняем в CSV
            tags = fetch_tags(appid)
            time.sleep(REQUEST_DELAY)

            # Игра прошла — собираем отзывы
            games_collected += 1
            print(
                f"[{games_collected}/{TARGET_GAMES}] {candidate['name']} (appid={appid}) "
                f"| score={candidate['score']:.3f} | отзывов всего: {candidate['total']}"
            )

            reviews = fetch_reviews(appid)
            print(f"  → собрано отзывов: {len(reviews)}")

            for r in reviews:
                writer.writerow(flatten(r, candidate, genres, tags))
            f.flush()

            done_set.add(appid)
            save_checkpoint(list(done_set), games_collected)
            time.sleep(REQUEST_DELAY)

    print(f"\n✓ Готово! Собрано игр: {games_collected}")
    print(f"✓ CSV сохранён: {OUTPUT_CSV}")

    if games_collected >= TARGET_GAMES:
        Path(CHECKPOINT_FILE).unlink(missing_ok=True)
        print("✓ Checkpoint удалён")


if __name__ == "__main__":
    main()
