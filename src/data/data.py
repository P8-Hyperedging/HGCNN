import psycopg2
import numpy as np
from psycopg2 import sql
from utils.config import config


class Business:
    def __init__(self, business_id: int, categories: list[int], name: str, stars: float, review_count: int, longitude: float, latitude: float):
        self.business_id = business_id
        self.categories = categories
        self.name = name
        self.stars = stars
        self.review_count = review_count
        self.longitude = longitude
        self.latitude = latitude

    def __repr__(self):
        return f"Business({self.name}, {self.stars}, {self.review_count} reviews, Location=({self.latitude}, {self.longitude}))"


class Review:
    def __init__(self, review_id, business, user, stars):
        self.review_id = review_id
        self.business = business
        self.user = user
        self.stars = stars

    def __repr__(self):
        return f"Review({self.review_id}, {self.business.business_id}, {self.user.user_id}, {self.stars})"

class User:
    def __init__(self, user_id, name):
        self.user_id = user_id
        self.name = name

    def __repr__(self):
        return f"User({self.user_id}, {self.name})"


class OpeningHours: 
    def __init__(self, business_id, hours):
        self.business_id = business_id
        self.hours = np.array(hours)

    def __repr__(self):
        return f"OpeningHours({self.business_id}, {self.hours})"

params = config()


def load_postgres_business_list_data(business_ids) -> list[Business]:
    businesses = []
    conn = psycopg2.connect(**params)
    try:
        with conn.cursor() as cur:
            cur.execute("""
                        SELECT b.business_id,
                               ARRAY_AGG(DISTINCT bc.category ORDER BY bc.category) AS categories,
                               b.name,
                               b.stars,
                               b.review_count,
                               b.longitude,
                               b.latitude
                        FROM business b
                                 LEFT JOIN business_categories bc
                                           ON b.business_id = bc.business_id
                        WHERE b.business_id = ANY (%s)
                        GROUP BY b.business_id, b.name, b.stars, b.review_count, b.longitude, b.latitude
                        """, (list(business_ids),))

            business_rows = cur.fetchall()

            cur.execute("""
                        SELECT business_category
                        FROM business_category_frequency""", (list(business_ids),))

            business_category_frequency = cur.fetchall()
            all_categories = [row[0] for row in business_category_frequency]  # ordered list

            for row in business_rows:
                business_id = row[0]
                business_cats = row[1] or []

                category_vector = [1 if cat in business_cats else 0 for cat in all_categories]

                businesses.append(Business(
                    business_id=business_id,
                    categories=category_vector,
                    name=row[2],
                    stars=row[3],
                    review_count=row[4],
                    longitude=row[5],
                    latitude=row[6]
                ))
    finally:
        conn.close()

    return businesses

def load_postgres_business_list_opening_hours(business_ids):
    businesses = []
    conn = psycopg2.connect(**params)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                WITH input_ids AS (
                    SELECT UNNEST(%s) AS business_id
                )
                SELECT
                    input_ids.business_id,
                    MAX(CASE WHEN day = 'Monday'    THEN open  END) AS monday_open,
                    MAX(CASE WHEN day = 'Monday'    THEN close END) AS monday_close,
                    MAX(CASE WHEN day = 'Tuesday'   THEN open  END) AS tuesday_open,
                    MAX(CASE WHEN day = 'Tuesday'   THEN close END) AS tuesday_close,
                    MAX(CASE WHEN day = 'Wednesday' THEN open  END) AS wednesday_open,
                    MAX(CASE WHEN day = 'Wednesday' THEN close END) AS wednesday_close,
                    MAX(CASE WHEN day = 'Thursday'  THEN open  END) AS thursday_open,
                    MAX(CASE WHEN day = 'Thursday'  THEN close END) AS thursday_close,
                    MAX(CASE WHEN day = 'Friday'    THEN open  END) AS friday_open,
                    MAX(CASE WHEN day = 'Friday'    THEN close END) AS friday_close,
                    MAX(CASE WHEN day = 'Saturday'  THEN open  END) AS saturday_open,
                    MAX(CASE WHEN day = 'Saturday'  THEN close END) AS saturday_close,
                    MAX(CASE WHEN day = 'Sunday'    THEN open  END) AS sunday_open,
                    MAX(CASE WHEN day = 'Sunday'    THEN close END) AS sunday_close
                FROM input_ids
                LEFT JOIN business_hours ON input_ids.business_id = business_hours.business_id
                GROUP BY input_ids.business_id
                ORDER BY input_ids.business_id
                """,
                (list(business_ids),)
            )
            for row in cur.fetchall():
                business_id = row[0]
                hours = row[1:15]
                businesses.append(OpeningHours(business_id, hours))
    finally:
        conn.close()
    return businesses

def load_postgres_user_data(limit=200000):
    users = []

    conn = psycopg2.connect(**params)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT user_id, name FROM users LIMIT %s", (limit,))
            for row in cur.fetchall():
                u = User(
                    user_id=row[0],
                    name=row[1]
                )
                users.append(u)
    finally:
        conn.close()

    return users

def load_postgres_review_data(db_table_name="user_business_reviews_sw_restaurants"):
    reviews = []
    with psycopg2.connect(**params) as conn:
        with conn.cursor() as cur:
            query = sql.SQL(
                """
                SELECT t.review_id, t.user_id, t.business_id, review.stars
                FROM {} AS t
                INNER JOIN review ON t.review_id = review.review_id
                """
            ).format(sql.Identifier(db_table_name))

            cur.execute(query)

            for review_id, user_id, business_id, stars in cur:
                reviews.append({
                    "review_id": review_id,
                    "user_id": user_id,
                    "business_id": business_id,
                    "stars": stars
                })

    return reviews

from collections import defaultdict

def sort_businesses_by_review_count(businesses, users, reviews):
    review_count = defaultdict(int)

    for review in reviews:
        review_count[review.business_id] += 1

    sorted_businesses = sorted(
        businesses,
        key=lambda b: review_count[b.business_id],
        reverse=True
    )

    return sorted_businesses, review_count

def group_reviews_by_user(reviews):
    user_to_businesses = defaultdict(set)

    for r in reviews:
        user_to_businesses[r['user_id']].add((r['business_id'], r['stars']))

    return user_to_businesses