import json
import psycopg2
from utils.config import config


class Business:
    def __init__(self, business_id, name, stars, review_count, longitude, latitude):
        self.business_id = business_id
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

    
params = config()


def load_postgres_business_data(limit=200000):
    businesses = []

    conn = psycopg2.connect(**params)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT business_id, name, stars, review_count, longitude, latitude FROM business LIMIT %s", (limit,))
            for row in cur.fetchall():
                b = Business(
                    business_id=row[0],
                    name=row[1],
                    stars=row[2],
                    review_count=row[3],
                    longitude=row[4],
                    latitude=row[5]
                )
                businesses.append(b)
    finally:
        conn.close()

    return businesses

def load_postgres_business_list_data(business_ids):
    businesses = []
    conn = psycopg2.connect(**params)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT business_id, name, stars, review_count, longitude, latitude
                FROM business
                WHERE business_id = ANY(%s)
                """,
                (list(business_ids),)
            )
            for row in cur.fetchall():
                businesses.append(Business(
                    business_id=row[0],
                    name=row[1],
                    stars=row[2],
                    review_count=row[3],
                    longitude=row[4],
                    latitude=row[5]
                ))
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

def load_postgres_review_data(limit=100000):
    reviews = []

    conn = psycopg2.connect(**params)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT 
                    review_id, 
                    review.user_id, 
                    review.stars, 
                    business.business_id, 
                    business.name, 
                    business.stars, 
                    business.review_count, 
                    business.longitude, 
                    business.latitude, 
                    yelp_user.user_id,
                    yelp_user.name
                FROM review 
                JOIN business ON review.business_id = business.business_id 
                JOIN yelp_user ON review.user_id = yelp_user.user_id 
                LIMIT %s
                """,
                (limit,)
            )
            for row in cur.fetchall():
                business = Business(
                    business_id=row[3],
                    name=row[4],
                    stars=row[5],
                    review_count=row[6],
                    longitude=row[7],
                    latitude=row[8]
                )
                user = User(
                    user_id=row[1],
                    name=row[9]
                )

                r = Review(
                    review_id=row[0],
                    business=business,
                    user=user,
                    stars=row[2]
                )
                reviews.append(r)
    finally:
        conn.close()

    return reviews

def load_business_data(base_path, limit=200000): # Total businesses in dataset is around 150k, so default loads all.
    businesses = []

    with open(
        base_path + '/data/yelp_dataset/yelp_academic_dataset_business.json',
        'r',
        encoding='utf-8'
    ) as file:
        for i, line in enumerate(file):
            if i >= limit:
                break
            data = json.loads(line)
            b = Business(
                business_id=data["business_id"],
                name=data["name"],
                stars=data["stars"],
                review_count=data["review_count"],
                longitude=data["longitude"],
                latitude=data["latitude"]
                # More will probably be required.
            )
            businesses.append(b)

    return businesses

def load_user_data(base_path):
    users = []

    with open(
        base_path + '/data/yelp_dataset/yelp_academic_dataset_user.json',
        'r',
        encoding='utf-8'
    ) as file:
        for i, line in enumerate(file):
            data = json.loads(line)
            user = User(
                user_id=data["user_id"],
                name=data["name"]
            )
            users.append(user)

    return users

def load_review_data(businesses, users, base_path, limit=1000):
    reviews = []

    business_lookup = {b.business_id: b for b in businesses}
    user_lookup = {u.user_id: u for u in users}

    with open(
        base_path + '/data/yelp_dataset/yelp_academic_dataset_review.json',
        'r',
        encoding='utf-8'
    ) as file:
        for i, line in enumerate(file):
            if i >= limit:
                break

            data = json.loads(line)

            r = Review(
                review_id=data["review_id"],
                business_id=data["business_id"],
                user_id=data["user_id"],
                stars=data["stars"]
            )

            reviews.append(r)

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
    user_to_businesses = defaultdict(list)

    for r in reviews:
        user_to_businesses[r.user.user_id].append(r.business.business_id)

    return user_to_businesses
