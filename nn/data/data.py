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
    def __init__(self, review_id, business_id, business, user_id, user, stars):
        self.review_id = review_id
        self.business_id = business_id
        self.business = business
        self.user_id = user_id
        self.user = user
        self.stars = stars

    def __repr__(self):
        return f"Review({self.review_id}, {self.stars}, Business={self.business.name})"

class User:
    def __init__(self, user_id, name):
        self.user_id = user_id
        self.name = name

    def __repr__(self):
        return f"User({self.user_id}, {self.name})"

    
params = config()

conn = psycopg2.connect(**params)

def load_postgres_business_data(limit=200000):
    businesses = []

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

def load_postgres_user_data(limit=200000):
    users = []

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
                r = Review(
                    review_id=row[0],
                    business_id=row[3],
                    business=Business(
                        business_id=row[3],
                        name=row[4],
                        stars=row[5],
                        review_count=row[6],
                        longitude=row[7],
                        latitude=row[8]
                    ),
                    user_id=row[1],
                    user=User(
                        user_id=row[1],
                        name=row[9]
                    ),
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

            business = business_lookup.get(data["business_id"])
            user = user_lookup.get(data["user_id"])

            r = Review(
                review_id=data["review_id"],
                business_id=data["business_id"],
                business=business,
                user_id=data["user_id"],
                user=user,
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