import json


class Business:
    def __init__(self, business_id, name, stars):
        self.business_id = business_id
        self.name = name
        self.stars = stars

    def __repr__(self):
        return f"Business({self.name}, {self.stars})"


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


def load_business_data(base_path):
    businesses = []

    with open(
        base_path + '/yelp_dataset/yelp_academic_dataset_business.json',
        'r',
        encoding='utf-8'
    ) as file:
        for i, line in enumerate(file):
            data = json.loads(line)
            b = Business(
                business_id=data["business_id"],
                name=data["name"],
                stars=data["stars"]
            )
            businesses.append(b)

    return businesses

def load_user_data(base_path):
    users = []

    with open(
        base_path + '/yelp_dataset/yelp_academic_dataset_user.json',
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
        base_path + '/yelp_dataset/yelp_academic_dataset_review.json',
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