import json


class Business:
    def __init__(self, business_id, name, stars):
        self.business_id = business_id
        self.name = name
        self.stars = stars

    def __repr__(self):
        return f"Business({self.name}, {self.stars})"


class Review:
    def __init__(self, review_id, business_id, user_id, stars):
        self.review_id = review_id
        self.business_id = business_id
        self.user_id = user_id
        self.stars = stars

    def __repr__(self):
        return f"Review({self.review_id}, {self.stars})"

class User:
    def __init__(self, user_id, name):
        self.user_id = user_id
        self.name = name

    def __repr__(self):
        return f"User({self.user_id}, {self.name})"


def load_business_data(base_path):
    businesses = []

    with open(base_path + '/yelp_dataset/yelp_academic_dataset_business.json', 'r') as file:
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

    with open(base_path + '/yelp_dataset/yelp_academic_dataset_user.json', 'r') as file:
        for i, line in enumerate(file):

            data = json.loads(line)

            user = User(
                user_id=data["user_id"],
                name=data["name"]
            )

            users.append(user)

    return users


def load_review_data(base_path, limit=1000):
    reviews = []

    with open(base_path + '/yelp_dataset/yelp_academic_dataset_review.json', 'r') as file:
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
        user_to_businesses[r.user_id].append(r.business_id)

    return user_to_businesses
