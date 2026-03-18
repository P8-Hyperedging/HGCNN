def calculate_mean_stars(reviews):
    mean_stars = 0
    for review in reviews:
        mean_stars += review.stars
    mean_stars /= len(reviews)

    return mean_stars

def calculate_review_variance(reviews, mean_stars):
    variance = 0
    for review in reviews:
        variance += (review.stars - mean_stars) ** 2
    variance /= len(reviews)

    return variance