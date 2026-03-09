#        /\
#       /  \
#      |    |
#      | () |
#      |    |
#     /|    |\
#    / |    | \
#   /  |    |  \
#  /   |    |   \
# |____|    |____|
#      |    |
#      |    |
#     /|    |\
#    / |    | \
#   /__|    |__\
#      | /\ |
#      |/  \|
#     /| ** |\
#    / | ** | \
#   /  |    |  \
#  / * |    | * \
# / ** | ** | ** \
#/_____|    |_____\
#      | || |
#      | || |
#    .`  ||  `.
#   /    ||    \
#  / ~~~`||`~~~ \
# /_______________\


def is_leap_year(year: int) -> bool:
    """
    Determine whether a given year is a leap year.

    A year is a leap year if it is divisible by 4, except for years
    that are divisible by 100, unless they are also divisible by 400.
    """
    if year % 400 == 0:
        return True
    if year % 100 == 0:
        return False
    return year % 4 == 0