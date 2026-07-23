"""Sample data for the two demo entry points. Nothing here is a real dataset --
see the README for pointing training at real data."""

MOCK_PROMPT = (
    "Q: I am an odd number. Take away one letter and I become even. "
    "What number am I?\nA: Let's think step by step."
)

# A handful of original short word problems (written for this project, not a
# real dataset) -- enough to see the training mechanism run and the loss move
# the right way. Replace with your own loader for anything beyond that.
TRAIN_TEXTS = [
    "Q: A baker made 3 trays of muffins with 8 muffins on each tray. She sold "
    "11 muffins. How many muffins are left?\nA: 3 trays of 8 muffins is 24 "
    "muffins total. Selling 11 leaves 24 minus 11, which is 13 muffins.",

    "Q: Tom has 5 dollars. He earns 2 dollars every day for 4 days. How much "
    "money does he have now?\nA: 2 dollars for 4 days is 8 dollars earned. "
    "Adding that to his original 5 dollars gives 13 dollars total.",

    "Q: A train travels 60 miles in 2 hours, then 90 miles in 3 hours. What "
    "is its average speed for the whole trip?\nA: The trip covers 60 plus 90 "
    "which is 150 miles, taking 2 plus 3 which is 5 hours. 150 divided by 5 "
    "is 30 miles per hour.",

    "Q: A garden has 6 rows of 9 tomato plants. Each plant produces 4 "
    "tomatoes. How many tomatoes grow in total?\nA: 6 rows of 9 plants is 54 "
    "plants. Each makes 4 tomatoes, so 54 times 4 is 216 tomatoes.",

    "Q: Maria reads 15 pages every night. Her book has 180 pages. How many "
    "nights will it take her to finish?\nA: 180 pages divided by 15 pages "
    "per night is 12 nights.",

    "Q: A shop had 240 apples. It sold three quarters of them on Monday and "
    "half of what remained on Tuesday. How many apples are left?\nA: Three "
    "quarters of 240 is 180 sold on Monday, leaving 60. Half of 60 is 30 "
    "sold Tuesday, leaving 30 apples.",

    "Q: Two friends split a 45 dollar bill so that one pays twice as much as "
    "the other. How much does each pay?\nA: Splitting into 3 equal parts of "
    "15 dollars, one friend pays 1 part (15 dollars) and the other pays 2 "
    "parts (30 dollars).",

    "Q: A rectangular room is 12 feet long and 9 feet wide. What is its "
    "area, and how much longer is it than it is wide?\nA: Area is length "
    "times width, so 12 times 9 is 108 square feet. It is 12 minus 9, which "
    "is 3 feet longer than it is wide.",
]
