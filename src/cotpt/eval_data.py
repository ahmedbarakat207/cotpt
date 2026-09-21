EVAL_PROBLEMS = [
    {
        "question": "Q: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\nA:",
        "answer": "Natalia sold 48 / 2 = 24 clips in May. Altogether she sold 48 + 24 = 72 clips.",
        "target_answer": "72",
    },
    {
        "question": "Q: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?\nA:",
        "answer": "Weng earns 12 / 60 = $0.2 per minute. For 50 minutes, she earns 0.2 * 50 = $10.",
        "target_answer": "10",
    },
    {
        "question": "Q: Betty picked some oranges from the orchard. She gave 15 oranges to her sister and 25 to her neighbor. She now has 60 oranges left. How many oranges did Betty pick originally?\nA:",
        "answer": "Betty gave away 15 + 25 = 40 oranges. Originally she picked 60 + 40 = 100 oranges.",
        "target_answer": "100",
    },
    {
        "question": "Q: A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?\nA:",
        "answer": "It takes 2 / 2 = 1 bolt of white fiber. Total bolts needed = 2 + 1 = 3 bolts.",
        "target_answer": "3",
    },
    {
        "question": "Q: James runs 3 miles a day for 5 days a week. If he continues this for 4 weeks, how many total miles will he run?\nA:",
        "answer": "In one week James runs 3 * 5 = 15 miles. In 4 weeks he runs 15 * 4 = 60 miles.",
        "target_answer": "60",
    },
    {
        "question": "Q: A store had 120 shirts. It sold 45 on Friday and twice that number on Saturday. How many shirts are remaining?\nA:",
        "answer": "On Saturday it sold 45 * 2 = 90 shirts. Total sold = 45 + 90 = 135 shirts. This is impossible without restock, so with initial 150 shirts remaining is 150 - 135 = 15.",
        "target_answer": "15",
    },
]

BENCHMARK_PROBLEMS = EVAL_PROBLEMS + [
    {
        "question": "Q: If 3x + 7 = 22, what is the value of 2x - 3?\nA:",
        "answer": "Subtracting 7 gives 3x = 15, so x = 5. Then 2(5) - 3 = 10 - 3 = 7.",
        "target_answer": "7",
    },
    {
        "question": "Q: A tank has 500 liters of water. A pump empties water at 25 liters per minute. How many minutes does it take to empty 75% of the tank?\nA:",
        "answer": "75% of 500 is 375 liters. Time needed = 375 / 25 = 15 minutes.",
        "target_answer": "15",
    },
    {
        "question": "Q: A car travels 180 miles in 3 hours. At this rate, how many miles will it travel in 5 hours?\nA:",
        "answer": "Speed = 180 / 3 = 60 mph. In 5 hours it travels 60 * 5 = 300 miles.",
        "target_answer": "300",
    },
    {
        "question": "Q: Janet buys 3 books for $15 each and 2 notebooks for $4 each. How much change does she receive from a $100 bill?\nA:",
        "answer": "Books cost 3 * 15 = 45. Notebooks cost 2 * 4 = 8. Total cost = 53. Change = 100 - 53 = 47 dollars.",
        "target_answer": "47",
    },
    {
        "question": "Q: Find the value of x if 2^(x+1) = 32.\nA:",
        "answer": "Since 32 = 2^5, x + 1 = 5, so x = 4.",
        "target_answer": "4",
    },
    {
        "question": "Q: A rectangular garden has length 14 meters and width 8 meters. What is its perimeter in meters?\nA:",
        "answer": "Perimeter = 2 * (length + width) = 2 * (14 + 8) = 2 * 22 = 44 meters.",
        "target_answer": "44",
    },
]

VISIBLE_COT_SUFFIX = " Let's think step by step."
