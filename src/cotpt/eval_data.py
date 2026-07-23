"""Held-out evaluation set -- disjoint from cotpt.data.TRAIN_TEXTS.
Original problems written for this project, arithmetic double-checked.
Each entry splits into a `question` (the prompt the model conditions on,
ending right after "A:") and an `answer` (the worked solution whose
log-likelihood gets measured) -- deliberately not a real benchmark; see
README for what a real evaluation would need beyond this.
"""

EVAL_PROBLEMS = [
    {
        "question": "Q: A farmer has 7 baskets with 12 apples in each basket. He gives away "
                    "20 apples. How many apples does he have left?\nA:",
        "answer": " 7 baskets of 12 apples is 84 apples total. Giving away 20 leaves 84 "
                  "minus 20, which is 64 apples.",
    },
    {
        "question": "Q: A movie theater sells tickets for 9 dollars each. If 34 tickets are "
                    "sold, how much money is collected?\nA:",
        "answer": " 34 tickets at 9 dollars each is 34 times 9, which is 306 dollars.",
    },
    {
        "question": "Q: A water tank holds 500 liters. It is currently 40 percent full. How "
                    "many more liters are needed to fill it completely?\nA:",
        "answer": " 40 percent of 500 liters is 200 liters currently in the tank. Filling "
                  "it completely needs 500 minus 200, which is 300 more liters.",
    },
    {
        "question": "Q: Sam runs 4 miles every morning and 3 miles every evening. How many "
                    "miles does he run in a full week?\nA:",
        "answer": " Each day Sam runs 4 plus 3, which is 7 miles. Over 7 days, that is 7 "
                  "times 7, which is 49 miles.",
    },
    {
        "question": "Q: A box of pencils contains 18 pencils. A school orders 15 boxes. How "
                    "many pencils is that in total?\nA:",
        "answer": " 15 boxes of 18 pencils is 15 times 18, which is 270 pencils.",
    },
    {
        "question": "Q: A recipe needs 250 grams of flour to make 5 cupcakes. How many grams "
                    "of flour are needed to make 20 cupcakes?\nA:",
        "answer": " 250 grams makes 5 cupcakes, so one cupcake needs 250 divided by 5, "
                  "which is 50 grams. For 20 cupcakes, that is 50 times 20, which is 1000 grams.",
    },
]

VISIBLE_COT_SUFFIX = " Let's think step by step."
