# About this repo
It is a retrospective report about research that was conducted by me. There i will upload report, code, and some tables and graphs about my research. Comms in code and graphs will have Russian text because Russian was more comfortable for me to write. Report will be on English because i want to write report on English.
# Context of Research
My researche theme was "Factors of Success in Indie Games: A Data-Driven Analysis of Steam Reviews" \. In this research I wanted to find some commons factors between "successfull indie games". I was interested in these theme because I had a simple question: Why some indie games cannot success? I have been interested in indie games and still am, so I decided to conduct this research for my FPIS (Field Projects in Information Systems) course.
My goal was to find success factors of indie games if they exist.
# Initial hypothesis
I had 4 initial hypothesis:

H1: Games with popular feature tags (for example Roguelike, Survival, Multiplayer) receive more positive reviews.  
H2: Most popular games have same tag combinations in common.  
H3: Games with unpopular combinations of tags are less popular. 
H4: Games with more reviews have more online players. 

Initially i thought that i will have at least some simmilar tags or genres that likely will have more chances to become "successfull".
# Methodology

To investigate statistical factors behind the “success” of indie games, I collected a large dataset containing game-related and user-generated features. These included game metadata (tags, genres, game names), review statistics (number of positive and negative reviews, review text), and user-level attributes (user language, number of owned games, whether the user voted up the game, helpfulness and funny votes, playtime, and review timestamps).

Although the dataset contained many variables, not all of them were used in the final analysis.

# Definition of “success”:
Game success was measured using the Wilson score interval, which provides a more reliable estimate of proportions, especially in cases of small or uneven sample sizes. This approach was chosen because it reduces bias compared to simple average ratings.

Only games with between 50 and 100 English-language reviews were included in the analysis. Games with fewer than 50 reviews were excluded to ensure statistical reliability. An attempt was made to collect up to 100 reviews per game.

To enable meaningful comparisons, games were divided into two groups based on the median Wilson score.
# Key findings
- The strongest negative influence on game ratings was related to technical issues (bugs, performance problems, and optimization).
- Tags and genres showed little to no significant influence on game ratings in this dataset.
- Identifying factors of “success” likely requires deeper analysis of additional game metadata beyond basic tags and review statistics.
# Tech stack
- Python
- Pandas
- Numpy
- Scikit-learn
- Maptolib
- Seaborn
# Report
Full report is available here:
- [Open report](Report.pdf)
