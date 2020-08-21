import glassdoor_scraper as gs
import pandas as pd
path = '/Users/oluyemijegede/Downloads/chromedriver'
df = gs.get_jobs('data scientist', 5, False, path, 15)