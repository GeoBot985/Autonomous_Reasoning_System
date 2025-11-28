import os
os.environ['PYTHONPATH'] = 'src'
from Autonomous_Reasoning_System.tools.web_search import perform_google_search
print(perform_google_search('Springboks next match'))
