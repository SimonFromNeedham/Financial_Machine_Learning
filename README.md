This module includes 2 distinct methods of artificially picking stocks:

The first is a supervised learning approach that predicts weather a security’s price will go up or down using the following steps:

  generator.py uses AlphaVantage API to gather information (e.g. RSI, MACD, and other indicators) about a set of stocks
  svm_trainer.py uses that information to train a variety of standard ML models, each using different indicators as features
  svm_tester.py tests that model’s accuracy on another set of stocks, creating models for each combination of indicators
  
This approach is mostly theoretical, as common technical analysis indicators are unlikely to generate market breaking results due to them being widely available. 
Even so, it provides a solid framework for future problems and could prove valuable if the user has an original way to measure a stock’s stability or if they alter the program to make it unsupervised.

The second approach is an exhaustive search through every stock listed on NASDAQ (etf.txt) to find stocks with the highest yields (yield_picker.py). 
This approach is more immediately practical because it generates stocks with very high yields (after discounting the top 20 or so stocks for having outlier dividends, the user can find others with %10+ yields) and can be changed to search for any attribute (e.g. P/E ratio).

Before you run any of the programs, make sure you install uncommonly used modules like sklearn, yfinance, and alpha_vantage to avoid errors.

If you find any bugs in the program, please let me know at traubsimon0@gmail.com, I am always looking to improve!
