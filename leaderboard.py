#!/usr/bin/env python

'''
Displays leaderboard for CS 446 Final Project. 
'''

import requests

IS_CSV = False

leaderboardUrl = 'http://104.43.250.70/'
page = requests.get(leaderboardUrl)
lines = page.content.split('\n')

if IS_CSV:
	print('Rank,NetID,RMSE')
else:
	print("{0:6s} {1:13s} {2:10s}".format('Rank', 'NetID', 'RMSE'))

for index in range(len(lines)):
	lineParts = lines[index].split(': ')
	if len(lineParts) > 1:
		if IS_CSV:
			print("{0},{1},{2}".format(index + 1, lineParts[0], lineParts[1]))
		else:
			print("{0:02d}     {1:13s} {2:9.3f}".format(index + 1, lineParts[0], float(lineParts[1][0:10])))

print('')