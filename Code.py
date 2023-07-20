import networkx as nx
import csv
import matplotlib.pyplot as plt
import numpy as np
import random
import time

V=0




INF = 99999


#Function to propagate the rumour
def propagate(source,arr,dp,t0):
	sourceLen=len(source)
	V=len(dp)
	for i in range(V):
		arr[i]=INF;
	for i in range(sourceLen):
		arr[source[i]]=t0
	for i in range(sourceLen):
		for j in range(V):
			arr[j]=min(arr[j],t0+dp[source[i]][j])
	return arr


def floydWarshall(graph):
	V=len(graph)

	dist = list(map(lambda i: list(map(lambda j: j, i)), graph))

	
	for k in range(V):

		# pick all vertices as source one by one
		for i in range(V):

			# Pick all vertices as destination for the
			# above picked source
			for j in range(V):

				# If vertex k is on the shortest path from
				# i to j, then update the value of dist[i][j]
				dist[i][j] = min(dist[i][j],
								dist[i][k] + dist[k][j]
								)
	
	return dist;



#Getting TimeStamp of all node by reverse traversing
def FindingTimeStamp(observer,TimeStamp,dp,arr):
	for i in range(len(observer)):
		for j in range(len(dp)):
			TimeStamp[i][j]=arr[observer[i]]-dp[observer[i]][j]
	return TimeStamp





# A utility function to print the solution
def printSolution(dist):
	print("Following matrix shows the shortest distances\
between every pair of vertices")
	for i in range(V):
		for j in range(V):
			if(dist[i][j] == INF):
				print("%7s" % ("INF"), end=" ")
			else:
				print("%7d" % (dist[i][j]), end=' ')
			if j == V-1:
				print()

def Print2d(arr):
	r=len(arr)
	c=len(arr[0])
	for i in range(r):
		for j in range(c):
			if(arr[i][j] == INF):
				print("%7s" % ("INF"), end=" ")
			else:
				print("%7d" % (arr[i][j]), end=' ')
			if j == c-1:
				print()

def print1d(dp):
		V=len(dp)
		for i in range(V):
			if(dp[i] == INF):
				print("%7s" % ("INF"), end=" ")
			else:
				print("%7d" % (dp[i]), end=' ')
			if i == V-1:
				print()

#function to get the index of source node 
def FindSourceNode(TimeStamp):
	print(" To find the minimum subset of columns that covers all the rows with a value of 1. This minimum subset is our answer")
	rows = len(TimeStamp)
	cols = len(TimeStamp[0])
	covered = set()
	result = []
	while len(covered) < rows:
		max_col = None
		max_count = 0
		for j in range(cols):
			if j not in result:
				count = 0
				for i in range(rows):
					if TimeStamp[i][j] == 1 and i not in covered:
						count += 1
				if count > max_count:
					max_col = j
					max_count = count
		if max_col is not None:
			result.append(max_col+1)
			for i in range(rows):
				if TimeStamp[i][max_col] == 1:
					covered.add(i)

	result.sort()
	print(result)
	return result
	

def CalculateScore(Result,Source):
	print("F1 score and othere measures for Our Algo are ")
	Result.sort()
	Source.sort()
	n1=len(Result)
	n2=len(Source)
	i=0
	j=0
	TP=0
	FP=0
	FN=0
	while i<n1 and j<n2:
		if(Result[i]==Source[j]):
			TP+=1
			i+=1
			j+=1
		elif(Result[i]<source[j]):
			FP+=1
			i+=1
		else:
			FN+=1
			j+=1
	if(i!=n1):
		FP=FP+(n1-i)
	if(j!=n2):
		FN=FN+(n2-j)
	print("True Positive are:")
	print(TP)
	print("False Positive are:")
	print(FP)
	print("False Negative are:")
	print(FN)
	Precision=TP/(TP+FP)
	Recall=TP/(TP+FN)
	# print("Precision is :")
	# print(Precision)
	# print("Recall is :")
	# print(Recall)
	if((Precision+Recall)==0):
		print("F1 Score Cannot be calculated as denomitor is 0")
	else:
		f1=2*(Precision*Recall)/(Precision+Recall)
		print("The F1 Score is ")
		print(f1)
	return


def DistanceError(Result,Source,dp):
	Result.sort()
	Source.sort()
	n1=len(Result)
	n2=len(Source)
	distance=0
	lim=min(n1,n2)
	if n2<n1:

		for i in range(n2):
			temp=INF
			for j in range(n1):
				temp=min(dp[Source[i]][Result[j]],temp)
			if(temp==INF):
				lim-=1
			else:
				distance+=temp
	else:
		for i in range(n1):
			temp=INF
			for j in range(n2):
				temp=min(dp[source[j]][Result[i]],temp)
			if(temp==INF):
				lin-=1
			else:
				distance+=temp
		
	return distance/lim


	
	

# Driver's code
if __name__ == "__main__":
  	graph = [[0, 3, INF,INF,INF,INF,INF,INF],
		[3,0,1,INF,2,1,2,INF],
		[INF,1,0,3,2,INF,INF,INF],
		[INF,INF,3,0,INF,INF,INF,INF],
		[INF,2,2,INF,0,3,INF,INF],
		[INF,1,INF,INF,3,0,2,3],
		[INF,2,INF,INF,INF,2,0,INF],
		[INF,INF,INF,INF,INF,3,INF,0]
		]
starttime=time.time()
g=nx.Graph()
with open("dolphins.csv", mode ='r')as file:
    csvFile = csv.DictReader(file)
    for line in csvFile:
       g.add_edge(int(line['Source'])-1,int(line['Target'])-1)


# Function call

#graph to 2d array A is a 2d Array of graph g
A = nx.adjacency_matrix(g).todense()

#node is the no of node in the graph
node=len(A)
print("This graph contain nodes:-")
print(node)
for i in range(node):
	for j in range(node):
		if(A[i][j]==0):
			A[i][j]=INF
		if(i==j):
			A[i][j]=0;
# print("The Graph is ")
# Print2d(A)
#dp contain shortest path length from all nodes to all other nodes
dp=floydWarshall(A)
# print("Shortest Distance between all nodes to all nodes")
# Print2d(dp)

#arr will contain the timestamp of all the node when the get infected
arr=[0]*node

sourcelen=10
observerlen=4
#source is the array containing all the source nodes
source=[0]*sourcelen
#observer array containing all the observer nodes 
observer=[0]*observerlen
for i in range(sourcelen):
	source[i]=random.randint(0,node-1)

for i in range(observerlen):
	observer[i]=random.randint(0,node-1)

source.sort()
observer.sort()
print("Sources are :")
print(source)
print("Observer are :")
print(observer)

#t0 is the initial timestamp of sources
t0=2;
arr=propagate(source,arr,dp,t0)

# print("The recieving time of all the node are")
# print1d(arr)


#Finding the timestamp of nodes using reverse diffusion
TimeStamp=[[0]*node for _ in range(len(observer))]
TimeStamp=FindingTimeStamp(observer,TimeStamp,dp,arr)

# print("Printing the timestamp of all node found by reverse traversal")
# Print2d(TimeStamp)

# Now we will calculate Maximum Timestamp of all node with respect to all observer
MaximumTimeStamp=[-INF]*node
# print1d(MaximumTimeStamp)
MinTimeStamp=INF;
for i in range(len(observer)):
	for j in range(node):
		MaximumTimeStamp[j]=max(MaximumTimeStamp[j],TimeStamp[i][j])
		
for i in range(node):
	MinTimeStamp=min(MinTimeStamp,MaximumTimeStamp[i]);

# print("The Maximum Timestamp of all the nodes are")
# print(MaximumTimeStamp)
print("According MMM the source node are are the nodes with maximum Maximum Timestamp")
ResultMMM=[]
for i in range(node):
	if(MaximumTimeStamp[i]==MinTimeStamp):
		ResultMMM.append(i)
print(ResultMMM)
# #we are deviding the node in two sets one which may contain source node and another which will not
check=[0]*node
minObserverTime=INF
for i in range(len(observer)):
	minObserverTime=min(minObserverTime,arr[observer[i]])

# print("Minimum Observe Time is")
# print(minObserverTime)

for i in range(node):
	if(MaximumTimeStamp[i]<=minObserverTime):
		check[i]=1
# print("Since the Source node can have initial time greater then minimum timestamp of observer.Possible source node are")
# for i in range(node):
# 	if(check[i]==1):
# 		print(i)
print("After applying the Integer programming the Timestamp matrix will be translated to")
for i in range(len(observer)):
	for j in range(node):
		if(check[j]==0):
			TimeStamp[i][j]=0
		else:
			if(TimeStamp[i][j]==MaximumTimeStamp[j]):
				TimeStamp[i][j]=1;
			else:
				TimeStamp[i][j]=0;
# Print2d(TimeStamp)
result=FindSourceNode(TimeStamp)
CalculateScore(result,source)
endtime=time.time()
print("The Distance Error for this input is ",DistanceError(result,source,dp))
print("Time taken for this is =", endtime-starttime,"seconds")
# CalculateScoreMMM(ResultMMM,source)