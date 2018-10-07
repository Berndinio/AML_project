import os
prefix = "7"
#naive_bayes
#os.system("screen -S "+prefix+"naive_bayes_0 -d -m -- sh -c 'date; exec $SHELL;'")
#os.system("screen -S "+prefix+"naive_bayes_0 -X stuff 'python3.5 -m scripts.naive_bayes --fMethod 0 ^M'")
#os.system("screen -S "+prefix+"naive_bayes_1 -d -m -- sh -c 'date; exec $SHELL;'")
#os.system("screen -S "+prefix+"naive_bayes_1 -X stuff 'python3.5 -m scripts.naive_bayes --fMethod 1 ^M'")


##naive_bayes_lib
#os.system("screen -S "+prefix+"naive_bayes_lib_0 -d -m -- sh -c 'date; exec $SHELL;'")
#os.system("screen -S "+prefix+"naive_bayes_lib_0 -X stuff 'python3.5 -m scripts.naive_bayes_lib --fMethod 0 ^M'")
#os.system("screen -S "+prefix+"naive_bayes_lib_1 -d -m -- sh -c 'date; exec $SHELL;'")
#os.system("screen -S "+prefix+"naive_bayes_lib_1 -X stuff 'python3.5 -m scripts.naive_bayes_lib --fMethod 1 ^M'")

#clustering - DBSCAN
for method in [0]:
    for metric in ["cosine"]:
        for eps in [0.01, 0.1, 0.5, 1, 10]:
            for minSamples in [2, 5, 10]:
                #os.system("screen -S "+prefix+"clustering_DBSCAN_"+metric+"_"+str(method)+"_"+str(eps)+"_"+str(minSamples)+" -d -m -- sh -c 'date; exec $SHELL;'")
                #os.system("screen -S "+prefix+"clustering_DBSCAN_"+metric+"_"+str(method)+"_"+str(eps)+"_"+str(minSamples)+" -X stuff 'python3.5 -m scripts.clustering --algorithm DBSCAN --eps "+str(eps)+" --minSamples "+str(minSamples)+" --fMethod "+str(method)+" --metric "+metric+" ^M'")
                pass
        #knn
        for numNeigh in [2,4,7,10]:
            #os.system("screen -S "+prefix+"kNN_"+metric+"_"+str(method)+"_"+str(numNeigh)+" -d -m -- sh -c 'date; exec $SHELL;'")
            #os.system("screen -S "+prefix+"kNN_"+metric+"_"+str(method)+"_"+str(numNeigh)+" -X stuff 'python3.5 -m scripts.kNN --numNeighbors "+str(numNeigh)+" --fMethod "+str(method)+" --metric "+metric+" ^M'")
            pass
        pass
    os.system("screen -S "+prefix+"clustering_kMeans_"+str(method)+" -d -m -- sh -c 'date; exec $SHELL;'")
    os.system("screen -S "+prefix+"clustering_kMeans_"+str(method)+" -X stuff 'python3.5 -m scripts.clustering --algorithm kMeans --fMethod "+str(method)+" ^M'")
