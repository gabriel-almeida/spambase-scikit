import pickle
from classifier import naive_bayes, naive_bayes_custom, svm, knn
from numpy import diag, sum, std, mean, var


def analize_file(f_name="results.pickle"):
	results, configuration_labels = pickle.load( open(f_name, 'rb') )
	tab_analize(results, configuration_labels)

def analize(results, configuration_labels=[]):
	for i in results.keys():
		print()
		if (len(configuration_labels) >= i):
			print(configuration_labels[i])
		else:
			print("Configuration", i)
		print("========================")
		for alg in results[i]:
			accuracy = [ sum(diag(cm)) for cm in results[i][alg] ]
			accuracy_mean = mean(accuracy)
			accuracy_std = std(accuracy)
			sample_size = len(results[i][alg])
			print(alg.__name__, ": Accuracy=", accuracy_mean,\
				 "Standard Deviantion:", accuracy_std, "Sample size", sample_size )

			cm_mean = mean(results[i][alg], axis=0)
			print(cm_mean)
			print()
def tab_analize(results, configuration_labels=[]):
	f= open("output.csv", 'w')
	for i in results.keys():
		if (len(configuration_labels) >= i):
			conf = (configuration_labels[i])
		else:
			conf = "Configuration" + i
		for alg in results[i]:
			accuracy = [ sum(diag(cm)) for cm in results[i][alg] ]
			accuracy_mean = mean(accuracy)
			accuracy_std = std(accuracy)

			#sample_size = len(results[i][alg])

			cm_mean = mean(results[i][alg], axis=0)
			
			print(conf, alg.__name__, accuracy_mean,\
			  accuracy_std, cm_mean[0,0], cm_mean[0,1], cm_mean[1,0], cm_mean[1,1], file=f, sep='\t')
	f.close()

if __name__ == "__main__":
	analize_file()