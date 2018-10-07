# extract *all* labels from project gutenberg

from gutenberg.query import get_metadata

# when running this code for the first time, you have to create a cache of the meta data. This may take some time
# (according to the author of the package - compare https://pypi.org/project/Gutenberg/ -, it took 18hrs on his machine,
# on mine it was less than 4 - I didn't check earlier, because I expected it would be around 18 as well.
#
# from gutenberg.acquire import get_metadata_cache
#
# cache = get_metadata_cache()
# cache.populate()

# list supported types of metadata
# from gutenberg.query import list_supported_metadatas
# print(list_supported_metadatas())

results = {}

for i in range(1, 57700):
    try:
        if get_metadata('language', i) == frozenset({'en'}):
            print(i)
            labels = get_metadata('subject', i)
            results[i] = labels
    except:
        print("extracting labels: Error at index " + str(i) + " probably no file with id " + str(i) + " was found. Skipped.")
        with open("./log/label_extraction.txt", 'a+') as logfile:
            logfile.write("extracting labels: Error at index " + str(i) + " probably no file with id " + str(i) + " was found. Skipped.\n")

path = "../data/labels.txt"
with open(path, "w") as file:
    file.write(str(results))
