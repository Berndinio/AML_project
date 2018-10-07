# extract the corpus from project gutenberg

from gutenberg.acquire import load_etext
from gutenberg.cleanup import strip_headers
from gutenberg.query import get_metadata

import time

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

for i in range(1, 57700):
    time.sleep(1)
    print(i)
    try:
        if get_metadata('language', i) == frozenset({'en'}):
            path = "./corpus/" + str(i) + ".txt"
            with open(path, "w") as file:
                file.write(strip_headers(load_etext(i)).strip())
    except:
        print("Error at index " + str(i) + " probably no file with id " + str(i) + " was found. Skipped.")
        with open("./log/corpus_creation.txt", 'a+') as logfile:
            logfile.write("Error at index " + str(i) + " probably no file with id " + str(i) + " was found. Skipped.\n")
