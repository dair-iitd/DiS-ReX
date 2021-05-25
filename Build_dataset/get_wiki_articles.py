import sys
from gensim.corpora import WikiCorpus

import sys
import os
import multiprocessing

from gensim.corpora.wikicorpus import * 

def tokenize(content):
    # override original method in wikicorpus.py
    return [token for token in content.split()
            if len(token) <= 15 and not token.startswith('_')]

def process_article(args):
   # override original method in wikicorpus.py
    text, lemmatize, title, pageid = args
    text = filter_wiki(text)
    result = tokenize(text)
    return result, title, pageid


class MyWikiCorpus(WikiCorpus):
    def __init__(self, fname, processes=None, lemmatize=utils.has_pattern(), dictionary=None, filter_namespaces=('0',)):
        WikiCorpus.__init__(self, fname, processes, lemmatize, dictionary, filter_namespaces)

    def get_texts(self):
        articles, articles_all = 0, 0
        positions, positions_all = 0, 0
        texts = ((text, self.lemmatize, title, pageid) for title, text, pageid in extract_pages(bz2.BZ2File(self.fname), self.filter_namespaces))
        pool = multiprocessing.Pool(self.processes)
        # process the corpus in smaller chunks of docs, because multiprocessing.Pool
        # is dumb and would load the entire input into RAM at once...
        for group in utils.chunkize(texts, chunksize=10 * self.processes, maxsize=1):
            for tokens, title, pageid in pool.imap(process_article, group):  # chunksize=10):
                articles_all += 1
                positions_all += len(tokens)
                # article redirects and short stubs are pruned here
                if len(tokens) < ARTICLE_MIN_WORDS or any(title.startswith(ignore + ':') for ignore in IGNORED_NAMESPACES):
                    continue
                articles += 1
                positions += len(tokens)
                if self.metadata:
                    yield (tokens, (pageid, title))
                else:
                    yield tokens
        pool.terminate()

        logger.info(
            "finished iterating over Wikipedia corpus of %i documents with %i positions"
            " (total %i articles, %i positions before pruning articles shorter than %i words)",
            articles, positions, articles_all, positions_all, ARTICLE_MIN_WORDS)
        self.length = articles  # cache corpus length

def make_corpus(in_f, out_f):

	"""Convert Wikipedia xml dump file to text corpus"""

	output = open(out_f, 'w')
	wiki = MyWikiCorpus(in_f)

	i = 0
	for text in wiki.get_texts():
		output.write(bytes(" ".join(text),'utf-8').decode('utf-8') + '\n')
		i = i + 1
		if (i % 10000 == 0):
			print('Processed ' + str(i) + ' articles')
	output.close()
	print('Processing complete!')


if __name__ == '__main__':

	if len(sys.argv) != 3:
		print('Usage: python make_wiki_corpus.py <wikipedia_dump_file> <processed_text_file>')
		sys.exit(1)
	in_f = sys.argv[1]
	out_f = sys.argv[2]
	make_corpus(in_f, out_f)
