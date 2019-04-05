from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from scipy.special import psi, gammaln, gamma, digamma, polygamma
from utils import sample_data_random, sample_probabilities
import scipy


def read(path):
    file = open(path, "r")
    docs = file.readlines()
    print("Read {} documents".format(len(docs)))
    return docs


def bow(docs):
    '''

    :param docs:
    :return: vectorizer, data MxV
    '''
    cv = CountVectorizer()
    X = cv.fit_transform(docs)
    return cv, X.toarray()


class VariationalInference():
    def __init__(self, corpus=[], stopwords=[], K=2, alpha_init=1, V=0, M=0):
        self.__vocabulary, self.__rev_vocabulary = self.__parse_vocab(corpus, stopwords)  # [word: id], [id: word]
        self.__corpus_ids, self.__corpus_counts = self.__parse_data(corpus)  # documents with id words

        self.M = len(self.__corpus_ids)  # number of documents
        self.V = len(self.__vocabulary)  # vocabulary size
        self.K = K  # number of topics

        if len(corpus) is 0:
            self.V = V
            self.M = M

        # corpus parameters
        self.alpha = np.full((1, self.K), alpha_init)
        # self.beta = np.tile([1.0/ self.V] * self.V, (self.K, 1)) # rows (topics) sum up to 1 (KxV)
        # self.beta = np.random.gamma(100., 1. / 100., (self.K, self.V));
        self.beta = np.random.random((self.K, self.V))
        self.beta /= self.beta.sum(axis=1)[:, np.newaxis]

        # variational parameters
        self.gamma = np.zeros((self.M, self.K))

    def __parse_vocab(self, corpus, stop_words):
        '''
        A word is an item from the vocabulary indexed by {1, ..., V}.
        E.g.
        1. find a word index: index=word_index['a']
        2. find an word: word = index_word[3]
        :param corpus: list of raw set of documents
        :param stop_words: list of words to avoid
        :return: return lists which are the vocabulary and reversed vocabulary
        '''
        word_index = {}
        index_word = {}
        for doc in corpus:
            for word in doc.split():
                if word in stop_words:
                    continue

                if word not in word_index:
                    word_index[word] = len(word_index)
                    index_word[len(index_word)] = word
        return word_index, index_word

    def __parse_data(self, corpus):
        '''
        Map corpus with its vocabulary and get the counts of words.
        :param corpus: list of raw set of documents
        :return: list of corpus index by vocabulary, corpus counts, and total of documents
        '''
        corpus_id = []
        corpus_word = []

        corpus_count = 0

        for doc in corpus:
            doc_word_count = {}
            for word in doc.split():
                if word not in self.__vocabulary:
                    continue

                word_id = self.__vocabulary[word]
                if word_id not in doc_word_count:
                    doc_word_count[word_id] = 0
                doc_word_count[word_id] += 1
            corpus_id.append(np.array(list(doc_word_count.keys())))
            corpus_word.append(np.array(list(doc_word_count.values())))

            corpus_count += 1

        print("Parsed {} documents".format(corpus_count))

        return corpus_id, corpus_word

    def expectation(self, corpus_ids, corpus_counts, beta, max_iterations=50, threshold=1e-10, verbose=False):
        corpus_loglikelihood = np.zeros(self.M)

        # sufficient statistics
        self.phi_ss = np.zeros((self.K, self.V))
        e = 0

        for d in range(self.M):
            doc_ids = corpus_ids[d]
            doc_counts = corpus_counts[d]
            N = len(doc_ids)

            # initialize gamma
            self.gamma[d] = self.alpha + N * 1.0 / self.K
            log_phi = np.log(np.full((doc_ids.shape[0], self.K), 1. / self.K)) # doc_ids.shape[0]=N

            # estimate variational parameters
            it = 0
            converged = 1
            # while change > threshold:
            while converged > threshold and it < max_iterations:
                gamma_old = self.gamma[d].copy()
                for n in range(N):
                    for i in range(self.K):
                        log_phi[n][i] = np.log(beta[i, doc_ids[n]]) + digamma(self.gamma[d][i])
                    log_phi[n] -= np.log(np.sum(np.exp(log_phi[n])))  # normalize to sum 1

                self.gamma[[d]] = self.alpha + np.exp(log_phi + np.log(doc_counts[:, np.newaxis])).sum(axis=0)

                # check convergence
                converged = np.max(np.abs(self.gamma[d] - gamma_old))
                it += 1

            # TODO: Check the right calculation for the likelihood
            # log likelihood - minka
            q = beta.T * np.exp(digamma(self.gamma[[d]]))
            q /= q.sum(axis=1, keepdims=True)
            e += gammaln(np.sum(self.alpha)) - gammaln(np.sum(self.gamma[[d]]));
            e += np.sum(gammaln(self.gamma[[d]])) - np.sum(gammaln(self.alpha));
            r = np.log(beta.T / q)
            e = e + np.sum(doc_counts * (q * r).sum(axis=1)[doc_ids])

            # blei simplify bound
            # compute the alpha terms
            log_likelihood = scipy.special.gammaln(np.sum(self.alpha)) - np.sum(scipy.special.gammaln(self.alpha))
            # compute the gamma terms
            log_likelihood += np.sum(scipy.special.gammaln(self.gamma[d])) - scipy.special.gammaln(
                np.sum(self.gamma[d]));
            # compute the phi terms
            log_likelihood -= np.sum(np.dot(doc_counts, np.exp(log_phi) * log_phi));

            # blei complete eq
            # # compute log-likelihood
            # log_likelihood = gammaln(self.alpha.sum()) - gammaln(self.alpha).sum() + ((self.alpha - 1) * (digamma(self.gamma[d]) - digamma(self.gamma[d].sum()))).sum()
            # log_likelihood += gammaln(self.gamma[d].sum()) - gammaln(self.gamma[d]).sum() + + ((self.gamma[d] - 1) * (digamma(self.gamma[d]) - digamma(self.gamma[d].sum()))).sum()
            # for n in range(N):
            #     for i in range(self.K):
            #         log_likelihood += np.exp(log_phi[n, i]) * doc_counts[n] * log_phi[n, i] * np.log(beta[i, doc_ids[n]]) * (digamma(self.gamma[d][i]) - digamma(self.gamma[d].sum()))

            if verbose:
                print("doc ({}), converged at {} after {} iterations. [likelihood: {}] [converged: {}]".format(d, it,
                                                                                                               log_likelihood,
                                                                                                               converged))

            corpus_loglikelihood[d] = log_likelihood
            self.phi_ss[:, doc_ids] += np.exp(log_phi + np.log(doc_counts[:, np.newaxis])).T

        # return corpus_loglikelihood.sum()
        return e

    def maximization(self):
        # alpha_ss = (digamma(self.gamma) - digamma(self.gamma.sum(axis=1)[:, np.newaxis])).sum(axis=0)
        alpha_ss = (digamma(self.gamma) - digamma(self.gamma.sum(axis=1)[:, np.newaxis])).sum()
        new_alpha = self.newton_rapson(alpha_ss, self.M, self.K)

        alpha = np.full((1, self.K), new_alpha)
        beta = np.exp(np.log(self.phi_ss) - np.log(self.phi_ss.sum(axis=1))[:, np.newaxis])

        return alpha, beta

    def newton_rapson(self, ss, D, K):
        init_alpha = 100;
        iter = 0;
        df = 2

        log_a = np.log(init_alpha);

        while ((np.abs(df) > 1e-5) and (iter < 1000)):
            iter += 1
            a = np.exp(log_a);

            f = (D * (gammaln(K * a) - K * gammaln(a)) + (a - 1) * ss);
            df = (D * (K * digamma(K * a) - K * digamma(a)) + ss)
            d2f = (D * (K * K * polygamma(1, K * a) - K * polygamma(1, a)))
            log_a = log_a - df / (d2f * a + df);
            print("alpha maximization : {}   {}", f, df);
        return (np.exp(log_a));

    def newton_rapson1(self, alpha_sufficient_statistics, hyper_parameter_iteration=100,
                       hyper_parameter_decay_factor=0.9, hyper_parameter_maximum_decay=10,
                       hyper_parameter_converge_threshold=1e-6):
        # assert(alpha_sufficient_statistics.shape == (1, self._number_of_topics));
        assert (alpha_sufficient_statistics.shape == (self.K,));
        alpha_update = self.alpha;

        decay = 0;
        for alpha_iteration in range(hyper_parameter_iteration):
            alpha_sum = np.sum(self.alpha);
            alpha_gradient = self.M * (scipy.special.psi(alpha_sum) - scipy.special.psi(
                self.alpha)) + alpha_sufficient_statistics;
            alpha_hessian = -self.M * scipy.special.polygamma(1, self.alpha);

            if np.any(np.isinf(alpha_gradient)) or np.any(np.isnan(alpha_gradient)):
                print
                "illegal alpha gradient vector", alpha_gradient

            sum_g_h = np.sum(alpha_gradient / alpha_hessian);
            sum_1_h = 1.0 / alpha_hessian;

            z = self.M * scipy.special.polygamma(1, alpha_sum);
            c = sum_g_h / (1.0 / z + sum_1_h);

            # update the alpha vector
            while True:
                singular_hessian = False

                step_size = np.power(hyper_parameter_decay_factor, decay) * (alpha_gradient - c) / alpha_hessian;
                # print "step size is", step_size
                assert (self.alpha.shape == step_size.shape);

                if np.any(self.alpha <= step_size):
                    singular_hessian = True
                else:
                    alpha_update = self.alpha - step_size;

                if singular_hessian:
                    decay += 1;
                    if decay > hyper_parameter_maximum_decay:
                        break;
                else:
                    break;

            # compute the alpha sum
            # check the alpha converge criteria
            mean_change = np.mean(abs(alpha_update - self.alpha));
            self.alpha = alpha_update;
            if mean_change <= hyper_parameter_converge_threshold:
                break;

        return 0

    def EM(self, threshold=1e-4, max_iterations=100):
        likelihood_old = 0.1
        converged = 1
        it = 0

        while converged > threshold and it < max_iterations:
            log_likelihood = self.expectation(self.__corpus_ids, self.__corpus_counts, self.beta, verbose=False)
            self.alpha, self.beta = self.maximization()

            converged = (likelihood_old - log_likelihood) / (likelihood_old)
            likelihood_old = log_likelihood
            it += 1

            print("log-likelihood={} converged={}".format(log_likelihood, converged))
        self.expectation(self.__corpus_ids, self.__corpus_counts, self.beta)

        return self.alpha, self.beta

    def print_topics(self, max_show=10):
        for k in range(self.K):
            print("Topic {}".format(k))
            max = 0
            for type_index in reversed(np.argsort(self.beta[k, :])):
                print("\t{} ({})".format(self.__rev_vocabulary[type_index], self.beta[k, type_index]))
                max += 1
                if max == max_show:
                    break


def parse_docs(docs):
    for doc in docs:
        doc_word_id = {}
        doc_id_word = {}
        for token in doc.split():
            if token not in doc_word_id:
                doc_word_id[token] = len(doc_word_id)
                doc_id_word[len(doc_id_word)] = token

        print(doc_id_word)
        print(doc_word_id)


if __name__ == "__main__":
    raw_docs = read('data/data_small1.txt');
    VariationalInference(raw_docs).EM()
